# 目标跟踪

## 任务分类

| 类型 | 说明 |
|------|------|
| SOT（单目标跟踪） | 给定初始框，跟踪整个序列 |
| MOT（多目标跟踪） | 同时跟踪视频中所有目标 |
| MOTS | 多目标跟踪 + 分割 |

## SOT 主流方法

| 模型 | 特点 |
|------|------|
| SiamFC | 孪生网络相似度匹配 |
| SiamRPN++ | 加 RPN，精度提升 |
| OSTrack | Transformer 一体化跟踪 |
| MixFormer | 混合注意力，SOTA |

## MOT 主流方法

| 模型 | 特点 |
|------|------|
| SORT | Kalman + 匈牙利算法，速度快 |
| DeepSORT | SORT + ReID 特征，抗 ID 切换 |
| ByteTrack | 低置信度检测框也参与匹配，SOTA |
| StrongSORT | DeepSORT 改进版 |
| BoT-SORT | ByteTrack + 相机运动补偿 |

---

## 卡尔曼滤波在目标跟踪中的状态方程

状态向量：`[cx, cy, s, r, vcx, vcy, vs]`（中心坐标、面积、宽高比及其速度）

```python
import numpy as np

class KalmanBoxTracker:
    """使用卡尔曼滤波对单个目标的边界框进行状态预测"""
    count = 0

    def __init__(self, bbox):
        """bbox: [x1, y1, x2, y2]"""
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 状态转移矩阵 F（匀速运动模型）
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        # 观测矩阵 H（只观测位置，不观测速度）
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        # 观测噪声协方差 R
        self.kf.R[2:, 2:] *= 10.
        # 过程噪声协方差 Q（速度分量不确定性更高）
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        # 初始化状态
        self.kf.x[:4] = self._xyxy_to_z(bbox)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak = 0
        self.age = 0

    @staticmethod
    def _xyxy_to_z(bbox):
        """[x1,y1,x2,y2] → [cx, cy, s, r]（面积、宽高比）"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        s = w * h
        r = w / (h + 1e-9)
        return np.array([[cx], [cy], [s], [r]])

    def predict(self):
        """预测下一帧位置"""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        return self._z_to_xyxy(self.kf.x)

    def update(self, bbox):
        """用新检测结果更新状态"""
        self.hit_streak += 1
        self.kf.update(self._xyxy_to_z(bbox))

    def _z_to_xyxy(self, x):
        w = np.sqrt(abs(x[2] * x[3]))
        h = x[2] / (w + 1e-9)
        return np.array([x[0] - w/2, x[1] - h/2, x[0] + w/2, x[1] + h/2]).flatten()
```

---

## ByteTrack：两阶段匹配算法

ByteTrack 的核心创新是**不丢弃低置信度检测框**，而是分两阶段进行匹配：

```
第一阶段：高置信度检测框（conf ≥ 0.5）+ 所有活跃轨迹
   ↓ IoU 匹配（匈牙利算法）
   → 匹配成功的轨迹：更新卡尔曼滤波
   → 未匹配轨迹（lost）+ 未匹配高置信度框（new_track）

第二阶段：低置信度检测框（0.1 ≤ conf < 0.5）+ 未匹配的 lost 轨迹
   ↓ IoU 匹配（更严格阈值）
   → 匹配成功：轨迹复活（避免遮挡导致 ID 切换）
   → 仍未匹配的低置信度框：直接丢弃（非目标）
```

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

def byte_track_step(trackers, detections, high_thresh=0.5, low_thresh=0.1,
                    iou_threshold_1=0.3, iou_threshold_2=0.5):
    """
    trackers:   活跃轨迹列表
    detections: [(bbox, score), ...]
    """
    # 分离高/低置信度检测框
    high_dets = [(b, s) for b, s in detections if s >= high_thresh]
    low_dets  = [(b, s) for b, s in detections if low_thresh <= s < high_thresh]

    # 第一阶段：高置信度 + 所有轨迹
    trk_preds = [t.predict() for t in trackers]
    matched_h, unmatched_trks, unmatched_high = \
        iou_matching(trk_preds, [d[0] for d in high_dets], iou_threshold_1)

    # 更新匹配成功的轨迹
    for trk_idx, det_idx in matched_h:
        trackers[trk_idx].update(high_dets[det_idx][0])

    # 第二阶段：低置信度 + 未匹配轨迹
    lost_trks = [trackers[i] for i in unmatched_trks]
    lost_preds = [trk_preds[i] for i in unmatched_trks]
    matched_l, still_lost, _ = \
        iou_matching(lost_preds, [d[0] for d in low_dets], iou_threshold_2)

    for trk_idx, det_idx in matched_l:
        lost_trks[trk_idx].update(low_dets[det_idx][0])

    # 新建轨迹（第一阶段未匹配的高置信度框）
    new_trackers = [KalmanBoxTracker(high_dets[i][0]) for i in unmatched_high]

    return trackers + new_trackers

def iou_matching(pred_boxes, det_boxes, threshold):
    """基于 IoU 的匈牙利匹配，返回匹配对、未匹配轨迹索引、未匹配检测索引"""
    if len(pred_boxes) == 0 or len(det_boxes) == 0:
        return [], list(range(len(pred_boxes))), list(range(len(det_boxes)))

    iou_matrix = compute_iou_matrix(pred_boxes, det_boxes)
    cost = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matched = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= threshold]
    matched_rows = {m[0] for m in matched}
    matched_cols = {m[1] for m in matched}
    unmatched_trks = [i for i in range(len(pred_boxes)) if i not in matched_rows]
    unmatched_dets = [i for i in range(len(det_boxes)) if i not in matched_cols]
    return matched, unmatched_trks, unmatched_dets
```

---

## DeepSORT：ReID 特征 + Mahalanobis 距离

DeepSORT 在 SORT 基础上引入 ReID 外观特征和马氏距离，有效抑制 ID 切换：

```python
import numpy as np

class DeepSORTTracker:
    """简化版 DeepSORT 核心逻辑"""

    def __init__(self, reid_model, max_age=70, n_init=3):
        self.reid_model = reid_model  # 提取 128-d 外观特征
        self.max_age = max_age         # 轨迹最多存活帧数
        self.n_init = n_init           # 至少命中 n_init 次才确认轨迹
        self.tracks = []

    def extract_reid_features(self, image, bboxes):
        """从检测框中裁剪图像块，提取 ReID 特征"""
        crops = [image[int(y1):int(y2), int(x1):int(x2)]
                 for x1, y1, x2, y2 in bboxes]
        return self.reid_model(crops)  # [N, 128]

    def mahalanobis_distance(self, track, detection_bbox):
        """
        计算轨迹卡尔曼预测与检测框之间的马氏距离
        用于过滤状态空间中不一致的匹配（位置约束）
        """
        mean = track.kf.x[:4].flatten()   # 预测的观测均值
        cov  = track.kf.P[:4, :4]          # 预测的观测协方差
        z    = convert_to_measurement(detection_bbox)
        d    = z - mean
        # 马氏距离 = d^T * S^{-1} * d
        S = np.linalg.inv(cov)
        return float(d @ S @ d)

    def cosine_distance(self, feat1, feat2):
        """外观特征余弦距离（越小越相似）"""
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-9)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-9)
        return 1 - feat1 @ feat2

    def gate_cost_matrix(self, cost_matrix, tracks, detections,
                         maha_threshold=9.4877):
        """使用马氏距离门控代价矩阵，过滤不合理匹配"""
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                maha = self.mahalanobis_distance(track, det['bbox'])
                if maha > maha_threshold:
                    cost_matrix[i, j] = 1e9  # 不可匹配
        return cost_matrix
```

**踩坑记录**：
- ReID 模型对裁剪框质量敏感，检测框过小（< 20px）时特征不可靠，建议过滤
- 马氏距离阈值 9.4877 对应 95% 置信区间（卡方分布 df=4），直接使用即可
- 长遮挡后 ID 切换是 DeepSORT 主要失败模式，ByteTrack 在该场景更鲁棒

---

## MOT 评估工具：TrackEval 使用说明

```bash
# 安装
git clone https://github.com/JonathonLuiten/TrackEval.git
cd TrackEval && pip install -r requirements.txt

# 数据目录结构
# data/gt/mot_challenge/MOT17-train/MOT17-02/gt/gt.txt
# data/trackers/mot_challenge/MOT17-train/my_tracker/data/MOT17-02.txt

# 评估（输出 MOTA、IDF1 等全部指标）
python scripts/run_mot_challenge.py \
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL my_tracker \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1
```

输出结果字段说明：

| 指标 | 含义 |
|------|------|
| HOTA | 综合检测与关联质量，更全面 |
| MOTA | 多目标跟踪精度（FP/FN/ID Switch 惩罚） |
| IDF1 | ID 级别的 F1（关联一致性） |
| IDs | ID 切换次数（越低越好） |
| FPS | 推理速度 |

## 评估指标（MOT）

- **MOTA**：多目标跟踪精度
- **MOTP**：多目标跟踪准确度
- **IDF1**：ID 一致性 F1
- **IDs**：ID 切换次数（越低越好）
