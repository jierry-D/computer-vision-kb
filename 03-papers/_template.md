# 论文标题

> **作者**：  
> **发表**：（会议/期刊，年份）  
> **链接**：[arxiv]() | [代码]() | [项目页]()

---

## 一句话总结

<!-- 用一句话说清楚这篇论文解决了什么问题、怎么解决的 -->

---

## 背景与动机

<!-- 之前的方法有什么不足？这篇论文为什么要做这个工作？ -->

---

## 核心方法

<!-- 方法的关键设计，可配图说明 -->

### 整体框架

### 关键模块

---

## 实验结果

<!-- 在哪些数据集上测试？与哪些方法对比？关键指标是多少？ -->

| 数据集 | 指标 | 本方法 | 对比方法 |
|--------|------|--------|---------|
| | | | |

---

## 优点

- 

## 局限性

- 

---

## 个人思考

<!-- 这篇论文对你的工作有什么启发？有哪些可以改进或延伸的方向？ -->

---

## 相关论文

- 

---

## 论文阅读方法：三遍读法

高效阅读一篇论文建议遵循"三遍读法"，从整体到细节，避免陷入细节泥潭。

### 第一遍：快速扫描（5-10 分钟）

目标：判断是否值得深读，搞清楚论文的大方向。

- 阅读标题、摘要、结论
- 看一遍所有图表（Figure/Table），理解主要结果
- 略读各节的小标题和第一段
- 完成后能回答：这篇论文做了什么？结论是什么？

### 第二遍：细读主体（1-2 小时）

目标：理解方法和实验，但可以跳过推导细节。

- 认真阅读 Introduction、Method、Experiments
- 记录不理解的地方，先跳过，不纠结
- 重点关注核心模块的设计动机
- 关注实验设置（消融实验说明了什么？）
- 完成后能回答：怎么做的？为什么这样设计？

### 第三遍：精读复现（3-5 小时，视需要）

目标：完全理解，能够复现或改进。

- 深入理解每个公式推导
- 对照代码逐行理解实现
- 思考每个设计决策的必要性
- 尝试找出方法的局限性和改进空间

💡 **Tips**：初学阶段不必每篇都读到第三遍，第一遍筛选、第二遍精读高价值论文即可。

---

## 模板填写示例（以 ResNet 为例）

以下是一个完整填写的示例，可供参考格式和详细程度：

---

### 示例：Deep Residual Learning for Image Recognition（ResNet）

> **作者**：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
> **发表**：CVPR 2016（Best Paper Award）  
> **链接**：arxiv 搜索 "1512.03385" | 代码：torchvision.models.resnet

#### 一句话总结

通过在网络中引入**残差连接（Skip Connection）**，使得极深网络（100+ 层）也能稳定训练，在 ImageNet 上以 3.57% top-5 错误率夺冠。

#### 背景与动机

**问题**：随着网络深度增加，出现"退化问题"——更深的网络在训练集上的误差反而比浅网络更高，这并非过拟合，而是优化困难。  
**动机**：如果能让深层网络至少学到恒等映射（identity mapping），就不会比浅网络差。

#### 核心方法

**整体框架**：堆叠残差块（Residual Block），每个块学习残差 `F(x) = H(x) - x`，而非直接学习映射 `H(x)`。

**关键模块**：残差块

```
输入 x
  │
  ├──> Conv → BN → ReLU → Conv → BN ──> F(x)
  │
  └──────────────────────────────────> x（shortcut）
                                          │
                                       F(x) + x
                                          │
                                        ReLU
```

当维度不匹配时，shortcut 使用 1×1 卷积进行投影。

#### 实验结果

| 数据集 | 指标 | ResNet-152 | VGG-16 |
|--------|------|------------|--------|
| ImageNet | top-5 错误率 | 3.57% | 7.3% |
| CIFAR-10 | 错误率 | 6.43% | ~7% |

#### 优点

- 解决了深度网络的退化问题，使 1000+ 层网络成为可能
- 残差连接成为后续几乎所有深度网络的标配设计
- 实现简单，几行代码即可

#### 局限性

- 残差连接对内存带宽有额外需求
- 对于极浅的网络（<20 层）收益不明显

#### 个人思考

残差连接的本质是提供了一个"梯度高速公路"，缓解了梯度消失。这一思想被后来的 DenseNet（密集连接）、Transformer（残差 + LayerNorm）等广泛采用。值得思考：为什么恒等映射比零映射更容易优化？

---

## 论文检索资源

### arXiv

- 地址：arxiv.org，选择 `cs.CV`（计算机视觉）分类
- 每日新论文在当地时间 20:00 更新（UTC）
- 搜索技巧：直接搜论文编号（如 `1512.03385`）或标题关键词
- 推荐配合 **Semantic Scholar**（semanticscholar.org）查看引用关系

### 顶会论文查找

| 会议 | 全称 | 检索方法 |
|------|------|----------|
| CVPR | IEEE Conference on Computer Vision and Pattern Recognition | openaccess.thecvf.com |
| ICCV | International Conference on Computer Vision | openaccess.thecvf.com（奇数年） |
| ECCV | European Conference on Computer Vision | ecva.net（偶数年） |
| NeurIPS | Neural Information Processing Systems | proceedings.neurips.cc |
| ICLR | International Conference on Learning Representations | openreview.net |
| ICML | International Conference on Machine Learning | proceedings.mlr.press |

💡 **Tips**：openaccess.thecvf.com 汇集了 CVPR/ICCV/ECCV 所有年份的全文 PDF，免费下载，是 CV 研究者最常用的检索入口。

### 论文推荐聚合

- **Papers With Code**（paperswithcode.com）：论文 + 代码 + 榜单，按任务分类
- **Hugging Face Papers**（huggingface.co/papers）：每日热门论文精选
- **Connected Papers**（connectedpapers.com）：可视化论文引用网络，找相关论文神器

---

## Zotero 论文管理建议

Zotero 是免费开源的文献管理工具，强烈推荐用于管理 CV 论文。

### 推荐配置

1. **安装 Zotero**（zotero.org）+ 浏览器插件（Zotero Connector）
2. **安装插件**：
   - `Better BibTeX`：生成规范引用 key，方便 LaTeX 使用
   - `ZotFile`：自动重命名 PDF，同步到云盘/平板
   - `Zotero PDF Translate`：PDF 内划词翻译

### 推荐工作流

```
1. 在 arXiv/ACM/IEEE 页面点击浏览器插件 → 自动抓取元数据+PDF
2. 用 ZotFile 将 PDF 重命名为 "作者_年份_标题缩写.pdf"
3. 在 Zotero 中添加笔记（与本模板对应）
4. 用标签区分：#待读 / #在读 / #精读完成 / #值得复现
5. 创建集合（Collections）按任务分类：检测/分割/多模态/...
```

### 文件夹结构建议

```
Zotero 我的文库/
├── 01-必读经典/
├── 02-检测/
├── 03-分割/
├── 04-多模态/
├── 05-工业视觉/
├── 06-三维视觉/
└── _待整理/
```

⚠️ **注意**：Zotero 免费存储空间为 300MB，PDF 较多时建议配合 WebDAV（坚果云免费 1GB/月）或 ZotFile+本地存储。
