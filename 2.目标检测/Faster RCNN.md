# 目标检测算法介绍 CNN、RCNN、Fast RCNN、Faster RCNN

CNN 在目标检测上的弊端
需要将图片分割成多个区域进行训练

## 基于区域的卷积神经网络

### 1. RCNN

使用 RCNN 检测目标物体的步骤如下：

1. 选取一个预训练卷积神经网络
2. 根据需要检测的目标类别，训练网络的最后一层
3. 得到每张图片的 ROI , 对这些区域重新改造，符合 CNN 输入 size
4. 训练 二元SVM 判别目标物体和背景
5. 训练线性回归模型，为每个辨识物体生成**更精确**的边界框

![RCNN](https://i.loli.net/2020/08/01/chAb5eoiJkLWrw1.png)

RCNN的问题：

1. 计算量大，对每张图片提取 2000 个单独的区域
2. 目标检测阶段还有三个模型，导致速度慢
   1. 用于提取特征的 CNN
   2. 用于目标物体辨别的线性 SVM 分类器
   3. 调整边界框的回归模型

为了解决上述问题，突破技术限制的新方法

### 2. Fast RCNN

$idea$: 每张图片提取 2000 个单独区域 $\rightarrow$ 通过一次 CNN 得到全部重点关注的区域

![Fast RCNN示例](https://i.loli.net/2020/08/01/Q1MfxRtE8XqPwTD.png)

Fast RCNN 计算步骤：

1. 输入图片
2. 输入到**卷积网络**中，生成 RoI (通过一个 CNN 快速生成 RoI)
3. 利用 RoI 池化层，对这些区域重新挑战尺寸 使得适配全连接网络
4. 在网络的顶层用 $softmax$ 层输出类别。同样使用一个线性回归模型，输出相对应的边界框。

### 3. Faster RCNN

![20200801153934](https://i.loli.net/2020/08/01/J7CtyZL8awoImDA.png)

Fast RCNN 的改进版，区别在于感兴趣区域的生成方法。Fast RCNN 采用选择性搜索。Faster RCNN 使用 RPN （Region Proposal网络）。RPN 将图像映射作为输入，生成一系列 object proposals, 每个都带有评分。

![Faster RCNN示例](https://i.loli.net/2020/08/01/YG9vz2yqMPuFdI4.png)

Faster RCNN 工作过程：

1. 输入图像到卷积网络中，生成该图像的特征映射
2. 在特征映射上应用 RPN, 返回 object proposals 和相应分数
3. 应用 RoI Pooling 将 proposals 修正到相同 size
4. 最后将 proposals 传递到完全连接层，生成目标物体的边界框

![算法总结](https://i.loli.net/2020/08/01/zOChrEg72W6MFJe.png)

### ResNet-FPN

多尺度检测在目标检测中的重要性与日俱增。Feature Pyramid Network FPN 是一种精心设计的多尺度检测方法。

FPN 结构包括 自下而上，自上而下 横向连接 三个部分。 如下所示，将各个层级的特征进行融合，使其具有强语义和强空间信息。

![20200801161248](https://i.loli.net/2020/08/01/84ZyOV7DuXgFLn9.png)

1. 自下而上 简单的特征提取过程。ResNet 做主干网络，根据feature map的大小分为5个stage。各自输出 conv。相对于原始图片其 stride 依次翻倍。
2. 自上而下
3. 横向连接 将上采样的结果和自底向上生成相同的 feature map 进行融合。(经过一个 $1\times1$ conv)

![20200801190327](https://i.loli.net/2020/08/01/UGmPTo5gjMnHIrR.png)

- [ ] ==最近邻上采样是什么==

### ResNet-FPN + Fast RCNN

![Faster RCNN](https://i.loli.net/2020/08/01/IlSyhpWw9Lzn5bv.png)

ResNet-FPN + Fast RCNN = Faster RCNN

### ResNet-FPN + Fast RCNN + mask

ResNet-FPN + Fast RCNN + mask = Mask RCNN

![Mask RCNN](https://i.loli.net/2020/08/01/jpxy2Ncmd8e6ka4.png)

1. 骨干网络 ResNet-FPN
2. 头部网络，边界框识别（分类和回归） + mask预测

头部结构如下：

![20200801205213](https://i.loli.net/2020/08/01/ti3bmFRnJIvraTq.png)
