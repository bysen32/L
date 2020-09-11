dp# 2018 Learning to Navigate for Fine-grained Classiﬁcation

学习如何进行细粒度分类
Navigator-Teacher-Scrutinizer Network (NTS-Net):

## 概要

自监督机制定位区域信息，不需要边界框与部件标注。

强化学习

![NTS-Net网络模型](images/NTS-Net_structure.png)

NTS-Net 网络由 __三部分组成__：

- __Navigator agent 导航者（参与者）__ 导航模型以关注信息最丰富的区域
    对于图像中的每个区域，Navigator 预测该区域的__信息量__，并且该预测被用来提出信息量最丰富的区域。
- __Teacher agent 教师（批评家）__ 评估 Navigator 提出的区域并提供反馈
    对于每个建议的区域，教师评估其属于真实类的概率,__置信度__
    可信度评估指导 Navigator 使用我们新颖的 __有序一致损失函数__ 提出更多信息量的区域
- __Scrutinizer agent__ 审查 Navigator 提出的区域，并进行细粒度分类
    每个建议的区域被放大到同样的大小，并提取其中特征。
    区域特征和整个图像联合处理以进行细粒度分类。

可视为多智能体的合作，各 agent 相互受益共同进步。

主要贡献：

- __多智能体合作学习机制__
- __新颖的损失函数__
- 端到端训练，良好的预测结果

> Learning to rank
> 机器学习和信息检索领域。

$R_1,R_2 \in A,\text{if } \mathcal{C}(R_1) > \mathcal{C}(R_2), \mathcal{I}(R_1) > \mathcal{I}(R_2)$

![Fig.3](images/NTS-Net_Fig3.png)

对于输入图像，特征提取器提取深度特征映射。特征映射fed into导航这网络计算所有区域的信息量。在NMS后选择top-M信息区域，将信息量指定为 $I_1,I_2,I_3$。 取自整图，resize到预指定的尺寸，将其输入TeacherNetwork，然后获得置信度$C_1,C_2,C_3$。优化网络使NavigatorNetwork使得 信息量大的 置信度高。

![Fig.4](images/NTS-Net_Fig4.png)

模型的推理过程。输入图像输入到特征提取器，导航网络提出输入信息最丰富的区域。
从输入图像中裁剪出这些区域，并将其调整到预定义的大小，然后使用特征提取器计算这些区域的特征并将其与输入图像的特征融合。最后审查器网络处理融合后的预测标签。

==这里左图的 Navigator network 不太懂？==

## 2. 相关工作

__首个将 FPN 引入细粒度分类，并消除人工标注。__

## 3. 方法

### 3.1 Overview

方法基于假设，__信息区域有助于更好地描述目标__，因此将信息区域的特征与完整图像进行融合，将获得更好的性能。

### 3.2 Navigator and Teacher

![Fig.2](images/NTS-Net_Fig2.png)

Navigator 提取信息量大的区域，Teacher Network 评判区域的置信度。通过网络训练，将 Navigator 和 Teacher 的评判顺序一致：信息量大的区域置信度高。

### 3.3 Scrutinizer

随着 Navigator 网络逐渐收敛，它将产生信息丰富的目标特征区域，以帮助 Scrutinizer 做决策。使用 Top-K 信息区域和完整图像相结合 作为输入来训练 Scrutinizer 网络。

使用信息区域能降低类间变化，在正确标签上生成高置信分数。实验表明，添加信息区域可显著提高细粒度分类结果。

### 3.4 network architecture

为了获得特征图中区域建议与特征向量之间的对应关系，采用完全卷积网络作为特征提取器，不需要全连接层。

__Navigator network__

