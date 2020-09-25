# Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification from the Bottom Up

[toc]

![Figure 2.](https://i.loli.net/2020/07/28/bKqePN8RxvD6ArB.png)

## 方法

其他的细粒度识别任务通常用图像级标签训练的深度神经网络倾向于**聚焦有识别力的部分**，而**忽略了其他目标部件**的作用。

弱监督的方式建立互补局部模型，以检索由卷积神经网络检测到的被域目标部件抑制的信息。

1. 采用基于 Mask R-CNN 和 CRF 的分割方法，通过弱监督 **目标检测和实例分割**，提取粗目标实例。
2. 估计并搜索每个对象实例的最佳部件模型
3. 构建双向长短期记忆 LSTM 网络，将这些互补部分的部件信息融合并编码成图像分类的综合特征

- [ ] ==MASK R-CNN 是什么？==
- [ ] ==CRF 又是什么？==
- [ ] ==双向长短期记忆 LSTM 的作用？==

## 2 相关工作

1. **弱监督目标检测和分割**，在这篇文章中，通过提供一个粗糙的分割 MASK 来执行弱监督对象实例的检测和分割，并提出使用 CAM 实现 MASK R-CNN，使用 CRF 迭代地修正目标位置和掩模。避免在后续对象部件建模中丢失重要的对象部件。

2. **基于部件的细粒度图像分类**，此篇文章，目标不在于建立强健的目标检测器，来对最终分类决定提供局部外观信息。**互补部件模型** 的目标在于有效地利用在目标检测阶段被隐藏在对象 proposal 下的**丰富信息**。

3. **使用 LSTM 进行上下文编码**，多标签图像分类中，采用 LSTM 子网络递归发现注意区域。LSTM 子网络依次预测定位区域的**语义标记得分**，同时捕获空间相关性。文章利用 **双向 LSTM** 来学习所有图像 patch 的深层层次表示。性能**显著**优于单层 LSTM 。

## 3. 弱监督互补部件模型

如 图 2 模型主要分为 3 个阶段：

1. 弱监督对象检测
2. 实例分割
3. 互补部件挖掘和上下文图像分类

### 3.2 弱监督对象检测与实例分割

![Stage 1.](https://i.loli.net/2020/07/29/mSjBMuCIdtl9Oz6.png)

#### 3.2.1.粗对象 Mask 初始化

符号说明：

- 图片 $I$
- 图片标签 $c$
- 分类网络的最后一个卷积层特征图 $\phi(I,\theta) \in \R^{K\times h \times w}$
- 全局平均池化后，池化的特征 $F_k = \sum_{x,y}\phi_k(x,y)$

_这里是标准的 CAM 解释_ 后接分类层，对于类 $c$ 该 **类激活映射 class activation map(CAM)** 如下 $$M_c(x,y) = \sum_{k} w^c_k\phi_k(x,y), \tag{1}$$

- $w_k^c$ 相应类 $c$ 对于第 $k$ 个通道在全局平均池化层中的权重
- 类激活映射 $M_c$ 通过双线性插值上采样到原始图片的尺寸 $\R^{H\times W}$

**这里将从主干网络中的卷积层进行逐像素的加权合并
得到一张热力图(Class Activation Map)**

由于一张图片具有多个对象实例、多个局部最大化响应，可在类激活映射 CAM $M_c$ 中观察到。
在这个 map 上引用 ==多区域级 集合分割== 以分割候选对象实例。
接下来，对于每个实例，归一化类激活至 $[0,1]$。

假定我们在 CAM 中有 $n$ 个对象实例，我们根据被归一化的 CAM 建立一个 **对象概率映射** $F \in \R^{(n+1) \times H \times W}$

前 n 个物体概率 map 表示图像中存在某个物体的概率，第 (n+1) 个概率图表示背景的概率。背景概率图计算如下：

$$F^{n+1}_{i\in\R^{H\times W}} = \max(1-\sum^n_{\tau=1}F^\tau_{i\in\R^{H\times W}},0) \tag{2}$$

**CRF** 被用来提取高质量的目标实例，为了能使用 CRF，标签图 $L$ 根据公式 3 产生，其被视为用于 Mask-RCNN 训练的伪 ground truth 注释。

$$
L_{i\in R^{h\times w}} = \begin{cases}
\lambda, \argmax_{\lambda} F^{\lambda}_{i\in\R^{H\times W}} > \sigma_c \\
0,\text{otherwise}
\tag{3} \end{cases}$$

标签图 $L$ 输入 CRF 来生成对象实例分割。 
在 Mask-RCNN 训练中，这些被当作伪 ground truth 注释。
用于判定某个点是属于对象还是属于背景，进行分割。

- [x] ==GAP(Global Average Pooling Layer) 和 CAM(Class Activation Mapping) 是什么？==

> ##### 1.GAP 全局平均池化
>
> **Global Average Pooling** 是指计算整个 channel 区域的平均值，仅用一个值来表示整个区域。如下图
![GAP](https://i.loli.net/2020/07/29/9KDjLmZGpMYk5uc.png)
> 上图卷积 $6\times 6\times 3$，经过 GAP，特征图变成 $1\times 1\times 3$ 的特征向量。
>
> 作用：用 GAP 代替全连接层，不仅可以降低维度，防止过拟合，而且减少大量参数，网络性能不错。
>
> ##### 2.CAM 类激活映射
>
> CAM 是帮助我们可视化 CNN 的工具。使用 CAM 可以清楚地观察到，网络关注图片的区域，根据哪一部分进行推断，可以帮助我们理解网络。就是注意力机制中常用的热力图。
>
> 类激活图仅仅是在不同空间位置处，存在这些视觉团的加权线性和。通过简单地将类激活映射上采样到输入图像的大小，识别于特定类别最相关的图像区域。数学语言描述如下：
>
> $F_k = \sum_{x,y} \phi_k(x,y)$，将 $F_k$ 插入到 类得分 $S_c$ 中
$$S_c=\sum_kw^c_k\sum_{x,y} \phi_k(x,y) = \sum_{x,y}\sum_{k}w^c_k \phi_k(x,y) \tag{1}$$
>
> 定义 $M_c$ 为 类$c$ 的类激活映射 CAM，每个空间元素通过下式得出:
$$M_c(x,y) = \sum_kw^c_k\phi_k(x,y) \tag{2}$$
>
> 因此，
$$S_c=\sum_{x,y}M_c(x,y)$$
>
> 直观解释：
![CAM interpresion](https://i.loli.net/2020/07/30/4MqANaeTJkfQBgi.png)

- [x] ==Q: GAP 如何代替全连接层?==

> ![GAP replace FC](https://i.loli.net/2020/07/29/VM9FEYjUpm6XeIi.png)
> 上图网络为 VGG16 ，若要使用 GAP 则需要改变网络的结构。把最大池化和全连接层丢弃，换成一个 $1\times 1\times 512$ 的 GAP，之后与原网络一致添加 $softmax$ 函数即可
> 将通道数较大的卷积 通过 GAP 转化为一维向量(类似于拉直操作) 做全连接

- [x] ==这个式子 3 啥意思？==

> 做像素阈值判定，从而实现实例分割。$\sigma_c$ 设为0.8, 用来决定像素属于对象还是背景

- [ ] ==CRF 是什么？==

> condition random filed
CRF 条件随机场 是判别式无向图模型，目标是构建条件概率模型。在给定输入的情况下 条件随机场的定义是：假设 $X$ 与 $Y$ 是随机变量，$P(Y|X)$ 是在给定 $X$ 的条件下 $Y$ 的条件概率。若随机变量 $Y$ 构成一个无向图 $G=(V,E)$ 表示马尔可夫随机场，即 $(_v|,_w,\neq) = (_v|,)$ 对于任意结点 $v$ 成立，则称条件概率分布 $P(Y|X)$ 为条件随机场。其中 $Y$ 表示结点 $V$ 对应的随机变量。$w \text{~} v$ 表示在图 $G=(V,E)$ 中与结点 $v$ 有边连接的所有结点 $w$.
CRF 的优势在于可以容纳上下文的信息

---------------------------

#### 3.2.2 联合检测和分割对象实例

从上一阶段中生成，给定分割对象实例集合 $\mathcal{S} = [S_1, S_2, \dots S_n]$ 与他们相应的类标签。获得每个分段最小化 $\text{bounding box}$ 形成建议集合 $\mathcal{P} = [\mathcal{P_1},\mathcal{P_2},\dots\mathcal{P_n}]$。

建议 $\mathcal{P}$，分割 $\mathcal{S}$，与相应类标签 $L$
被用于 Mask R-CNN 进一步建议与 mask 改进。
以这种方式，我们将目标检测与实例分割 转变为全监督学习。训练 Mask R-CNN。

- [ ] ==Mask R-CNN是什么？==

##### 3.2.2.1 Mask R-CNN

###### 1. Faster RCNN

![20200730110226](https://i.loli.net/2020/07/30/dY4xICJLSDKARs5.png)

- [x] ==ROI Pooling 是什么？==

> 在 FPN 中用于对生成的一系列 proposals 特征进行统一的表征。
> 从大小不同的 bbox 提取特征使得输出结果等长。
>
> 1. **ROI Pooling** ![ROI Pooling](https://i.loli.net/2020/07/31/bk2rsxWHq3nPBip.png)
>
> 1. **ROI Align** ![ROI Align](https://i.loli.net/2020/07/31/OsTdft9U64gaPAM.png)

###### 2. ResNet-FPN

###### 3. ResNet-FPN+Fast RCNN

#### 3.2.3 CRF-Based 分割
- [x] ==公式 2 是什么意思?==

> 背景概率图等于 1 - 图中所有 n 个对象存在的概率。全图概率和为 1。


假定类 $c$ 有 $m$ 个对象建议 $\mathcal{P}^*= [\mathcal{P}^*_1,\mathcal{P}^*_2,\dots,\mathcal{P}^*_m]$，与相应的分割 $\mathcal{S}^*=[\mathcal{S}^*_1,\mathcal{S}^*_2,\dots,\mathcal{S}^*_m]$，
其 $\text{classification score}$ 高于 $\sigma_0$ (用于删除异常proposal)。

一个非极大值抑制 NMS(non-maximum suppression) 将重叠阈值 $\tau$ 应用于 $m$ 个 proposals
假定随后仍然有 $n$ 个对象 proposals 保留。$\mathcal{O} = [\mathcal{O}_1, \mathcal{O}_2,\dots, \mathcal{O}_n]$，$n \ll m$。

现有的大多数研究使用 NMS 来抑制大量共享同一类标签的方案，以便获得少量特殊的目标 proposals 。
然而在我们弱监督设置中，实际上包含的丰富的对象部件信息被抑制在 NMS 过程中。

特别是被对象提议 $\mathcal{O}_j$ 抑制的每个方案 $\mathcal{P}^*_i \in \mathcal{P}^*$，可被考虑为对 $\mathcal{O}_j$ 的补充。
$\mathcal{O}$ 由 $\mathcal{P}$ 聚合而来，将 $\mathcal{P}$ 作为 $\mathcal{O}$ 的补充说明。因此被抑制的提议 $\mathcal{P}^*_i$ 可在将来用于提炼 $\mathcal{O}_j$

我们通过初始化类概率映射 CAM ==$F^* \in \R^{(n+1)\times H\times W}$== 实现这个 $idea$。

部分proposals $\mathcal{P}^*$ 被抑制为 $\mathcal{O}_j$。通过双线性插值 添加它的提议分段遮罩 $\mathcal{S}^*_i$ 的概率图到 $F^*_j$ 相应的位置。
类概率图稍后归一化到 $[0,1]$， 对于第 $(n+1)$ 个表示背景的概率图，它定义为

$$
F^{*,n+1}_{i\in\R^{H\times W}} = \max(1-\sum^n_{\tau=1} F^{*,\tau}_{i\in \R^{H\times W}}, 0) \tag{4}
$$

- [x] ==$F^*_{i\in\R^{H\times W}}$是什么？==

> 类概率映射 CAM。形成某个对象 $\mathcal{O}_j$ 的多个概率图集合。

- [x] ==4 式与 2 式有什么关联？==

> 2式是明确的对象，而4式是通过 **部件集合** 组成的对象。 这里可以保留在其他方法中舍弃的丰富的细节信息。加入后续计算，对提高最终结果的精度提供一定的帮助。

给定类概率映射 $F^*$ ， CRF 再次被用来 提炼与纠正实例分割结果，如同在上一阶段中描述的那样。

#### 3.2.4.迭代实例提炼

多次交替使用基于 CRF 的分割和基于 Mask R-CNN 的检测和实例分割，以逐步确定对象实例的定位和分割。如图 2

### 3.3 互补部件模型

![20200801230431](https://i.loli.net/2020/08/01/f361tAWsnUvyqTH.png)

#### 3.3.1. 模型定义

给定检测对象 $\mathcal{O}_i$，它抑制相应的 proposals 集合 $\mathcal{P}^{*,i} = [\mathcal{P}^{*,i}_1, \mathcal{P}^{*,i}_2,\dots,\mathcal{P}^{*,i}_k]$，其中包含由有用的对象信息与正确定位物体的位置。

对图像分类提出补充部件模型 $\mathcal{A}$， $\mathcal{A} = [A_1,\dots,A_n, A_{n+1}]$ 模型由一个 $size$ 为 $n+1$ 的元组 组成：

1. 根部分$A_{n+1}$：覆盖整个对象及其上下文
2. 中心部分$A_1$：覆盖对象的核心区域
3. 固定数目的周边 proposals $A_i$： 覆盖不同对象部件，但保留足够的判别信息

其中每个部件模型通过一个元组定义 $A_i=[\phi_i,u_i]$，其中 $\phi_i$ 是第 $i$ 个部件的特征。$u_i$ 是一个描述部件几何信息的 $\R^4$ 维元组 $(x_i,y_i,w_i,h_i)$

一个没有任何部件丢失的潜在部件模型称为 **对象假设**。
为了使对象部件互补，其外观特征或位置特征差异应当尽量大，部件组合的分数也应当尽量大。
在寻找互补的有判别部分时，以上标准是约束条件。

- [x] ==部件组合的分数是什么？==

对象假设的分数 $\mathcal{A}$ 为 所有对象部件的分数之和 减去 不同部件的外观相似性和空间重叠。 如下式：
$$\begin{aligned}
\mathcal{S(A)} &= \sum^{n+1}_{\iota=1} f(\phi_{\iota}) \\ &- \lambda_0 \sum^n_{p=1}\sum^{n+1}_{q=p+1}[d_s(\phi_p,\phi_q)+\beta_0 IoU(u_p,u_q)] \tag{5}
\end{aligned}$$

- $f(\phi_k)$ 是在 Mask R-CNN 分类分支中第 $k$ 个部件的分数
- $d_s(\phi_p,\phi_q) = \|\phi_p-\phi_q\|^2$ 语义相似性
- $IoU(u_p,u_q)$ 在部件 $p$ 和 $q$ 中的空间覆盖

给定目标假设集合，选择一个最大 score 的假设作为最终目标部件模型。搜索优化 上述得分最大化的方案的最优子集是一个组合优化问题，计算代价大。
接下来 使用快速启发式算法 寻找一个近似解。

- [x] ==$IoU$ 是什么？==

> **$IoU$ Intersection over Union 交并比**
目标检测中的常见评价标准，主要是衡量模型生成的 $\text{bounding box}$ 和 $\text{ground truth box}$ 之间的重叠程度。公式如下：
$$IoU = \frac{Detection Result \cap GroundTruth}{Detection Result \cup GroundTruth}$$
>
> 直观的图像解释:
![20200730204336](https://i.loli.net/2020/07/30/UdiJgSQ2j9NFlca.png)

- [x] ==NMS 是什么==

> **Non-maximum suppression 非极大抑制**
在目标检测中常出现对同一个物体预测出多个 $\text{bounding box}$ 的情况。
NMS 所作的就是除掉多余的 $\text{bounding box}$，
**只保留和 $\text{gound truth}$ 重叠度最高的 $\text{bounding box}$**

- [x] ==NMS的主要流程==

> 在目标检测中，分类器会给每个 $\text{bounding box}$ 计算一个 $\text{class score}$，即该 $bb$ 属于每一类的 **概率**。
> NMS根据这些值进行。主要流程：
>
>- 对于每一类，把所有 score < thresh 的 bb 的 score 设置为0
>- 将所有的 bb 按照 score 排序，选择得分最高的 bb
>- 变量其余 bb，如果和最大 score bb 的 IoU 超过一定阈值 则删除该 bb
>- 从未处理的 bb 继续选择上述过程，直到找到所有保留 bb
>- 根据所有保留的 bb class score 和 class color 画出最后的预测结果

#### 3.3.2 部件位置初始化

为了初始化部件模型，我们通过设计一个遵守两条规基础规则的网格对象部件模板来简化部件评估:

1. 每个部件必须包含足够的判别信息
2. 部件 $\text{pair}$ 之间的差异尽量大

如图2，深度卷积神经网络在定位对象最具判别性的部件上展现了出众的能力。因此我们将 根部件 $A_{n+1}$ 设置为代表整个对象的对象建议 $\mathcal{O}_i$。
以 $A_{n+1}$ 为中心，创建一个 $s\times s(=n)$ 的栅格。 每个栅格单元的尺寸为 $\frac{w_{n+1}}{s}\times \frac{h_{n+1}}{s}$

- $w_{n+1},h_{n+1}$ 为根部件 $A_{n+1}$ 的宽高

中心栅格单元被指定给对象中心部分。余下的栅格单元被指配给部件 $A_i, i\in [2,3,\dots, n]$

然后我们将每个部件 $A_i \in A$ 初始化为最接近该栅格单元的方案 $\mathcal{P}^*_j \in \mathcal{P}^*$

- [x] ==用人话解释一下==

> 这里在原始图片中定义了一些栅格单元，并为栅格分配部件。根部件 $A_{n+1}$，中心部件 $A_1$，周围建议部件 $A_i$。再将这些部件分配给最近的方案 $\mathcal{P}^*_j$

#### 3.3.3 部件模型搜索

对于有 n 个对象部件 和 k 个候选的 **抑制proposals** 模型。 对象函数定义为 $$\hat{\mathcal{A}} = \argmax_{\mathcal{A\in S_A}} \mathcal{S(A)} \tag{6}$$

- $K=C^n_k, k \gg n$ 为对象假设的总数
- $\mathcal{S_A} = [\mathcal{A^1,A^2,\dots,A}^K]$ 对象假说集合

直接搜索部件模型优化很棘手。
此处采用 **贪心搜索策略** 来搜索 $\hat{\mathcal{A}}$。
具体来说，我们依次检查 $\mathcal{A}$ 中的每一个 $\mathcal{A}_i$
并**在 $\mathcal{P}^*$ 中寻找能使得 $\mathcal{\hat{A}}$ 最小化的 优化对象部件 $\mathcal{A}_i$**。
将总体指数级时间复杂度降低到线性级 $O(nk)$。

图 2 中，可看到 在搜索过程中生成的对象假说 覆盖了对象的不同部分。
而且不仅只关心对象的核心区域。

- [x] ==Eq.6 是什么?==

> 在所有对象部件的组合方案中，选择一个使得目标函数值最大的组合作为最优化的方案 $\mathcal{\hat{A}}$

### 3.4 上下文编码图像分类

#### 3.4.1.CNN特征提取器微调

给定图像 $I$ 和 部件模型 $\mathcal{A} = [A_1, \cdots, A_n, A_{n+1}]$
相应部件的 **图像补丁** $I(\mathcal{A}) = [I(A_1), I(A_2), \cdots, I(A_n), I(A_{n+1})]$

因此，除了 $n+1$ 块，我们附加了原始图像的随机裁剪作为第 $n+2$ 块图像补丁。
目的是为了 在训练过程中增加更多的上下文信息。
因为与对象部分相应的补丁主要集中在对象本身。

每个补丁与它被裁剪的原始图像共享相同的标签。
来自所有原始训练图像的所有补丁构成一个新的训练集，用于微调在 ImageNet 上预训练的模型。
此微调模型用作所有图像补丁的特征提取程序。

- [x] ==图像补丁是什么?==

> 在原始图像上进行随机裁剪以便增加上下文信息。
在图像分类中，图像的随机裁剪常用于模型训练。

- [ ] ==图像补丁在哪里使用了？==

#### 3.4.2 特征融合的堆叠 LSTM

![Figure 3.](https://i.loli.net/2020/07/31/icd3XS9ZbJrkqp7.png)

为了 特征融合和性能增强 提出堆叠 LSTM 模型 $\phi_l(\cdot ;\theta_l)$，如 图3。

1. 补充部件模型中的 (n+2) 个补丁 通过输入到前一步的CNN特征提取器中。 该步的输出定义为 $\Psi(I) = [\phi_c(I;\theta_c),\phi_c(I(A_1);\theta_c),\dots,\phi_c(I(A_{n+2});\theta_c)]$。
2. 接下来建立两层的堆叠 LSTM 来融合提取的特征 $\Psi(I)$，第一层 LSTM 的隐态作为第二层 LSTM 的输入。 第二层与第一次的次序相反。 设 $D(=256)$ 作为隐态的维度。使用 $softmax$ 针对每一个部件 $A_i$ 来生成类概率向量。$f(\phi_l(I(A_i);\theta_l)) \in \R^{\mathcal{C}\times1}$

最终图像分类的损失函数：

$$\begin{aligned}
\mathcal{L}(I, y_I) &= -\sum^{\mathcal{C}}_{k=1} y^k\log f^k(\phi_l(I;\theta_l)) \\ & - \sum^{n+2}_{i=1}\sum^{\mathcal{C}}_{k=1}\gamma_i y^k\log f^k(\phi_l(I(A_i);\theta_l)), \tag{7}
\end{aligned}
$$

- $f^k(\phi_l(I;\theta_l))$： 图像 $I$ 属于第 $k$ 类的概率。
- $f^k(\phi_l(I(A_i);\theta_l))$：图像补丁 $I(A_i)$ 属于第 $k$ 类的概率。
- $\gamma_i$ 补丁 $i$ 的常数权重，两种设置方案
  - 单损失集 $\gamma_i = 0(i=2,\dots,n+2)$
  - 多损失集 $\gamma_i = 1(i=2,\dots,n+2)$ 相较于单损失集，性能更优

## 4.实验结果

### 4.1 实现细节

$n=9$

在 mask 初始化阶段，在 ImageNet 上的预训练模型 GoogleNet + BN
标准的 SGD 优化器 每 $40000$ 次迭代 初始学习率 $0.001 / 10$
70000 次迭代后训练收敛

在 Mask R-CNN 阶段，使用 ResNet-50 特征金字塔网络 FPN 作为主干网络。在 COCO 数据集上预训练。在目标数据集上 fine-tune

![Figure 4.](https://i.loli.net/2020/07/31/FkWNaSlBCeLwzIh.png)

### 4.2 细粒度图像识别

#### 4.2.1 Stanford Dogs 120

![Stanford Dogs](https://i.loli.net/2020/07/31/BFn74HGtT9QENO3.png)

- [x] ==SJFT 是什么？==

> selective joint fine-tuning(SJFT) **有选择的联合微调**
> 一种迁移学习方法，从其他相关任务中汲取知识来辅助当前任务学习的一系列方法。减少训练集不足带来的过拟合风险，提升模型精度。
> 使用训练集丰富的的预训练模型，将其中与当前任务训练集相似的部分，辅助训练，提升精度。
> 训练过程解决问题：**如何在辅助任务上挑选出和主要任务数据集相似的图像作为辅助任务的训练集**。(特征提取，比较相似特征)
> 两种方法：
>
> 1. 构造 Gabor 滤波器提取图像特征
> 2. 预训练卷积神经网络的第一二层作为图像的特征（包含了丰富的底层信息）
>
> 提取完特征后，对于当前任务训练集的每一张图像在辅助任务的数据集中逐一进行距离度量( KL散度 )，即可找到一批相似的图像。组成新的训练集来初始化当前任务，使用挑选后的相似图训练辅助任务，辅助任务和主要任务共享卷积参数，进行训练实现性能的提升。

#### 4.2.2 Caltech-UCSD Birds 2011-200

![UCB200](https://i.loli.net/2020/07/31/hPG9a82ulAjLM4T.png)

- [ ] ==文中的 baseline 包括了图2中的哪些部分？==

### 4.3 通用对象识别

![Caltech 256](https://i.loli.net/2020/07/31/rWujBTn52VoYMkH.png)

### 4.4 消融学习
