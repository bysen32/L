# Bilinear CNN Models for Fine-grained Visual Recognition

## 1. Introduction

提出了一个双线性模型，由两个基于CNNs特征提取器组成的识别架构。其输出在图像的每个位置使用外积相乘，并合并以获得图像描述符。以平移不变性的方式对局部特征进行建模，对细粒度分类特别有效。

![20200528082114](https://raw.githubusercontent.com/bysen32/PicGo/master/20200528082114.png)

使用外积捕捉特征通道间的成对关系，并对零件特征交互进行建模。

## 2. 双线性模型 - 图像分类 简介

### 2.1 双线性模型进行图像分类的通用公式

图像分类双线性模型$\mathcal{B} = (f_A, f_B, \mathcal{P}, \mathcal{C})$

- $f_A, f_B$ 特征函数 映射：$f: \mathcal{L} \times \mathcal{I} \rightarrow R^{c\times D}$
  - $\mathcal{I}$ 图像
  - $\mathcal{L}$ 位置
  - 输出特征尺寸$c\times D$
- $\mathcal{P}$ 池化函数
- $\mathcal{C}$ 分类函数

例如，$f_A$和$f_B$在位置$l$的双线性特征合并 $bilinear(l, \mathcal{I}, f_A, f_B) = f_A(l, \mathcal{I})^Tf_B(l, \mathcal{I})$

为了获得图像描述，池化函数$\mathcal{P}$聚合图像中各个位置的双线性特征。

1. 求和池化 $\phi(\mathcal{I}) = \sum_{l \in \mathcal{L}} bilinear(l, \mathcal{I}, f_A, f_B)$
2. 最大池化

上式忽略特征的位置，因此无序。

将的到的池化结果$\phi(\mathcal{I})$拉直，输入到分类函数$\mathcal{C}$中

### 2.2 使用CNNs的实现模型

![20200528101233](https://raw.githubusercontent.com/bysen32/PicGo/master/20200528101233.png)

使用CNN作为特征函数$f$

- $l$为损失函数
- 双线性向量$x = \phi (\mathcal{I})$
- $y \leftarrow sign(\bold{x})\sqrt{|\bold{x}|}$
- $z \leftarrow \bold{y}/||\bold{y}||_2$

### 2.3 CV中常用的无序池化方法的双线性模型写法

(略) 暂时没有需求

## 目前存在的疑问

- Fisher Vector

## 小结

## time log

2020.06.10 概念不清，回头看
2020.06.11 了解实现原理 tf

