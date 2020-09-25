# Compact Bilinear Pooling

## 1. Abstract

可视化任务中线性模型的大量使用：

- 语义分割
- 细粒度识别
- 人脸识别

缺点：维度过高（十万、百万）。

本文提出：；两个紧凑的线性表示，降维（数千），不影响性能。
==对双线性池进行新的核化分析，得到紧凑双线性表示，为双线性池化的判别能力提供新见解。==

## 2. Introduction

语义图像分析方法的重要部分：可视化特征的编码和池化。
CNN层，特征提取器。全连接层，池化编码机制。
用双线性池化取代全连接层，在细粒度视觉识别任务上取得显著改善。

贡献：

1. 两个紧凑双线性池化方法
   较之全双线性池化，其**特征维度降低两个数量级**(two orders of magnitude)，且不影响性能。
2. 紧凑双线性池化**支持反向传播**，允许识别网络的端到端优化
3. 提出双线性池化的**核化**观点，并提供理论支持

## 3. Relatedwork

双线性模型：用以分割style和内容。
其他用于视觉识别的聚类方法：

1. BoVW(Bag of Visual Words)framework 纹理分类方法，之后被视觉任务采用。
2. VLAD
3. Improved Fisher Vector

训练大型网络，降低参数规模相当重要。
降低全连接层的参数数量:

1. **Deep Fried Convnets**
2. **Fast Circulant Projection**

本文提出的方法高效（如bilinear pooling）但参数更少（compact）

## 4. Compact bilinear models

双线性池化或second order池化，通过下式形成**全局图片描述**：

$$B(\mathcal{X}) = \sum_{s\in\mathcal{S}} x_sx_s^T \tag{1}$$

- $\mathcal{X} = (x_1,\cdots,x_{|\mathcal{S}|},x_s \in \R^c)$ 局部描述符集合
- $x_s$ **局部描述符**，提取于 SIFT,HOG 或通过CNN前向传播
- $\mathcal{S}$ 空间位置（行和列组合）集合
- $B(\mathcal{X})$ 为 $c\times c$ 矩阵,本文中视作长为$c^2$的向量

### 4.1 双线性池化的核观点

双线性描述符进行图像分类通常用SVM或逻辑回归实现，可视作线性核机器：
$$\begin{aligned}
   \lang B(\mathcal{X}), B(\mathcal{Y}) \rang &= \lang\sum_{s\in \mathcal{S}}x_sx_s^T, \sum_{u \in \mathcal{U}}y_uy_u^T\rang \\
   &= \sum_{s\in \mathcal{S}} \sum_{u \in \mathcal{U}} \lang x_sx_s^T, y_uy_u^T\rang \\
   &= \sum_{s\in\mathcal{S}}\sum_{u\in\mathcal{U}}\lang x_s,y_u\rang^2
\end{aligned} \tag{2}$$

- $\mathcal{X},\mathcal{Y}$ 局部描述符集合

### 4.2 紧凑双线性池化

- 使得$k(x,y)$表示比较核 $i.e$ 二阶多项式核(second order polynomial kernel)
- $\phi(x) \in R^d$ 低维映射函数 $d << c^2$
  - 满足 $\lang\phi(x),\phi(y)\rang \approx k(x,y)$

$(2)$ 式可化为：
$$\begin{aligned}
   \lang B(\mathcal{X}), B(\mathcal{Y}) \rang &= \sum_{s\in \mathcal{S}} \sum_{u \in \mathcal{U}} \lang x_s, y_u\rang^2 \\
   &\approx \sum_{s\in \mathcal{S}} \sum_{u \in \mathcal{U}} \lang \phi(x), \phi(y)\rang \\
   &\equiv \lang C(\mathcal{X}), C(\mathcal{Y})\rang
\end{aligned} \tag{3}
$$ $where$
$$C(\mathcal{X}):=\sum_{s\in\mathcal{S}}\phi(x_s) \tag{4}$$

其中$C(\mathcal{X})$为紧凑双线性特征。
从上式子可看出，任何多项式核的低维近似都可以用来创建一个紧凑的双线性池方法。

两个算法：

1. **RM (Random Maclaurin Projection) 随机麦克劳林投影**
   ![20200528134022](https://raw.githubusercontent.com/bysen32/PicGo/master/20200528134022.png)
   早期方法，低维显示特征图逼近似多项式核
   $w_1,w_2 \in \R^c$是两个随机的$-1,+1$向量，$\phi(x)=\lang w_1,x\rang\lang w_2,x\rang$
   对于非随机的$x,y\in\R^c$,$E[\phi(x)\phi(y)]=E[\lang w_1,x\rang\lang w_1,y\rang]^2=\lang x,y\rang^2$
   输出d项，估计方差可以由1/d的因子降下来
   因此，RM中的每个预计条目都有一个近似数量的期望。

1. **TS (Tensor Sketch Projection)**
   ![20200528152607](https://raw.githubusercontent.com/bysen32/PicGo/master/20200528152607.png)
   使用草图函数提高投影过程中的计算复杂度，在实践中提供更好的近似。
   $\Psi(x,h,s)$具有性质$E[\lang\Psi(x,h,s),\Psi(y,h,s)\rang] = \lang x,y\rang$
   额外性质：$\Psi(x\bigotimes y,h,s) = \Psi(x,h,s)*\Psi(y,h,s)$
   上式含义：两向量外积的计数素描等于个体计数素描的卷积

## 5. Experiments

## 6. Conclusion

核框架下的双线性池化 + 提出两种紧凑表示（支持梯度反向传播，分类通道的端端优化）降低特征维度。
**TS feature** 不损失精度，大大降低参数维度。

## 7. Question

- [x] bilinear pooling
- [ ] spatial pyramid pooling
- [x] TS feature
- [ ] hidden Markov models
- [ ] circular convolution
- [x] second order pooling
- [ ] Fisher Vector
- [x] kernel 性质
- [ ] count sketch
- [ ] VGG
- [ ] 紧凑双线性池化是如何提取图像特征的？

## 8. Extension

Time Log

- 2020.05.26 完成部分阅读工作
- 2020.06.03 新问题：如何提取图像特征？