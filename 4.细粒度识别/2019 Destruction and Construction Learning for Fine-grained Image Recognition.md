# Destruction and Construction Learning for Fine-grained Image Recognition

## 3. Proposed Method

DCL 的网络结构：

![Fig 2.](https://i.loli.net/2020/07/23/wyQHe4gdB8kXOhD.png)

### 3.1 Destruction Learning

对于细粒度识别任务来说，局部的细节远比全局的结构更重要。(细粒度识别任务的关键在于对局部细节的识别，不同的类别通常具有相似的全局结构。也就是研究难点：类间相似，类内差异大)

#### 3.1.1 Region Confusion Mechanism 区域混淆机制

![20200723213659](https://i.loli.net/2020/07/23/Otu1zbdVKAEJfqC.png)

$N\times N$
$1\leq i,j \leq N$
$R_{i,j}$
$R_{\sigma(i,j)}$
$q_{j,i} = i+r, r\sim U(-k,k), 1\leq k \leq N$

$$\forall i\in \{1,\cdots,N\}, |\sigma^{row}_j(i)-i|<2k \tag{1}$$

$$\forall j\in \{1,\cdots,N\}, |\sigma^{col}_i(j)-j|<2k \tag{2}$$

$$\sigma(i,j) = (\sigma^{row}_j(i), \sigma^{col}_i(j)) \tag{3}$$

$$\mathcal{L}_{cls}=-\sum_{l\in\mathcal{I}}l \cdot\log[C(I)C(\phi(I))] \tag{4}$$

#### 3.1.2 Adversarial Learning 对抗学习

使用 RCM 来损坏图像时，不总是对分类任务有益处。
比如在 shuffle 局部区域的时候，会引入噪声。
从这些带噪声的可视化部分进行学习，对分类任务有坏处。
因此提出了另一个对抗损失$\mathcal{L}_{adv}$ 来避免 RCM 导致过拟合。

> 由于 RCM 随机混洗操作会带来噪声，影响分类任务的精度。在此加入了对抗损失 $\mathcal{L}_{adv}$ 对 RCM 进行限制。

将原始图像和 destructed 图像分为两个域。
对抗损失 $\mathcal{L}_{adv}$ 和分类损失 $\mathcal{L}_{cls}$ 以 ==对抗的方式工作==。

1. 保持域不变模式
2. 拒绝习得 $I$ 和 $\phi(I)$ 之间的 domain-specific 模式

==这里的 domain 是什么意思？==
是不是区域图像的分布模式？

将图像标号为 one-hot 向量 $d \in \{0, 1\}^2$，表示图片是否被损坏。

> one-hot

判别器可在框架中被添加为新的分支来评判图像 $I$ 是否被 destructed

$$D(I,\theta_{adv})=\text{softmax}(\theta_{adv}C(I,\theta^{[1,m]}_{cls})) \tag{5}$$

==这个式子不是太了解==

> softmax
>
> - $C(I,\theta^{[1,m]}_{cls})$ 从主干分类网络的第 $m^{th}$ 层的输出提取出的特征向量
> - $\theta^{[1,m]}_{cls}$ 主干分类网络 $1^{st}$ 到 $m^{th}$ 层的可学习参数
> - ==$\theta_{adv} \in \R^{d\times 2}$ 是线性映射== 什么意思？

判别网络的损失：

$$\mathcal{L}_{adv} = -\sum_{I\in\mathcal{I}}d\cdot\log[D(I)]+(1-d)\cdot\log[D(\phi(I))] \tag{6}$$

==Todo 1== **Justification** 有点难以理解（暂略）
如何理解对抗损失如何调节特征学习的解释

![20200723221431](https://i.loli.net/2020/07/23/j9cHBWGXalJo5OC.png)

可视化主干网络的特征图 取消对抗损失，来理解它的作用。

给定输入图像 $I$ ,$F^k_m(I)$ 表示第 m 个卷积层的第 k 个特征图。 对 ResNet-50 从最后一个卷积池化层的输出来提取特征 来做对抗损失。 因此在最后的卷积层的第 k 个 filter 的处理，可被表示为
$r^k(I,c) = \bar{F}^k_m(I) \times \theta^{[1,m]}_{cls}[k,c]$，
$\theta^{[m+1]}_{cls}[k,c]$是第$k^{th}$个特征图和$c^{th}$个输出标签之间的权重。

我们在 **散点图** 中比较不同 filters 对于原始图和被毁坏的图的响应。每一个 filter 的积极响应 映射到数据点$(r(I,c), r(\phi(I), c))$
我们发现由 $\mathcal{L}_{cls}$ 训练出的特征分布比由 $\mathcal{L}_{cls}+\mathcal{L}_{adv}$ 计算出来的分布更紧密。

这意味着 过滤器 对由 RCM 产生的噪声模式由很大响应，对原图也可能有很大响应。(例如可视化部分 ABC,拥有很多 过滤器 对 edge-style 和 不相关模式 有响应)

通过使用对抗损失，可以区分过滤器对嘈杂视觉模式的响应（D VS. F)

D: 倾向于对 **噪声模式** 做出响应的过滤器 RCM-induced 的图像特征
F: 倾向于响应 **全局上下文** 的过滤器 原始图像特定的图像特征
E: 绝大多数过滤器都与通过 $\mathcal{L}_{cls}$ 增强的详细的 **局部区域描述** 有关

两个损失 $\mathcal{L}_{cls}$ 和 $\mathcal{L}_{adv}$ 共同作用于 destruction 学习，增强具有判别性的局部细节，过滤掉不相关的特征。

### 3.2 Construction Learning 构造学习

考虑到图像中相关区域复杂多变的视觉模式，我们提出另一个学习方法来对局部区域之间的关系进行建模。提出区域对齐网络 **区域构建损失** $\mathcal{L}_{loc}$
测量图像中不同区域的 **定位精度**，通过端到端训练诱导骨干网络对 **区域间的语义关系** 进行建模。

![20200723214250](https://i.loli.net/2020/07/23/piB6LStdY5TDhKI.png)

区域对齐网络工作在 分类网络网络的第 $n^{th}$ 层卷积层的 输出特征上。 该输出特征通过 $1\times 1$ 卷积获得两个输出通道，后通过 ReLU 和平均池化层处理 得到一个映射图 $2\times N \times N$, 分别代表行和列坐标。

$$ M(I) = h(C(I,\theta^{[1,n]}_{cls}),\theta_{loc})\tag{8}$$

$\mathcal{L}_{loc}$公式:

$$\mathcal{L}_{loc} = \sum_{l\in\mathcal{I}}\sum^N_{i=1}\sum^N_{j=1}|M_{\sigma(i,j)}(\phi(I))- \begin{bmatrix} i\\j \end{bmatrix}|_1+|M_{i,j}(I)-\begin{bmatrix}i\\j \end{bmatrix}|_1 \tag{9}$$

> $L_1$范数：
> 向量中各元素的元素绝对值之和

定位图像中的主要对象，发现子域之间的关系。
帮助分类主网络建立对对象的深刻理解 和 对结构信息进行建模。例如对象部件的 形状 和 语义关系 。

### 3.3 Destruction and Construction Learning 损失构建

帮助网络习得判别性区域

此框架中，分类、对抗和区域对准损失以 end-to-end 方式进行。
模型可以利用增强的局部细节、良好建模的对象部件关系 进行细粒度识别。

$$\mathcal{L} = \alpha\mathcal{L}_{cls} + \beta\mathcal{L}_{adv} + \gamma\mathcal{L}_{loc} \tag{10}$$

这里需要了解三个损失函数的意义

1. $\mathcal{L}_{cls}$
2. $\mathcal{L}_{adv}$
3. $\mathcal{L}_{loc}$
