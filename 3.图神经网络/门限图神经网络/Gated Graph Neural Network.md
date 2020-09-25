# Gated Graph Sequence Neural Network

以图结构数据为输入的机器学习任务。
主要贡献：

- 图神经网络拓展到序列结构上，以往的工作结果仅仅输出单一模型（如图级别的分类任务）
- 强调图神经网络的泛用性

图特征学习的两种设置：

- 学习输入图的表示
- 学习处理序列输出过程中的内部状态表示

GGSNN = GNN + RNN 将原本的单个输出变成**序列输出**。

## 2 Graph Neural Networks

GNN概念定义

- 图结构 $\mathcal{G}=(\mathcal{V},\mathcal{E})$
- 节点 $v \in \mathcal{V}$从$1,\dots,|\mathcal{V}|$中取唯一值
- 边对 $e = (v,v')\in \mathcal{V} \times\mathcal{V}$，代表有向边$v \rightarrow v'$
- $D$维节点向量（节点表示、节点嵌入） $\bold h_v\in\R^D$
- 图包含的节点标签集合 $l_v\in\{1,\dots,L_{\mathcal{V}}\} \rArr h_{\mathcal{S}} = \{h_v|v\in\mathcal{S}\}$ 当$\mathcal{S}$为节点集
- 图包含的边标签集合 $l_e\in\{1,\dots,L_{\mathcal{E}}\} \rArr h_{\mathcal{S}} = \{l_e|e\in\mathcal{S}\}$ 当$\mathcal{S}$为边集
- 函数$IN(v)=\{v'|(v',v)\in\mathcal{E}\}$ 代表$v$的前继节点集合
- $OUT(v)=\{v'|(v,v')\in\mathcal{E}\}$
- $NBR(v)=IN(v)\cup OUT(v)$
- $CO(v)=\{(v',v'')\in\mathcal{E}|v=v'\bigvee v=v''\}$ 与顶点$v$相接的边集

GNN的两个步骤：

1. 传播步骤：计算节点的表示
2. 输出模型：节点表示和相应标签映射到一个输出：$$o_v=g(\bold h_v,l_v)$

参数的依赖性隐式保留。由于系统端到端可微，所以所有参数都是基于梯度优化的联合学习。

### 2.1 传播模型

计算与结点$v$相关所有变量得到节点表示，不断重复直至收敛。

$$\begin{aligned}
\bold h^{(t)}_v&=f^*(l_v, l_{Co(v)},l_{NBR(v)},h^{(t-1)}_{NBR(v)}) \\
&= \sum_{v'\in IN(v)} f(l_v,l_{(v',v)}, h^{(t-1)}_{v'})+\sum_{v'\in OUT(v)}f(l_v, l_{(v,v')},l_{v'},h^{(t-1)}_{v'})
\end{aligned}$$

$f(\cdot)$可以为线性函数或神经网络

$$\sum_{v'\in IN(v)} f(l_v,l_{(v',v)}, h^{(t)}_{v'}) = A^{(l_v,l_{(v',v)},l_{v'})}h^{(t-1)}_{v'} + b^{(l_v,l_{(v',v)},l_{v'})}$$

### 2.2 输出模型与学习

输出模型定义各节点的可微函数$g(\bold h_v,l_v)$
最终的输出函数$o_v=g(\bold{h}^T_v,l_v)$通过映射最终节点表示$\bold{h}^{(T)}_v$

## 3 GGNN(Gated Graph Neural Networks)

### 3.1 结点标注

为了区分作为输入的结点标签，使用向量$\bold{x}$表示标注。这些标注在传播模型中会用到。

### 3.2 传播模型

传播模型的基本递推如下：
![20200529222744](https://raw.githubusercontent.com/bysen32/PicGo/master/20200529222744.png)

1. 公式1 为初始化步骤，复制节点的标注至隐藏态$\bold h_v$的第一个组件，余下的用0填充
1. 公式2 是传递信息的步骤，从图中不同节点间由带参的出入边传递信息，参数取决于边的类型与方向
   $\bold{a}_v^{(t)}\in\R^{2D}$包含双向边的激活
1. 余下部分公式3~6 为GRU-like(门限循环单元)的更新策略，从其他节点合并信息，从前一个$timestep$去更新每个节点的隐藏态
   $z$为**更新门限**，$r$为**重置门限**，$\sigma(x)=1/(1+e^{-x})$为logistic $sigmoid$函数

![20200529222834](https://raw.githubusercontent.com/bysen32/PicGo/master/20200529222834.png)
矩阵$A = [A^{(OUT)}, A^{(IN)}]\in\R^{D|\mathcal{V}|\times2D|\mathcal{V}|}$定义了图节点之间的交互方式。
图1展示了图的稀疏结构，每个子矩阵的参数由边的类型与方向决定。
$A_{v:}\in\R^{D|\mathcal{V}|\times 2D}$为节点$v$在矩阵$A^{out}$和$A^{in}$中对应的块
其中$A^T_{v:}\in\R^{2D}$表示为点$v$在$A^{(out)}$和$A^{(in)}$中的两个列块

==门限是做啥用的？==
通过学习，用向量点乘保留重要信息，过滤不重要的信息。可以视为一种注意力机制

### 3.3 输出模型

一步输出：

1. GG-NNs 支持一下形式的节点选择任务$o_v=g(\bold h^{(T)}_v,x_v), v\in\mathcal{V}$, 输出节点分数，并应用一个$softmax$函数
2. 对于图级输出，定义图级表示向量
   $$h_{\mathcal{G}}=tanh(\sum_{v\in\mathcal{V}}\sigma(i(\bold h^{(T)}_v,\bold x_v))\odot tanh(j(\bold h^{(T)}_v,\bold x_v)) \tag{7}$$
   1. $\sigma(i(\bold h^{(T)}_v, \bold x_v))$ 作为与当前图级任务的soft注意力机制，决定当前图任务的相关节点。
   2. $h^T_v, x_v$作为输入，经由神经网络$i,j$ 输出实值向量
   物理含义：每个节点通过两个神经网络训练，分别得到一个$score$向量，将两个向量进行关联。累加所有输出结果，最终得到一个**图级节点向量**

## 4 GGS-NNs (Gated Graph Sequence Neural Networks)

若干个GG-NNs输出序列 $o^1,o^2,\dots,o^K$
对于第$k$个输出步，定义节点注释$\mathcal{X}^k=[x^k_1;\dots;x^k_{|\mathcal{V}|}]^T \in\R^{|\mathcal{V}|\times L_{\mathcal{V}}}$
使用两个GG-NNs，$\mathcal{F}^k_o,\mathcal{F}^k_{\mathcal{X}}$，都包含传播模型和输出模型

- $\mathcal{F}^k_o$：从$\mathcal{X}^k$预测$\bold{o}^k$
- $\mathcal{F}^k_{\mathcal{X}}$：从$\mathcal{X}^k$预测$\mathcal{X}^{k+1}$
- 节点向量矩阵 $\mathcal{H}^{(k,t)} = [h^{(k,t)_1};\dots;h^{(k,t)}_{|\mathcal{V}|}]^T\in\R^{|\mathcal{V}|\times D}$
  - $k$：第$k$个输出步
  - $t$：第$t$个传播步
- 另外，$\mathcal{F}^k_o,\mathcal{F}^k_o$可以共享同一个传播模型，单独的输出模型

![20200601203514](https://raw.githubusercontent.com/bysen32/PicGo/master/20200601203514.png)

==先看到这里==
