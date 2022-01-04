> 这一章的知识点其实相当于知识总结，因为基础的内容基本已经在前面的章节中探讨过。但是我这两天思考了一下，觉得有必要把神经元的知识点全部抓出来讨论一下，做一个相当于系统性的概述，这样我们在之后的讨论中，就不会涉及到太多数学性的话题，而更多的可以放在拓扑结构或其他内容上了，比如具体的应用上。

@[toc]

# 关于前面内容的回顾

首先，在前面的章节里，我论述过一个完整的神经元的计算单元，其运算过程如下：

```mermaid
flowchat
st=>start: 输入权重
e=>end: 输出权重
op1=>operation: 向前传播并更新参数
op3=>operation: 通过激活函数映射到新函数空间
op2=>operation: 向后传播并更新权重
cond=>condition: 损失函数->符合期望？

st->op1->op3->cond
cond(yes)->e
cond(no)->op2
op2->op1
```

这里，向前传播的过程，是线性的矩阵运算，而向后传播的过程，则是对导数的运算，并通过学习率 $\Delta$ 进行参数更新，然后周而复始，直到迭代到损失函数最小、符合期望时输出最后的权重。

反向传播的这个过程中，由于出现了梯度下降，所以又把这一过程称为梯度下降算法。这在我前面写的博客里已经提到过，所以不再赘述。

[深度学习知识总结——1.1. 什么是梯度下降](https://seagochen.blog.csdn.net/article/details/116401539) 
[深度学习知识总结——1.2.梯度下降算法实现](https://seagochen.blog.csdn.net/article/details/117419033)

然后接下来，我们探讨了计算图对于帮助我们构建运算模型以及如何在反向传播中进行求导。并且简单的论证了，所谓神经元的网络拓扑结构，本质上就是一堆复杂函数的计算图的各种叠加。

[深度学习知识总结——2. 计算图与反向传播](https://seagochen.blog.csdn.net/article/details/118082114)

但是这里呢，和框架，例如Pytorch和Tensorflow所对应的神经元拓扑结构还是稍稍有点出入，至于这个出入在哪里也是我这个文章即将要介绍的。并且将告诉你，如果你自己要构建一套复杂的神经元网络模型，你还应该掌握的基本知识。

然后，就是为了便于我们打破线性输出，为了让模型适用于非线形问题，而引入了激活函数的概念，并且用了一个非线形的分类问题，证明了这种思路是可行的。

[深度学习知识总结——3. 激活函数与非线性输出](https://blog.csdn.net/poisonchry/article/details/118526467?spm=1001.2014.3001.5501)

那么接下来我们要讨论一个比较现实的问题，就是现实的应用场景，如果可以被观测、被建模的，通常都倾向是一个复合函数。换句话说，我们不太可能只用一个一次函数、或者二次函数就能求解问题。

# 构建复杂的函数模型

比如说，对于医学生来说，某些场合下需要使用神经元网络学习并调试心电ECG信号的滤波参数。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709101024134.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
仅仅是生成这个模拟信号本身就无法简单的用几个一次函数线性叠加就能做出来。通常情况下，一个业务模型，通常可以被表示为如下形式

$$z(g(f(x)))$$

也就是函数 $y = f(x)$ 的输出将作为 $w = g(y)$ 的输入，而 $z(w)$ 的最终结果，由 $w$ 决定。如果这个神经元网络不能并行的处理数据，只能一次一个数据的输入和输出，那么在其拓扑结构上，就会表现为如下形式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709140153682.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

假如，我们在 $f(x)$ 节点上，执行的是线性模型计算，即 $f(\omega) = \omega x + b$, 在 $g(Y)$ 以及 $z(W)$ 上也执行的是线性模型计算，即：

$$f(\omega) \rightarrow g(y) \rightarrow z(w)$$

如果我们需要处理的数据很简单，比如说只有1个的情况，那么网络复杂程度也就如上图一样所示。但是，实际上我们知道，我们需要处理的数据量经常特别大，而计算机比人脑最大的优势在于它可以同时并行的处理数据，所以这个时候，我们通常会考虑能不能把矩阵应用到神经元当中去。

## 矩阵与并行计算

如果详细探讨矩阵与并行计算，这将会是一个可以展开成另外一个系列的话题，在我所知道的范畴，有一个专门的学科叫——数值计算与并行处理。这里我只提一点它的内容，这对于我们进一步理解神经元网络结构十分有帮助。

矩阵是一种很重要的工具，可以把它理解为函数或者运算。对于 $y= \omega x + b$ 这样一个线性模型，如果我们要一次性处理多组数据的话，那么就需要使用到矩阵工具。此时原函数我们就要变成这样的形式

$$y = \begin{bmatrix}
\omega_1 \\ 
\omega_2 \\ 
\omega_3 \\
\cdots \\
\omega_n
\end{bmatrix} \times \begin{bmatrix}
x_1 & x_2 & \cdots & x_n
\end{bmatrix} + b$$

为什么要这样写而不是反过来，你可以试一试调换行后是什么结果。这里再次强调一遍，$\omega$ 是我们需要求解线性模型的权重，而 $x$ 是样本数据。根据外积规则，所以有：

$$y = \begin{bmatrix}
\omega_1 x_1 & \omega_1 x_2 & \cdots & \omega_1 x_n \\ 
\omega_2 x_1 & \omega_2 x_2 & \cdots & \omega_2 x_n \\ 
\vdots & \cdots & \ddots & \vdots \\
\omega_n x_1 & \omega_n x_2 & \cdots & \omega_n x_n
\end{bmatrix} + b$$

b是常数，根据广播规则，有

$$y = \begin{bmatrix}
\omega_1 x_1 + b & \omega_1 x_2 + b & \cdots & \omega_1 x_n + b \\ 
\omega_2 x_1 + b & \omega_2 x_2 + b & \cdots & \omega_2 x_n + b \\ 
\vdots & \cdots & \ddots & \vdots \\
\omega_n x_1 + b & \omega_n x_2 + b & \cdots & \omega_n x_n + b
\end{bmatrix} $$


关于矩阵的基本运算，请参考我之前写过的一篇文章内容 [线性代数—— 基本矩阵运算公式](https://seagochen.blog.csdn.net/article/details/114661205)

然后可以发现，上式从左往右，上下相邻的列元素彼此之间逻辑上不存在着关联关系，每一行的元素则是迭代运算。所以，我们可以用并行的思路来处理问题了

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709151147466.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
我们可以把上面一大坨的运算全部压入一个线程里执行，也可以分成几个Batch，分别在不同的线程里执行。通常使用CPU做运算的话，由于Intel内核限制，目前主流是4核16线程，理论上最大性能输出就是分成16个Batch，分别到16个不同线程上执行运算了。如果使用GPU的话，那么可以成百倍的计算了。

如果我们计划把 $\omega$ 分割成4个Batch，并且启用4个线程做运算：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709152148773.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
那么根据矩阵的运算法则，就需要对原来的数据集进行适度的划分。例如对于全部在一个Batch时，有


$$y = \begin{bmatrix}
\omega_1 \\ 
\omega_2 \\ 
\omega_3 \\
\cdots \\
\omega_n
\end{bmatrix} \times \begin{bmatrix}
x_1 & x_2 & \cdots & x_n
\end{bmatrix} + b$$

那么，变成了4个Batch时，此时，这个就要变成：


$$y_{11}= \begin{bmatrix}
\omega_1 \\ 
\omega_5 \\ 
\cdots \\
\omega_{n-3}
\end{bmatrix} \times \begin{bmatrix}
x_1 & x_5 & \cdots & x_{n-3}
\end{bmatrix} + b$$

$$y_{12}= \begin{bmatrix}
\omega_1 \\ 
\omega_5 \\ 
\cdots \\
\omega_{n-3}
\end{bmatrix} \times \begin{bmatrix}
x_2 & x_6 & \cdots & x_{n-2}
\end{bmatrix} + b$$

$$y_{13}= \begin{bmatrix}
\omega_1 \\ 
\omega_5 \\ 
\cdots \\
\omega_{n-3}
\end{bmatrix} \times \begin{bmatrix}
x_3 & x_7 & \cdots & x_{n-1}
\end{bmatrix} + b$$

$$y_{14}= \begin{bmatrix}
\omega_1 \\ 
\omega_5 \\ 
\cdots \\
\omega_{n-3}
\end{bmatrix} \times \begin{bmatrix}
x_4 & x_8 & \cdots & x_n
\end{bmatrix} + b$$


$$y_{21} = \begin{bmatrix} 
\omega_2 \\ 
\omega_6 \\
\cdots \\
\omega_{n-2}
\end{bmatrix} \times \begin{bmatrix}
x_1 & x_5 & \cdots & x_{n-3}
\end{bmatrix} + b$$


$$\cdots$$


$$y_{44} = \begin{bmatrix}
\omega_4 \\ 
\omega_8 \\ 
\cdots \\
\omega_n
\end{bmatrix} \times \begin{bmatrix}
x_4 & x_8 & \cdots & x_n
\end{bmatrix} + b$$

即，从原来1行1列共1组输出，变成了4行4列共16组输出。由于，

$$f(\omega) \rightarrow g(y) \rightarrow z(w) \rightarrow Output$$

那么后面跟随而来的 $g(y)$ 以及 $z(w)$ 也要根据矩阵运算法则相应的调整输出和输入，于是可以变成这样的表达形式：

$$I_{1 \times 4}F_{4\times 4} G_{4 \times 8} Z_{8 \times 1} = O_{1 \times 1}$$

如果这个时候，我们再构建一次神经元的拓扑结构的时候，就变成了这样的形式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210709160057735.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

这里稍微多说一点，比如在Pytorch里，线性模型公式可能写作

$$y= x \omega^T  + b$$

这都是这些框架在当初构建时，定义输入维度的方向，但问题的根本思考方式是一样的。


## 计算图与神经元网络的关系

我之前的一些文章里提到过计算图和神经元网络拓扑结构的关系，但是后来想了想其实觉得不是很妥当，也不准确。尽管整个神经元网络的拓扑结构，如果拆开细微观察的话，确实是由一个个不同的函数组合在一起的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210703081721954.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

神经元网络通过层，决定了构建的数学模型中，使用了哪些基本函数模型，例如一次线性模型，卷积模型等，而每一层的节点数则决定了在整个网络模型中，参数、权重是如何划分和传导的。

我们利用计算图构建导数的链式法则，它在网络中扮演的更多的类似于细胞之于组织器官的作用。就好像肌肉和肌肉细胞，通过有机的组织在一起，从而使的器官具备了某些特殊的生理功能一样。计算图和节点也是通过这种有机的组合后，具备了非凡的“智能”。