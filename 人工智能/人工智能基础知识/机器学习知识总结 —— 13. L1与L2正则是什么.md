@[toc]

# 先从拉格朗日约束「Lagrange Constraint」开始

要理解L1、L2正则是什么含义，我们就要回归最基础的一个概念——拉格朗日数乘。提出拉格朗日数乘的根本原因，就是**寻找多元函数在其变量受到一个或多个约束条件限制时的极值问题**。就是把 $n$ 个变量与 $k$ 个约束条件转化为 $n + \lambda k$ 的方程组问题。

所以，从拉格朗日数乘（或者拉格朗日约束）的定义出发，对于方程 $f(x_1, x_2, \cdots, x_n)$ 的诸多个约束条件 $g(x_1, x_2, \cdots, x_n)$ 的极值问题，就可以表示为下面这样的形式：

$$
L(x_1, x_2, \cdots, x_n; \lambda_1, \lambda_2, \cdots, \lambda_n) = f(x_1, x_2, \cdots, x_n) + \sum_{i=1}^{n} \lambda_i g(x_1, x_2, \cdots, x_n)
$$ 

对于拉格朗日数乘基础概念不理解的朋友，可以看这篇基础[文章](https://seagochen.blog.csdn.net/article/details/121614426)。

所以，对于MSE方程。如果在没有约束的前提下，那么对于方程 $\left \| y - \omega x - b\right \|^2 < \varepsilon$，它求解出的解在欧拉几何空间中，倾向于一条直线，或者一个平面。

在机器学习中，这通常意味着满足  $\left \| y - \omega x  - b \right  \|^2 < \varepsilon$ 的 $(\omega, b)$ 的组合其实可以有很多。

如果对于一般的数学问题来说，这样的结果不能说不好。

但是对于机器学习，如果我们试图在趋向 $y$ 的 $\omega x + b$ 中找到一个最适解，就要考虑引入约束条件。

比方说，对于椭圆问题 $f(x, y)$，我们希望在一个曲线方程 $g(x, y)$的约束下找到极值点，而满足这样的极值点通常只有一个或数个点（两凸函数相切的时候通常只有一个点，凸函数与必入双曲线相切时，可能有两个或数个点）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1bd383ba80b94f61b81d373da4a26941.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
比方说对于上面这个问题，在给定约束条件 $g(x, y)$，向内收缩的 $f(x, y)$的极值点显然只有当两个函数相切。所以，在理解这个问题后，我们再回来看L1，L2正则就能很快理解它为什么要这样定义的了。


# L1正则与L2正则「L1 and L2 Regularization」

我们从拉格朗日约束出发，得到一般形式，即：

$$
L = f(x_1, x_2, \cdots, x_n) + \sum_{i=1}^{n} \lambda_i g(x_1, x_2, \cdots, x_n)
$$ 

对于机器学习来说，我们期望找到与目标方程误差最小的预测方程，在这目标的影响下，我们通常会使用方差函数作为我们的评价函数，也称损失函数，于是有

$$
f(\cdots) =  \sum (y - \omega x)^2
$$

当然也把上式写成矩阵的形式

$$
f(\cdots) =  \sum (y - \omega x) (y - \omega x) ^{T}
$$

不过我个人一般更喜欢第一种写法。然后我们来套入约束方程，也就是 $g(\cdots)$。目前已公布的约束方程主要有两类，一类是 $\left \| W \right \|_1$，还有一类是 $\left \| W \right \|_2^2$。

这里的 $\left \| W \right \|_1$ 和 $\left \| W \right \|_2$ 不是求摸也不是求绝对值，而是另一个数学概念 「**范数**」。

根据定义，范数是指

>对于向量 $x = [x_1, x_2, \cdots, x_n]^T$，它的P范数就定义为：
> $$
> \left \| x \right \|_p = (|x_1|^p + |x_2|^p + \cdots + |x_n|^p)^{\frac{1}{p}}
> $$

所以，对于 $\left \| W \right \|_1$ 即关于W的L1范数，它表示为

$$
\left \| W \right \|_1 = |W_1| + |W_2| + \cdots + |W_n|
$$

而 $\left \| W \right \|_2$ 则是关于W的L2范数，它表示为

$$
\left \| W \right \|_2 = \sqrt{ |W_1|^2 + |W_2|^2 + \cdots + |W_n|^2 }
$$

但是L2正则又多了一个平方，于是  $\left \| W \right \|_2^2$ 自然就变成了

$$
\left \| W \right \|_2^2 = |W_1|^2 + |W_2|^2 + \cdots + |W_n|^2 
$$

在欧式空间中，L1和L2正则的约束条件，反映为图像分别为棱形和圆形，在下图所示的淡黄色部分

![在这里插入图片描述](https://img-blog.csdnimg.cn/046b6b46bdd247f99b85955ccc8528b4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


对于机器学习，我们一般只用到L1范数和L2范数，而且只有 $P \ge 1$ 的时候，范数的点所构成的点才有可能是一个凸集合，而针对凸集的优化问题，是一个数学上极为简单的问题。

我们在前面已经说过，如果网络权重和偏置量不做限制，那么对于损失函数 $\left \| y - \omega x  - b \right  \|^2 < \varepsilon$ 它一定能找到多组满足最小误差的组合 $(\omega, b)$，所以我们要给这个损失函数做一个限制，也就是要求 $(\omega, b)$ 必须在一定范围内才可以有效，那么我们就可以在损失函数 $f(\omega, b)$ 之后加入限制条件，而这个限制条件可以是L1和L2范数，于是我们得到：

L1正则式

$$
L(\omega, b) = J(\omega, b) + \lambda \sum \left \| \omega \right \|_1
$$


L2正则式

$$
L(\omega, b) = J(\omega, b) + \lambda \sum \left \| \omega \right \|_2^2
$$

这里的 $J(\omega, b)$ 是损失函数，可以是我们先前提到的MSE，也可以是交叉熵或者其他。求解过程就是在搜索最小值 $\min J(\omega, b) \to \varepsilon$ 的过程中，确保网络权重 $\omega$ 能够被限制在 $L1$ 和 $L2$ 范数里。

也就是说，在求解网络权重 $\omega$ 的过程中
* 确保 $\omega$ 与原点之间的距离不超过 $L1$ 距离 $| \omega | \leq C$（L1正则）；
* 或不超过$L2$距离 $\omega^2 \leq C$ （L2正则）。
