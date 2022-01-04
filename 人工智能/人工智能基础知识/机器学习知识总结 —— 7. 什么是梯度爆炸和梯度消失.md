@[toc]

# 因「激活函数（Activation Functions）」而带来的新问题

在 [《深度学习知识总结 —— 5. 什么是过拟合和欠拟合》](https://seagochen.blog.csdn.net/article/details/121042549) 里，我们为了避免「过拟合」的情况出现，有时候我们需要删减一些不必要的数据，或者对数据进行某种程度的正则化。这通常是通过一组功能和特点各异的 **「激活函数（activation functions）」** 完成的。

但是又因为激活函数本身的特点，在我们输出参数过小，或过大时又会带来  **「梯度消失」** 和  **「梯度爆炸」** 的问题。

## 从一个简单的线性模型开始
由于大部分复杂的神经网络模型可以用数个简单的线性公式组合得到（这类似于积分的原理），所以为了说明什么导致了我们用最基本的线性公式为例：

$$
f(x) = \omega * x + b
$$

为了更好说明本章所述内容，我们再加上一个比较常见的 sigmoid 函数作为它的非线性输出

$$
\sigma = \frac{1}{1 + e^{-z}}
$$

现在我们构成了一个网络模型最基本的形式：

$$
Output = \sigma(\omega, x, b) = \sigma f(x)
$$

在我们进一步展开讨论前，先记住 sigmoid 函数图像，是下面这个样子

![在这里插入图片描述](https://img-blog.csdnimg.cn/1660db336de9466e81e7436cc754a8aa.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)
已知，如果要让预测数据 $\hat y$ 趋近于观测数据 $y$，我们可以通过更新权重 $\omega$。而更新 $\omega$ 则需要依赖「损失函数」作为评价指标，所以这里我们使用最常见的MSE函数

$$
L(\omega) = \frac{1}{n} \sum (\sigma (x \omega + b) - y)^2
$$

然后通过评价函数与链式法则，我们得到了下面这个权重更新公式

$$
\hat \omega = \omega - \lambda \frac{\partial L}{\partial \omega}
$$

在很多教科书或者资料里 $\partial L / \partial \omega$ 又被写成梯度算子的形式 $\triangledown_{\omega}$ L 。

> 对于这个过程不是很清楚的朋友，可以通过章节 [《深度学习知识总结—— 2. 计算图与反向传播》](https://seagochen.blog.csdn.net/article/details/118082114) 了解具体的信息。

我们的目标是找到最优解，它的过程经常被直观地表述为从山顶下降到山底的过程，也就是所谓的「梯度下降」，如同下面这张图

![在这里插入图片描述](https://img-blog.csdnimg.cn/ba806bb1cf63424796ce7b66e661cc71.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 网络中，输出预测值的权重与实际被观测值的权重差，经常被描述为二维，或者更高维度空间的「高度」；我们从空间中任意一点开始，通过比较临近梯度大小，最终实现网络权重 $\omega$ 的收敛，即找到「最优解」的过程，被称为「梯度下降」。

## 梯度是怎么消失的

现在，我们来看看这个简单的网络模型是什么样的。首先数据经由一个线性单元，进行权重和映射处理：

$$f(x) = \omega x + b$$

我们知道 $f(x) = \omega x + b$ 输出的是线性图像。当 $\omega > 0$ 时，它的函数图像位于1、3象限，$b$ 可以调节函数关于 $y$ 轴的相位，当 $b = 0$ 时的图像穿过原点。

如果把计算过程想象成映射，同时我们假定输入的 $x$ 的范围限定在 $[-10, 10]$ 之间，且我们不采用 $b$ 调整线性模型的相位，那么由上述公式可以知道，$y$ 的范围也必然限定在 $[-10 \omega, 10 \omega]$ 之间。

$f(x)$ 的输出变成了 $\sigma$ 的输入，由 $\sigma$ 的公式可知，它的输出范围最终被限定在 $(0, 1)$。之后我们来到计算损失函数的步骤，我们观察损失函数

$$
L(\omega) = \frac{1}{n} \sum (\sigma - y)^2
$$

如果我们不求和每个 $\sigma$ 与 $y$ 的方差，而是把它们作为矩阵存储起来，即

$$
\mathbf{L} = \left | \begin{matrix}
l_{11} & l_{12} & \cdots & l_{1n} \\
l_{21} & l_{22} & \cdots & l_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
l_{m1} & l_{m2} & \cdots & l_{mn}
\end{matrix} \right |
$$

于是它得到了一个关于 $\mathbf{L}$ 的导数矩阵 $\triangledown_{\omega} L$， 其中 $\partial l / \partial \omega$ 由链式法则展开后得到：

$$
\frac{\partial l_{ij}}{\partial \omega} = \frac{\partial l}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial y} \cdot \frac{\partial y}{\partial \omega}
$$

而它所对应的梯度矩阵就变成了

$$
\mathbf{\frac{\partial L}{\partial \omega}} = \left | \begin{matrix}
l_{11}' & l_{12}' & \cdots & l_{1n}' \\
l_{21}' & l_{22}' & \cdots & l_{2n}' \\
\vdots & \vdots & \ddots & \vdots \\
l_{m1}' & l_{m2}' & \cdots & l_{mn}'
\end{matrix} \right |
$$

我们依据导数的一般定义可以得到

$$
\frac{\partial l}{\partial \sigma} = 2 l_{ij} \\
\frac{\partial \sigma}{\partial y} = \frac{e^{-x}}{(1 + e^{-x})^2} \\
\frac{\partial y}{\partial \omega} = x
$$

于是我们可以得到下面这个公式：

$$
\frac{\partial l}{\partial \omega} = 2 l_{ij} x \frac{e^{-\omega x}}{(1 + e^{-\omega x})^2} 
$$

代入学习率 $\lambda$ 后得到一个新的权重 $\hat \omega$

$$
\hat \omega_{ij} = \omega_{ij} - \lambda \times 2 l_{ij} x \frac{e^{-\omega x}}{(1 + e^{-\omega x})^2}
$$

现在我们来分析这个计算过程，会很容易发现上述过程，很容易在两个地方出现异常。

* 第一个容易出现异常的是在正向传播时，由于 sigmoid 这类函数的原因，会导致 $\sigma(f(x))$ 输出的权重过小，而使得 $l_{ij}$过小，从而导致 $\triangledown_{\omega} L$ 过小，权重更新缓慢。

> 我们令 $K=2 l_{ij} x \frac{e^{-\omega x}}{(1 + e^{-\omega x})^2}$ , 若 $K \ll 1$ 就会导致 $\lambda \times K \ll 1$ （学习率 $\lambda$ 通常是一个小于1的数）。所以这会导致 $\hat \omega = \omega - \lambda \times K$ 更新过于缓慢。

* 第二个容易出现的问题是在反向传播时，由于初时函数过大或过小，导致 $\partial \sigma / \partial y$ 直接 “ 归0” ，从而导致 $\triangledown_{\omega} L$ 过小，权重更新缓慢。

> 这个问题也会直接导致 $K \ll 1$，从而使 $\hat \omega$ 更新过于缓慢。

对于权重更新来说，上述情况可以简化为 $\triangledown_{\omega} L = 0$，于是

$$
\hat \omega = \omega - \lambda \times 0 = \omega
$$

权重无法得到更新。所以，在观察 $\mathbf{L}$ 矩阵生成的过程，发现元素的导数迅速从有效值 $V \rightarrow 0$ 的过程被称为 **「梯度消失」**。

## 梯度又是如何爆炸的
现在我们讨论另外一个情况，就是线性模型输出了结果后，我们不经过「激活函数」又会出现什么情况呢。由公式 $f(x)$ 可知

$$f(x) = \omega x + b$$

没有经过「非线性处理」前，输出值和输入值之间保持线性关系。我们也知道对于深度学习网络来说，它可以是多个线性矩阵相乘，于是

$$\mathbf{M}_{out} = \mathbf{M}_1 \times \mathbf{M}_2 \cdots \times \mathbf{M}_n$$

根据矩阵叉乘的运算法则，最终我们得到的方差矩阵其元素 $l_{ij}$ 会变成一个极为大的数

$$l_{ij} = \omega_i \times \omega_j^T \cdot x$$ 

> 比如对于如下一个归一化权重矩阵，它一开始的值是
> ```bash
> tensor([[-0.2167,  1.0562, -1.1602, -0.5125],
>        [-0.9953, -0.3229,  0.6139, -0.3564],
>        [-0.5894,  0.6944, -0.7655, -0.3474],
>        [-0.0225,  1.2729, -0.4987,  0.8055]])
>```
> 如果这样的矩阵执行10次后，得到的值就会变成这样：
> ```bash
> tensor([[-117.1322,  -95.8194,   38.9671,  -73.8220],
>        [  52.0601,   43.9106,  -17.0622,   33.3657],
>        [ -78.9374,  -64.8063,   26.1892,  -49.8487],
>        [-129.1926, -110.5344,   41.8324,  -83.4628]])
>```

我们的系数 $K$ 由于 $l_{ij}$ 被放大，$\lambda \times K$ 也可以得到很大的数，使得权重更新公式的「节奏感」被破坏；$\hat \omega = \omega - \lambda \times K$ 的每一次计算，都会导致新权重远大于旧权重，$|\hat \omega| \gg |\omega|$，这简直像在建筑顶做「信仰之跃」。

![在这里插入图片描述](https://img-blog.csdnimg.cn/199b6800582a4f858f875a7cbcce214d.png#pic_center)
阿萨辛会不会摔死我不知道，但是「梯度下降」算法一定会摔个粉碎，函数收敛曲线出现剧烈抖动的情况。

# 预防梯度消失、爆炸的办法
经验性的，我们也是可以通过以下方法来干预梯度的消失或爆炸。

## 逐层微调
在训练时，我们可以把模型逐层微调；以上一层的输出作为本层的输入，观察本层的输出是在快速增大还是减少。

## 梯度剪切

对梯度剪切，其核心思想是设定一个映射阈值，比如我们只允许输出值在 [0, 10]，它可以用以下公式表示

$$
f(x) = \left\{\begin{matrix}
a  & x < a \\
b & b > b \\
x & a \leq x \leq b
\end{matrix}\right.
$$

梯度裁剪是一种能防止「梯度爆炸」的很简单但十分有效的方法。

## 正则化
通常情况下，这里提到的正则化不是对输入参数 $x$ 或者输出参数 $f(x)$ 的正则，而是对于网络权重 $\omega$ 的正则，所以如果一旦发生梯度爆炸的情况，能够在一定程度上起到限制梯度爆炸的作用。一般情况下，在机器学习领域，用的比较多的是 L1 正则和 L2 正则。

## ReLu、LeakyReLu 等激活函数
ReLu等非线形激活函数，某种程度有点类似一个单向阀或者二极管，具备单向导通性。

![在这里插入图片描述](https://img-blog.csdnimg.cn/06eb6e9f2548457da59252fe3752e073.png#pic_center)
> 对于二极管来说，电流只能通过二极管的正极流向负极，除非负极有极大的电压，才可能会导致二极管被击穿。

这一类函数在 $x \ge 0$ 时，$f(x) = x$，而 $x < 0$时，$f(x) = 0$ 或 $f(x) = a x$，其中 $a < 1$。

由于Relu函数的特性决定了它的导数在 $x>0$ 时恒为1，所以对于由 sigmoid 函数构成的网络来说，它可以解决梯度消失和爆炸的问题，这是因为这类网络的导数通常可以被表示为

$$
\frac{d L}{d \omega} =2 l_{ij} x 
$$

不过，这类函数由于由于剪掉了负数部分，所以对于一些特殊情况可能无法正确反应。所以可以使用改良型的 LeakyReLu 或 eLu。
