@[toc]

> 这一章节实际上是对之前 [《深度学习知识总结—— 3. 激活函数与非线性输出》](https://seagochen.blog.csdn.net/article/details/118526467) 的补充。随着自己的工作内容的深入，发现自己在一些概念的理解上过于浅薄，在参考了[《深度学习领域最常用的10个激活函数，详解数学原理及优缺点》](https://cloud.tencent.com/developer/article/1800954) 基础上，做一些必要的补充说明。

我们使用激活函数的主要目的，有三：
* 打破矩阵运算之间的「线性关系」；
* 避免或降低模型「过拟合」；
* 调整模型梯度生成情况。

然后我们接下来就常用的十类激活函数进行说明。


# 1. sigmoid 函数

## 1.1. 函数原型

$$
\sigma(x) = \frac{1}{1+ e^{-x}}
$$

输出是S型曲线，具备打破网络层与网络层之间的线性关系，可以把网络层输出非线形地映射到$(0, 1)$ 区间里。函数的特性，决定了它能够避免或降低网络模型过拟合情况的发生，但是这种函数最大的缺陷在于容易出现「梯度消失」的情况。

## 1.2. 函数图与梯度图
* 红色为原始函数图像
* 蓝色为函数导数图像

![在这里插入图片描述](https://img-blog.csdnimg.cn/590e4167cd71488b86c27347ebfdf5b6.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 原函数的值域区间为 $(0, 1)$，从导数来看其在 $[-2, 2]$ 区间尤其是接近0轴的导数较大，但是函数最大值依然小于1，所以当多个 $d \sigma$ 相乘时很容易导致梯度变为极小值，使权重更新缓慢。


# 2. tanh 函数
## 2.1. 函数原型

$$
\tanh x = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

输出是S型曲线，具备打破网络层与网络层之间的线性关系，可以把网络层输出非线形地映射到$(-1, 1)$ 区间里。函数的特性，决定了它能够避免或降低网络模型过拟合情况的发生，相较于 sigmoid 函数不容易出现梯度变为极小值，导致权重更新缓慢的问题。

> 可作为 sigmoid 函数的替代函数。

## 2.2. 函数图与梯度图

* 绿色为原始图像
* 紫色为导数图像

![在这里插入图片描述](https://img-blog.csdnimg.cn/98df081249564476902c8c2b7ec68ce8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 原函数的值域区间为 $(-1, 1)$，从导数来看其在 $[-2, 2]$ 区间尤其是接近0轴的导数较大，但是函数最大值为1，不容易出现梯度消失的情况，但是对于x在 $[-2, 2]$ 之外的值，其导数接近0，所以 **一定要确保 $f(x)$ 输出的数据在进入到 $\tanh$ 之前，都已经做了正则化处理**。

# 3. ReLu 函数
## 3.1. 函数原型

$$
ReLu = \left \{ \begin{matrix}
x & x > 0 \\
0 & x \leq 0
\end{matrix} \right.
$$

尽管计算速度快，但是需要注意一点，由于 $f(x) \ge 0$ 时 $ReLu(f(x)) = f(x)$，在求导时很可能因为多个 $\frac{d}{dx}f(x)$ 连续相乘而出现梯度爆炸出现，所以必要的时候应该配合 $\tanh$ 或 sigmoid 函数使用。

> ### Dead ReLu 问题
> 在输入数据小于0的时候，输出为0。如果 $f(x)$ 的输出中有不需要处理的负值，可以考虑使用这个函数。另外由于它的梯度恒为1，所以函数本身不存在梯度消失或爆炸的问题，通常配合 sigmoid 函数或 $\tanh$ 函数使用，也可以单独使用。


## 3.2. 函数图
![在这里插入图片描述](https://img-blog.csdnimg.cn/fdcb14cdb2ef432299ff83b4a19b0915.png#pic_center)

> 虽然大多数情况下，我们更关心的是概率问题 $[0, 1]$，但是在网络层传递过程中有些特殊情形是一定需要负值参与的。比如某些条件的成立需要某两个参数之间是「负相关」，而由于 $x < 0$ 时 $y = 0$，它会导致模型对这部分输入没有相应，从而影响精度。

# 4. Leaky ReLu 函数
## 4.1. 函数原型

$$
LeakyReLu = \left \{ \begin{matrix}
x & x > 0 \\
c x & x  \leq 0
\end{matrix} \right.
$$

$c$ 是可调节权重允许 $[0, 1]$，但是通常习惯上只使用到 0.01 左右。

不会出现 Dead ReLu 问题，但是关于输入函数 $f(x)$ 的部分容易出现梯度爆炸的情况是一样的，所以必要时，也可以搭配 sigmoid 或 tanh 使用。
## 4.2. 函数图

![在这里插入图片描述](https://img-blog.csdnimg.cn/896c512e1aff4e3ebbdcaae915ae925d.png#pic_center)
> 允许负值一定程度上参与到计算中，比 ReLu 函数稍微温和一些，所以不存在 Dead ReLu 问题。

# 5. ELU 函数
## 5.1. 函数原型

$$
ELU = \left \{ \begin{matrix}
x & x > 0 \\
c(e^x - 1) & x  \leq 0
\end{matrix} \right.
$$

eLu 也是为了解决 Dead ReLu 而提出的改进型。计算上稍微比 Leaky ReLu 复杂一点，但从精度看似乎并未提高多少。

## 5.2. 函数图
![在这里插入图片描述](https://img-blog.csdnimg.cn/b0fea9abf88f4eec805973221110b8fd.png#pic_center)

# 6. PReLu 函数
## 6.1. 函数原型

$$
PReLu = \left \{ \begin{matrix}
x & x > 0 \\
\beta x & x  \leq 0
\end{matrix} \right.
$$

公式与 LeakyReLu 相似，但并不完全一样。$\beta$ 可以是常数，或自适应调整的参数。也就是说，如果让 $\beta$ 自适应，那么 **PReLu会在反向传播时更新参数 $\beta$**。

# 7. Softmax 函数

## 7.1. 函数原型

$$
Softmax(Z_j) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
$$

概率论和相关领域中出现的一种 **「归一化（normalize）」** 函数。它可以把一个 **「K维」** 数据压入到 「e指数」 空间中，使得每一个元素的范围都在 $(0, 1)$ 之间，并且所有元素的和为1。


![在这里插入图片描述](https://img-blog.csdnimg.cn/fecc493a1fdd4b1ca02e8997664cb2b5.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
Softmax 可以确保较小的值具有较小的概率，并且不会直接丢弃。由于Softmax 函数的分母结合了所有因子，这意味着 Softmax 函数获得的各种概率彼此相关。另一方面，由于 e 指数的限制，对于负值的梯度趋近于0，所以这部分权重不会在反向传播期间更新。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f74fe4d2d1c541808603aa1c4635f6a1.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 注意，对于 $e^x$ 的导数等于 $e^x$本身，所以在负值时，函数左侧数值趋向于0，这会导致数据在反向传播期间无法有效更新。


# 8. Swish 函数
## 8.1. 函数原型

$$
Swish(x) = x \sigma(\beta x) = x \frac{1}{1 + e^{- \beta x}}
$$

$\beta$ 可以是常数或自适应。

如果令 $\beta = 1$，那么方程等价于 「权重 sigmoid 函数（Sigmoid- weighted Linear Unit Function）」可起到如下图所示，类似 ELU的效果
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/ebaf190e9b2545c8ba7b3eaf73b258f0.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 其中：
> * 绿色为原始图像
> * 红色色为导数图像

当 $\beta = 0$时，方程变成 $f(x) = \frac{x}{2}$ 线性方程。

如果我们令 $\beta \rightarrow \infty$，方程会变成如下所示，类似 ReLu 函数的效果。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/ee89091b182d4988abc3e384142e3ea3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
因此，随着 $\beta$ 的变化，函数会非线性地在「线性函数」 和 「ReLu函数」函数间变化。

# 9. Maxout 函数
## 9.1. 函数原型

$$
Maxout(x) = max(\omega_1 x + b_1, \omega_2 + b_2, \cdots, \omega_n x + b_n)
$$

它是极为特殊的一类激活函数，与其他激活函数一开始固定了函数输出的形式不同，它采用分段线性组合，对任意 **「凸函数（convex function）」** 进行线性逼近。

> 注意：
> 国内教材对于凹凸函数的定义与国际相反。国际一般定义凸函数的图像形如开口向上的杯，形似 $\cup$ ，而凹函数则形如开口向下的帽 $\cap$。

我们需要在训练开始前确定使用的线性单元数量，为了获得理想的激活函数，Maxout 使用这些线性单元，采用分段地逼近策略（piece-wise linear approximation），并在最终取值时从分段函数选取最大值作为输出。

![在这里插入图片描述](https://img-blog.csdnimg.cn/d5fae2054ec74f83bc7b6e8d4e1ebd0f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 上图示例了 Maxout 如何逼近 ReLu 函数，绝对值函数，以及任意凸函数。

# 10. Softplus 函数
## 10.1. 函数原型

$$
softplus(x) = \log (1 + e^{x})
$$

它是一种和 ReLu 函数功能作用极象的函数，并且在很多新的模型里，作为 ReLu 的替代。相对于ReLu 或 LeakyReLu 来说，Softplus 有个非常「致命」的优点，就是它在0点处是可导的。

不过相对于 ReLu 的粗暴简单，这个函数的运算耗费时间相对较多。

## 10.2. 函数图

![在这里插入图片描述](https://img-blog.csdnimg.cn/ecc611315bf44d1893377ee6a7607934.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> * 蓝色线条是 Softplus 函数
> * 绿色线条是 ReLu 函数