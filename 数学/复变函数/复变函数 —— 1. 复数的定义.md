@[TOC]

# 0. 什么是「复数」

我们都知道，如果数 $\sqrt {-1}$ 不扩展的话，那么这个计算就毫无意义。所以，为了解决 $\sqrt {-1}$ 的问题，我们让数的概念从实数域扩展到复数域，以 $j^2 = -1$ 表示（某些教材以 $i^2 = -1$表示虚数，但我个人建议你最好习惯使用 $j$ 表示虚数）。这定义有非常重要的作用，它使得任意多项式的方程都有根，如果一个数包含虚数部分，那么可以被表示为

$$
z = x + y j
$$

而 $x$ 称为「实部」，$y$ 称为「虚部」。

# 1. 什么是「共轭复数」

在数学中，复数的共轭复数（常简称共轭）是对虚部变号的运算，因此一个复数 $z = x + yj$ 的共轭可以表示为 $\bar z = x - yj$。如果把复数也像实数那样投影到欧氏空间中，那么互为「共轭」的复数，可以被表示如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/e73deb563ddc4107b143fd552ff43667.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_9,color_FFFFFF,t_70,g_se,x_16#pic_center)
我们以横轴表示实数部，纵轴表示虚数部。所谓共轭的复数就是关于X轴方向对称的两个线段（或者说向量）。因此，它便有了：

* 与实数轴X的夹角为 $\varphi$；
* 模长相等，且长度为 $r = \sqrt{ x^2 + y^2}$；
* 在实数轴X上的投影长为 $|x|$
* 在虚数轴Y上的投影长为 $|y|$
* 或者在球坐标系里，$z = x + yj$ 可以被表示为 $z = r(\cos \varphi + j \sin \varphi)$
* 亦或者在欧拉方程中，以自然指数形式表示为 $z = r e^{j \varphi}$

> 球坐标系下，x，y，z都可以表示为
> 
> $$ 
> \left \{ \begin{matrix}
> x = r \cos \phi \cos \theta \\
> y = r  \cos \phi \sin \theta \\
> z = r \sin \theta
> \end{matrix} \right .
> $$
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/91a7c0784143409d85bb11aa99e0cf9c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

这些性质给复数带来了一些很有意思的应用，比如不同坐标系之间的映射关系：

> 复数 $\frac{1}{2} - \frac{\sqrt 3}{2} j$ 的指数形式为 _______ 三角形式为_____
> 解：
>$$
> r = \sqrt{ \left[ \frac{1}{2} \right ]^2 + \left [  \frac{3}{2} \right ] } = 1
> $$
> 由 $\phi = \arctan (\frac{\sqrt 3}{2} / \frac{1}{2}) = \arctan \sqrt 3$ 得到夹角 $\phi = \pi /3$。
> 又可以从欧拉公式得到
> $$
> z = r e^{j \phi} = e^{\frac{\pi}{3} j}
> $$
>
> 以及由球坐标系得到
> $$
> z = \cos \frac{\pi}{3} + j \sin{\pi}{3}
> $$

# 2. 复数的基本运算规则
## 2.1. 加减运算
$$
(x_1 + y_1 j) \pm (x_2 + y_2 j) = (x_1 \pm x_2) + (y_1 \pm y_2) j
$$

这个运算性质与向量的加减类似。

## 2.2. 乘法运算
$$
(x_1 + y_1 j) (x_2 + y_2 j) = x_1 x_2 + x_1 y_2 j + y_1 x_2 j + y_1 y_2 j^2 \\
= ( x_1 x_2 - y_1 y_2) + (x_1 y_2  + y_1 x_2 ) j
$$

## 2.3. 除法运算

$$
\frac{x_1 + y_1 j}{x_2 + y_2 j} = \left ( \frac{x_1 + y_1 j}{x_2 + y_2 j} \right ) \left( \frac{x_2 - y_2 j}{x_2 - y_2 j} \right )
$$

注意，由于分子分母同时乘以了分母的共轭，所以分母直接等于了 $x_2^2 + y_2^2$，于是上面的除法就变成了

$$
\frac{x_1 + y_1 j}{x_2 + y_2 j}  = \frac{(x_1 x_2 + y_1 y_2) + (x_2 y_1 - x_2 y_2) j}{x_2^2 + y_2^2}
$$

这个很容易推导，不必记忆。

