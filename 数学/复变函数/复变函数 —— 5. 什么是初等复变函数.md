@[toc]

# 关于什么是初等函数

以高等数学，或者更高一级的解析数学的角度看，所谓的初等函数是由 「幂函数 （power function）」「指数函数 （exponential function）」「 对数函数 （logarithmic function）」「三角函数 （trigonometric function）」「反三角函数 （inverse trigonometric function）」与常数经过有限次的有理运算（比如，加、减、乘、除、乘方、开方运算等）及有限次函数复合产生，并能用一个解析式表示的函数。

自然，复变函数的初等函数也是遵循上述规则，只不过参与演算的数从原来的实数域扩展至了复数域，于是被称为——「**初等复变函数**」。在大学里所涉及的复变函数的「初等复变函数」主要有四类，即指数函数、对数函数、幂函数以及三角函数。

现在我们来逐一分析。


# 指数函数

在[《复变函数 —— 0. 连接复数与三角函数的欧拉公式》](https://seagochen.blog.csdn.net/article/details/120963316) 和 [《复变函数 —— 1. 复数的定义》](https://seagochen.blog.csdn.net/article/details/121660877) 我们分别用泰勒展开和球坐标公式，证明了重要的欧拉公式，而 **欧拉公式** 本身也是复变函数中极为重要的一类初等复变函数。

如果有复函数 $z = x + j y$，有 

$$
f(z) = e^z = e^{x + j y} = e^{x}( \cos y + j \sin y)
$$

通常如果不特别指定，它的模长 $|\omega| = |e^{2 k \pi j}| = |e^x| = 1$，函数周期是 $2 k \pi j$。

指数函数有如下一些基本性质。
* $(e^z)' = e^z$
* $|e^z| = e^x$，它的幅角函数 $\arg e^z = y + 2 k \pi$
* $e^{z + 2k\pi j} = e^z$，即 $e^z$ 是以 $2 k \pi j$ 为周期的周期函数。

> 「arg(x)」即所谓的 **幅角** 函数。在数学中，它是指复数在复平面上对应的向量和正向实数轴所成的有向角。
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/bd84a6a144224261a9709009b4e1a81b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_9,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 对数函数




# 幂函数


# 三角函数