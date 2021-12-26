@[toc]

在我们求解微积分公式时，经常会遇到积分公式难以求解的情况，所以这个时候我们通常需要用到类似「映射」的技巧，把函数积分项或者积分域换成相对容易的形式，这一章节我们来复习一下高数教材中常提到的两类换元法。

# 从链式法则出发

无论对于第一类还是第二类换元法，都遵循链式法则，例如对原函数为 $f(g(x))$ 求导，它的导数有如下定义：

$$
(f \circ g(x))'=f'(g(x)) g'(x)
$$

> **注意**，原函数 $f(g(x))$ 是可以写成 $f \circ g(x)$ 或 $(f \circ g)(x)$，它可以被认为 $f(z)$ 的输入 $z$ 是上一个函数 $g(x)$ 的输出，或函数 $f$ 对 $g(x)$ 做卷积。

所如，如果对原导数函数求积分，也就等于对它的链式导数部分求积分，即：

$$
\int_a^b f(X)' dX = \int_{\alpha}^{\beta} (f \circ g(t))' dt = \int_{\alpha}^{\beta} f'(g(t)) g'(t) dt
$$

> **注意**，需要注意的是，对于原导数来说，它的积分区域如果是 $[a, b]$，通过换元法后其积分区域会变到 $[\alpha, \beta]$。$\alpha$, $\beta$ 与 $a$、$b$，之间是线性相关，而它们是否相等，则取决于 $g(t)$ 的映射选取范围，而这也是第一类和第二类换元法最大的区别。


# 第一类换元法

为了更好说明第一类积分换元法，我们先来看一到例题：

## 例题
> 求 $2 \int \cos 2x dx$

如果看上式，它的导数部分 $2 \cos 2x$ 可以写成 $\cos 2x \cdot (2x)'$， 它可以表示为 $[\cos 2x]'$ 的导数，于是我们可以令 $u = 2x$，而 $u$ 的积分范围也仍然是 $[- \infty, + \infty]$ 于是可以直接得到：

$$
\int 2 \cos 2x dx = \int \cos 2x \cdot 2 dx \\
= \int \cos 2x \cdot (2x)' dx = \int \cos u du  \\
= \sin u + C
$$

代入 $u = 2x$，于是可得：

$$
\cdots = \sin 2x + C
$$


> 求 $\int \frac{x^2}{(x+2)^3} dx$

如果直接求解，会特别困难，于是我们可以令 $u = x+2$，那么 $x = u-2$，积分域是 $[- \infty, + \infty]$，对于 $u$ 的积分域也自然是 $[- \infty, + \infty]$

$$
\int \frac{x^2}{(x+2)^3} dx = \int \frac{(u - 2)^2}{u^3} du = \int \frac{u^2 - 4u + 4}{u^3} du \\
= \int (u^{-1} - 4 u^{-2} + 4 u^{-3}) du \\
= \int u^{-1} du - \int 4 u^{-2} du + \int 4 u^{-3} du
$$

然后我们查一下[《常用积分表》](https://blog.csdn.net/poisonchry/article/details/121268019)，然后填入上面的积分项后，有

$$
\cdots = \ln |u| + C + 4 u^{-1} - C - 2 u^{-2} + C \\
= \ln |u| + 4 u^{-1} - 2u^{-2} + C
$$

再代回 $u = x + 2$，得

$$
\cdots = \ln |x+ 2| + \frac{4}{x + 2} - \frac{2}{(x+2)^2} + C
$$

# 第二类换元法

第二类换元法与第一类一样，也是从链式法则出发，

$$
(f \circ g(x))'=f'(g(x)) g'(x)
$$

我们依然可以得出

$$
\int_a^b f(X)' dX = \int_{\alpha}^{\beta} f'(g(t)) g'(t) dt
$$

为了说明两者之间最大的区别，我们还是先看一些例题。

> 求 $\int \frac{1}{2 + \sqrt{x - 1}} dx$

这里最要命的部分是开根号的 $\sqrt{x - 1}$， 所以我们先令 $t = \sqrt{x - 1}$，于是有

$$
t^2 = x - 1 \rightarrow \\
t^2 + 1 = x \Rightarrow \\
dx = 2tdt
$$

其中，x的范围是 $[- \infty, + \infty]$，所以 $-\infty \leq t \leq +\infty$，于是有

$$
\cdots = \int \frac{1}{2 + t} 2t dt \\
= 2 \int \frac{t}{2 + t} dt
$$

我们虽然不知道 $\frac{t}{t+2}$ 的原函数是什么，但是可以通过查表知道 $\int \frac{1}{ax + b} dx = \frac{1}{a} \ln |ax + b|$，所以可以朝着这个方向进行变换，于是

$$
\cdots = 2 \int \frac{t + 2 - 2}{t + 2} dt \\
= 2 \int (1 - \frac{2}{t + 2}) dt \\
= 2 \int dt - 2 \int \frac{2}{t + 2} dt \\
= 2t - 4 \int \frac{1}{t + 2} dt \\
= 2t - 4 \ln |t + 2| + C 
$$

代入 $t = \sqrt{x - 1}$，于是

$$
\cdots = 2 \sqrt{x -1} - 4 \ln |\sqrt{x - 1} + 2| + C
$$

# 总结

经过上面的例题之后，可知对于第一类换元法，尽管我们令 $g(t) = x$，使得原本 $f$ 对 $x$ 的积分，被转换到了 $g$ 对 $t$ 的积分关系，但是由于 $g(t)$ 与 $x$ 之间是线性关系，所以 $t$ 可以直接线性的投影到原本 $x$ 的取值区间内

$$
a \leq x \leq b \Rightarrow  a \leq g(t) \leq b \Rightarrow  g(t_a) \leq g(t) \leq g(t_b)
$$

由于线性关系存在，即 $g(t) = \omega t + \gamma$，其中$\omega$ 和 $\gamma$ 都是常数，所以当 $- \infty \leq g(t) \leq + \infty$ 时，$- \infty \leq t \leq + \infty$。 换句话说，如果当 $t$ 与 $x$ 差异不是很大时，且取值范围足够覆盖 $t$ 与 $x$ 由于参数带来的差异时，我们甚至可以直接在求积分取值范围时，认为 $t$ 与 $x$ 的取值范围是一样的。

$$
\int_a^b f(x) dx \rightarrow \int_a^b f(g(t)) g'(t) dt 
$$

但是对于第二类换元法，它更多的像是非线性映射关系，就好像欧拉空间坐标系被转换成了球坐标系一样；在欧拉坐标系中起到决定性的是相对于x轴、y轴以及z轴的模长，而在球坐标系中则变成了相对于x、y、z的夹角以及半径。所以，对于第二类变换，我们在积分函数变换后，也需要同时明确此时  $a$、$b$、$\alpha$、$\beta$ 的对应关系。