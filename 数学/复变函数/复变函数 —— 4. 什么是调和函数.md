@[toc]

# 1. 调和函数的定义

在[《浅谈矢量场 —— 1. 梯度、散度与拉普拉斯算子》](https://seagochen.blog.csdn.net/article/details/114387703) 这篇文章中提到过「拉普拉斯算子」，它的表达形式一般如下：

$$
\Delta = \nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}+\frac{\partial^2}{\partial z^2}
$$

在物理上，它是 $n$ 维欧几里德空间中的一个二阶微分算子，定义为梯度 $\nabla f$ 的散度 $\nabla \cdot \nabla f$。注意，通常表示梯度时，我们使用 $\nabla f$，而表示散度时，我们习惯使用 $\nabla \cdot f$，旋度则一般表示为 $\nabla \times f$。所以，拉普拉斯算子的二阶形式，经常被简写为 $\nabla^2 f$，很少使用 $\Delta f$ 形式，因为这容易与微量弄混淆，所以现在一些较新的出版论文或教材里，已经较多的使用 $\nabla^2$ 替换了原有的 $\Delta f$ 形式。

而「调和函数」的形式可以从「拉普拉斯算子」出发，被认为是当 「拉普拉斯算子」等于0的特殊情况的一类函数，即：

$$
\nabla^2 \varphi = \frac{\partial^2 \varphi}{\partial x^2} + \frac{\partial^2 \varphi}{\partial y^2}+\frac{\partial^2 \varphi}{\partial z^2} = 0
$$

而且一般对于复数域来说，我们只讨论到实数域和虚数域两个维度，所以：

$$
\nabla^2 \varphi = \frac{\partial^2 \varphi}{\partial x^2} + \frac{\partial^2 \varphi}{\partial y^2} = 0
$$

*我个人感觉，「调和函数」这种函数形式，对于研究物理「场」是一种特别重要的工具，但是说实话在数学范畴上，是比较少见到具体应用的。* 

那么对于一个复变函数 $f(z) = u + j v$ 来说，如果它自身满足

$$
\left \{ \begin{matrix}
		\nabla^2 u = 0 \\
 		\nabla^2 v = 0
\end{matrix} \right .
$$

那么我们称其为调和函数。

现在我们来看一些例题

## 例1.

> 函数 $f(z) = u + j v$ 在区域 $D$ 内解析，则下列命题中错误的是________
> A. 函数 $f(z)$ 在区域 $D$ 内可导；
> B. 函数 $u$、 $v$ 时区域 $D$ 的调和函数；
> C. 函数 $u$、 $v$ 在区域 $D$ 内满足柯西黎曼方程；
> D. 函数 $u$、 $v$ 在区域 $D$ 内的共轭调和函数。

解：这题主要考察对复变函数相关概念的掌握，我们现在一一分析：

首先对于答案A，由于题干给出了在 $D$ 内解析，那它必然在 $D$ 内处处可导（对这问题不熟悉的朋友，可以看 [《复变函数 —— 3. 什么是解析函数》](https://seagochen.blog.csdn.net/article/details/121663513) ），并且可以直接得到  $u$、 $v$必然也满足柯西黎曼方程，所以C也是正确的。

接下来对于B来说，由于A和C正确，所以对于复变函数的一阶导必然是一个复常数 $a$

$$
\nabla f = a
$$

这是因为如果说复变函数在点 $(x, y)$ 存在导数，也就意味着当 $z$ 趋于 $z_o$ 时，$f(z)$ 有极限a存在，即 $lim_{z \to z_o} \frac{f(z) - f(z_o)}{z - z_o} = a$。注意这里的 $a$ 必须是一个确定的「复常数」，即 $3-j$ 或者 $1/4j$这样，而不是 $x - j$这种类型的。

所以如果我们再对「复常数」$a$ 取导，它一定等于0，所以在满足区域 $D$ 内解析的同时，$u$、 $v$也同时满足调和函数的定义要求，B因此也是正确的；这样错误的只有D了。


## 例2.

> 验证 u(x, y) = x^2 - y^2 + xy 是调和函数，并求相应的解析函数，$f(z) = u + j v$，使 $f(0) = 0$。

解：验证调和函数，首先要求上式的二阶导，所以

$$
\frac{\partial}{\partial x} \frac{\partial (x^2 - y^2 + xy)}{\partial x} = \frac{\partial}{\partial x} \cdot (2x + y) = 2
$$

$$
\frac{\partial}{\partial y} \frac{\partial (x^2 - y^2 + xy)}{\partial y} = \frac{\partial}{\partial y} \cdot (-2y + x) = -2
$$

由于 $2-2 =0$，所以$u$是调和函数。接下来在已知实数域函数 $u$ 的前提下，我们需要推导出虚数域的函数 $v$，先从CR方程，可以得到 $\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}$， 注意这两个都是导数形式，所以要想得到原函数，可以把导数代入积分中，即：

$$
v = \int v' dy = \int u' dy
$$

$u'$ 其实已经在验证调和函数过程中得到，所以直接代入

$$
v = \int (2x + y) dx = 2xy + \frac{1}{2} y^2 + C(x)
$$

于是得到 $\frac{\partial v}{\partial x} = 2y + C'(x)$，然后再代入CR方程，$\frac{\partial u}{\partial y}= - \frac{\partial v}{\partial x}$，

$$
x - 2y = -2y - C'(x) \to C'(x) = -x
$$

然后求$C(x)$ 的原函数，通过 $C(x) = \int -x dx = -\frac{1}{2} x^2 + C$，最终 $v = 2xy + \frac{1}{2} y^2 -\frac{1}{2} x^2 + C$，然后对于 $f(z) = u + jv$ ，可得到：

$$
f(z) = x^2 - y^2 + xy + j(2xy + \frac{1}{2} y^2 -\frac{1}{2} x^2 + C) 
$$

然后带入条件 $f(0) = 0$，且 $z = x + j y$ 可知 $x = y = 0$，于是

$$
f(z) = x^2 - y^2 + xy + j(2xy + \frac{1}{2} y^2 -\frac{1}{2} x^2 + C)  \Rightarrow  jC = 0 \Rightarrow  C= 0
$$