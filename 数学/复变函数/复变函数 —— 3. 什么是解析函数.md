@[toc]

# 1. 解析函数

## 1.1. 对区域 $D$ 的「解析函数」是什么意思？
> 
> 设 $f(z)$ 在区域 $D$ 有定义，当 $z_o \in D$ 时，若存在一个 $z_o$ 的一个邻域，使得 $f(z)$ 在邻域内处处可导，则 $z_o$ 为 $f(z)$ 的解析点。当 $D$ 上每一个点都解析时，则称 $f(z)$ 是 $D$ 的解析函数。

**可以看到，解析和可导具有一定的等价性，但他们的意义不同，解析是指某一邻域可导，而可导只是某一点可导**[^1]。

[^1]: 复变函数与解析函数，https://zhuanlan.zhihu.com/p/133249193

所以从上述定义出发，我们可以得到「函数解析」的充要条件：

> 对于 $f(z) = u(x, y) + j y(x, y)$ 在区域 $D$ 内解析的充要条件为：$u(x, y)$、$v(x, y)$ 在区域 $D$ 内可微；在 $D$ 内满足等式 C-R方程。

所以，我们可以通过CR方程，得到下面方程
$$
f'(z) = \frac{\partial u}{\partial x} + j \frac{\partial v}{\partial x} = \frac{\partial v}{\partial y} - j \frac{\partial u}{\partial y}
$$

为了更好理解这个概念，我们先从函数 $f(z)$ 对于某一点可导开始说起

## 例1
> 函数 $f(z) = xy + j y$ 仅在 $z=$_______ 处可导，且该点的导数值为_________。

解：对于问题一来说，我们可以从解析函数的充要条件出发，得到

$$
\left \{ \begin{matrix}
\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y} \\
\frac{ \partial u}{\partial y } = -\frac{ \partial v}{ \partial x } 
\end{matrix} \right. \rightarrow \left \{ \begin{matrix}
y = 1 \\
x = 0
\end{matrix} \right.
$$

得到 $z = j$，而该点的导数值即

$$
f'(z) = \frac{\partial u}{\partial x} + j \frac{\partial v}{\partial x} = y + j 0 \left |_{z=j} \right . \rightarrow f'(z) = 1
$$ 

从上述例题中可以知道，当函数在点 $(0, 1)$ 时，可以求导，并且导数等于 1。现在我们再来看看另外一个例子。

## 例2

> 设函数 $f(z) = m y^3 + n x^2 y + j (x^3 + k xy^2)$ 在 $z$ 平面上解析，求 $m$、$n$、$k$ 的值。

解：由于函数 $f(z)$ 在平面 $z$ 解析，也就是说它可导。所以我们可以代入 C- R方程，得到：

$$
\left \{ \begin{matrix}
	\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y} \\
	\frac{\partial u}{\partial y} = - \frac{\partial v}{\partial x}
\end{matrix} \right. 
\to 
\left \{ \begin{matrix} 
	2nxy = 2kxy \\
	3my^2 + n x^2 = -3x^2 - k y^2
\end{matrix} \right.
\to
\left \{  \begin{matrix}
	m = 1 \\
	n = -3 \\
	k = -3 \\
\end{matrix} \right .
$$