@[toc]

# 定义

拉格朗日乘数法（Lagrange multiplier，以数学家约瑟夫·拉格朗日命名），在数学中的最优化问题中，是一种寻找多元函数在其变量受到一个或多个条件的约束时的极值的方法。这种方法可以将一个有 $n$ 个变量与 $k$ 个约束条件的最优化问题转换为一个解有 $n + k$ 个变量的方程组的解的问题。这种方法中引入了一个或一组新的未知数，即拉格朗日乘数，又称拉格朗日乘子，或拉氏乘子，它们是在转换后的方程，即约束方程中作为梯度（gradient）的线性组合中各个向量的系数。

比如在要求 $f(x, y)$ 在 $g(x, y) = 0$ 时的极值，我们可以引入拉格朗日乘数 $\lambda$ ，得到如下新方程

$$
L(x, y, \lambda) = f(x, y) - \lambda \cdot g(x, y)
$$

求解方式就是对上述新函数求导，由于新函数包含多个变量，所以变成了

$$
\left \{ \begin{matrix}
\frac{\partial L}{\partial x} = 0 \\
\frac{\partial L}{\partial y} = 0 \\
\frac{\partial L}{\partial \lambda} = 0
\end{matrix} \right .
$$

得出的x，y值，就是在满足 g(x, y) = 0 这一约束条件下，函数 $f(x, y)$ 的极值（可能极大或极小）。

#  理解「拉格朗日乘数法」

比方说对于函数 $C = x^2 + y^2$ 它在没有任何一个限制条件时，函数图像呈现如下所示的圆形，理论上这个函数中心面积在欧拉平面可以从 $[0, +\infty)$，也就是在没有限制条件，比如没有常数 $C$ 给定的情况下，它可以无限大

![在这里插入图片描述](https://img-blog.csdnimg.cn/f982a1c7e8fb41af812bb04a1ae53c41.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
而当我们出现一个限制条件，比如 $xy = 3$ 时，我们希望找到此时圆形曲线对于函数$g(x, y) = xy = 3$ 时函数 $f(x, y) = x^2 + y^2$ 的变量 $x$ 和 $y$ 所能取到的极值。一般来说我们可以联立函数，有

$$
\left \{ \begin{matrix}
xy = 3 \\
x^2 + y^2 = C
\end{matrix} \right .
$$

但是我们可以观察图像，发现满足上面这个条件时，红色圆和蓝色的双曲线一点会在某个点上相切。
![在这里插入图片描述](https://img-blog.csdnimg.cn/6c7cffacb93a472982d6a84dc2e41588.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
你也可以想象为红色圆持续膨胀，当触及了蓝色曲线上的点时停止膨胀。在函数图像上，就表示为两个函数图像彼此相切。说到相切，它不就是红色函数或者蓝色函数在彼此函数空间上的极值点嘛（也就是映射的关系）。

于是我们自然而然的联想到数学的一个关于极值点的概念，也是导数的一种应用形式，即，**当导数为0时，原函数可以求到极值点这么一个特性**。所以意味着我们可以把上面这个问题转化为对两个函数 $f(x, y)$ 和 $g(x, y)$ 的求导

但是这里又带来一个问题，$f(x, y)$ 不一定和 $g(x, y)$ 共享同一个函数平面，这会导致什么问题呢，比如对于 $f'(x, y)$ 来说，它可能得出的解例如 $(x_i, y_i)$ 不是 $g(x, y)$ 函数的极大或极小值（当然，对于我们这个例子来说，它两是同一个函数空间的，但是对于更多应用来说可能并不在同一个映射空间）。

所以这时，我们需要做一个函数映射，把其中一个函数的空间映射到另一个函数的空间里去。关于映射，这时你可能会想到一些其他方法，比如计算两个函数的 $\cos \theta$ 这样的夹角。但是我们可以用线代的特征向量的概念，因为这样更通用一点，于是我们可以得到这样一个表示形式

$$
L(x, y, \lambda) = f(x, y) - \lambda g(x, y)
$$

> 特征向量是一种线性代数式的映射表示方法，它与我们习惯的 $a b = |a||b|\cos \theta$ 相比更通用一些，也能适用于多组连续向量向其他空间映射的问题，其基本表示形式如下：
> $$
> \mathbf{V} = \lambda \mathbf{v}
> $$

函数的顺序其实没有影响，不过我们通常习惯上让限制条件 $g(x, y)$ 朝着 $f(x, y)$ 上做映射。这样一来，对于 $f(x, y)$ 的极值问题，就演化为求函数 $L(x, y)$ 的极值问题，此时联系前面提到的导数概念，于是可以得到下面这个公式

$$
\nabla f(x, y) = \lambda \nabla g(x, y) = 0
$$

$\nabla$ 是梯度算子，也叫nabla算子，它相当于对函数 $f(x, y)$ 连续偏微分求导，一般来说

$$
\nabla f(x, y) = \frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}
$$

然后我们把这个公式和限制条件 $g(x, y)$ 联立，可以得到

$$
\left \{ \begin{matrix} 
\nabla f(x, y) = \lambda \nabla g(x, y) = 0 \\
g(x, y)
\end{matrix} \right .
$$

然后就可以得出关于 $f(x, y)$ 的解了。

# 一些例题
为了巩固知识点，我们来试着做一些例题。

> 求 $f(x, y) = x^2 + y^2 = C$ 时，$g(x, y) = xy = 3$ 的极值点。

对于上面这个例子，我们来看看该怎么做，首先列出 $L(x, y, \lambda)$ 的偏微分方程

$$
\left \{ \begin{matrix}
\frac{\partial L}{\partial x} = \left \{ (x^2 + y^2 - C) - [\lambda (x y - 3)] \right \}'_x = 0
\\
\frac{\partial L}{\partial y} = \left \{ (x^2 + y^2 - C) - [\lambda (x y - 3)] \right \}'_y  = 0
\\
xy =3
\end{matrix} \right .
$$

对于 $\left \{ (x^2 + y^2 - C) - [\lambda (x y - 3)] \right \}'_x$ 我们可以得到 $2x - \lambda y = 0$；
对于 $\left \{ (x^2 + y^2 - C) - [\lambda (x y - 3)] \right \}'_y$ 我们可以得到 $2y - \lambda x = 0$；

于是 $4xy = \lambda^2 xy$ 得到 $\lambda = \pm 2$。当 $\lambda = 2$ 时，有 $x = y$，于是可以得到 $x = y = \sqrt{3}$；

而当 $\lambda = -2$ 时，上式无解。所以对于函数 $x^2 + y^2 = C$ 来说，当存在限制条件 $xy =3$ 时，它存在一个最大值，且 $x = y = \sqrt 3$，此时的 $C = 6$。
