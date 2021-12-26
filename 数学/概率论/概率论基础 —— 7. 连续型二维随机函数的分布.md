> 对于连续型随机变量经常喜欢考察的一个知识点，就是其函数的分布，以及变换函数后的分布，比如 Z = X + Y。在浙大版的《概率与数理统计》这本教材里，主要重点考察了三类 Z = X + Y型，Z = X Y型，Z = X/Y型，以及 Z = min(X, Y) 和 Z = max(X, Y)这几类型函数的分布。
> 为什么教材特别喜欢反复考察这些知识点，其中一个问题是对于我们连续型随机变量来说，有时不一定能顺利的使用连续的密度函数对概率事件进行建模，反而此时使用概率函数能够简化我们的工作量，所以这也就是要求学习概率论的朋友必须掌握的基础概念。
> 概率函数其实有很多，但是如果拆分它的基本型也就是以上这些。

@[toc]

# Z = X + Y型
浙大版的《概率论与数理统计》这本书呢，列举了好几类函数分布。这里，我也对照着教材上把相关的类型罗列出来。

首先，你需要保持清晰的头脑，弄明白对于连续型的概率密度函数，与分布函数的区别，如果这方面你还是一团浆糊，建议先回到我之前的章节里，仔细看一看。

那么首先，对于 $Z = X + Y$ 的分布，它的概率密度为：

$$f_{X + Y}(z) = \int_{- \infty}^{+ \infty} f(z - y, y) dy = \int_{- \infty}^{+ \infty} f(x, z-y) dx$$

若X和Y互为独立事件，那么可以得到：

$$f_{X + Y}(z) =  \int_{- \infty}^{+ \infty} f_X(z - y)f_Y(y) dy = \int_{- \infty}^{+ \infty} f_X(x)f_Y(z-y) dx$$

这里的 $f_X$ 和 $f_Y$ 是 $(X, Y)$ 的边缘密度，同时也是卷积公式。


## 例题

>（1）：设X和Y是相互独立的随机变量，其概率密度分别如下，求Z = X + Y的概率密度
> $$f_x(x) = \left\{\begin{matrix}
e^{-2x} & x > 0\\ 
0 & else
\end{matrix}\right.$$
> $$f_y(y) = \left\{\begin{matrix}
1/2 & 0 \leq y < 2\\ 
0 & else
\end{matrix}\right.$$

**解（1）**

**首先我们来确定被积函数**，我们遵循谁简单替换谁的原则，替换y，且由于X和Y相互独立，所以可以直接带入卷积公式，有：

$$f_z(z) = \int f_x(x) f_y(z - x) dx \Rightarrow \left\{\begin{matrix}
\frac{1}{2} \int e^{-2x} dx& x > 0, 0 \leq y < 2\\ 
0 & else
\end{matrix}\right.$$

**然后来确定被积函数的积分范围**，由于上述积分式是关于x的积分，所以我们要确定对于上述积分式，它对应的x积分范围，即：

$$\left\{\begin{matrix}
x > 0 \\
0 \leq y < 2
\end{matrix}\right. \Rightarrow \left\{\begin{matrix}
x > 0 \\
0 \leq z - x < 2
\end{matrix}\right. \Rightarrow \left\{\begin{matrix}
x > 0 \\
z - 2 < x \leq z
\end{matrix}\right. $$

然后我们根据上面的范围，来确定当z分别为以下情况时，密度函数为多少：

对于第一种情况，$z - 2 < x \leq z$ 与 $x > 0$ 不想交时， $f_z(z) = 0$

![在这里插入图片描述](https://img-blog.csdnimg.cn/img_convert/a0361aae9764320c85e6117a5e474d06.png#pic_center)
对于第二种情况，部分相交时，有：$z-2 < 0 < z$，于是 $0 < z < 2$ 而对于密度函数的积分有效区域是 (0, z)

![在这里插入图片描述](https://img-blog.csdnimg.cn/img_convert/e891194493009d265da06028e424e492.png#pic_center)
于是：
$$f_z(z) = \frac{1}{2} \int_0^z e^{-2x} dx = -\frac{1}{4} e^{-2x} \bigg|_0^z = \frac{1}{4}(1 - e^{-2z})$$

然后对于第三个情况，当z区域处于x区域中时：

![在这里插入图片描述](https://img-blog.csdnimg.cn/img_convert/c6a5cf9b96a13ef8b6548a6b7bb0b801.png#pic_center)
积分有效区域为 $(z-2, z)$， z的有效范围是：$z \geq  2$

于是：
$$f_z(z) = \frac{1}{2} \int_0^z e^{-2x} dx = -\frac{1}{4} e^{-2x} \bigg|_{z-2}^z = \frac{1}{4}(e^{4-2z} - e^{-2z})$$

然后我们整理一下结果：

$$f_z(z) = \left\{\begin{matrix}
\frac{1}{4}(1 - e^{-2z}) & 0 < z < 2 \\ 
\frac{1}{4}(e^{4-2z} - e^{-2z}) & z \geq 2\\
0 & else
\end{matrix}\right.$$

> （2）设随机变量（X，Y）的概率密度为：
>  $$f(x, y) = \left\{\begin{matrix}
>  x + y & 0 < x < 1, 0 < y < 1 \\
>  0 & else
>  \end{matrix}\right.$$
> 求Z = X + Y的概率密度

**解（2）：**

由于不满足X与Y相互独立，于是根据 Z= X + Y 型的密度函数可知：

$$f_z(z) = \int f(x, z-x) dx$$

对于上述概率密度，只有当 $f(x, z-x) \neq 0$ 时才有意义，而从从题干可知 $f(x, y)不为0时，必须 $0 < x < 1, 0 < y < 1$。

即 $f(x, y) \neq 0 \Rightarrow  \left\{\begin{matrix}
0 < x < 1\\ 
0 < y < 1
\end{matrix}\right. \Rightarrow  \left\{\begin{matrix}
0 < x < 1\\ 
0 < z - x < 1
\end{matrix}\right. \Rightarrow  \left\{\begin{matrix}
0 < x < 1\\ 
z - 1 <  x < z
\end{matrix}\right.$ 

然后剩下的解体过程与上面的那个例题一样，最终结果是：

$$f_z(z) = \left \{ \begin{matrix} 
z^2 & 0 < z < 1\\
2z - z^2 & 1 \leq z < 2\\
0 & else
\end{matrix} \right.$$

中间的解体过程我不详列，感兴趣的朋友可以试一试。


# Z = XY 型

对于 $Z = XY$ 的分布，它的概率密度为：

$$f_{X Y}(z) = \int_{- \infty}^{+ \infty} \frac{1}{|x|}  f(x, \frac{z}{x}) dx = \int_{- \infty}^{+ \infty} \frac{1}{|y|} f(\frac{z}{y}, y) dy$$

若X与Y相互独立，其边缘密度分别为：$f_x(x)$ 和 $f_y(y)$，那么其概率密度为：

$$f_{X Y} = \int_{- \infty}^{+ \infty} \frac{1}{|x|} f_x(x) f_y(\frac{z}{x}) dx =  \int_{- \infty}^{+ \infty} \frac{1}{|y|} f_x(\frac{z}{y}) f_y(y) dy$$ 

## 例题
> （1）设随机变量（X，Y）的概率密度为：
>  $$f(x, y) = \left\{\begin{matrix}
>  x + y & 0 < x < 1, 0 < y < 1 \\
>  0 & else
>  \end{matrix}\right.$$
> 求 $Z = XY$ 的概率密度

**解：** 这题和上面的解体过程大体上都是一样的。由于X 和 Y 并未指定是否相互独立，所以我们只能使用一般型，即：

$$f_{XY} = \int \frac{1}{|x|} f(x, z/x) dx$$

然后我们代入被积函数：

$$f_{XY} = \int \frac{1}{|x|} (x + \frac{z}{x}) dx$$

由于 x 的有效区间为正，所以：

$$f_{XY} = \int \frac{1}{x} (x + \frac{z}{x}) dx = \int (1 + \frac{z}{x^2}) dx = x - z x^{-1} \bigg|_{\alpha}^{\beta}$$

接下来我们确定积分的合理范围：

$$ \left \{ \begin{matrix}
0 < x < 1 \\
0 < y < 1
\end{matrix} \right. \Rightarrow \left \{ \begin{matrix}
0 < x < 1 \\
0 < \frac{z}{x} < 1
\end{matrix} \right. \Rightarrow \left \{ \begin{matrix}
0 < x < 1 \\
0 < z < x
\end{matrix} \right.$$

然后简单的做个图，或者直接代入不等式，可以得到：

![在这里插入图片描述](https://img-blog.csdnimg.cn/img_convert/5eb9b9b131f5f1f8cf80cb5f6b53551c.png#pic_center)
z和x的合理范围只有 $0 < z < x < 1$这个范围，所以

$$x - z x^{-1} \bigg|_{z}^{1} = 2 - 2z$$

于是我们最终得到关于 $Z = XY$ 型的概率密度是：

$$\left \{ \begin{matrix} 
2 - 2z & 0 < z < 1\\
0 & else
\end{matrix} \right.$$


# Z = X/Y 型

对于 $Z = \frac{X}{Y}$ 的分布，它的概率密度为：

$$f_{X/Y}(z) = \int_{- \infty}^{+ \infty} |x|  f(x, zx) dx = \int_{- \infty}^{+ \infty} |y| f(zy, y) dy$$

若X与Y相互独立，其边缘密度分别为：$f_x(x)$ 和 $f_y(y)$，那么其概率密度为：

$$f_{X/Y}(z) = \int_{- \infty}^{+ \infty} |x|  f_x(x)f_y(zx) dx = \int_{- \infty}^{+ \infty} |y| f_x(zy)  f_y(y) dy$$ 

## 如何记住 XY 及 X/Y 型的概率密度

这里借用一点物理的概念：$m = \rho V$，概率密度相当于质量方程的$\rho$，积分区域让它等价于体积V。如果我们令被测量的物体体积一样，而且它们也满足 Z = X Y这样的属性，于是我们为了获得被测量物体的密度，就可以衍生出这样的公式：

$$Z = XY \Rightarrow V_z \rho_z = V_x \rho_x \times V_y \rho_y \rightarrow \rho_z = \frac{1}{V_z} \times V_x \rho_x \times V_y \rho_y$$

因为$V_z$，$V_x$，$V_y$我们令他们体积一样，所以 $V_z = V_x = V_y$ 是不是很合理的？

又因为 $Z = XY$，所以密度之间有 $\rho_z = \rho_x \rho_y$ 是不是也是合理的？

然后上面的式子是不是就可以变成：

$$\rho_z = \frac{1}{V} V_x \rho_x V_y \frac{\rho_z}{\rho_x}$$

是不是也是合理的？

再然后由于上面式子所表述的三个物体的密度，使用的是同一个计算公式，是不是简化为：$f(V, \rho)$，由于V是已知的，所以又可以进一步简化一下：$f(\rho)$，于是上面的式子是不是可以这样写：

$$\rho_z = \frac{1}{V} f(\rho_x) f(\frac{\rho_z}{\rho_x})$$

然后把 1/V 换成积分的形式，嗯…… 然后

$$\rho_z = \int \frac{1}{v_x} f(\rho_x) f(\frac{\rho_z}{\rho_x}) dx$$

然后再换个符号

$$f_{XY}(z) = \int \frac{1}{|x|} f(x, \frac{z}{x}) dx$$ 

又或者

$$f_{XY}(z) = \int \frac{1}{|x|} f_X(x) f_Y(\frac{z}{x}) dx$$ 

为什么加绝对值，因为体积没有为负的道理啊……

![在这里插入图片描述](https://img-blog.csdnimg.cn/img_convert/aa949ab471abb47bb977527cf3b75159.png#pic_center)
虽然不是很严谨吧……但你这样记公式，那就错不了。考试的时候如果不会写了，用30s的时间推一下公式就有了。至于另外比如 Z = X / Y 型，Z = X + Y型，甚至考官心血来潮发明的  $Z = X^2 + Y^5$ 你都可以用这个方法推导出它的密度函数原型。

不信，自己试一试？

# Z = min(X, Y) 和 Z = max(X, Y) 型

如果X与Y是两个相互独立的随机变量，它们的分布函数分别为 $F_x(x)$ 和 $F_y(y)$，那么对于 **$Z = max \{X, Y\}$ 和 $Z = min\{X, Y\}$**型的分布函数，它们分别是:

$$Z_{max} = F_{max}(z)= F_x(z)F_y(z)$$

与

$$Z_{min} = F_{min}(z)= 1 - [1 - F_x(z)][1 - F_y(z)]$$

其对应的概率密度为：

$$f_z(z) = F(z)'$$

**Z = min(X, Y) 和 Z = max(X, Y) 型和之前的不太一样，不太容易理解。为了更好的理解这种概率是在什么状况使用的，我们来做下面的这个例题吧。**

## 例题
> 设系统L由两个独立的子系统 $L_1$，$L_2$ 连接而成，连接的方式分别为串联、并联、备用，设 $L_1$，$L_2$ ，已知它们的概率密度分别为：
> $$f_x(x) = \left \{ \begin{matrix} 
\alpha e^{-\alpha x} & x > 0\\
0 & x \leq 0
\end{matrix} \right.$$
> $$f_y(y) = \left \{ \begin{matrix} 
\beta e^{-\beta y} & y > 0 \\
0 & y \leq 0
\end{matrix} \right.$$
> 其中 $\alpha > 0$，$\beta > 0$ 且 $\alpha \neq \beta$。试分别就以上三种连接方式写出L的寿命Z的概率密度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b16541907c174ace9d6e6b105849d200.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

**解（i）** 针对串联情况，如果X或Y有任何一路出现问题，那么线路就会停止工作。

由于串联关系，发生事件概率大，预期使用寿命最短，所以使用 $Z = min\{X, Y\}$

为了更好的带入公式，我们先求解X和Y对应的分布函数：

$$F_x(x) = \int \alpha e^{-\alpha x} dx = -e^{-\alpha x} \bigg|_0^1 = 1 -e^{- \alpha x}$$

$$F_y(y) = \int \beta e^{-\beta y} dy = -e^{-\beta y} \bigg|_0^1 = 1 - e^{-\beta y}$$

并且分布函数存在的情况，有且只有当 $x > 0, y> 0$。

然后我们带入公式，得到关于z的分布函数

$$F_{min}(z) = 1 - [1 - F_x(z)][1 - F_y(z)] = 1 -  e^{- \alpha z} e^{-\beta z} = 1 - e^{-z(\alpha + \beta)}$$

现在我们来确定下z的有效区间，因为 $x > 0, y > 0$ z 若需要另 $F_{min}$ 有效，必然 $z > 0$，所以：

$$F_{min}(z) = \left \{ \begin{matrix}
1 - e^{-z(\alpha + \beta)} & z > 0 \\
0 & z \leq 0
 \end{matrix} \right.$$

然后我们对原函数求导，就可以得到它的密度函数了

$$f_{min}(z) =  \left \{ \begin{matrix}
(\alpha + \beta) e^{-z(\alpha + \beta)} & z > 0 \\
0 & z \leq 0
 \end{matrix} \right.$$


**解（ii）** 针对并联情况，只有X和Y同时出现问题，线路才会停止工作。

由于并联关系，发生事件概率小，预期使用寿命长，所以使用 $Z = max\{X, Y\}$

我们直接把上面得到的关于X和Y的分布函数拿下来用，于是：

$$F_{max}(z) = F_x(z)F_y(z) = (1 -e^{- \alpha z})(1 - e^{-\beta z})$$

所以，

$$F_{max}(z) = \left \{ \begin{matrix}
(1 -e^{- \alpha z})(1 - e^{-\beta z}) & z > 0 \\
0 & z \leq 0
 \end{matrix} \right.$$

同样，求导后有

$$f_{min}(z) =  \left \{ \begin{matrix}
\alpha e^{-\alpha z} + \beta e^{-\beta z} - (\alpha + \beta) e^{-z(\alpha + \beta)}& z > 0 \\
0 & z \leq 0
 \end{matrix} \right.$$

**解（iii）** 情况和并联相似，但是有先后关系。即先X出问题，然后启动Y，如果Y再出问题，则线路停止工作。

由于X与Y是先后投入使用，所以预期使用寿命应该为 Z = X + Y。

$$f_{X+Y}(z) = \int f_x(x) f_y(z - x) dx = \int \alpha e^{-\alpha x} \beta e^{-\beta (z - x)} dx$$

$$= \alpha \beta e^{-\beta z} \int e^{(\beta - \alpha) x} dx =  \alpha \beta e^{-\beta z} \cdot \frac{1}{\beta - \alpha} e^{(\beta - \alpha)x} \bigg|_0^z = \frac{\alpha \beta}{(\beta - \alpha)} (e^{-\alpha z} - e^{-\beta z})$$

于是：

$$f_{X+Y}(z) = \left \{ \begin{matrix} 
\frac{\alpha \beta}{(\beta - \alpha)} (e^{-\alpha z} - e^{-\beta z}) & z > 0\\
0 & else
\end{matrix} \right.$$