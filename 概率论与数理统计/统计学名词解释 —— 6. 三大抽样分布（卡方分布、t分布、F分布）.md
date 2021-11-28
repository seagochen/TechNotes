@[toc]

从经验可知，大部分的样本分布服从或近似服从「正态分布」。现在我们要看看和正态分布有所异同，也是非常常见的三大分布都是什么样的。


# $x^2$ 分布
$Y \sim X^2(n)$ 分布又称卡方分布，它的定义如下：

## 基本概念

> 设 $X_1, X_2, \cdots, X_n$ 来自正态分布总体 $N(0, 1)$ 的样本，则称统计量
>
> $$
> Y = X_1^2 + X_2^2 + \cdots X_n^2
> $$
>
> 服从自由度为 $n$ 的 $X^2$ 分布，记为 $Y \sim X^2(n)$，$X^2(n)$ 分布的概率密度函数为：
> 
> $$
> f(y) = \left \{ \begin{matrix}
\frac{1}{2^{n/2} \Gamma (n / 2)} y^{n/2-1} e^{-y / 2} & y > 0 \\
0 & otherwise
\end{matrix} \right .
$$


## 函数密度图像

![在这里插入图片描述](https://img-blog.csdnimg.cn/fe332402299b427b8487d763416341cf.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

> 这张图主要说明，随着样本数增加，卡方分布的概率密度图像逐渐从类似 $log$ 的对数图像逐渐接近柏松分布。使得「概率密度图像（PDF）」呈现出和「泊松等待」相类似的特征。
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/3fa195235d654bb1b750627ca7fe81de.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_14,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 由于组成卡方分布的每个样本 $X$ 来自标准正态分布，所以每个独立样本的期望 $E(X) = 0$，方差 $D(X) = 1$。

## 基本性质
对于 $X^2$ 分布来说它有两个性质

其一：
> 当 $X^2$ 分布的期望 $E(Y) = n$时，它的方差 $D(Y) = 2n$

其二：
> $X^2$ 分布具有可加性。
> 比如，有 $X \sim Y^2(m)$ 和 $Y \sim Y^2(n)$，且 X 和 Y 相互独立，有 $X+Y \sim X^2(m+n)$

## 例题
> 设 $(X_1, X_2, \cdots, X_6)$ 为取自标准正态总体 $N(0, 1)$ 的一个样本，求下列三个统计量的分布
>（1） $X_1^2 + X_2^2$
>（2）$X_1^2$
>（3）$X_1^2 + a(X_2 + X_3)^2 + b(X_4 + X_5 + X_6)^2$

解（1）：
由样本定义可知，$X_1, X_2, \cdots X_6$ 彼此相互独立，且服从 $N(0,1)$，所以 $X_1^2 + X_2^2 \sim X^2(2)$

解（2）：
由样本定义可知，$X_1, X_2, \cdots X_6$ 彼此相互独立，且服从 $N(0,1)$，因此对于单个元素它的卡方分布为 $X_1^2 \sim X^2(1)$

解（3）：
从卡方分布的定义出发，我们令

$$
Y_1 = X_1^2 \\
Y_2 = a(X_2 + X_3)^2 \\
Y_3 = b(X_4 + X_5 + X_6)^2$$

对于 $Y_1 = X_1^2$来说，由于元素来自标准正态总体，所以 $Y_1$ 的期望 $E(Y_1) = 0$，方差 $D(Y_1) = 1$，所以 $Y_1 \sim N(0, 1)$

对于 $Y_2 = a(X_2 + X_3)^2$ 来说，它有两个离散的样本，在 [《概率论基础 —— 8.数学期望、方差、协方差》](https://blog.csdn.net/poisonchry/article/details/119027117) 一节中，我们可以知道由样本 $(X_2, X_3)$ 组成的离散集合，我们可以通过离散型期望、方差的计算方法得到 $E(X_2, X_3) = E(X_2) + E(X_3) = 0$，其方差 $D(X_2, X_3) = D(X_2) +D(X_3) = 2$，于是有 $(X_2 + X_3) \sim N(0, 2)$ ，我们对正太分布进行标准化，代入如下公式：

$$
\frac{X - \mu}{\sigma} = \frac{X - 0}{\sqrt{2}} = \frac{X}{\sqrt 2}
$$

于是我们得到标准正态分布 $\frac{X_2 + X_3}{\sqrt 2} \sim N(0, 1)$

同理，对于 $Y_3 = b(X_4 + X_5 + X_6)^2$，它的样本集合 $(X_4, X_5, X_6)$ 的期望为0，方差为3，其标准正态分布为 $\frac{X_4 + X_5 + X_6}{\sqrt 3}$

再从卡方分布的基本概念出发，拼凑出它应该为

$$
X^2 = X_1^2 + \left (\frac{X_2 + X_3}{\sqrt 2} \right )^2 + \left ( \frac{X_4 + X_5 + X_6}{\sqrt 3} \right )^2 = X_1^2  + \frac{(X_2 + X_3)^2}{2} + \frac{(X_4 + X_5 + X_6)^2}{3}
$$

所以，$a=\frac{1}{2}$， $b = \frac{1}{3}$

# $t$ 分布

## 基本概念

> 设 $X \sim N(0, 1)$，$Y \sim X^2(n)$，且 X, Y 相互独立，则称随机变量
> $$
> t = \frac{X}{\sqrt{Y / n}}
> $$
> 服从自由度为 $n$ 的 $t$ 分布，记为 $t \sim t(n)$。$t(n)$ 分布的概率密度函数函数为：
> 
> $$
> h(t) = \frac{\Gamma [(n+1) / 2]}{\sqrt{\pi n} \Gamma(n / 2)} (1 + \frac{t^2}{n})^{-(n+1) / 2}, -\infty < t < \infty
> $$

## 函数密度图像
![在这里插入图片描述](https://img-blog.csdnimg.cn/ad74f7e8158e4da4a0908186c6589654.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 例题
> 假设总体 $X \sim N(0, 3^2)$，$X_1, X_2, \cdots X_n$ 是来自总体X的简单随机样本，则统计量
> $$Y = \frac{X_1 + X_2 + X_3 + X_4}{\sqrt{X_5^2 + X_6^2 + X_7^2 + X_8^2}}$$ 服从自由度为____ 的 __________ 分布。

解：

我们从t分布的基本定义入手

> $$
> t = \frac{X}{\sqrt{Y / n}}
> $$
>
> 注意对于t分布的要求，其中的元素必须服从 $X \sim N(0, 1)$，分母的Y是卡方分布，$Y \sim X^2(n)$。

所以令 $Z=X_1 + X_2 + X_3 + X_4 \sim N(0, 36)$，我们可以标准化这个分布后得到 $\frac{Z}{6} \sim N(0, 1)$。

分母虽然看起来很像卡方分布，但是由于假设的总体 $X \sim N(0, 3^2)$，所以我们要先对它进行标准化后，可以得到 $\frac{X_i}{3} \sim N(0, 1)$，然后凑出一个卡方分布得到

$$
Y' = \left ( \frac{X_5}{3} \right )^2 +   \left ( \frac{X_6}{3} \right )^2 +  \left ( \frac{X_7}{3} \right )^2 +  \left ( \frac{X_8}{3} \right )^2 = \frac{X_5^2 + X_6^2 + X_7^2 + X_8^2}{9} \sim X^2(4)
$$

然后分别把得到的 $Z$ 和 $Y'$ 代入 $t$ 分布公式中，于是得到

$$
t = \frac{X / 6}{\sqrt{Y' / 4}} = \frac{1}{6} \frac{X_1 + X_2 + X_3 + X_4}{\sqrt{ \frac{X_5^2 + X_6^2 + X_7^2 + X_8^2}{9 \times 4}}} = \frac{X_1 + X_2 + X_3 + X_4}{\sqrt{X_5^2 + X_6^2 + X_7^2 + X_8^2}} \sim t(4)
$$

所以它是自由度为4的t分布。

# $F$ 分布

## 基本概念
> 设 $U \sim X^2(n_1)$，$V \sim X^2(n_2)$，且 $U$，$V$ 相互独立，则称随机变量 
>
> $$
> F = \frac{U / n_1}{V / n_2}
> $$
> 
> 服从自由度为 $(n_1, n_2)$ 的 $F$ 分布，记为 $F \sim F(n_1, n_2)$。$F(n_1, n_2)$ 分布的概率密度函数为：
> 
> $$
> \varphi (y) = \left \{ \begin{matrix}
\frac{\Gamma [(n_1 + n_2) / 2] (n_1 / n_2)^{n_1 / 2} y^{(n_1 / 2) - 1}}{1} & y > 0 \\
0 & otherwise
\end{matrix} \right .
> $$
> 
## 函数密度图像

![在这里插入图片描述](https://img-blog.csdnimg.cn/43e8b2bf0a4c4e009890b14eed5c340a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 例题

> 设随机变量 $T \sim t(n)$，$F = \frac{1}{T^2}$ 求随机变量F的分布

解：

先从 $t$ 分布的定义出发，它是 

$$
t = \frac{X}{\sqrt{Y / n}}
$$

其中 $X \sim N(0, 1)$，$Y \sim X^2(n)$，所以我们得到 $T = \frac{X}{\sqrt{Y / n}}$。代入 $F = \frac{1}{T^2}$ 后，我们有

$$
F = \frac{Y / n}{X^2}
$$

由于我们前面已经假设了 $X \sim N(0, 1)$，所以当 $Y' = X^2$ 时，它自然也是卡方分布，且只有一个元素，于是有 $Y' \sim X^2(1)$，参考F分布的定义，我们有

$$
F' = \frac{U / n_1}{V / n_2}
$$

且 $U$，$V$ 均是卡方分布，我们代入已知的 $Y / n$ 到 $U / n_1$，$Y'$ 可等价于 $Y' / 1$ 并且 $Y$ 和 $Y'$互相独立，于是也可以代入到 $V/n_2$，得到最终 $F'$ 的分布

$$
F' = \frac{Y / n}{ Y' / 1} = \frac{Y / n}{X^2}
$$

所以 $F = F'$，于是 $F \sim F(n , 1)$。