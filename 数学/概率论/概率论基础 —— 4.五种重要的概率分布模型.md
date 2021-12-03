> 说到概率分布模型，通常是基于实际研究和观察而总结出来的。你的数学老师可能只会要求你记住这些模型的公式，而不太关心其背后的应用场景。而在我之前的文章里有聊到过数字信号当中关于常见噪音的形式。
> [[数字图像学笔记] 5.1. 噪音生成——椒盐噪音、高斯噪音](https://seagochen.blog.csdn.net/article/details/110847127)
> [[数字图像学笔记] 5.2. 噪音生成——泊松噪音](https://seagochen.blog.csdn.net/article/details/113576027)
> 除了噪音的处理，在实际工作中，对事件发生概率的处理和统计，也会发现它们与这里即将介绍的五种概率模型高度相关。所以你如果是数据专业方向的学生，或者相关领域的，在学习概率论这门课程中，你也应当特别重视这几个常见的概率分布模型。

@[toc]

# 基础的五种概率模型

## 二项分布（离散型）
二项分布是一种在统计学里常常用到的数学模型，比如统计一个地区物种数量的分布情况，一个班级里学生成绩的分布，一个国家居民收入、支出情况等。很多数据在采集并统计后，往往呈现二项分布，或柏松分布的特点。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715215701245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
其数学记号表示为：$X \sim  b(n,p)$，其对应的概率分布律函数为： 

$$P \left \{ X = k \right \} = C_n^k p^k(1-p)^{n-k}$$

* 期望$E(X) = np$
* 方差$D(X) = npq$

这个式子，表示事件A发生的概率为P，在N次事件中发生了K次。换句话说，即对于独立事件A，其事件发生的概率是P，然后在某一时刻发生了K次。

比方说对于一个有N个小区的片区，每个小区发生停电的概率是P，然后问在某一天有M个小区都发生了停电的概率。


## 柏松分布（离散型）

柏松分布和二项式分布在图形上表现十分相似，但是区别在于从记录事件时间T开始，在某一时刻t时事件发生的概率会达到最大可能性，之后概率会迅速的减少。比如说，某个时刻天空中恰好有流星划过，问在接下来时间T里，再次有流星出现的概率。

通常，柏松分布也叫等待概率，是一种比二项分布应用场景更为丰富的概率模型，在数控、电商优化中也经常能见到它的影子。

其数学记号表示为：$X \sim  \pi (\lambda)$，其对应的概率分布律函数为： 

$$P \left \{ X = k \right \} =\frac{\lambda ^k}{k!} e^{-\lambda}$$

* 期望$E(X) = \lambda$
* 方差$D(X) = \lambda$

比如对于一场足球比赛来说，进球的期望是每场2.5个球。那么在一次比赛中，进球数K和这个事件发生的概率就可以表示为下面这张表：

K  | 0 | 1 | 2 | 3 | ... 
---|---|---|---|----|-----
P | 0.082 | 0.205 | 0.257 | 0.213 | ...


## 均匀分布（连续型）

均匀分布感觉不是特别常见，，因为在实际的场景中，很少有样本事件概率是同时保持一致的。之所以这样说，是指它指在一段区间内，事件在相同长度间隔的分布概率是等可能的。

不过对于，比如投硬币、投骰子等这种等概率事件时，均匀分布才比较容易见到。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715225902890.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

这里，连续型和离散型最大的一个区别，在于连续型通常会优先使用概率密度函数进行表示。

其数学记号表示为：$X \sim  U (a, b)$，其对应的概率密度函数为： 

$$f(x) = \left\{\begin{matrix}
\frac{1}{b - a} & a < x < b \\ 
0 & else
\end{matrix}\right.$$

* 期望$E(X) = \int x f(x) dx = \frac{1}{2}(a+b)$
* 方差$D(X) = \frac{(b-a)^2}{12}$

## 指数分布（连续型）

指数分布是另外一种比较少见的，因为更多的样本分布概率是以泊松分布的形式出现，指数分布很大程度被认为是柏松分布的特殊的近似形式。

也就是说大多数情况下，事件多少都会有泊松分布的函数图像所有的先升再降的特点，而不是完全光滑曲线图像。

另外，考虑到概率事件通常加和上限为1，所以提到指数分布，它指的是单调下降的指数，而不是上升的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210715231301586.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
其数学记号表示为：$X \sim  E (\lambda)$，其对应的概率密度函数为： 

$$f(x) = \left\{\begin{matrix}
\lambda e^{-\lambda x}  & x > 0 \\ 
0 & else
\end{matrix}\right.$$

* 期望$E(X) =  \frac{1}{\lambda}$
* 方差$D(X) = \frac{1}{\lambda^2}$

## 正态分布（连续型）
经典分布类型，也是除泊松分布，二项分布以外最常见的一类。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021071523210556.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)


其数学记号表示为：$X \sim  N (\mu, \sigma^2)$，其对应的概率密度函数为： 

$$f(x) = \frac{1}{\sqrt{2 \pi } \sigma} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}$$

* 期望$E(X) = \mu$
* 方差$D(X) = \sigma^2$
# 做点练习题
## 离散型二项分布

> 一个小区有7台供电设备，每台设备都是独立使用，且每台被使用到的概率为0.1，问某一时刻：
> （1）恰有3台设备被使用的概率
> （2）至少有2台设备被使用的概率

**解（1）** 我们根据二项分布$X \sim  b(7, 0.1)$ 可以得到：

$$P \left \{ X = 3 \right \} = C_7^3 0.1^3 0.9^4 \approx  0.023$$

**解（2）**$P \left \{ X >= 2 \right \} = 1 - P \left \{ X < 2 \right \} = 1- P \left \{ X = 0 \right \} - P \left \{ X = 1 \right \}$ 可知：

于是我们可以得到：

$$P \left \{ X = 0 \right \} = C_7^0 0.1^0 0.9^7 \approx 0.478$$

$$P \left \{ X = 1 \right \} = C_7^1 0.1^1 0.9^6 \approx 0.372$$

$$P \left \{ X >= 2 \right \} = 1 - 0.372 - 0.478 \approx 0.15$$


## 离散型泊松分布

> 假设某地区地震发生次数服从 $\lambda = 2$ 的泊松分布，则未来一年，该地区至少发生一次地震的概率是多少？

**解** 我们根据泊松分布$X \sim  \pi (\lambda)$，以及它的分布函数，$P \left \{ X = k \right \} = \frac{\lambda ^k}{k!} e^{-\lambda}$ 可以得到：

因为 $P \left \{ X >= 1 \right \} = 1 - P \left \{ X < 1 \right \} = 1- P \left \{ X = 0 \right \}$，所以可以得到：

$$P \left \{ X = 0 \right \} = \frac{2 ^0}{0!} e^{-2} = e^{-2}$$

$$P \left \{ X >= 1 \right \} = 1 - e^{-2}$$


## 连续型均匀分布

> 随机变量X在区间（0，1）服从均匀分布，求 $Y=-3lnX$ 的概率密度

**解** 定义出发：$X \sim  U (a, b)$，它的对应概率密度函数为： 

$$f(x) = \left\{\begin{matrix}
\frac{1}{b - a} & a < x < b \\ 
0 & else
\end{matrix}\right.$$

即：

$$f(x) = \left\{\begin{matrix}
1 & 0 < x < 1 \\ 
0 & else
\end{matrix}\right.$$

因为$Y = -3 ln X$， 而它的函数图像如下，由于x只在 $(0, 1)$区间内有效，所以可以得到 $Y > 0$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210716121343522.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
然后我们从上式子可以得到关于X的函数，即Y的逆函数：$$X = e^{-\frac{1}{3}Y}$$

然后由于原函数是单调的，所以直接用公式法可以得到：

$$f_y(y) = f_x(x) |x'| = f_x(x) |(e^{-\frac{1}{3}Y})'| = \frac{1}{3} e^{- \frac{1}{3}Y}$$

因为$f_x$只有在 $0 < x < 1$区间内有值，其余为0，而在同一区间内的y函数是 $y > 0$，所以最终关于Y的密度函数 $f_y$ 表示为：

$$f_y(y) = \left\{\begin{matrix}
\frac{1}{3} e^{- \frac{1}{3}Y} & Y > 0 \\ 
0 & else
\end{matrix}\right.$$

这题不是很容易通过一般概念做得出来，所以可以在Y函数满足单调可导这个条件时，使用公式法直接进行求解。公式法的推导在我上一章里有提到，可以用定积分换元法+链式法则推导出来的。

## 连续型指数分布
> 若X服从 $E(\lambda)$，且 $P \left \{ X > 3 \right \} = e^{-6}$。
> （1）求$\lambda$ 
> （2）$P \left \{ X > 9 | X > 4 \right \}$

**解（1）:** 我们从指数分布的定义出发：

$$f(x) = \left\{\begin{matrix}
\lambda e^{-\lambda x}  & x > 0 \\ 
0 & else
\end{matrix}\right.$$

从题干 $P \left \{ X > 3 \right \} = e^{-6}  \rightarrow \int_3^{+ \infty} \lambda e^{-\lambda x} dx = e^{-6}$

所以：$$ \int_3^{+ \infty} \lambda e^{-\lambda x} dx = -e^{- \lambda x} |_3^{\infty}$$

即：

$$-e^{- \infty} + e^{-3 \lambda} = e^{-6} \rightarrow \lambda = 2$$


**解（2）:** 由几何分布的无记忆性，可以得到：

$$P \left \{ X > 9 | X > 4 \right \} = P \left \{ X > 5 \right \} =  \int_5^{+ \infty} 2 e^{-2 x} dx$$

即：

$$ \int_5^{+ \infty} 2 e^{-2 x} dx = -e^{-2x} |_5^{\infty} = e^{-10}$$


# 关于几何分布的无记忆性

所谓几何分布，就是指每次事件发生时，前后事件之间不存在关联性，而且概率不随着时间或者其他因素发生变化，概率值一直恒定。就比如抛硬币，正面朝上的概率一直是0.5一样。

所以从几何分布出发，会发现指数分布和均匀分布其实很多时候是描述一个事物的两种不同的角度。那么，怎么去理解呢。

比方说，抛十次硬币，每一次正面朝上的概率都是0.5，因此画出概率密度图就是一根直线。但是如果问，第一次正面朝上的概率是1/2，那么第二次也是正面朝上的概率就是1/4，第三次就是1/8... 这样画出的概率密度图就表现为指数形式。所以可以看到，描述问题的方式，决定了我们采用什么模型，但是问题本身并不会因为描述方式的改变而改变，比如抛硬币，无论我们怎么描述抛硬币的事件，正面朝上的客观概率依然保持在0.5

给几何分布一个比较明确的定义，即如下：

* 进行一系列相互独立的试验。
* 每一次试验都既有成功的可能，也有失败的可能，且单次试验的成功概率相同。
* 为了取得第一次成功需要进行的试验次数。

所以，这样的分布形式就被称为几何分布。那么几何分布一个最关键的性质，即几何分布的无记忆性。也就是说m次发生事件的概率与 m-1 次无关。因此在涉及到条件概率的问题时，这就可以很容易的简化为以下形式：

$$P \left \{ X > a + b | X > a \right \} =  P \left \{ X > b  \right \}$$

比方说，前10次出现了反面的情况下，第11次出现反面的情况，就可以用这个概率模型直接求出等于 $P \left \{ X = 1 \right \} = 0.5$

关于几何分布更多的讨论，你可以看看[某乎](https://www.zhihu.com/question/36965252)。
