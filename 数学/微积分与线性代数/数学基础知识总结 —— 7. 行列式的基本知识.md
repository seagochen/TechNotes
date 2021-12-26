> 行列式是一种特殊的运算形式，跟一维的加减乘除这类运算类似。只不过它所针对的是对高维度数据的求解，它是从方程组的概念发展而来，是一种比较常见的数学工具。
行列式和 **「线性代数」** 里的其他概念，比如 **「矩阵」**、**「向量」** 比较容易混淆。但是实际上在概念上它们的差异很大，你可以说「矩阵是高维度的向量，而向量是一维度的矩阵」，但行列式是行列式，它既不是向量也不是矩阵，它就是一种特殊的 **运算规则** 。

@[toc]

行列式，也称方阵，是一个元素以 $n \times n$ 存在的特殊运算形式，记为 $\det (A)$ 或 $|A|$

# 1. 二阶行列式

$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{matrix} \right | = a_{11} a_{22} - a_{12} a_{21}
$$

# 2. 三阶行列式
$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{matrix} \right | 
$$

对于二阶、三阶行列式，它的运算过程可以非常粗暴地图像化的表示如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/b527d1f8b88742238496c4a1827854db.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
即

$$
\det (A) = a_{11} a_{22} a_{33} + a_{12} a_{23} a_{31} + a_{13} a_{21} a_{32} - a_{11} a_{23} a_{32} - a_{12} a_{21} a_{33} - a_{13} a_{22} a_{31}
$$

# 3. 高阶行列式
**注意，二阶以及三阶行列式的运算规则不适用于高阶行列式，对于正负号的判断要考虑下标的「逆序数」**

$$
\det(A) = \left | \begin{matrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{matrix} \right |  = \sum_{p_1 p_2 \cdots p_n} (-1)^{\tau (p_1 p_2 \cdots p_n)} a_{1p_1} a_{2} \cdots a_{n p_n}
$$

这是一个 $n$ 阶行列式，其中 $\tau (p_1 p_2 \cdots p_n)$ 表示由行列式元素第二下标构成的逆序数。

> 知识补充：
> **逆序数**
> 在一个排列中，如果一对数的前后位置与大小顺序相反，即前面的数大于后面的数，那么它们就称为一个逆序。一个排列中逆序的总数就称为这个排列的逆序数。也就是说，对于n个不同的元素，先规定各元素之间有一个标准次序（例如n个 不同的自然数，可规定从小到大为标准次序），于是在这n个元素的任一排列中，当某两个元素的实际先后次序与标准次序不同时，就说有1个逆序。一个排列中所有逆序总数叫做这个排列的逆序数[^1]。

> 例如对于下面的行列式 $\det (A)$ 
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/0ec243ed5e0f4d899b21bf9a3f6b3232.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 由元素 $a_{12} a_{21} a_{34} a_{43}$ 的第二下标组成序列 $(2143)$，我们根据公式，可以知道要确定正负号就要先确定逆序数是多少。对于 $(2143)$ 来说它的逆序对有 $(2, 1), (4, 3)$，逆序数 $\tau (2143) = 2$，所以
> $$
> (-1)^{\tau (2143)} a_{12} a_{21} a_{34} a_{43} = a_{12} a_{21} a_{34} a_{43}
> $$

> 我们可以再选一组，比如它后面的 $a_{13} a_{22} a_{31} a_{44}$，对于 $(3214)$ 来说，它的逆序对为 $(3, 2), (3, 1), (2, 1)$，所以 $\tau(3214) = 3$，最终
> $$
> (-1)^{\tau (3214)} a_{13} a_{22} a_{31} a_{44} = - a_{13} a_{22} a_{31} a_{44}
> $$

[^1]: https://baike.baidu.com/item/%E9%80%86%E5%BA%8F%E6%95%B0/3334502

# 4. 行列式的性质

## 4.1. 行列式转置后，值不变

$$
\det(A) = \det(A^{T}) =  \left | \begin{matrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{matrix} \right |  = \left | \begin{matrix} 
a_{11} & a_{21} & \cdots & a_{n1} \\
a_{12} & a_{22} & \cdots & a_{n2} \\
\vdots & \vdots & \ddots & \vdots \\
a_{1n} & a_{2n} & \cdots & a_{nn}
\end{matrix} \right |
$$

>  我们可以用三阶段行列式验证
>  $$ 
>  \left | \begin{matrix}
> a_{11} & a_{12} & a_{13} \\
> a_{21} & a_{22} & a_{23} \\
> a_{31} & a_{32} & a_{33}
> \end{matrix} \right | 
> $$
> 它的转置行列式 $\det (A^T)$ 为
> $$
>  \left | \begin{matrix}
> a_{11} & a_{21} & a_{31} \\
> a_{12} & a_{22} & a_{32} \\
> a_{13} & a_{23} & a_{33}
> \end{matrix} \right |  
>$$
>$$
>\det(A^T) =  a_{11} a_{22} a_{33} + a_{21} a_{32} a_{13} + a_{31} a_{12} a_{23} - a_{31} a_{22} a_{13} - a_{11} a_{32} a_{23} - a_{21} a_{12} a_{33}
> $$
> 可以发现结果和 $\det(A)$ 是一样的

## 4.2. 有两行（列）元素成比例，行列式的值为0
我们以三阶行列式为例

$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{matrix} \right | 
$$

令第三行为第二行的倍数，于是

$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
ka_{21} & ka_{22} & ka_{23}
\end{matrix} \right | 
$$

展开后，得到

$$
\det (A) = a_{11} a_{22} [k \cdot a_{23}] + a_{12} a_{23} [k \cdot a_{21}] + a_{13} a_{21} [k \cdot a_{22}] - \\
a_{11} a_{23} [ k\cdot a_{22}] - a_{12} a_{21} [k \cdot a_{23}] - a_{13} a_{22} [k \cdot a_{21}] \\
= k (a_{11} a_{22} a_{23} + a_{12} a_{23} a_{21} + a_{13} a_{21} a_{22} - a_{11} a_{23} a_{22} - a_{12} a_{21} a_{23} - a_{13} a_{22} a_{21} )
$$

然后，我们可以发现

$$
= k (a_{11} a_{22} a_{23} - a_{11} a_{23} a_{22} + a_{12} a_{23} a_{21}  - a_{12} a_{21} a_{23} + a_{13} a_{21} a_{22}  - a_{13} a_{22} a_{21} ) = 0
$$

同理，列也是一样的。

## 4.3. 行列式中某一行（列）元素全为0时，行列式值为0

我们以三阶行列式为例

$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
0         & 0          & 0
\end{matrix} \right |
$$

基本上不用计算，可以直接凭直觉得出它的值为0。
## 4.4. 如果某行（列）的元素都是两个数的和，可以把行列式拆分为两个单独的行列式

$$
\det(A) = \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
b_{i1} + c_{i1}    & \cdots & b_{in} + c_{in} \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right | =   \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
b_{i1}  & \cdots & b_{in}  \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right |  +  \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
c_{i1}  & \cdots & + c_{in} \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right | 
$$

## 4.5. 行列式的两行（列）互换，行列式变号
这里，我们为了方便，直接用二阶行列式做说明

$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{matrix} \right | = a_{11} a_{22} - a_{12} a_{21}
$$

交换行或列后，得到

$$
\det(A') = \left | \begin{matrix}
a_{12} & a_{11} \\
a_{22} & a_{21}
\end{matrix} \right | = a_{12} a_{21} - a_{11} a_{22} 
$$

所以为了让 $\det (A')$ 和 $\det(A)$ 的值一致，我们要令 $\det (A) = -\det (A')$

## 4.6. 如果某行（列）的元素有公因子，可将公因子提到行列式外

$$
\det(A) = \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
k a_{i1}& \cdots & k a_{in} \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right | = k \cdot \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
a_{i1}  & \cdots & a_{in} \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right |
$$

## 4.7.  行列式中某一行（列）元素的k 倍加到另一行（列），其值不变
$$
\det(A) = \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
a_{i1}  & \cdots & a_{in} \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right | = \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
a_{i1} + ka_{j1}  & \cdots & a_{in}  + ka_{jn}\\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right |
$$

这其实很容易求证，根据第4条性质，上面的行列式可以表示为


$$
\det(A) = \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
a_{i1} + ka_{j1}  & \cdots & a_{in}  + ka_{jn}\\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right | =   \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
a_{i1}  & \cdots & a_{in}  \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right |  +  \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
ka_{j1}  & \cdots & + ka_{jn} \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right | 
$$

然后后面的行列式的 $ka_{jn}$ 行又因为是 $a_{jn}$ 的倍数，于是


$$
\det(A) = \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
a_{i1} + ka_{j1}  & \cdots & a_{in}  + ka_{jn}\\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right | =   \left | \begin{matrix} 
a_{11} & \cdots & a_{1n} \\
\vdots &            & \vdots \\
a_{i1}  & \cdots & a_{in}  \\
\vdots &            & \vdots \\
a_{n1} & \cdots & a_{nn}
\end{matrix} \right |
$$

# 5. 行列式的展开定理

## 5.1. 行列式的代数余子式
在 $n$ 阶行列式中，将元素 $a_{ij}$ 所在的 **「$i$ 行」** 与 **「$j$ 列」** 的元素划去，其余元素按照原来的相对位置构成 $n-1$ 阶行列式 $M_{ij}$

$$
A_{ij} = (-1)^{i + j} M_{ij}
$$

怎么理解呢，比方说对于三阶行列式

$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{matrix} \right | 
$$

其元素 $a_{11}$ 的代数余子式表示为

$$
A_{11} = (-1)^{1+1} \left | \begin{matrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{matrix} \right |  = \left | \begin{matrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{matrix} \right |
$$

其元素 $a_{23}$ 的代数余子式为

$$
A_{23} = (-1)^{2+3} \left | \begin{matrix}
a_{11} & a_{12} \\
a_{31} & a_{32} 
\end{matrix} \right |  = - \left | \begin{matrix}
a_{11} & a_{12} \\
a_{31} & a_{32} 
\end{matrix} \right | 
$$


这有什么用呢？显然它可以简化运算，比方说我们展开三阶行式

$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{matrix} \right | 
$$

它可以表示为某行或列的元素的代数余子式的和，即


$$
\det(A) = \left | \begin{matrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{matrix} \right | \\
= a_{11} A_{11} + a_{12} A_{12} + a_{13} A_{13} \\
= a_{12} A_{12} + a_{22} A_{22} + a_{32} A_{32} = \cdots
$$

我们随便以 $a_{11} A_{11} + a_{12} A_{12} + a_{13} A_{13}$ 为例

$$
= a_{11} \left | \begin{matrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{matrix} \right | - a_{12} \left | \begin{matrix}
a_{21} & a_{23} \\
a_{31} & a_{33}
\end{matrix} \right | + a_{13}  \left | \begin{matrix}
a_{21} & a_{22} \\
a_{31} & a_{32}
\end{matrix} \right | \\
= a_{11} a_{22} a_{33} + a_{12} a_{23} a_{31} + a_{13} a_{21} a_{32} - a_{11} a_{23} a_{32} - a_{12} a_{21} a_{33} - a_{13} a_{22} a_{31}
$$


# 6. 特殊的行列式

## 6.1. 「上三角」或「下三角」行列式

如果行列式「上三角」或「下三角」的元素全为0，那么行列式等于对角线上元素的乘积

$$
\det(A) =  \left | \begin{matrix} 
a_{11}  & 0           & \cdots & 0          & 0 \\
a_{21}  & a_{22}  & \cdots & 0          & 0 \\
\vdots  & \vdots  & \ddots & \vdots & \vdots \\
a_{m1} & a_{m2}  & \cdots & a_{mm} & 0 \\
a_{n1}  & a_{n2}  & \cdots & a_{mn}   & a_{nn}
\end{matrix} \right |  = a_{11} a_{22} a_{33} \cdots a_{mm} a_{nn}
$$

## 6.2. n 阶范德蒙德行列式

$$
\det(A) =  \left | \begin{matrix} 
1      & 1      & \cdots & 1      & 1 \\
x_1  & x_2  & \cdots & x_m  & x_n \\
x_1^2  & x_2^2  & \cdots & x_m^2  & x_n^2 \\
\vdots  & \vdots  & \ddots & \vdots & \vdots \\
x_1^{n-1}  & x_2^{n-1}  & \cdots & x_m^{n-1}  & x_n^{n-1} \\
\end{matrix} \right |  = \Pi_{1 \leq i \leq j \leq n} (x_i - x_j)
$$