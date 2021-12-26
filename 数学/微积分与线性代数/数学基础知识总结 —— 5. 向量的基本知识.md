@[toc]

# 1. 什么是向量的模
向量的模，即向量的大小或长度。比如，对于向量 $\vec{V} = (v_i, v_2, \cdots, v_n)$ 它的模表示为：

$$
\left \|  \vec{V} \right \| = \sqrt{\sum_{i=1}^n v_i^2}
$$

# 2. 什么是单位向量

在线性代数及相关数学领域中，单位向量就是长度为 1 的向量。单位向量的符号通常有个“帽子”，$\hat n$。通常一个非零向量的正则化向量就是该向量的单位向量。比如，对于向量 $\vec{V} = (v_i, v_2, \cdots, v_n)$ 它的单位向量 $\hat v$ 表示为

$$
\hat v = \frac{\vec{V}}{\left \|  \vec{V} \right \| }
$$

此外，三维直接坐标系里，$\hat i$，$\hat j$，$\hat k$ 分别是x，y，z轴的单位向量

$$
\hat i = \left [ \begin{matrix}
1 \\
0 \\
0
\end{matrix} \right ] 
$$

$$
\hat j = \left [ \begin{matrix}
0 \\
1 \\
0
\end{matrix} \right ] 
$$

$$
\hat k = \left [ \begin{matrix}
0 \\
0 \\
1
\end{matrix} \right ]
$$

在其他坐标系中，如极坐标系、球坐标系，使用不同的单位向量，符号也会不一样。

# 3. 什么是零向量

在线性代数及相关数学领域中，**零向量**（也称 **退化向量**）即所有元素都为 0 的向量  $\vec{V} =(0, 0, …, 0)$。零向量可以表示为大写字母 $O$，或 $\vec{0}$。

$$
\vec{0} = \left [ \begin{matrix}
0 \\
0 \\
\vdots \\
0
\end{matrix} \right ] 
$$

# 4. 什么是反向量

一个向量 $\vec{V}$ 的反向量是指与它大小相等，但方向相反的向量。它有如下定义：

$$
\vec{a} + \vec{b} = \vec{0}
$$

向量 $\vec a$ 和向量 $\vec b$ 互为 **「反向量」**。

# 5. 什么是等向量

不论起点终点，只要有两个或多个向量，彼此方向、大小相等，则称这些向量是 **「等向量」** 或 **「相等的向量」**。


# 6. 什么是方向向量

对于任意向量 $\vec a$ ，若存在一个向量 $\vec b$，两者的方向相同，大小不一定相同，则 $\vec b$ 是 $\vec a$ 的一个方向向量。


# 7. 向量运算
## 7.1. 向量与常数的乘积运算 （可以计算向量倍长）

对于向量 $\vec{V} = (v_i, v_2, \cdots, v_n)$ 它与常数 $k$ 的乘积表示为：

$$
k \cdot \vec{V} = k \cdot \left [  \begin{matrix} 
v_1 \\
v_2 \\
\vdots \\
v_n
\end{matrix} \right ] = \left [ \begin{matrix} 
k \cdot v_1 \\
k \cdot v_2 \\
\vdots \\
k \cdot v_n
\end{matrix} \right ]
$$
## 7.2. 向量的加法和减法 （向量的线性组合）
两个向量 $\vec{a}$ 和 $\vec{b}$ 相加或相减，得到的都是另一个向量，即

$$
\vec a \pm \vec b = \left [ \begin{matrix} 
a_1 \\
a_2 \\
\vdots \\
a_n
\end{matrix} \right ] \pm \left [ \begin{matrix} 
b_1 \\
b_2 \\
\vdots \\
b_n
\end{matrix} \right ] = \left [ \begin{matrix} 
a_1 \pm b_1 \\
a_2 \pm b_2 \\
\vdots \\
a_n \pm b_n
\end{matrix} \right ]
$$

从空间图像上，向量之间的加减法遵循 **「三角形法则」** 或 **「四边形法则」** ，它们可以表示为 $\vec{a}$ 和 $\vec{b}$ 的起点重合后，以它们为邻边构成的平行四边形的一条对角线，或者表示为将$\vec{a}$ 的终点和 $\vec{b}$ 的起点重合后，从 $\vec{a}$ 的起点指向 $\vec{b}$ 的终点的向量：
![在这里插入图片描述](https://img-blog.csdnimg.cn/caba22d12fcf42caa34d7f6b22c7afd4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)

对于 $\vec a - \vec b$ 来说，它也可以理解为 $\vec a + (-\vec b)$ ，也就是在向量 $\vec b$ 的顶点做一条方向相反、大小相等的反向量，然后执行向量 $\vec a$ 与向量 $-\vec b$ 的加法运算。


## 7.3. 数量积（点乘，可以求向量的投影）

数量积也叫点积，它是向量与向量的乘积，其结果为一个标量（非向量）。几何上，数量积可以定义如下：

$$
\vec a \cdot \vec b = |\vec a| |\vec b| \cos \theta
$$

即向量 $\vec a$ 在向量 $\vec b$ 方向上的投影长度，$\theta$ 表示为两向量的夹角。

![在这里插入图片描述](https://img-blog.csdnimg.cn/718f80e2b7454acead27da0368c39e4d.png#pic_center)


## 7.4. 向量积（叉乘，可以求向量面积或垂直向量）

向量积也叫叉积，外积，它也是向量与向量的乘积，不过需要注意的是，它的结果是个向量。它的几何意义是所得的向量与被乘向量所在平面垂直，方向由右手定则规定，**大小是两个被乘向量张成的平行四边形的面积**，所以向量积不满足交换律。

$$
\vec a \times \vec b = \left | \begin{matrix}
\vec i & \vec j & \vec k \\
a_x    & a_y    & a_z     \\
b_x    & b_y    & b_z
\end{matrix} \right | = (a_y b_z -  a_z b_y)\vec i + (a_x b_z - a_z b_x) \vec j + (a_x b_y - a_y b_x) \vec k
$$

> 我个人感觉这个概念应该还是最早出自物理研究。在比较后数学和物理的记忆方法后，我个人认为物理关于记忆磁通量 $B$ 的方法其实更合适。也就是说：
> **伸出你的右手，四指指向 $\vec a$ 的方向，然后朝着 $\vec b$ 的方向握紧拳头，此时大拇指的方向即 $\vec c$ 的方向**。
> 然后，这个记忆方法可以扩展到比如「旋度」等各种概念上，非常好用。

![在这里插入图片描述](https://img-blog.csdnimg.cn/ee81ef5e15ef4ac3a48a69f53a176921.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 7.5. 混合积（可以求三个向量组成的六面体的体积）

几何上，由三个向量 $\vec a$，$\vec b$，$\vec c$ 定义的平行六面体，其体积等于这三个向量的混合积，其表示为：

$$
V = |\vec a \vec b \vec c| = (\vec a \times \vec b) \cdot \vec c = (\vec b \times \vec c) \cdot \vec a = (\vec c \times \vec a) \cdot \vec b
$$

> 其证明是这样：
> 我们知道一般体积公式表示为
> $$
> 体积(V) = 底面积(S) \times 高(h)
> $$
> 对于地面积S，可以由
> $$
> S = |\vec a \times \vec b|
> $$
> 得到。尽管叉乘得到的是一个垂直于 $\vec a$ 和 $\vec b$ 的向量 $\vec S$，但是它的模，也就是大小等于 $|\vec a| \times |\vec b|$ ,也就是的面积。
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/ba5ff233f03f4c79a8dddd0f7ddfdcf5.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 此时，$\vec S \cdot \vec c$ 得到的是 $\vec c$ 在向量 $\vec S$ 的投影长 $|\vec c| \cos \theta$ 也就是高。
> 于是，由点乘公式可以得到
> $$
> \vec S \cdot \vec c = |\vec S| |\vec c| \cos \theta
> $$
> 即
> $$
> (\vec a \times \vec b) \cdot \vec c = 面积(|S|) \cdot 高 (|\vec c| \cdot \cos \theta)
> $$