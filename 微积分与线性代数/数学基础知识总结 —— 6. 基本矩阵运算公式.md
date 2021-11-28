常量与变量的运算公式非常简单，这里不做赘述。所以我们重点会放在矩阵、行列式，以及向量的运算公式上。

@[toc]

# 矩阵运算公式
矩阵运算主要分为加减乘，而没有除法。除法运算通常是计算除数矩阵的逆矩阵，然后再用乘法。而矩阵乘法，常用的主要有两种，点乘和叉乘。

通常矩阵（Matrix）之间要想进行某种数学运算，都多少有其维度的限制性要求。比如要求矩阵之间行列数相同，或者要求前矩阵的列数和后矩阵的行数相等。

## 矩阵加减法（两矩阵之间要求维度相同）
假设有如下矩阵：

$$
A =\begin{bmatrix}
a_{00} & a_{01} & \cdots &a_{0j} \\ 
a_{10} & a_{11} & \cdots & a_{1j} \\ 
\cdots & \cdots  & \cdots & \cdots\\
a_{i0} & a_{i1} & \cdots  & a_{ij}
\end{bmatrix}
$$

你会发现我这里用了一个跟传统教科书不太一样的下标表示。如果你写过程序，就能明白矩阵的元素遍历，和程序里常见的二维数组遍历的形式很像：

用C系语言描述，大概就是这一个样子的

```c
for (int i = 0; i < X; i++) {
	for (int j = 0; j < Y; j++) {
	///
	}
}
```

$i$ 通常被定义为矩阵的行， $j$ 通常被定义为矩阵的列。所以通常矩阵A和矩阵B进行加减法运算时，新矩阵的元素就是这样的：

$$
A \pm B =\begin{bmatrix}
a_{00} \pm b_{00}, & a_{01} \pm b_{01}, & \cdots &a_{0j} \pm b_{0j} \\ 
a_{10} \pm b_{10}, & a_{11} \pm b_{11}, & \cdots & a_{1j} \pm b_{1j} \\ 
\cdots & \cdots  & \cdots & \cdots\\
a_{i0} \pm b_{i0}, & a_{i1} \pm b_{i1}, & \cdots  & a_{ij} \pm b_{ij}
\end{bmatrix}
$$

### 运算法则
满足交换律
$$
A \pm B = B \pm A
$$

满足结合律
$$
A \pm B \pm C = A \pm (B \pm C) = (A \pm B) \pm C
$$

## 矩阵乘法——哈达玛积(Hadamard product)（两矩阵之间要求维度相同）

这个运算是比较常用的一种运算，通常当作掩码运算规则使用。

$$
A \cdot B =\begin{bmatrix}
a_{00} \cdot b_{00}, & a_{01} \cdot b_{01}, & \cdots &a_{0j} \cdot b_{0j} \\ 
a_{10} \cdot b_{10}, & a_{11} \cdot b_{11}, & \cdots & a_{1j} \cdot b_{1j} \\ 
\cdots & \cdots  & \cdots & \cdots\\
a_{i0} \cdot b_{i0}, & a_{i1} \cdot b_{i1}, & \cdots  & a_{ij} \cdot b_{ij}
\end{bmatrix}
$$

数学上常以点号表示这个计算，也偶尔可以在一些国外教科书上见到 $A \circ B$ 这一类的表示。不过要想国际都能理解的话，你可以用函数式的形式，即 $hadamard(A,B)$ 进行表示。

### 运算法则
满足交换律
$$
A \cdot B = B \cdot A
$$

满足结合律
$$
A \cdot B  \cdot C = A \cdot (B  \cdot C) = (A \cdot B)  \cdot C
$$


## 矩阵乘法——叉乘/向量外积（要求前列与后行元素数一致）
叉乘是高数或者线性代数课本上最常考的一类运算，通常用 $A \times B$ 的形式进行表示，函数式写作 $cross(A,B)$ 。它是否能够执行有一个要求，对于n行m列的矩阵 $A_{2 \times 3}$，它只能跟行数与A的列数相等的 $B_{3 \times n}$ 相乘。

例如：

$$
A_{23} \times B_{32} = \begin{bmatrix}
a_{00} & a_{01} & a_{02} \\
a_{10} & a_{11} & a_{12}
\end{bmatrix} \times \begin{bmatrix}
b_{00} & b_{01} \\
b_{10} & b_{11} \\
b_{20} & b_{21} 
\end{bmatrix}
$$ 

$$
= \begin{bmatrix}
a_{00} \cdot b_{00} + a_{01} \cdot b_{10} + a_{02} \cdot b_{20}, &
a_{00} \cdot b_{01} + a_{01} \cdot b_{11} + a_{02} \cdot b_{21} \\
a_{10} \cdot b_{00} + a_{11} \cdot b_{10} + a_{12} \cdot b_{20}, &
a_{10} \cdot b_{01} + a_{11} \cdot b_{11} + a_{12} \cdot b_{21}
\end{bmatrix}
$$

### 运算法则
不满足交换律

$$
A \times B \neq B \times A
$$

满足结合律

$$
A \times B \times C = A \times (B \times C)  = (A \times B) \times C 
$$

## 矩阵乘法——内积（两矩阵之间要求维度相同）
也是比较常用的一种运算。大多数的空间滤波函数的实现方式都依赖这个内积，函数式写作 $dot(A, B)$

假设：

$$
A =\begin{bmatrix}
a_{00} & a_{01} & a_{02} \\ 
a_{10} & a_{11} & a_{12} 
\end{bmatrix}
$$

$$
B =\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} 
\end{bmatrix}
$$

则 $AB= a_{00} \cdot b_{00} + a_{01} \cdot b_{01} + a_{02} \cdot b_{02} + a_{10} \cdot b_{10} + a_{11} \cdot b_{11} + a_{12} \cdot b_{12}$

### 运算法则
满足交换律
$$
AB = BA
$$

满足结合律
$$
ABC = (AB)C = A(BC)
$$

## 矩阵乘法——克罗内科积（Kronecker product）（维度没有要求）

这是唯一个对矩阵维度没有要求的积，可以认为是一种排列组合。

$$
A\otimes B={\begin{bmatrix}a_{{11}}B&\cdots &a_{{1n}}B\\\vdots &\ddots &\vdots \\a_{{m1}}B&\cdots &a_{{mn}}B\end{bmatrix}}
$$

$$
{\begin{bmatrix}1&2\\3&1\\\end{bmatrix}}\otimes {\begin{bmatrix}0&3\\2&1\\\end{bmatrix}}={\begin{bmatrix}1\cdot 0&1\cdot 3&2\cdot 0&2\cdot 3\\1\cdot 2&1\cdot 1&2\cdot 2&2\cdot 1\\3\cdot 0&3\cdot 3&1\cdot 0&1\cdot 3\\3\cdot 2&3\cdot 1&1\cdot 2&1\cdot 1\\\end{bmatrix}}={\begin{bmatrix}0&3&0&6\\2&1&4&2\\0&9&0&3\\6&3&2&1\end{bmatrix}}
$$

### 运算法则
满足交换律
$$
A \otimes B = B \otimes A
$$

满足结合律
$$
A \otimes B \otimes C = (A \otimes B) \otimes C = A \otimes (B \otimes C)
$$

# 矩阵与常数的运算
常数与矩阵的加法、减法、乘法，都可以认为是常数对矩阵的全部元素的运算。通常我们以空心字母表示矩阵比如 $\mathbb{M}$，以实心小写字母表示常数比如 $c$，它们的运算关系表示如下。因此，可以得到这样的一个模板：

$$
\mathbb{M} \star c   = \begin{bmatrix}
M_{00} \star c, & M_{01} \star c, & \cdots, & M_{0m}  \star c \\ 
M_{10}  \star c, & M_{11} \star c, & \cdots, & M_{1m}  \star c \\
\vdots                 &                           &               & \vdots \\
M_{n0} \star c,  & M_{n1} \star,   & \cdots  & M_{nm}  \star c         
\end{bmatrix}
$$

$\star$ 可以是 $+ - \times \div$。

### 运算法则
满足交换律，结合律