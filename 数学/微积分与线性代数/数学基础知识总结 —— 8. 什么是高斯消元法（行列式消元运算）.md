@[toc]

# 先提问题

我们先提出这样一个问题，对于如下这样一组方程，应该如何求出它的 $x$, $y$, $z$？

$$
\left\{\begin{matrix}
x + 2y + z = 2 & eq. 1 \\
3x + 8y + z = 12 & eq. 2\\
4y + z = 2 & eq. 3
\end{matrix}\right.
$$

我们必然首先假设这样一组方程它是有解的（当然，你也可以假设它无解，然后右上角点关闭离开这个页面  Σ( ° △ °|||) ）。然后我们会尽可能让上面的方程组简化为下面形式：

$$
\left\{\begin{matrix}
x + y + z = 2 & eq. 1 \\
z +?y = ? & eq. 2\\
y = ? & eq. 3
\end{matrix}\right.
$$

也就是说，只要我们计算出x, y, z中的任何一个，就可以通过公式之前的比例关系最终求解出全部的未知数。而这，就是高斯消元法的基本思路。

# 高斯消元法

我们先回顾一下初中就学过的基本知识，就是方程两边同时加减乘除一个数，不改变结果，所以

$$
\left\{\begin{matrix}
x + 2y + z = 2 \\
3x + 8y + z = 12 \\
4y + z = 2
\end{matrix}\right.\Longrightarrow \left\{\begin{matrix}
3x + 6y + 3z = 6 \\
3x + 8y + z = 12 \\
4y + z = 2
\end{matrix}\right.
$$

而初中的基本知识也告诉过我们，方程组的方程之间相加减也不改变最终结果，所以

$$
\left\{\begin{matrix}
3x + 6y + 3z = 6 \\
3x + 8y + z = 12 \\
4y + z = 2
\end{matrix}\right. \Longrightarrow \left\{\begin{matrix}
x + 2y + z = 2 \\
2y - 2z = 6 \\
4y + z = 2
\end{matrix}\right.
$$

我们把上面这个计算过程简化一下，于是

$$
\left\{\begin{matrix}
x + 2y + z = 2 \\
3x + 8y + z = 12 \\
4y + z = 2
\end{matrix}\right. \overset{r2 - 3 \times r1}{\Longrightarrow}  \left\{\begin{matrix}
x + 2y + z = 2 \\
2y - 2z = 6 \\
4y + z = 2
\end{matrix}\right.  \overset{r3 - 2 \times r2}{\Longrightarrow}  \left\{\begin{matrix}
x + 2y + z = 2 \\
2y - 2z = 6 \\
5z = -10
\end{matrix}\right. 
$$

在最终我们得到的方程组，由于得到了 $z = -2$ 这个解，于是我们就可以倒回去依次求出 y, x 的解。**这个直观的过程，就是高斯消元法本尊了，而以上计算过程是可以推广的。**

需要注意，从最初的方程组出发，每一次计算，我们一定要保证下面一行方程组相较于上面一行方程组少了一个未知数，而消去未知项的顺序，可以是顺序或逆序的，这样才能保证到最后一步时，我们能倒推回去得到全部未知项。

**例外情况：**

当然你可能会问，到某一步的时候，会不会出现某个计算突然消失了多于一项的情况。答案是当然存在的，所以这个时候你可能需要交换方程组的行，或列。而如果出现了下面这种情况：

$$
 G(x, y, z) = \left\{\begin{matrix}
x + 2y + z = 2 \\
2y - 2z = 6 \\
2y -2z = 5
\end{matrix}\right. 
$$

方程组可能有解也可能无解（通常是无解的），那么就可以把方程组设为

$$G(x, y, z) = 0$$

因为这个方程组无意义。

# 简化模型

现在我们简化模型，而且仅考虑有解的情况。我们现在用矩阵的形式表示这个计算过程

$$
\left\{\begin{matrix}
x + 2y + z = 2 \\
3x + 8y + z = 12 \\
4y + z = 2
\end{matrix}\right. \overset{r2 - 3 \times r1}{\Longrightarrow}  \left\{\begin{matrix}
x + 2y + z = 2 \\
2y - 2z = 6 \\
4y + z = 2
\end{matrix}\right.  \overset{r3 - 2 \times r2}{\Longrightarrow}  \left\{\begin{matrix}
x + 2y + z = 2 \\
2y - 2z = 6 \\
5z = -10
\end{matrix}\right. 
$$

首先是原始的方程组，这个过程我相信大多数读者应该都能看懂

$$\left\{\begin{matrix}
x + 2y + z = 2 \\
3x + 8y + z = 12 \\
4y + z = 2
\end{matrix}\right. \Longrightarrow \begin{bmatrix}
 1 & 2 & 1 & 2\\
 3 & 8 & 1 & 12\\
 0 & 4 & 1 & 2
\end{bmatrix} $$

这样计算过程就可以表示为

$$\begin{bmatrix}
 1 & 2 & 1 & 2\\
 3 & 8 & 1 & 12\\
 0 & 4 & 1 & 2
\end{bmatrix} \overset{r2 - 3 \times r1}{\Longrightarrow} \begin{bmatrix}
 1 & 2 & 1 & 2\\
 0 & 2 & -2 & 6\\
 0 & 4 & 1 & 2
\end{bmatrix}  \overset{r3 - 2 \times r2}{\Longrightarrow} \begin{bmatrix} 
 1 & 2 & 1 & 2\\
 0 & 2 & -2 & 6\\
 0 & 0 & 5 & -10
\end{bmatrix} \Longrightarrow \begin{bmatrix} 
1 & 2 & 1 & 2 \\
0 & 1 & -1 & 3 \\
0 & 0 & 1 & -2
\end{bmatrix} $$

然后，反向代入即

$$\begin{bmatrix} 
1 & 2 & 1 & 2 \\
0 & 1 & -1 & 3 \\
0 & 0 & 1 & -2
\end{bmatrix}  \Longrightarrow \begin{bmatrix} 
1 & 0 & 0 & 2 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & -2
\end{bmatrix}   \Longrightarrow \left\{\begin{matrix}
x = 2 \\
y = 1 \\
z = -2
\end{matrix}\right. 
$$ 


如果只看系数部分，它就是一个典型的上三角矩阵，而从上三角矩阵得到单位矩阵的过程，就完成了对所有未知数的求解，是不是很简单？

**这就是高斯消元法的矩阵方法！**

当然，在这个过程中，你或许会注意到我们对矩阵的行列有做变换，这部分内容属于矩阵运算的基本知识，我不在这里展开做过多的探讨。
