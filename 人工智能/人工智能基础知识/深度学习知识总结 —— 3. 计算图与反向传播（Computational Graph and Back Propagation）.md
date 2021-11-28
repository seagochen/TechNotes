
@[toc]

# 计算图与正向运算（Computational Graph）

计算图的概念其实很好理解，比方说对于如下的运算过程

1. 首先计算： $Z_0 = x+B$
2. 其次计算：$Z_1 = Z_0 \times A$
3. 最后计算：$y = Z_1 + C$

我们可以用计算图的形式表示这个过程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210621112920892.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
这样，从图上看，涉及到的运算过程一共是三层（每一步的运算过程都可以作为计算图的一层运算节点），涉及到的变量只有1个，而常量有3个。

所以，计算图的本质就是—— **「运算过程的结构图」** 。此外，在某些计算图中你或许会看到常量跟运算符绑定在一起，共同作为运算节点出现，就像下面这样：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210621120509233.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
不过，垂直的方式一般不太符合工程的习惯，我们通常用遵循 **「从左往右」** 的顺序表示运算过程。所以对于上面这个运算过程，你更多的会看到它是以下面这个形式表现的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210703081438590.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
接下来，我们可以把这个网络做复杂点，加入更多的运算，于是得到下面这图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210703085622819.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

例如对于上面这图来说，它的运算过程有两个：
* 其中黑色箭头表示的，是 **正向运算** 过程；
* 其中橙红色箭头部分，则是反向传播过程。

正向传播比较好理解，对于 $y = ax + b$ 执行 $ax + b$ 得到 $y$ 的过程就是 **正向运算** 过程。那么反向传播过程又是什么呢？

# 反向传播与导数（Back Propagation）
为了说明什么是反向传播，我们先假定存在这样一个线性模型

$$\hat y = x \times \omega$$

然后再给定一组数据：

x | $y$
---|---------
1 | 1
2 | 2
3 | 3
... | ...

我们现在比较想知道，如果 $x=9$ 时，$y$ 应该等于多少。假定我们不知道模型里权重 $\omega$ 的值，我们应该用什么方法求 $\omega$ 的值呢？

> 补充知识：
> 这里涉及到梯度下降算法，如果你不清楚梯度下降过程，那么请参考我前面的文章，对于你理解这部分内容会有帮助：
> [梯度下降算法——1. 什么是梯度下降](https://blog.csdn.net/poisonchry/article/details/116401539?spm=1001.2014.3001.5502)
> [梯度下降算法——2.梯度下降算法实现](https://blog.csdn.net/poisonchry/article/details/117419033?spm=1001.2014.3001.5501)

为了说明情况，现在我们用计算图来表示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/202107031809545.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

这里的 $Y*$ 因为绘图软件的关系，其实是 $\hat y$。我们得到了新的值后，需要先与观测值进行比对，于是产生了新的一层：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210703181650589.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
为了更好便于我们思考，我们把上述问题简化为 **「一个点」** 的情况，并且选用方差作为损失函数（评价函数），那么就有如下公式：

$$
Loss =  \frac{1}{n} \sum (\hat y - y) ^ 2
$$

如果假设此时得到的 $\hat y_x \neq  y_x$， 即每一项$\hat y$ 与 y之间都存在误差，自然MSE方程的和不可能为0，为了令 $Loss \rightarrow 0$，就会在这个时候激活反向传播过程。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210703182444284.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

由于令 $Loss \rightarrow 0$ 的这个过程，很像以前学过的关于极限的形式。换句话说，要想让MSE方程为0，就需要让MSE方程中涉及的权重尽可能的接近被观测的数据值 ，而推动这个计算的，最有效的数学工具就是导数，于是

$$
\frac{\partial L}{\partial \omega} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial \omega} 
$$

当然，这个链可以非常长，如果涉及到的参数足够多，它还会像波一样向每个参数传播出去，所以叫 **「反向传播」**。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210703085622819.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
那么对于一个简单的节点来说，它的参数值是怎么更新的呢？

# 从梯度下降开始聊起

## 使用链式法则更新参数

尽管我们已经在前面的章节里介绍了什么是 [梯度下降](https://seagochen.blog.csdn.net/article/details/116401539) 并且实现了一个简单的 [梯度下降程序](https://seagochen.blog.csdn.net/article/details/117419033) 。在这一节里，我们将进一步扩展内容，介绍梯度下降和反向传播是如何帮助我们快速锁定解的。

首先，我们引入最常见的线性方程，它也是神经元网络使用最频繁的基础函数。

$$
y = \omega x + b
$$

它被表述为权重 $\omega$ 和实验参数 $x$ 的乘积，和偏见 $b$ 之和。现在我们通过上述公式得到了推测值 $\hat y$，要与观测值 $y$ 进行比对，判断权重 $\omega$ 好坏的标准，落到了名为「损失函数」的测试函数，它可以是均方差函数，也可以是交叉墒，或者其他什么函数。

> 常见均方差函数
> $$
> Loss = \frac{1}{n} \sum (y - \hat y)^2
> $$

总之，这类函数的名字叫 $Loss(y, \hat{y})$ 就对了。由于 $y = \omega x$，所以 Loss 函数可以被改写为：

$$Loss(\omega) = \frac{1}{n} \sum (y - \omega x)$$

现在我们希望，新的权重 $\omega_{new}$ 比 之前的 $\omega_{old}$ 有更好的表现力，使得

$$
Loss(\omega_{new}) < Loss(\omega_{old})
$$

换句话说，我们希望能找到一个十分接近理想值 $\omega_o$ 的权重 $\omega_o + \Delta \omega$ 使的斜率接近0，于是有

$$
\lim_{\Delta \omega \rightarrow 0} \frac{Loss(\omega_o + \Delta \omega) - Loss(\omega_o)}{\Delta \omega} \rightarrow 0
$$

于是我们得到了关于Loss的导数

> **知识补充**
> 导数公式的定义为：
> $$
> f'(x_0) = \lim_{\Delta x \rightarrow 0} \frac{f(x_o + \Delta x) - f(x_o)}{\Delta x}
> $$

于是我们可以通过「链式法则」得到下面这串东西

$$
\frac{dL}{d\omega} = \frac{dL}{dy} \frac{d y}{d \omega} 
$$

然后我们带入公式，可以得到


$$
\frac{\partial L}{\partial \omega} = \sum \frac{2}{n} (\omega x - y) \cdot \frac{\partial}{\partial \omega} (\omega x - y) = \sum \frac{2x}{n} (\omega x - \omega^* x) 
$$

这里的 $\frac{2x^2}{n}$ 是一个常数，所以我们可以用 $\lambda$ 表示学习率，然后这里还有个很关键的问题，就是确定梯度下降的方向。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210703200601373.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
对于上面这个红色的点来说，它可以朝两个方向更新自己的权重。我们需要让它下降到图中最低点，所以，我们应该令

$$\hat \omega = \omega - \Delta \omega$$

于是得到最终的更新函数

$$
\hat \omega = \omega - \lambda \frac{\partial L}{\partial \omega}
$$

通过这个函数，我们终于可以让权重朝着设计的方向收敛，并找出最优解了。


![在这里插入图片描述](https://img-blog.csdnimg.cn/ba806bb1cf63424796ce7b66e661cc71.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 参数的更新流程

```mermaid
flowchat
st=>start: Start
e=>end: End
op1=>operation: 计算参数
op2=>operation: 更新参数
cond=>condition: 符合期望？

st->op1->cond
cond(yes)->e
cond(no)->op2
op2->op1
```

基本上来说，参数的更新过程就是上面这个流程。