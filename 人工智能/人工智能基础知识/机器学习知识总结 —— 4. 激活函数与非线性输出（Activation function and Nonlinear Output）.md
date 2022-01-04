> 在前面的章节里，分别介绍了声名远扬的梯度下降算法，计算图，以及反向传播。现在我们要思考这样一个问题。对于$y = ax +b$这样一个线性模型来说，无论x有多复杂，都无法改变它线性输出的特点。
> 线性模型对于线性问题的求解是可行，但是我们需要用倒神经元网络的实际应用场景来说，有很多是无法直接套用线性模型来求解的。举个例子来说，对于分类问题，就无法很好的依靠线性模型进行求解。
> 所以，为了解决这个问题，就有科学家提出在每一层神经元的输出上，根据需要套一个激活函数，从而破坏神经元的线性结构，使它有求解非线形问题的能力。 


@[toc]

# 为什么需要非线性

我们首先要搞明白什么是线性输出？对于公式

$$
y = ax + b
$$

它是线性函数。它意味着所有的输入，经过相同的权重后，都能得到同等的输出。这也就是为什么图像上它表现为一条直线。那么自然，非线形的概念，就是所有的输入，在不同位置上有不同程度的输出。

也就是说，非线性用的好，我们能降低不重要的数据影响权重，而节省精力关注重点信息。如果用照片为例，就是突出主体，虚化背景了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/985c50e0a73e41a887a3b9580ce1117c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_13,color_FFFFFF,t_70,g_se,x_16#pic_center)
所以，非线性在所有的算法中都有不同程度的体现，我们的目的就是让我们的资源能够 **Focus** 在重要的事情上，减低对不重要数据的敏感性。

# 关于激活函数
对于深度学习来说，我们可以在线性模型的输出上增加名为「激活函数」的工具，使得线性模型具备非线性特征。以流程图为例，如果说原来的线性计算模型是这样的：

```mermaid
flowchat
st=>start: 随机输入权重
e=>end: 输出权重
op1=>operation: 向前传播并更新参数
op2=>operation: 向后传播并更新权重
cond=>condition: 损失函数->符合期望？

st->op1->cond
cond(yes)->e
cond(no)->op2
op2->op1
```

那么增加了激活函数的模型就变成了这样：

```mermaid
flowchat
st=>start: 随机输入权重
e=>end: 输出权重
op1=>operation: 向前传播并更新参数
op3=>operation: 通过激活函数映射到新函数空间
op2=>operation: 向后传播并更新权重
cond=>condition: 损失函数->符合期望？

st->op1->op3->cond
cond(yes)->e
cond(no)->op2
op2->op1
```

那么我们常用的激活函数有哪些呢？

# 常见的几个激活函数

## Sigmoid 函数

严格来说，Sigmoid函数不止一个，任何符合$\sigma(x) = \frac{1}{1+ e^{-x}}$函数图像的，都可以被归类为Sigmoid函数，这是一种经常被用来做 0-1 分类问题的函数。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210706223953383.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

其函数式写为：

$$
\sigma(x) = \frac{1}{1+ e^{-x}}
$$


除它之外，还有比较相似的tanh函数，其输出图像是：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210706224305397.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
这一类的都是Sigmoid函数家族，但是一般提到Sigmoid，一般指的就是老大哥$\sigma(x) = \frac{1}{1+ e^{-x}}$了。

这类函数的优点：

在于它可导，并且值域在有限范围区间，如 [0, 1]，可以使神经元的输出标准化。

缺点：
* 容易导致梯度消失。
* 不以零为中心：函数的输出恒为正值，不是以零为中心的，这会导致权值更新时只能朝一个方向更新，从而影响收敛速度。
* 计算成本高昂：exp() 函数与其他非线性激活函数相比，计算成本高昂。

## tanh 函数

属于Sigmoid函数的一种，但是和老大哥所不同，它的取值范围是 [-1, 1]
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210706224305397.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
其函数式为：

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

缺点和Sigmoid函数一样，但是优点是解决了Sigmoid只能在[0, 1]之间的问题，其取值范围是[-1 , 1]

## ReLu 函数
有些地方叫它整流函数，它可以对神经元输出进行有针对性的数据过滤作用。比方说，数据经过一定的处理后，输出的值包含符号为正的有效数据和符号为负的噪音数据，为了避免噪音数据输入到下一层的网络，所以可以使用这个函数，过滤需要的数据。

从函数图像上看，它是分段函数，基本的ReLu函数的函数式表达为：

$$
ReLu(x) =\left\{\begin{matrix} 
0 & x < 0\\
x & x  \geq 0\\
\end{matrix}\right.
$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210706225618367.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
常见的是最左侧的。但是有时候总有人想适当放一些另外一侧的数据，所以提出了其他的改进形式。

优点：
* 收敛速度快
* 能够滤除不需要的噪音
* 降低运算成本

缺点：
* 单向性
* 如果在负轴存在有效数据，可能导致神经元没有有效反馈

# 来做个简单的应用吧
假设我们有这么一张表，这是一张我比较喜欢的辉光管工作时长表，这种前苏联时代的老玩意由于适用寿命都不长，所以如果持续点亮，那么在一定时候就会坏掉。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210706233122877.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)


Duration(Year.) | Work
---------|---------
0.1 | True
0.3 | True
0.5 | True
0.7 | True
0.9 | True
1.1 | True
1.3 | True
1.5 | False
1.7 | False
1.9 | False

也就是说，当时间到1.5年后，辉光管一定会坏掉，那么问题是1.4年的时候，它坏掉的机率有多大？

我们用Torch来执行一下，首先使用一个线性模型对数据进行预测，并且为了增加一点玄学的成分，用Sigmoid作为激活函数，BCE函数作为损失函数，然后看看预测情况如何


```python
import torch

x_data = torch.Tensor([[0.1], [0.3], [0.5], [0.7], [0.9], [1.1], [1.3], [1.5], [1.7], [1.9]])  # 1 column x 10 rows
y_data = torch.Tensor([[1], [1], [1], [1], [1], [1], [1], [0], [0], [0]])  # 0 - false, 1 - true


class SimpleLogisticModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        """
        Regression analysis of the data was
        continued using the linear model
        """

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


if __name__ == "__main__":
    # fitting modle
    model = SimpleLogisticModel()

    # LOSS function
    criterion = torch.nn.BCELoss(size_average=False)

    # parameters optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # stochastic gradient descent

    # training and do gradient descent calculation
    for epoch in range(500):
        # forward
        # y_predict_value = model(x_data)
        y_predicted_value = model.forward(x_data)

        # use MSE to check the deviation
        loss = criterion(y_predicted_value, y_data)

        # print for debug
        print(epoch, loss.item())

        # set calculated gradient to zero
        optimizer.zero_grad()

        # call backward to update the parameters
        loss.backward()

        # optimize parameters
        optimizer.step()

    # finally
    print("omega = ", model.linear.weight.item())
    print("bias = ", model.linear.bias.item())

    # test values
    x_test = torch.Tensor([1.4])
    y_test = model(x_test)

    # print out result
    print("final y = ", y_test.data)
```

输出结果是：

```
omega =  -7.137760162353516
bias =  9.893336296081543
final y =  tensor([0.4751])
```

如果四舍五入，那么y = 0，也就是说1.4年后，坏掉的可能性是53%，其实概率还是挺大的呢～


# 后记
如果你刚接触Ai框架，你可能看到这里有点懵，这些函数都是干嘛的，有什么用。其实我计划把如何使用这些Ai框架的内容放在后面，这部分属于理论范畴，我们弄明白神经元是怎么工作的先。

另外，关于激活函数就我介绍的这么些吗？其实目前公布的有很多，这里只是提到了一些对于初学者来说常用和简单的类型。在[某乎](https://zhuanlan.zhihu.com/p/260970955)上有这么一个帖子目前总结了目前常用的激活函数总类，你可以去那帖子上面看看。