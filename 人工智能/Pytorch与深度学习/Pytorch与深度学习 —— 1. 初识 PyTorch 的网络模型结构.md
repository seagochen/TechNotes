@[toc]

在没有任何基础的前提下，直接学习如何搭建神经网络，意义其实不大。我建议你如果因为读研或者好奇而开始学神经元网络，建议你先看看我前面写的基础知识内容后，再回来学习内容。

[深度学习知识总结—— 1.1. 什么是梯度下降](https://seagochen.blog.csdn.net/article/details/116401539)
[深度学习知识总结—— 1.2.梯度下降算法实现](https://seagochen.blog.csdn.net/article/details/117419033)
[深度学习知识总结—— 2. 计算图与反向传播](https://seagochen.blog.csdn.net/article/details/118082114)
[深度学习知识总结—— 3. 激活函数与非线性输出](https://seagochen.blog.csdn.net/article/details/118526467)
[深度学习知识总结—— 4. 神经元网络与矩阵运算](https://seagochen.blog.csdn.net/article/details/118598093)

当然，理解以上内容需要一定的线性代数方面的知识。不过既然你都想掌握AI技术了，这点门槛应该不是什么大问题。

# 构建神经元网络模型的基本范型

所谓范式，就是说用代码要怎么构建神经元网络模型的基本套路。对于Pytorch来说，基本分为以下四步：

* 构建网络模型
* 选择合适的优化函数和损失函数
* 构建训练迭代过程
* 对计算结果进行准确度验证

所以，我们用代码的形式，表示为

~~~python
# 第一步，声明网络结构、损失函数（评价函数）、优化函数
nn_layout1 = torch.nn.Linear(l, m) # 声明一个 (l x m) 的网络层
nn_layout2 = torch.nn.Linear(m, n) # 声明一个 (m x n) 的网络层

criterion = torch.nn.MSELoss() # 声明一个均方差损失函数

optimizer = torch.optim.SGD(...) # 声明一个SGD优化函数

# 第二步，构建网络组织结构
m = nn_layout1(l)
n = nn_layout2(m)

# 第三步，构建训练迭代过程
for epoch in range(...):

	# 获取向前传播得到的新参数
	y_predict = model.forward(x_train)

	# 获取新参数与目标值之间的误差水平
	loss = criterion(y_predict, y_train)

	# 反向传播
	loss.backward()

	# 调用优化器更新参数
	optimizer.step()

	# 清空优化器的梯度
	optimizer.zero_grad()

# 第四步，验证数据
y_predict = model(x_test)
print(y_predict， y_test)
~~~


注意这里的结构，这就是我提到过的这么一种逻辑过程：

```mermaid
flowchat
st=>start: 输入权重
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

我们虽然是按照这4个步骤来编写 torch 网络，但不意味着实际代码就是这样的，接下来以全连接神经元网络为例，了解一般范型是怎样的。


# 构建网络模型

构建网络模型，说到底就是确定一个神经元网络是以什么结构来处理任务。以 torch 为代表的AI框架，封装了大量底层的操作，和基础模型，所以这让我们能用非常简单的方法构建出所需的训练网络。

和前面提到的稍微有点不太一样的地方，是 torch 在这方面提供了已经是实现好的框架，我们使用者只需要按要求填东西就好了。

那么构建网络模型，如果不想自己一步步实现，那么就一定要自己继承 **torch.nn.Module** 这个基类。除了声明网络模型中用到的层和定义层结构外，用户自己还需要实现一个 forward 向前计算的函数。

比如说这样：

~~~python
import torch

class DefinedModelNet(torch.nn.Module):

	def __init__(self):
		super().__init__()

		# 定义我们需要使用的网络层
		self.linear = torch.nn.Linear(n, m) # 输入维数为 n，输出为 m；这个地方参考矩阵乘法运算的基本规则
		...
	
	def forward(self, input):
		"""
		网络前向运算，这里也是构建计算图的过程
		"""
		y = self.linear(x)
		return y
~~~

除了例子中用到的 **torch.nn.Linear** 线性层模型外， **torch.nn** 也提供了其他常用的模型，例如卷积层、循环层等。

模型创建完毕后，就需要对模型进行实例化：

~~~python
	model = DefinedModelNet()
~~~


# 选择优化和损失函数

损失函数，以均方差函数 (MSE) 为例。

$$Loss = \frac{1}{N} \sum (\hat{y} - y)^2$$

它所做的主要工作，就是把观测值和预测值之间做一个比较，并且计算出误差。在机器学习领域，类似功能的又被称为评价函数。

在 **torch.nn** 包里，提供了很多有用的，不同类型的损失函数。定义损失函数的方法，以MSE为例：

~~~python
    # LOSS function
    criterion = torch.nn.BCELoss(size_average=False)
~~~

而优化函数，本质上是对梯度下降求导的优化、学习率等的定义（以MSE为例）。

$$
\omega *  = \omega -  \Delta \frac{1}{N} \sum_{i=1}^{n} 2(\omega -y_i)
$$

除了torch提供的几种优化函数外，你也可以通过 **torch.optim.Optimizer** 自定义自己的优化算法，如果仅调用 torch 提供的优化函数，那么调用方式一般为：

~~~python
    # parameters optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # stochastic gradient descent
~~~

# 构建迭代过程

迭代的过程，说到底就是缩小误差的过程，而这个过程也称训练过程。这里，我们连同上面提到的优化函数和损失函数一起封装到一个函数里，命名为train，于是有了

~~~python
def train(...):
    # LOSS function
    criterion = torch.nn.BCELoss(size_average=False)
    
    # parameters optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # stochastic gradient descent

	for epoch in range(training_cycles):
		# zero the parameter gradients
		optimizer.zero_grad()
		
		# forward + backward + optimize
		# y_predict = model().forward(x_train) # Call function explicitly
		y_predict = model(x_train) # forward computation
		loss = criterion(y_predict, y_train) # loss function computation
		loss.backward() # backward computation
		optimizer.step() # update parameters

		# foo
~~~

如果需要在训练过程中打印或者绘制训练结果，可以在foo后面加入执行语句。另外，就是torch在每一轮计算的时候，都需要做一次梯度清零，建议是放在 forward+loss+backward 前执行，避免第一次启动的内存没有规制为0导致的错误。

# 结果验证

当训练结束后，model 会得到一个与训练误差极少的模型权重，那么是否符合我们实际任务需要，就需要对模型结果进行验证。通常情况下，对于某个数据集，我们一般采用随机选取数据集中80%的数据作为训练集，20%的数据作为验证用的结果。

验证的方法可以有人工和自动评估两种。自动评估的概念就是我对所有的数据样本都建立对应的标签，然后执行模型后看输出的参数和标签重合情况（这一般多用于分类、拟合问题）。另一种就是直接看输出的参数，或者把训练好的模型扔到线上和实际运行情况进行对比。

这里我先不展开介绍，我们先来看看对于结果验证，我们可以这样做一个函数

~~~python
def test(...):
	for x_test, y_test in samples.items():
		y_predict = model(x_test)
		ret = torch.sum((y_test - y_predict)**2)
		print("final MSE is", ret)
~~~

至此，使用 torch 构建一个最简单的网络模型的主要工作已经完成，剩下的就是如何准备数据之类的任务了，在下一章节里，我们来尝试构建一个最简单的全连接网络，看看效果如何吧。


# 关于文档

关于 PyTorch 最权威的解释和说明，自然是参考官方的说明是最好的。

PyTorch 官方说明文档： https://pytorch.org/docs/stable/index.html
PyTorch 官方教学手册：https://pytorch.org/tutorials/

除此之外，还有中文圈的大佬对官方文档的翻译，如果觉得英文的比较难懂，可以看看

PyTorch 中文文档：https://pytorch-cn.readthedocs.io/zh/latest/

PyTorch 最主要的包，包含有已经实现的各类层类型外，还有比如损失函数、激活函数等，基本上都在 torch.nn 里， 关于这部分详细内容，可以查看文档：https://pytorch.org/docs/stable/nn.html