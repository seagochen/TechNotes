
@[toc]

# 创建 Tensor 的一般方法

在很多时候，我们需要创建一些 tensor，PyTorch 为我们提供了丰富的 tensor 创建方案，在这篇文章里，让我们来看看都有哪些可以使用。


## 一般函数共用参数
这主要是指，针对 randn、empty、zeros、ones 等函数时，我们可以在创建 tensor 时，可以指定的其他参数。

**size**
定义输出张量的维度，输入形式为列表或者元组。

~~~python
import torch

tensor = torch.ones(size=(4, 4))
~~~

输出结果如下：

~~~
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
~~~

**out (Tensor, optional)**
输出张量指向的引用，一般不使用

**dtype (torch.dtype, optional)**
输出的张量数据类型，不指定的时候，使用默认数据类型。采用默认的函数创建的tensor，一般情况下都是 **float32** 类型，或者根据输入的原始数据类型，来决定tensor的类型。不同类型的tensor在计算上存在着精度丢失的问题，所以有时候需要你手工指定。

~~~python
import torch

tensor = torch.ones(size=(4, 4), dtype=torch.float32)
~~~

我们可以使用并定义的数据类型如下：

Data type |dtype | CPU tensor | GPU tensor
--------------|--------|-----------------|------------------
32-bit floating point | torch.float32 or torch.float	| torch.FloatTensor |  torch.cuda.FloatTensor
64-bit floating point | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor
16-bit floating point | torch.float16 or torch.half | torch.HalfTensor	| torch.cuda.HalfTensor
8-bit integer (unsigned) | torch.uint8 | torch.ByteTensor | torch.cuda.ByteTensor
8-bit integer (signed) | torch.int8 | torch.CharTensor | torch.cuda.CharTensor
16-bit integer (signed) | torch.int16 or torch.short |  torch.ShortTensor | torch.cuda.ShortTensor
32-bit integer (signed)	| torch.int32 or torch.int | torch.IntTensor | torch.cuda.IntTensor
64-bit integer (signed)	| torch.int64 or torch.long | torch.LongTensor | torch.cuda.LongTensor


**layout (torch.layout, optional)**
这个是比较涉及底层的操作，它指定某个tensor底层的数据逻辑组织方式。通常不需要单独指定，你如果需要指定layout，应该确保所有参与计算的每个tensor都遵循同样的数据布局。默认为 torch.strided.

```python
>>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>>> x.stride()
(5, 1)

>>> x.t().stride()
(1, 5)
```

**device (torch.device, optional)**
指定创建的tensor，它的数据存储的物理存储位置，默认是在CPU内存上划分区块，不过如果你的程序需要全部跑在GPU上，那么为了减少不必要的内存开销，在创建tensor的时候，一并指定设备内存会更好一些。默认值：”cpu“。

**requires_grad (bool, optional)**
求梯度的时候，是否需要保留梯度信息，默认为关闭。建议没事别动这个参数，如果你把这个参数设置为 **True**，那么就会在每一次运算的时候都会保留计算图，从而占用大量的内存。它的默认值：False. 那么是一直使用它的默认设置吗? 也不是, 通常在做网络迭代的时候, 为了验证参数的正确性, 会打开这个设置, 不过在你熟悉了这个框架后, 你再根据具体情况进行设置, 刚入门的时候就不用管了。

**pin_memory (bool, optional)**
它表示返回的tensor，会被存储在固定内存/锁页内存（the Pinned Memory）上。如果你有GPU的话，可以使用它来加速数据从CPU拷贝到GPU的过程。不过由于此块内存区域有限，使用时要谨慎一些。默认值：False。


**memory_format (torch.memory_format, optional)**
tensor的物理存储组织形式，默认数据是以连续形式创建，通常没必要碰这玩意。默认值：torch.contiguous_format.


## 创建空的张量 torch.empty(...)

有些时候我们需要创建空的张量，对于torch来说，可以使用 torch.empty 方法创建空白张量，函数原型如下：

~~~swift
 torch.empty(*size, *, 
 	out=None, 
 	dtype=None, 
 	layout=torch.strided, 
 	device=None, 
 	requires_grad=False, 
 	pin_memory=False, 
 	memory_format=torch.contiguous_format) -> Tensor
~~~

### 例程
~~~python
>>> a = torch.empty((2,3), dtype=torch.int32, device = 'cuda')
>>> torch.empty_like(a)
tensor([[0, 0, 0],
        [0, 0, 0]], device='cuda:0', dtype=torch.int32)
~~~

## 创建随机数值的张量 torch.rand(...)

尽管我个人认为包含随机变量的张量一般没什么用，不过用来做测试的时候，还是可以用的。其函数原型如下：

~~~swift
 torch.rand(*size, *, 
 	out=None, 
 	dtype=None, l
 	ayout=torch.strided, 
 	device=None, 
 	requires_grad=False) -> Tensor
 ~~~

### 例程

~~~python
>>> torch.rand(4)
tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
>>> torch.rand(2, 3)
tensor([[ 0.8237,  0.5781,  0.6879],
        [ 0.3816,  0.7249,  0.0998]])
~~~

## 创建全0的张量 torch.zeros(...)
用处要更大一些，torch.empty 虽然常用，但是由于内部数值不一定都会初始化为0，所以在使用时有时候不小心会带入一些”脏东西“进去，所以我们更多的会使用创建时同时初始化的tensor。

函数原型为：

~~~swift
torch.zeros(*size, *, 
	out=None, 
	dtype=None, 
	layout=torch.strided, 
	device=None, 
	requires_grad=False) -> Tensor
~~~

### 例程

~~~python
>>> torch.zeros(2, 3)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])

>>> torch.zeros(5)
tensor([ 0.,  0.,  0.,  0.,  0.])
~~~

## 创建全1的张量 torch.ones(...)

它的作用跟 zeros 相似，函数原型为:

~~~swift
torch.ones(*size, *, 
    out=None, 
    dtype=None, 
    layout=torch.strided, 
    device=None, 
    requires_grad=False) -> Tensor
~~~

### 例程

~~~python
>>> torch.ones(2, 3)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])

>>> torch.ones(5)
tensor([ 1.,  1.,  1.,  1.,  1.])
~~~

## 创建顺序的张量 torch.arrange(...)

它的作用是创建一个顺序增加的一维张量。其函数原型：

```python
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

新建张量每一位的计算公式如下：

$$
out_{i+1} = out_i + step
$$

### 例程

```python
>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])
>>> torch.arange(1, 4)
tensor([ 1,  2,  3])
>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```


# Numpy 与 Tensor

在以Python为代表的数学框架，基本上都是建立在numpy之上的。

这是一种比python list更快，且用C实现的矩阵计算框架。所以Python并不擅长的数值计算，由于Numpy的存在而大大加强。所以也可以认为是Numpy成就了Python在今天数据数值分析、统计领域一哥的位置。

自然，在Numpy的加持下，大部分的科学计算框架都集成或融合了Numpy，这也使得原本在其他语言上需要花费很多时间编码的内存对齐这件事，Python上就变得十分轻巧而方便。

所以，Pytorch也不例外，尽管底层是自有的数据结构，但是依然提供了对Numpy的支持。所以我们可以轻易的把从诸如Pandas, OpenCV,  OpenGL， scikit-learn 等框架的数据与Torch，通过Numpy进行对接。

## 从Numpy到Tensor

函数原型为：

~~~python
torch.from_numpy(ndarray) → Tensor
~~~

### 例程

~~~python
>>> nparray = numpy.array([1, 2, 3])
>>> tensor = torch.from_numpy(nparray, dtype=torch.float32)
>>> tensor
tensor([ 1,  2,  3])
>>> tensor[0] = -1
>>> nparray
array([-1,  2,  3])
~~~

### 使用Numpy的Copy命令创建备份

上面这个例子可以看见，torch只是把numpy的数据封装了一遍，如果希望 Torch 处理的数据与 Numpy 在物理上是隔离的，可以使用 **copy** 命令

~~~python
>>> import torch
>>> import numpy
>>> arr1 = numpy.array([1, 2, 3])
>>> arr2 = arr1.copy()
>>> tensor = torch.from_numpy(arr2)
>>> tensor[0] = -1
>>> tensor
tensor([-1,  2,  3])
>>> arr1
[1 2 3]
~~~

### 创建Numpy时指定数据类型

由于Numpy到Tensor，不能指定Tensor的类型，所以最好在创建Numpy的时候，指定数据的类型。

~~~python
>>> import torch
>>> import numpy
>>> arr1 = numpy.array([1, 2, 3], dtype=numpy.float32)
>>> tensor = torch.from_numpy(arr1)
>>> tensor
tensor([1., 2., 3.])
~~~

### 修改Numpy的数据类型

如果不能做到从Numpy创建之初就指定数据类型，那么可以用 **astype** 命令。

~~~python
>>> import torch
>>> import numpy
>>> arr1 = numpy.array([1, 2, 3])
>>> arr2 = arr1.astype(numpy.float32)
>>> tensor = torch.from_numpy(arr2)
tensor
tensor([1., 2., 3.])
~~~

## 从Tensor到Numpy

数据处理完毕后，我们有时候会想把数据导回 Numpy，用于其他的后续处理，方法很简单，只要这样就可以了：

~~~python
t = torch.ones(5)
n = t.numpy()
~~~

# 从Python列表、元组赋值

除了以上方法，我们还可以通过Python自己的基本数据类型创建Tensor：

~~~python
torch.tensor(data, *, 
	dtype=None, 
	device=None, 
	requires_grad=False, 
	pin_memory=False) -> Tensor
~~~

## 例程

~~~python
>>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor([[ 0.1000,  1.2000],
        [ 2.2000,  3.1000],
        [ 4.9000,  5.2000]])

>>> torch.tensor([0, 1])  # Type inference on data
tensor([ 0,  1])

>>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
...              dtype=torch.float64,
...              device=torch.device('cuda:0'))  # creates a torch.cuda.DoubleTensor
tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

>>> torch.tensor(3.14159)  # Create a scalar (zero-dimensional tensor)
tensor(3.1416)

>>> torch.tensor([])  # Create an empty tensor (of size (0,))
tensor([])
~~~

## 从Tensor到List

我们有时候也会想这样一个问题，我能不能把Tensor转换回 Python.List

```python
data = tensor.tolist()
```

## 从Tensor到Scalar

与上面这个方法不太一样，这是把Tensor中的某个具体元素转换为Python整数或浮点数

```python
num = tensor[0].item()
```

但是一定要记住，这个函数只能对某个具体的值做转换，如果试图对一组Tensor，比如列表、矩阵做转化，那会导致最终的报错。


# 创建和其他Tensor维度一样的Tensor

如果我们想创建一个维度和其他Tensor一致的Tensor，主要用到这样这么几个函数。首先，假设我们有了一个Tensor

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

然后分别创建全零、全一、随机Tensor

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

x_zeros = torch.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")

x_empty = torch.empty_like(x_data)
print(f"Empty Tensor: \n {x_empty} \n")
```
