@[toc]

本章节主要介绍这几个函数

Function | Description | Detail
-------------|----------------|-----------
cat    |  Concatenates the given sequence of seq tensors in the given dimension | 最常用的张量拼接函数
stack | Concatenates a sequence of tensors along a new dimension. | 按新的dim，拼接张量
dstack  | Stack tensors in sequence depthwise (along third axis). |  按深度方向拼接张量
hstack | Stack tensors in sequence horizontally (column wise). | 按水平方向拼接张量
vstack | Stack tensors in sequence vertically (row wise). | 按垂直方向拼接张量
row_stack | Alias of torch.vstack(). |  和 torch.vstack 功能相似
column_stack | Creates a new tensor by horizontally stacking the tensors in tensors. | 和 torch.hstack 相似


# torch.cat

张量拼接是非常常见的操作，以OpenCV为例，有时候我们需要把彩色图片（通常为3通道数据）分别进行处理，然后再重新组合在一起，生成新的图片。对于类似的框架来说，也提供了类似的函数。

现在，让我们来看看Torch的张量拼接函数的原型：

~~~python
    torch.cat(tensors, dim=0, *, out=None) -> Tensor
~~~

* tensors，通常是一组张量，要求大小维度相同，否则会导致拼接失败。
* dim，是拼接的方向，默认是0.


## 例程

### 低维度时的拼接

这个函数本质上并不难理解，但是唯一比较麻烦的就是dim，也就是轴方向，这是来自Numpy的概念，我觉得对于这个概念最好的理解，还是直接看源码最好。

```python
>>> import torch
>>> ones = torch.ones(4, 5)
>>> zeros = torch.zeros(4, 5)

>>> torch.cat((ones, zeros), dim=0)
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
        
>>> torch.cat((ones, zeros), dim=1)
tensor([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]])
        
>>> torch.cat((ones, zeros), dim=2)
Traceback (most recent call last):
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)        
```

可以看到，对于低维度的拼接时，torch.cat 仅支持把两个张量按列、行方向进行拼接。那么对于高维度的拼接，比如三维度时，又是怎样的呢？

### 高维度时的拼接

```python
>>> ones = torch.ones(2, 3, 4, 5)
>>> zeros = torch.zeros(2, 3, 4, 5)

>>> cat1 = torch.cat((ones, zeros), dim=0)
>>> cat1.shape
torch.Size([4, 3, 4, 5])

>>> cat2 = torch.cat((ones, zeros), dim=1)
>>> cat2.shape
torch.Size([2, 6, 4, 5])

>>> cat3 = torch.cat((ones, zeros), dim=2)
>>> cat3.shape
torch.Size([2, 3, 8, 5])

>>> cat4 = torch.cat((ones, zeros), dim=3)
>>> cat4.shape
torch.Size([2, 3, 4, 10])
```

从这个例子可以很明显的看出，cat操作的dim，是依据张量围度从左往右进行计算的。所以很多网上所说的dim=0是沿着X轴、dim=1是沿着Y轴，dim=2是沿着Z轴这种说法是十分不准确的。

**所以更准确的说法应该是：dim，指定操作沿着张量的第N位执行指令。** 对于上面这个例子来说，执行cat操作时，dim=0，即指定以张量维度第0位，执行拼接操作。

## 验证
为了证明结果，这里执行一个小程序片段，我们分别打印 cat1 和 cat4，看看同样的执行顺序会分别输出什么内容

```python
dim0, dim1, dim2, dim3 = cat1.shape
for i in range(dim0):
    print("i:", i, end=" ")
    for j in range(dim1):
        print("j:", j)
        for k in range(dim2):
            for l in range(dim3):
                print(cat1[i, j, k, l].item(), end=" ")
            print("")
        print("")

print("-----------------------------")

dim0, dim1, dim2, dim3 = cat4.shape
for i in range(dim0):
    print("i:", i, end=" ")
    for j in range(dim1):
        print("j:", j)
        for k in range(dim2):
            for l in range(dim3):
                print(cat4[i, j, k, l].item(), end=" ")
            print("")
        print("")

i: 0 j: 0
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 

j: 1
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 

j: 2
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 

i: 1 j: 0
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 

j: 1
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 

j: 2
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 
1.0 1.0 1.0 1.0 1.0 

i: 2 j: 0
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 

j: 1
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 

j: 2
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 

i: 3 j: 0
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 

j: 1
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 

j: 2
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 
0.0 0.0 0.0 0.0 0.0 

-----------------------------
i: 0 j: 0
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 

j: 1
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 

j: 2
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 

i: 1 j: 0
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 

j: 1
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 

j: 2
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 
```


#  torch.stack
```python
torch.stack(tensors, dim=0, *, out=None) → Tensor
```

这个函数是个类似cat的函数，而且这个函数官方也没有给出很明确的说明，所以我们还是来直接看看代码好了。

## 例程

```python
>>> a = torch.arange(0, 16).reshape(1, 2, 8)
>>> b = torch.arange(0, 16).reshape(1, 2, 8)
>>> c1 = torch.stack((a, b))
>>> c1.shape
torch.Size([2, 1, 2, 8])
```

可以看到stack，增加了一个维度。如果我们直接使用cat，会表现怎样呢？

```python
>>> a = torch.arange(0, 16).reshape(1, 2, 8)
>>> b = torch.arange(0, 16).reshape(1, 2, 8)
>>> c2 = torch.cat((a, b))
>>> c2.shape
torch.Size([2, 2, 8])
```

那么，怎么做可以得到一样的效果呢？

```python
>>> a = torch.arange(0, 16).reshape(1, 2, 8)
>>> b = torch.arange(0, 16).reshape(1, 2, 8)
>>> a1 = torch.unsqueeze(a, dim=0)
>>> b1 = torch.unsqueeze(b, dim=0)
>>> c2 = torch.cat((a1, b1))
>>> c2.shape
torch.Size([2, 1, 2, 8])
```
如果用 torch.equal 可以发现结果是正确的。

# dstack、hstack、vstack、row_stack、column_stack
单独把这几个放到一起介绍，主要是这些拼接函数主要是针对三维张量，尽管我们仍然可以用在高维度或者低于三维的张量里，但是当一个张量的维度表示为 $(Columns \times Rows \times Depth)$ 时，会更容易理解这几个函数是怎么使用的。

首先来看看各函数的原型：

```python
torch.dstack(tensors, *, out=None) → Tensor
```

```python
torch.hstack(tensors, *, out=None) → Tensor
```

```python
torch.vstack(tensors, *, out=None) → Tensor
```

```python
torch.row_stack(tensors, *, out=None) → Tensor
```

```python
torch.column_stack(tensors, *, out=None) → Tensor
```

然后我们看看例程

```python
import torch

COLUMNS = 5
ROWS = 10
DEPTH = 8

a1 = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)
b1 = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)

# 深度相加
c1 = torch.dstack((a1, b1))
print(c1.shape)  # torch.Size([10, 5, 16])

# 水平相加（行相加）
c2 = torch.hstack((a1, b1))
print(c2.shape)  # torch.Size([10, 10, 8])

# 垂直相加（列相加）
c3 = torch.vstack((a1, b1))
print(c3.shape)  # torch.Size([20, 5, 8])

# 列相加
c4 = torch.row_stack((a1, b1))
print(c4.shape)  # torch.Size([20, 5, 8])

# 行相加
c5 = torch.column_stack((a1, b1))
print(c5.shape)  # torch.Size([10, 10, 8])
```