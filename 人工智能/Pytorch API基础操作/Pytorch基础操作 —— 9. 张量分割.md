@[toc]

Function | Description | Detail
-------------|----------------|-----------
chunk  | Attempts to split a tensor into the specified number of chunks. | 按指定数量分割张量
tensor_split | Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections. | 按指定引索分割张量
split | Splits the tensor into chunks. | 分割张量
unbind | Removes a tensor dimension. |  对张量进行解耦操作
dsplit   | Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. | 按深度方向分割张量
hsplit   | Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections. | 按水平方向分割张量
vsplit | Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections. | 按垂直方向分割张量

# 需要注意的地方

## 返回的都是 list of view

在本章中提到的函数，返回的结果都是原张量的view，换句话说如果修改了结果的值，会导致原本的张量被修改。所以如果要想子张量和原张量数据隔离，可以使用 

> tensor.clone().detach()

或

> torch.clone(tensor)

的方式，创健一个备份。

## 数据的连续性

另外就是关于数据切分的方式，是按照顺序的形式进行拆分，比如说有一个一维的数据，

> [[1, 2, 3, 4, 5, 6], [7,  8, 9, 10, 11, 12], ...]

现在想把它们拆分成3组，那么这个时候的顺序就是这样的了：

第一次拆分时，先按照某个维度区分出3组（例如维度dim=1）
>[1, 2], [3, 4], [5, 6]

然后发现还有数据，于是：
> [[1, 2], [7, 8], ...], 
> [[3, 4], [9, 10], ...],
> [[5, 6], [11, 12], ...]

直到全部数据都切分完毕，这就是最终的数据形式。

# torch.chunk

如果说数据可以通过cat粘合，那么chunk就可以把tensor按维度方向进行分割。我们来看看这个函数原型：

## 函数原型

```python
torch.chunk(input, chunks, dim=0) → List of Tensors
```

## 例程

为了更好说明这个函数是怎么使用的，不如直接看看它的执行结果如何。

```python
>>> tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)
>>> tensor
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
        
>>> list_of_tensors = torch.chunk(tensor, 3, dim=0)
>>> list_of_tensors
(tensor([[1, 2, 3]]), tensor([[4, 5, 6]]), tensor([[7, 8, 9]]))

>>> list_of_tensors = torch.chunk(tensor, 3, dim=1)
>>> list_of_tensors
(tensor([[1],
        [4],
        [7]]), tensor([[2],
        [5],
        [8]]), tensor([[3],
        [6],
        [9]]))
```

从这个例子来看，chunk命令，实际上是按照维度进行切分，但不是我们通常理解的按x轴、y轴或z轴这种定义。这在所有Numpy为基础的框架，都是一样的道理。为了更好理解这个问题，我们再看一个例子就好了。

```python
COLUMNS = 5
ROWS = 10
DEPTH = 8

a = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)

ts = torch.chunk(a, 10, dim=0)
print(ts[0].shape)  # torch.Size([1, 5, 8])

ts = torch.chunk(a, 5, dim=0)
print(ts[0].shape)  # torch.Size([2, 5, 8])

ts = torch.chunk(a, 2, dim=2)
print(ts[0].shape)  # torch.Size([10, 5, 4])
```

chunks与所对应的dim应该能整除，如果不能整除，那么就返回一个自认为合适的划分，保证这组除最后一个，其他块的维度都是一样。为了更好说明这个情况，看下面这个例子：

```python
COLUMNS = 5
ROWS = 10
DEPTH = 8

a = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)

ts = torch.chunk(a, 4, dim=1) # 5 / 4 无法整除，所以会返回函数认为最合适的划分
print(len(ts))  # 只返回了3个划分

for t in ts:
    print(t.shape) 
    # torch.Size([10, 2, 8])
	# torch.Size([10, 2, 8])
	# torch.Size([10, 1, 8])
```


# torch.tensor_split

## 函数原型

```python
torch.tensor_split(input, indices_or_sections, dim=0) → List of Tensors
```
它的作用和 chunk 很相似，我们来看看具体的代码吧。

## 例程

```python
COLUMNS = 5
ROWS = 10
DEPTH = 8

a = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)

ts = torch.tensor_split(a, 4, dim=2)
print(len(ts)) # 8 / 2 = 4

for t in ts:
    print(t.shape)
    # torch.Size([10, 5, 2])
	# torch.Size([10, 5, 2])
	# torch.Size([10, 5, 2])
	# torch.Size([10, 5, 2])
```

和chunk最大的区别，在于如果某维度无法整除时，它会忠实的按照给定的维度进行划分，余数部分会被平分加入到列表的前几位。

```python
COLUMNS = 5
ROWS = 10
DEPTH = 8

a = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)

ts = torch.tensor_split(a, 4, dim=1) # 5 / 4 无法整除，所以会返回函数认为最合适的划分
print(len(ts))  # 只返回了4个划分

for t in ts:
    print(t.shape) 
	# torch.Size([10, 2, 8])
	# torch.Size([10, 1, 8])
	# torch.Size([10, 1, 8])
	# torch.Size([10, 1, 8])
```

#  torch.split

## 函数原型

```python
torch.split(tensor, split_size_or_sections, dim=0)
```

功能总体和上面提到的都很相似，只不过最大的区别在于它不是划分有多少块，而是指定每个view包含多少条数据，为了更好说明，我们依然来直接看看代码好了。

## 例程

```python
COLUMNS = 5
ROWS = 10
DEPTH = 8

a = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)

ts = torch.split(a, 5, dim=0)
print(len(ts)) # 10条数据，按每个块包含5条，一共划分成了2块

for t in ts:
    print(t.shape)
	# torch.Size([5, 5, 8])
	# torch.Size([5, 5, 8])
```

那么如果维度大小不能整除时怎么办？

```python
a = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)

ts = torch.split(a, 6, dim=0)
print(len(ts))  # 10条数据，按每个块包含6条，一共划分成了2块

for t in ts:
    print(t.shape)
	# torch.Size([6, 5, 8])
	# torch.Size([4, 5, 8])
```

可以看到，它和整除很像，会优先保证前面的块有足够的数据，最后的块往往不足这个数。

# torch.unbind

## 函数原型

```python
torch.unbind(input, dim=0) → seq
```

它的作用有一点像 **torch.split** 但是又有所不同的是，torch.split 会把所有的数据拆分成等分的全部压成一维的。而这个函数，并不会做那么具体，而是你指定某个维度，它直接就会把张量拆成一组张量。

## 例程

```python
>>> torch.unbind(torch.tensor([[1, 2, 3],
>>>                            [4, 5, 6],
>>>                            [7, 8, 9]]))
(tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
```


# dsplit、vsplit、hsplit
这三个函数功能很相似，不过主要是针对三维数据进行划分的，其各自的函数原型如下：

## 函数原型

```python
torch.dsplit(input, indices_or_sections) → List of Tensors
```

```python
torch.vsplit(input, indices_or_sections) → List of Tensors
```

```python
torch.hsplit(input, indices_or_sections) → List of Tensors
```

现在我们来看看例子吧：

## 例程

```python
COLUMNS = 12
ROWS = 10
DEPTH = 8

a = torch.arange(0, COLUMNS * ROWS * DEPTH).reshape(ROWS, COLUMNS, DEPTH)

# 垂直方向划分，vsplit
vs = torch.vsplit(a, 2)
for s in vs:
    print(s.shape)
    # torch.Size([5, 12, 8])
	# torch.Size([5, 12, 8])

# 水平方向划分，hsplit
hs = torch.hsplit(a, 2)
for s in hs:
    print(s.shape)
   	# torch.Size([10, 6, 8])
	# torch.Size([10, 6, 8])

# 深度方向划分，dsplit
ds = torch.dsplit(a, 2)
for s in ds:
    print(s.shape)
    # torch.Size([10, 12, 4])
	# torch.Size([10, 12, 4])
```