
@[toc]

Function | Description | Abstract
-------------|----------------|-----------
index_select | Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor. | 引索选取
masked_select | Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor. | 掩码选取
take | Returns a new tensor with the elements of input at the given indices. | 以1维的方式按引索选取
take_along_dim | Selects values from input at the 1-dimensional indices from indices along the given dim. | 以某轴，按引索选取
gather | Gathers values along an axis specified by dim. |  -
where | Return a tensor of elements selected from either x or y, depending on condition. | -


# torch.index_select
函数原型

```python
torch.index_select(input, dim, index, *, out=None) → Tensor
```

从输入的张量里，按照某个维度方向，选取出数据并组成一个新的张量，其返回的数据长度，或深度信息，与原始的输入保持一致。

## 例程

```python
>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-0.4664,  0.2647, -0.1228, -1.1068],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
        
>>> indices = torch.tensor([0, 2])
>>> torch.index_select(x, 0, indices)
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
        
>>> torch.index_select(x, 1, indices)
tensor([[ 0.1427, -0.5414],
        [-0.4664, -0.1228],
        [-1.1734,  0.7230]])
```

# torch.masked_select
函数原型

```python
torch.masked_select(input, mask, *, out=None) → Tensor
```
这是另外一种数据选取的方式，我们可以让原输入的张量执行某种比较运算后得到一个MASK，然后通过这个函数可以迅速地选取合适的数据，并组成一个新的 1D 张量。

## 例程

```python
>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
        [-1.2035,  1.2252,  0.5002,  0.6248],
        [ 0.1307, -2.0608,  0.1244,  2.0139]])
        
>>> mask = x.ge(0.5)
>>> mask
tensor([[False, False, False, False],
        [False, True, True, True],
        [False, False, False, True]])
        
>>> torch.masked_select(x, mask)
tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
```

# torch.take
函数原型

```python
torch.take(input, index) → Tensor
```

它会把输入的张量先理解成一个同等元素量的1D张量，类似于：

> tensor.reshape(-1)

然后允许使用者使用一组下标，从原始输入中摘去需要的数据，并重新组成一个新的张量。

## 例程

```python
>>> src = torch.tensor([[4, 3, 5],
                        [6, 7, 8]])
>>> torch.take(src, torch.tensor([0, 2, 5]))
tensor([ 4,  5,  8])
```

# torch.take_along_dim
函数原型

```python
torch.take_along_dim(input, indices, dim, *, out=None) → Tensor
```
它会把输入的张量先理解成一个同等元素量的1D张量，类似于：

> tensor.reshape(-1)

然后允许用户使用 **torch.argmax()** 或 **torch.argsort()** 或自定义下标，然后沿着某个维度方向选取数据，得到的新数据会重新组合成一个新的张量，有点像 **torch.take()** 函数的升级版。

## 例程

```python
>>> t = torch.tensor([[10, 30, 20], [60, 40, 50]])
>>> max_idx = torch.argmax(t)
>>> torch.take_along_dim(t, max_idx)
tensor([60])

>>> sorted_idx = torch.argsort(t, dim=1)
>>> torch.take_along_dim(t, sorted_idx, dim=1)
tensor([[10, 20, 30],
        [40, 50, 60]])
```

# torch.gather
函数原型

```python
torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
```

沿着某个指定的轴方向，选取数据。对于3D张量来说，其选取数据的方式如下。

```python 
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```
它有一个要求，输入的张量和查询的张量应该有一样的维度，并且 **index.size(d) <= input.size(d)**，并且每一个维度的 **d != dim** 输出的张量维度和查询输入的维度一样。

另外，这个函数不是很常用。

## 例程
```python
>>> t = torch.tensor([[1, 2], [3, 4]])
>>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
tensor([[ 1,  1],
        [ 4,  3]])
```

# torch.where
函数原型

```python
torch.where(condition, x, y) → Tensor
```

从输入的张量的 x 和 y 选取数据，规则遵循下面的规则

$$
\text{out}_i = \begin{cases}
\text{x}_i & \text{if } \text{condition}_i \\ 
\text{y}_i & \text{otherwise}
\end{cases}
$$


## 例程

```python
>>> x = torch.randn(3, 2)
>>> y = torch.ones(3, 2)       
>>> torch.where(x > 0, x, y)
tensor([[ 1.0000,  0.3139],
        [ 0.3898,  1.0000],
        [ 0.0478,  1.0000]])
        
>>> x = torch.randn(2, 2, dtype=torch.double)
>>> x
tensor([[ 1.0779,  0.0383],
        [-0.8785, -1.1089]], dtype=torch.float64)
        
>>> torch.where(x > 0, x, 0.)
tensor([[1.0779, 0.0383],
        [0.0000, 0.0000]], dtype=torch.float64)
```