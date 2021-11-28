@[toc]

Function | Description | Detail
-------------|----------------|-----------
narrow | Returns a new tensor that is a narrowed version of input tensor. | 返回缩略信息
conj     | Returns a view of input with a flipped conjugate bit. |  创建共轭张量
nonzero | | 返回张量中非0的引索



# torch.narrow

## 函数原型

```python
torch.narrow(input, dim, start, length) → Tensor
```

它的具体作用是，从输入的张量返回一个摘要性质的新张量。使用者可以指定维度，也可以指定摘要的开始和结束位置，有一点类似于张量分片的效果 **[a:b]**。不过通常我们喜欢直接使用分片，而比较少用到这个函数。

## 参数说明
* input (Tensor) – the tensor to narrow
* dim (int) – the dimension along which to narrow
* start (int) – the starting dimension
* length (int) – the distance to the ending dimension

具体例程如下：

## 例程

```python
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> torch.narrow(x, 0, 0, 2)
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])

>>> torch.narrow(x, 1, 1, 2)
tensor([[ 2,  3],
        [ 5,  6],
        [ 8,  9]])
```

# torch.conj

## 函数原型

```python
torch.conj(input) → Tensor
```

根据输入的张量返回一个共轭张量。这里所谓的共轭概念，主要是指复平面上，对于复轴的数的翻转。比方说：

$$3-4j$$

的共轭就是：

$$3+4j$$

其中，j是虚数。另外，根据官方说明，如果输入的数不是复数，那么这个函数只会返回输入的张量本身。

> **注意**
>
> torch.conj() 执行 lazy conjugation，但实际共轭张量可以随时使用 torch.resolve_conj() 实现。

> 不过说实话，我还没听说过 lazy conjugation，也不太清楚实际执行上会有什么区别，等到有一天用到这个函数后有什么新发现我会再更新这个部分的内容的。

> **警告**
>
> 将来，torch.conj() 可能会为非复杂数据类型的输入返回一个不可写的视图。 当输入为非复杂 dtype 时，建议程序不要修改 torch.conj_physical() 返回的张量以与此更改兼容。

## 例程

```python
>>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
>>> x.is_conj()
False

>>> y = torch.conj(x)
>>> y
tensor([-1.-1.j, -2.-2.j,  3.+3.j])

>>> y.is_conj()
True
```

# torch.nonzero

## 函数原型

```python
torch.nonzero(input, *, out=None, as_tuple=False) → LongTensor or tuple of LongTensors
```

> **注意**
> 
> torch.nonzero(..., as_tuple=False) （默认）它会返​​回一个二维张量，其中每一行都是非零值的索引。
> torch.nonzero(..., as_tuple=True) 返回一维索引张量的元组，允许高级索引，因此 x[x.nonzero(as_tuple=True)] 给出张量 x 的所有非零值。在返回的元组中，每个索引张量包含特定维度的非零索引。
> 此外，当输入在 CUDA 上时，torch.nonzero() 会导致主机设备同步。


## 例程

```python
>>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
tensor([[ 0],
        [ 1],
        [ 2],
        [ 4]])
>>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
...                             [0.0, 0.4, 0.0, 0.0],
...                             [0.0, 0.0, 1.2, 0.0],
...                             [0.0, 0.0, 0.0,-0.4]]))
tensor([[ 0,  0],
        [ 1,  1],
        [ 2,  2],
        [ 3,  3]])
        
>>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
(tensor([0, 1, 2, 4]),)

>>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
...                             [0.0, 0.4, 0.0, 0.0],
...                             [0.0, 0.0, 1.2, 0.0],
...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
(tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))

>>> torch.nonzero(torch.tensor(5), as_tuple=True)
(tensor([0]),)
```


