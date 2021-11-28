@[toc]

Function | Description | Detail
-------------|----------------|-----------
squeeze | Returns a tensor with all the dimensions of input of size 1 removed. | 
unsqueeze | Returns a new tensor with a dimension of size one inserted at the specified position. | 

#  torch.squeeze
维度压缩，这个函数会把张量中所有为1的维度全部删除，以此达到降维操作。如果输入的维度是 $(A \times 1 \times B \times C \times 1 \times D)$ 函数会输出维度为 $(A \times B \times C \times D)$。如果定义了维度dim的参数，那么函数只会处理对应的维度。

举例来说，如果维度为 $(A \times 1 \times B)$ ，

> squeeze(input, 0) 

输出的张量由于dim=0时，维度是A，所以不会发生改变，但是如果

> squeeze(input, 1) 

最终输出的维度就变成 $(A \times B)$

```python
torch.squeeze(input, dim=None, *, out=None) → Tensor
```

## 例程

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])

>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([2, 2, 2])

>>> y = torch.squeeze(x, 0)
>>> y.size()
torch.Size([2, 1, 2, 1, 2])

>>> y = torch.squeeze(x, 1)
>>> y.size()
torch.Size([2, 2, 1, 2])
```

#  torch.unsqueeze
与 torch.squeeze 正好相反，它允许用户在指定的位置扩张张量的维度。

其中，dim 的范围是 $[-input.dim() - 1, input.dim() + 1)$ 也就是允许用户以顺序、逆序的方式插入维度。

```python
torch.unsqueeze(input, dim) → Tensor
```

举例来说，如果 $dim=-1$，张量的维度会从 $(A \times B)$ 变成 $(A \times B \times 1)$；如果 $dim=0$，维度会从 $(A \times B)$ 变成 $(1 \times A \times B)$；如果 $dim=1$ ，张量会从 $(A \times B)$ 变成 $(A \times 1 \times B)$。


## 例程

```python
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])

>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```