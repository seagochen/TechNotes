@[toc]

Function | Description | Detail
-------------|----------------|-----------
reshape | Returns a tensor with the same data and number of elements as input, but with the specified shape. |


# torch.reshape

维度改变是很常见的一种操作，合理修改张量维度能帮我们省很多事，但是需要注意一点的是，维度修改，并不改变底层数据的顺序和存储结构，它只是修改了引索范围而已。所以如果你希望两个维度之间的数据进行顺序交换，就不能使用reshape命令，而应该使用transpose，或者自己写个函数，实现类似的功能。

```python
torch.reshape(input, shape) → Tensor
```

## 例程

```python
>>> a = torch.arange(4.)
>>> torch.reshape(a, (2, 2))
tensor([[ 0.,  1.],
        [ 2.,  3.]])

>>> b = torch.tensor([[0, 1], [2, 3]])
>>> torch.reshape(b, (-1,))
tensor([ 0,  1,  2,  3])
```