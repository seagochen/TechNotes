@[toc]

Function | Description | Detail
-------------|----------------|-----------
transpose | Returns a tensor that is a transposed version of input. | 多维张量转置
t | Expects input to be <= 2-D tensor and transposes dimensions 0 and 1. | 1、2维张量转置
swapaxes | Alias for torch.transpose(). | 和 transpose 功能一致
swapdims | Alias for torch.transpose(). | 和 transpose 功能一致

# torch.transpose

我们有时候会遇到这样的问题，某个时刻我们希望把某个张量从横向量，转置成纵向量。因为对于线性代数来说，矩阵的基本计算法则告诉我们，两个矩阵叉乘必须满足 $[l \times m] \times [m \times n]$ 才可以进行计算。但有时候我们为了方便会创建出两个这样维度的矩阵 $[m \times l]$ 和 $[m \times n]$，为了执行叉乘，必须要对第一个矩阵进行转置。

如果直接用 reshape 或者 view 命令，虽然改变了维度，但是没有改变底层的数据顺序，一样得不到正确的结果。

```python
>>> tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).view(2, -1)
>>> tensor
tensor([[1, 2, 3, 4, 5],
        [6, 7, 8, 9, 0]])
>>> tensor.reshape(-1, 2)        
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 0]])
```

对于上面这个例子，我们实际上希望输出的是

```python
tensor([[1, 6],
		[2, 7],
		[3, 8],
		[4, 9],
		[5, 0]])
```


所以这个时候就需要转置函数，其函数原型为

~~~swift
    torch.transpose(input, dim0, dim1) -> Tensor
~~~

这个函数表示张量的某两维度之间进行转置，对于上面这个例子来说即：

```python
>>> tensor = torch.transpose(tensor, dim0=0, dim1=1)
tensor([[1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
        [5, 0]])
```

## 对于高维度张量
高维度张量也存在需要转置的情况，由于transpose不指定具体哪两个维转置，所以相对来说对使用者要方便很多。

```python
>>> tensor = torch.randn(1, 2, 3)
>>> tensor  # shape of tensor is [1, 2, 3]
tensor([[[ 0.1264, -0.7503,  0.5522],
         [ 0.0680,  1.0128,  0.1585]]])

>>> tensor = torch.transpose(tensor, dim0=1, dim1=2)
>>> tensor  # shape of tensor is [1, 3, 2]
tensor([[[ 0.1264,  0.0680],
         [-0.7503,  1.0128],
         [ 0.5522,  0.1585]]])

>>> tensor = torch.transpose(tensor, dim0=0, dim1=2)
>>> tensor  # shape of tensor is [2, 3, 1]
tensor([[[ 0.1264],
         [-0.7503],
         [ 0.5522]],

        [[ 0.0680],
         [ 1.0128],
         [ 0.1585]]])
```

# torch.t

根据描述，这个函数仅能用于小于或等于二维的张量转置

```python
torch.t(input) → Tensor
```

## 例程

```python
>>> x = torch.randn(3)
>>> x
tensor([ 2.4320, -0.4608,  0.7702])
>>> torch.t(x)
tensor([ 2.4320, -0.4608,  0.7702])
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.4875,  0.9158, -0.5872],
        [ 0.3938, -0.6929,  0.6932]])
>>> torch.t(x)
tensor([[ 0.4875,  0.3938],
        [ 0.9158, -0.6929],
        [-0.5872,  0.6932]])
```