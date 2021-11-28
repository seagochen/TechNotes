Function        | Description   | Detail
----------------|---------------|-----------
max             | Returns the maximum value of all elements in the input tensor. | 返回张量中的最大值
min             | Returns the minimum value of all elements in the input tensor.   | 返回张量中的最小值
maximum         | Computes the element-wise maximum of input and other. | 多个张量间执行元素级的最大值比较（建议使用fmax）
minimum         | Computes the element-wise minimum of input and other.   | 多个张量间执行元素级的最小值比较（建议使用fmin）
fmax            | Computes the element-wise maximum of input and other. | 和 maximum 相似，但支持更多数据形式输入
fmin            | Computes the element-wise minimum of input and other. | 和 minimum 相似，但支持更多数据形式输入
argmax          | Returns the indices of the maximum value of all elements in the input tensor. | 返回张量中最大值的引索
argmin          | Returns the indices of the minimum value(s) of the flattened tensor or along a dimension | 返回张量中最小值的引索
amax            | Returns the maximum value of each slice of the input tensor in the given dimension(s) dim. |  找出轴方向上的最大值
amin            | Returns the minimum value of each slice of the input tensor in the given dimension(s) dim. | 找出轴方向上的最小值
aminmax         | Computes the minimum and maximum values of the input tensor. | 同时找出最大和最小值


@[toc]

# torch.max
## 原型

```python
torch.max(input) → Tensor
```

## 说明
返回张量中的最大值

## 例子
```python
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.6763,  0.7445, -2.2369]])
>>> torch.max(a)
tensor(0.7445)
```
---

# torch.min
## 原型
```python
torch.min(input) → Tensor
```

## 说明
返回张量中的最小值

## 例子
```python
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.6750,  1.0857,  1.7197]])
>>> torch.min(a)
tensor(0.6750)
```

---

# torch.maximum
## 原型
```python
torch.maximum(input, other, *, out=None) → Tensor
```

## 说明
张量 *input* 和 *other* 之间执行元素级的最大值比较。

## 例子
```python
>>> a = torch.tensor((1, 2, -1))
>>> b = torch.tensor((3, 0, 4))
>>> torch.maximum(a, b)
tensor([3, 2, 4])
```

---

# torch.minimum
## 原型
```python
torch.minimum(input, other, *, out=None) → Tensor
```

## 说明
张量 *input* 和 *other* 之间执行元素级的最小值比较。

## 例子
```python
>>> a = torch.tensor((1, 2, -1))
>>> b = torch.tensor((3, 0, 4))
>>> torch.minimum(a, b)
tensor([1, 0, -1])
```
---

# torch.fmax
## 原型
```python
torch.fmax(input, other, *, out=None) → Tensor
```

## 说明
和 torch.maximum 相似，但支持更多形式的数据形式输入（整数、浮点、NaN等）。

## 例子

```python
>>> a = torch.tensor([9.7, float('nan'), 3.1, float('nan')])
>>> b = torch.tensor([-2.2, 0.5, float('nan'), float('nan')])
>>> torch.fmax(a, b)
tensor([9.7000, 0.5000, 3.1000,    nan])
```

---

# torch.fmin
## 原型
```python
torch.fmin(input, other, *, out=None) → Tensor
```

## 说明
和 torch.minimum 相似，但支持更多形式的数据形式输入（整数、浮点、NaN等）。

## 例子
```python
>>> a = torch.tensor([2.2, float('nan'), 2.1, float('nan')])
>>> b = torch.tensor([-9.3, 0.1, float('nan'), float('nan')])
>>> torch.fmin(a, b)
tensor([-9.3000, 0.1000, 2.1000,    nan])
```

---

# torch.argmax
## 原型
```python
torch.argmax(input) → LongTensor
```

## 说明
找出张量中最大的值，并返回该值所对应的引索。

## 例子
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
        [-0.7401, -0.8805, -0.3402, -1.1936],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])
>>> torch.argmax(a)
tensor(0)
```

---

# torch.argmin
## 原型
```python
torch.argmin(input, dim=None, keepdim=False) → LongTensor
```

## 说明
找出张量中最小的值，并返回该值所对应的引索。不过和上面 argmax 不同的是，它还可以沿着轴方向 dim 找出轴方向的最小值引索。

* dim，指定的轴方向，默认无（从张量里找出极小值）
* keepdim，通常配合dim一起使用，如果输入的张量为 $(m \times n \times l)$ ，并且我们指定的dim = 1，输出的结果就会变成 $(m \times 1 \times l)$ 

## 例子
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
        [ 1.0100, -1.1975, -0.0102, -0.4732],
        [-0.9240,  0.1207, -0.7506, -1.0213],
        [ 1.7809, -1.2960,  0.9384,  0.1438]])
>>> torch.argmin(a)
tensor(13)
>>> torch.argmin(a, dim=1)
tensor([ 2,  1,  3,  1])
>>> torch.argmin(a, dim=1, keepdim=True)
tensor([[2],
        [1],
        [3],
        [1]])
```

---

# torch.amax
## 原型
```python
torch.amax(input, dim, keepdim=False, *, out=None) → Tensor
```
## 说明
找出轴方向上的最大值。

* dim，指定的轴方向，默认无（从张量里找出极小值）
* keepdim，通常配合dim一起使用，如果输入的张量为 $(m \times n \times l)$ ，并且我们指定的dim = 1，输出的结果就会变成 $(m \times 1 \times l)$ 

## 例子
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
        [-0.7158,  1.1775,  2.0992,  0.4817],
        [-0.0053,  0.0164, -1.3738, -0.0507],
        [ 1.9700,  1.1106, -1.0318, -1.0816]])
>>> torch.amax(a, 1)
tensor([1.4878, 2.0992, 0.0164, 1.9700])
```
---

# torch.amin
## 原型
```python
torch.amin(input, dim, keepdim=False, *, out=None) → Tensor
```

## 说明
找出轴方向上的最小值。

* dim，指定的轴方向，默认无（从张量里找出极小值）
* keepdim，通常配合dim一起使用，如果输入的张量为 $(m \times n \times l)$ ，并且我们指定的dim = 1，输出的结果就会变成 $(m \times 1 \times l)$ 

## 例子
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.6451, -0.4866,  0.2987, -1.3312],
        [-0.5744,  1.2980,  1.8397, -0.2713],
        [ 0.9128,  0.9214, -1.7268, -0.2995],
        [ 0.9023,  0.4853,  0.9075, -1.6165]])
>>> torch.amin(a, 1)
tensor([-1.3312, -0.5744, -1.7268, -1.6165])
```
---

# torch.aminmax
## 原型

```python
torch.aminmax(input, *, dim=None, keepdim=False, out=None) -> (Tensor min, Tensor max)
```

## 说明
同时找出最大和最小值。

* dim，指定的轴方向，默认无（从张量里找出极小值）
* keepdim，通常配合dim一起使用，如果输入的张量为 $(m \times n \times l)$ ，并且我们指定的dim = 1，输出的结果就会变成 $(m \times 2 \times l)$ 

## 例子

```python
>>> torch.aminmax(torch.tensor([1, -3, 5]))
torch.return_types.aminmax(
min=tensor(-3),
max=tensor(5))

>>> # aminmax propagates NaNs
>>> torch.aminmax(torch.tensor([1, -3, 5, torch.nan]))
torch.return_types.aminmax(
min=tensor(nan),
max=tensor(nan))

>>> t = torch.arange(10).view(2, 5)
>>> t
tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])
>>> t.aminmax(dim=0, keepdim=True)
torch.return_types.aminmax(
min=tensor([[0, 1, 2, 3, 4]]),
max=tensor([[5, 6, 7, 8, 9]]))
```

---
