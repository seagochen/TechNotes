Function        | Description   | Detail
----------------|---------------|-----------
cummax          | Returns a namedtuple (values, indices) where values is the cumulative maximum of elements of input in the dimension dim. | 在数列中累计比较元素大小，找出每次比较结果的最大值和引索
cummin          | Returns a namedtuple (values, indices) where values is the cumulative minimum of elements of input in the dimension dim.  | 在数列中累计比较元素大小，找出每次比较结果的最小值和引索
cumsum          | Returns the cumulative sum of elements of input in the dimension dim. | 以累计形式返回张量内元算的连加运算
cumprod         | Returns the cumulative product of elements of input in the dimension dim. | 以累计形式返回张量内元算的连乘运算
logcumsumexp    | Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim. | 以累计形式返回张量内各元算的log运算

@[toc]

# torch.cummax
## 原型
```python
torch.cummax(input, dim, *, out=None)
```

## 说明
对数列的元素逐个进行比较，每次得出一个最大值，并最终返回结果和引索。它对每个输出元素 $y_i$ 执行以下运算

$$
y_i = max(x_1, x_2, \cdots, x_n)
$$

因此最终输出结果如下

$$
y = [max(x_1), max(x_1, x_2), max(x_1, x_2, x_3), \cdots]
$$

## 例子

```python
>>> a = torch.randn(10)
>>> a
tensor([-0.3449, -1.5447,  0.0685, -1.5104, -1.1706,  0.2259,  1.4696, -1.3284,
     1.9946, -0.8209])
>>> torch.cummax(a, dim=0)
torch.return_types.cummax(
    values=tensor([-0.3449, -0.3449,  0.0685,  0.0685,  0.0685,  0.2259,  1.4696,  1.4696,
     1.9946,  1.9946]),
    indices=tensor([0, 0, 2, 2, 2, 5, 6, 6, 8, 8]))
```

---

# torch.cummin
## 原型

## 说明
对数列的元素逐个进行比较，每次得出一个最小值，并最终返回结果和引索。它对每个输出元素 $y_i$ 执行以下运算

$$
y_i = min(x_1, x_2, \cdots, x_n)
$$

因此最终输出结果如下

$$
y = [min(x_1), min(x_1, x_2), min(x_1, x_2, x_3), \cdots]
$$

## 例子
```python
>>> a = torch.randn(10)
>>> a
tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220, -0.3885,  1.1762,
     0.9165,  1.6684])
>>> torch.cummin(a, dim=0)
torch.return_types.cummin(
    values=tensor([-0.2284, -0.6628, -0.6628, -0.6628, -1.3298, -1.3298, -1.3298, -1.3298,
    -1.3298, -1.3298]),
    indices=tensor([0, 1, 1, 1, 4, 4, 4, 4, 4, 4]))
```
---

# torch.cumsum
## 原型
```python
torch.cumsum(input, dim, *, dtype=None, out=None) → Tensor
```

## 说明
执行张量内元素连加运算，它最终返回的是一组张量，其各值如下

$$
y = [x_1, x_1 + x_2, \cdots, x_1  + x_2 + \cdots + x_n]
$$


## 例子

```python
>>> a = torch.randn(10)
>>> a
tensor([-0.8286, -0.4890,  0.5155,  0.8443,  0.1865, -0.1752, -2.0595,
         0.1850, -1.1571, -0.4243])
>>> torch.cumsum(a, dim=0)
tensor([-0.8286, -1.3175, -0.8020,  0.0423,  0.2289,  0.0537, -2.0058,
        -1.8209, -2.9780, -3.4022])
```

---

# torch.cumprod 
## 原型

```python
torch.cumprod(input, dim, *, dtype=None, out=None) → Tensor
```

## 说明

执行张量内元素连乘运算，它最终返回的是一组张量，其各值如下

$$
y = [x_1, x_1 \times x_2, \cdots, x_1 \times x_2 \times \cdots \times x_n]
$$


## 例程

所以，在下面这个例子里，序列最后的才是元素累乘后的最终值

```python
>>> a = torch.randn(10)
>>> a
tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
        -0.2129, -0.4206,  0.1968])
>>> torch.cumprod(a, dim=0)
tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
         0.0014, -0.0006, -0.0001])

>>> a[5] = 0.0
>>> torch.cumprod(a, dim=0)
tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
         0.0000, -0.0000, -0.0000])
```
---

# torch.logcumsumexp

## 原型
```python
torch.logcumsumexp(input, dim, *, out=None) → Tensor
```

## 说明

执行张量内各元素log运算

$$
y_{ij} = \log \sum_{j=0}^i \exp(x_{ij})
$$

## 例程
```python
>>> a = torch.randn(10)
>>> torch.logcumsumexp(a, dim=0)
tensor([-0.42296738, -0.04462666,  0.86278635,  0.94622083,  1.05277811,
         1.39202815,  1.83525007,  1.84492621,  2.06084887,  2.06844475]))
```