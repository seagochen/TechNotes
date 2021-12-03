Function        | Description   | Detail
----------------|---------------|-----------
abs             | Computes the absolute value of each element in input. | 返回元素的绝对值
absolute        | Alias for torch.abs()  | torch.abs 的别名
ceil            | Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element. | 对元素向上取整
round           | Returns a new tensor with each of the elements of input rounded to the closest integer. | 对元素四舍五入
floor           | Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element. | 对元素向下取整

@[toc]

# torch.abs
## 原型
```python
torch.abs(input, *, out=None) → Tensor
```

## 说明
对张量里的每一个元素计算绝对值

$$
out_i = | input_i |
$$

## 例子

```python
>>> torch.abs(torch.tensor([-1, -2, 3]))
tensor([ 1,  2,  3])
```

---

# torch.ceil
## 原型
```python
torch.ceil(input, *, out=None) → Tensor
```

## 说明
向上取整

$$
out_i = \left \lceil  input_i \right \rceil 
$$

## 例子

```python
>>> a = torch.randn(4)
>>> a
tensor([-0.6341, -1.4208, -1.0900,  0.5826])
>>> torch.ceil(a)
tensor([-0., -1., -1.,  1.])
```

---

# torch.round
## 原型
```python
torch.round(input, *, out=None) → Tensor
```

## 说明

四舍五入

## 例子

```python
>>> a = torch.randn(4)
>>> a
tensor([ 0.9920,  0.6077,  0.9734, -1.0362])
>>> torch.round(a)
tensor([ 1.,  1.,  1., -1.])
```

---

# torch.floor
## 原型
```python
torch.floor(input, *, out=None) → Tensor
```

## 说明
向下取整

$$
out_i = \left \lfloor input_i  \right \rfloor 
$$
## 例子
```python
>>> a = torch.randn(4)
>>> a
tensor([-0.8166,  1.5308, -0.2530, -0.2091])
>>> torch.floor(a)
tensor([-1.,  1., -1., -1.])
```
