Function        | Description   | Detail
----------------|---------------|-----------
add             | Adds other, scaled by alpha, to input.   | 多个张量相加运算
sum             | Returns the sum of all elements in the input tensor. | 元素的求和累加运算
sub             | Subtracts other, scaled by alpha, from input.   | 多个张量相减运算
prod            | Returns the product of all elements in the input tensor. | 以数值形式返回元素的连乘运算
subtract       | Alias for torch.sub(). | sub函数的别名
dot				| Computes the dot product of two 1D tensors. |  两个向量的内积
inner           | Computes the dot product for 1D tensors. | 多个向量的内积
cross           | Returns the cross product of vectors in dimension dim of input and other. Computes the dot product of two 1D tensors.   | 计算两个向量，或多个向量的外积
mv 				| Performs a matrix-vector product of the matrix input and the vector vec. | 计算矩阵与向量的点乘
mm              | Performs a matrix multiplication of the matrices input and mat2. | 计算矩阵叉乘，或矩阵与常数乘法
mul             | Multiplies input by other. | 矩阵元素乘法（可执行广播），或矩阵与常数乘法
multiply        | Alias for torch.mul(). | mul 函数的别名
matmul          | Matrix product of two tensors. | 根据输入的数据类型自动返回矩阵乘、向量叉乘、或矩阵与向量的点乘（不推荐）
div             | Divides each element of the input input by the corresponding element of other. | 矩阵元素除法运算（可执行广播）
divide          | Alias for torch.div(). | div的别名

@[toc]

---

# torch.add
## 函数原型
```python
torch.add(input, other, *, alpha=1, out=None) → Tensor
```
该函数的执行方式，可以用公式表示如下；

 $$
 out_i = input_i + alpha \times other_i
 $$

另外，也可以直接使用 + 符用于张量之间，张量与常数间的加法。

---

# torch.sum
## 函数原型
原型1:
```python
torch.sum(input, *, dtype=None) → Tensor
```

原型2:
```python
torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor
```

$$
\sum = x_1 + x_2 + \cdots + x_n
$$

求和累加是运算，不过该函数是针对张量内的元素求和，不指定方向的话，它会计算出所有元素的加和，如果制定了轴方向 dim，仅沿着轴方向累加数据。如果需要执行多个张量之间相加，使用 torch.add 函数。

## 参数说明
* input (Tensor) – the input tensor.
* dim (int or tuple of python:ints) – the dimension or dimensions to reduce.
* keepdim (bool) – whether the output tensor has dim retained or not.

## 例程

累加全部元素

```python
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.1133, -0.9567,  0.2958]])
>>> torch.sum(a)
tensor(-0.5475)
```
沿着轴方面 dim=1 累加
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
        [-0.2993,  0.9138,  0.9337, -1.6864],
        [ 0.1132,  0.7892, -0.1003,  0.5688],
        [ 0.3637, -0.9906, -0.4752, -1.5197]])
>>> torch.sum(a, dim=1)
tensor([-0.4598, -0.1381,  1.3708, -2.6217])
```
先沿着轴 dim=2 方向累加，再沿着 dim=1 方向累加；结果与 dim=(1, 2) 是一致的。
```python
>>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
>>> torch.sum(b, dim=(2, 1))
tensor([  435.,  1335.,  2235.,  3135.])
```

---

# torch.sub
## 函数原型

```python
torch.sub(input, other, *, alpha=1, out=None) → Tensor
```

该函数的执行方式，可以用公式表示如下；

$$
 out_i = input_i - alpha \times other_i
$$

另外，也可以直接使用 - 符用于张量之间，张量与常数间的加法。

---

# torch.prod
## 函数原型

```python
torch.prod(input, *, dtype=None) → Tensor
```


执行张量内元素连乘运算，并以值的形式返回结果。

$$
\Pi  = x_1 \times x_2 \times \cdots \times x_n
$$

---

# torch.dot
## 函数原型

```python
torch.dot(input, other, *, out=None) → Tensor
```

这个函数只能执行两个一维张量的内积。不需要指定张量为横、竖张量，也不需要转置。但是input和other都必须只能是一维的。

```python
>>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
tensor(7)
```

---

# torch.inner

## 函数原型

```python
torch.inner(input, other, *, out=None) → Tensor
```

函数可执行一维张量的内积，当有高维度张量时，它会把一维内积的结果沿着张量最后的维度方向累加起来。

## 例程
执行与dot类似的内积。
```python
# Dot product
>>> torch.inner(torch.tensor([1, 2, 3]), torch.tensor([0, 2, 1]))
tensor(7)
```

执行高维度运算。

```python
# Multidimensional input tensors
>>> a = torch.arange(1, 7).reshape(2, 3)
>>> a
tensor([[1, 2, 3],
        [4, 5, 6]])
>>> b = torch.arange(10, 34).reshape(2, 4, 3)
>>> b
tensor([[[10, 11, 12],
         [13, 14, 15],
         [16, 17, 18],
         [19, 20, 21]],

        [[22, 23, 24],
         [25, 26, 27],
         [28, 29, 30],
         [31, 32, 33]]])
>>> torch.inner(a, b)
tensor([[[ 68,  86, 104, 122],
         [140, 158, 176, 194]],

        [[167, 212, 257, 302],
         [347, 392, 437, 482]]])
```

看起来有点复杂，其实很好理解。我们用比较直观的方式，将这里的计算过程可以表示成下面这段代码

```python
d01, d02 = a.shape
d11, d12, d13 = b.shape

if d02 != d13:
	print("Invalid size of both tensors")
	exit()

# inner(a, b) ----> d01 x d11 x d12
# inner(b, a) ----> d11 x d12 x d01

c = torch.zeros((d01, d11, d12)) # perform inner(a, b)

for i0 in range(d01):
    for i1 in range(d11):
        for j1 in range(d12):
            temp = 0
            for k in range(3):
                temp += a[i0, k] * b[i1, j1, k]
            c[i0, i1, j1] = temp
```

执行与常量的计算。

```python
# Scalar input
>>> torch.inner(a, torch.tensor(2))
tensor([[1.6347, 2.1748, 2.3567],
        [0.6558, 0.2469, 5.5787]])
```

---

# torch.cross
## 函数原型

```python
torch.cross(input, other, dim=None, *, out=None) → Tensor
```

该函数可计算两个或多个向量的外积，要求 input 和 other 的维度最少是3（x, y, z），如果需要同时计算多个向量，可以用矩阵的形式输入参数。

## 例程

```python
>>> a = torch.randn(4, 3)
>>> a
tensor([[-0.3956,  1.1455,  1.6895],
        [-0.5849,  1.3672,  0.3599],
        [-1.1626,  0.7180, -0.0521],
        [-0.1339,  0.9902, -2.0225]])
>>> b = torch.randn(4, 3)
>>> b
tensor([[-0.0257, -1.4725, -1.2251],
        [-1.1479, -0.7005, -1.9757],
        [-1.3904,  0.3726, -1.1836],
        [-0.9688, -0.7153,  0.2159]])
>>> torch.cross(a, b, dim=1)
tensor([[ 1.0844, -0.5281,  0.6120],
        [-2.4490, -1.5687,  1.9792],
        [-0.8304, -1.3037,  0.5650],
        [-1.2329,  1.9883,  1.0551]])
>>> torch.cross(a, b)
tensor([[ 1.0844, -0.5281,  0.6120],
        [-2.4490, -1.5687,  1.9792],
        [-0.8304, -1.3037,  0.5650],
        [-1.2329,  1.9883,  1.0551]])
```
如果列方向是 (x, y, z) 那么指定 dim 没什么意义，除非向量的方向是以行或其他形式组成。

---

# torch.mv

## 函数原型

```python
torch.mv(input, vec, *, out=None) → Tensor
```

执行矩阵与向量的点乘，矩阵的维度如果是 $(n \times m)$，则要求向量的必须是 $m$ 长

## 例程

```python
>>> mat = torch.randn(2, 3)
>>> vec = torch.randn(3)
>>> torch.mv(mat, vec)
tensor([ 1.0404, -0.6361])
```

---

# torch.mm
## 函数原型
```python
torch.mm(input, mat2, *, out=None) → Tensor
```
执行矩阵乘法，如果input维度为 $(m \times n)$， mat2 维度为 $(n \times p)$ 输出结果为 $(m \times p)$。

## 例程
```python
>>> mat1 = torch.randn(2, 4)
>>> mat2 = torch.randn(4, 3)
>>> torch.mm(mat1, mat2) # output is 2 x 3
tensor([[ 0.4851,  0.5037, -0.3633],
        [-0.0760, -3.6705,  2.4784]])
```

---


# torch.mul
## 函数原型
```python
torch.mul(input, other, *, out=None) → Tensor
```

$$
out_i = input_i \times other_i
$$

它执行矩阵间的元素乘，也可以执行矩阵与常数的乘法，在遭遇行、列向量相乘时，可以对运算执行广播。


## 例程

矩阵与常数的乘法
```python
>>> a = torch.randn(3)
>>> a
tensor([ 0.2015, -0.4255,  2.6087])
>>> torch.mul(a, 100)
tensor([  20.1494,  -42.5491,  260.8663])
```

执行广播运算

```python
>>> b = torch.randn(4, 1)
>>> b
tensor([[ 1.1207],
        [-0.3137],
        [ 0.0700],
        [ 0.8378]])
>>> c = torch.randn(1, 4)
>>> c
tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
>>> torch.mul(b, c)
tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
        [-0.1614, -0.0382,  0.1645, -0.7021],
        [ 0.0360,  0.0085, -0.0367,  0.1567],
        [ 0.4312,  0.1019, -0.4394,  1.8753]])
```

---

# torch.matmul
## 函数原型
```python
torch.matmul(input, other, *, out=None) → Tensor
```

根据函数的说明，它是根据用户输入的张量类型，自动判断执行矩阵、向量外积，还是矩阵与向量乘。由于在使用时可能会出现遗忘或失误，极可能导致函数执行了不希望的运算，所以不推荐这种「隐式」乘法运算。

详细信息，请参考官方相关文档说明：

> https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul

---

# torch.div
## 函数原型
```python
torch.div(input, other, *, rounding_mode=None, out=None) → Tensor
```

矩阵是不能直接求除法的，但是我们可以求矩阵元素的除法。

$$
output_i = \frac{input_i}{other_i}
$$

它要求矩阵的维度大小一致，或正好满足可广播的维度。

## 特别说明
函数中有一个 「rounding_mode」可以指定运算结果的取整方式，字符串型，具体定义如下


* **None** - default behavior. Performs no rounding and, if both input and other are integer types, promotes the inputs to the default scalar type. Equivalent to true division in Python (the / operator) and NumPy’s np.true_divide.
* **"trunc"** - rounds the results of the division towards zero. Equivalent to C-style integer division.
* **"floor"** - rounds the results of the division down. Equivalent to floor division in Python (the // operator) and NumPy’s np.floor_divide.

## 例程

```python
>>> x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
>>> torch.div(x, 0.5)
tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])

>>> a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
...                   [ 0.1815, -1.0111,  0.9805, -1.5923],
...                   [ 0.1062,  1.4581,  0.7759, -1.2344],
...                   [-0.1830, -0.0313,  1.1908, -1.4757]])
>>> b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
>>> torch.div(a, b)
tensor([[-0.4620, -6.6051,  0.5676,  1.2639],
        [ 0.2260, -3.4509, -1.2086,  6.8990],
        [ 0.1322,  4.9764, -0.9564,  5.3484],
        [-0.2278, -0.1068, -1.4678,  6.3938]])
```
使用了 「rounding_mode」的结果

```python
>>> torch.div(a, b, rounding_mode='trunc')
tensor([[-0., -6.,  0.,  1.],
        [ 0., -3., -1.,  6.],
        [ 0.,  4., -0.,  5.],
        [-0., -0., -1.,  6.]])

>>> torch.div(a, b, rounding_mode='floor')
tensor([[-1., -7.,  0.,  1.],
        [ 0., -4., -2.,  6.],
        [ 0.,  4., -1.,  5.],
        [-1., -1., -2.,  6.]])
```