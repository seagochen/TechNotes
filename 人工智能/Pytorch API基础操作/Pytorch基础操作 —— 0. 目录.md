@[toc]

# PyTorch与深度学习入门手册

## 正文

[Pytorch与深度学习 —— 1. 初识 PyTorch 的网络模型结构](https://seagochen.blog.csdn.net/article/details/119678081)  
[Pytorch与深度学习 —— 2.用全连接神经网络识别手写数字数据集MNIST](https://seagochen.blog.csdn.net/article/details/119754293)  
[Pytorch与深度学习 —— 3. 如何利用 CUDA 加速神经网络训练过程](https://seagochen.blog.csdn.net/article/details/119897310)  
[Pytorch与深度学习 —— 4. 用卷积神经网络识别手写数字数据集MNIST](https://seagochen.blog.csdn.net/article/details/119940591)  
[Pytorch与深度学习 —— 5. 什么是循环神经网络](https://seagochen.blog.csdn.net/article/details/119974822)  
[Pytorch与深度学习 —— 6. 使用 RNNCell 做文字序列的转化之 RNN 入门篇](https://seagochen.blog.csdn.net/article/details/120091116)  
[Pytorch与深度学习 —— 7. 什么是长短期记忆网络](https://seagochen.blog.csdn.net/article/details/120016528)  
[Pytorch与深度学习 —— 8. 使用 LSTM 做文字分类预测之 RNN 提高篇](https://seagochen.blog.csdn.net/article/details/120358980)  
[Pytorch与深度学习 —— 9. 如何把训练好的网络部署到C/C++语言的应用中](https://blog.csdn.net/poisonchry/article/details/120229489)  

## 附录
[Pytorch与深度学习 —— 附录I. 如何在 Ubuntu 中使用 C/C++ 语言编写 Pytorch 程序](https://blog.csdn.net/poisonchry/article/details/120318273)  
[Pytorch与深度学习 —— 附录II. 如何在 Windows 中使用 MSVS 编写 Pytorch 程序](https://blog.csdn.net/poisonchry/article/details/120346716?spm=1001.2014.3001.5502)  


# PyTorch 函数手册

## PyTorch 基本介绍

[Pytorch基础操作 —— 1.什么是Tensor，以及Tensor的基本属性、数据通道、维度](https://seagochen.blog.csdn.net/article/details/119484725)  
[Pytorch基础操作 —— 2.数据与模型在CPU与GPU之间的拷贝与传递](https://seagochen.blog.csdn.net/article/details/120623258)  
[Pytorch基础操作 —— 3.保存和加载Torch模型和参数](https://seagochen.blog.csdn.net/article/details/120628550)  
[Pytorch基础操作 —— 4. 创建 Tensor 的一般方法](https://seagochen.blog.csdn.net/article/details/120628581)  
[Pytorch基础操作 —— 5. 标准化数据集接口 Dataset 与开源数据集](https://seagochen.blog.csdn.net/article/details/120628595)  
[Pytorch基础操作 —— 6. 如何使用自定义数据集](https://seagochen.blog.csdn.net/article/details/120721148)  

## 引索、切分、转置操作

### 张量拼接

Function        | Description   | Detail
----------------|---------------|-----------
cat             |  Concatenates the given sequence of seq tensors in the given dimension | 最常用的张量拼接函数
concat          | Alias of torch.cat(). | 和 torch.cat 功能相似
stack           | Concatenates a sequence of tensors along a new dimension. | 创建新的维度，并拼接张量
dstack          | Stack tensors in sequence depthwise (along third axis). |  按深度方向拼接张量
hstack          | Stack tensors in sequence horizontally (column wise). | 按水平方向拼接张量
vstack          | Stack tensors in sequence vertically (row wise). | 按垂直方向拼接张量
row_stack       | Alias of torch.vstack(). |  和 torch.vstack 功能相似
column_stack    | Creates a new tensor by horizontally stacking the tensors in tensors. | 和 torch.hstack 相似

[Pytorch基础操作 —— 7. 张量拼接操作](https://seagochen.blog.csdn.net/article/details/120628777)

### 张量转置

Function        | Description   | Detail
----------------|---------------|-----------
t               | Expects input to be <= 2-D tensor and transposes dimensions 0 and 1. | 1、2维张量转置
transpose       | Returns a tensor that is a transposed version of input. | 多维张量转置
swapaxes        | Alias for torch.transpose(). | 和 transpose 功能一致
swapdims        | Alias for torch.transpose(). | 和 transpose 功能一致

[Pytorch基础操作 —— 8. 张量转置操作](https://seagochen.blog.csdn.net/article/details/120825671)

### 张量分割
Function        | Description   | Detail
----------------|---------------|-----------
chunk           | Attempts to split a tensor into the specified number of chunks. | 按指定数量分割张量
tensor_split    | Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections. | 按指定引索分割张量
split           | Splits the tensor into chunks. | 分割张量
unbind          | Removes a tensor dimension. |  对张量进行解耦操作
dsplit          | Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. | 按深度方向分割张量
hsplit          | Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections. | 按水平方向分割张量
vsplit          | Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections. | 按垂直方向分割张量

[Pytorch基础操作 —— 9. 张量分割](https://seagochen.blog.csdn.net/article/details/120843877)

### 升降维度

Function        | Description   | Detail
----------------|---------------|-----------
squeeze         | Returns a tensor with all the dimensions of input of size 1 removed. | 降维操作
unsqueeze       | Returns a new tensor with a dimension of size one inserted at the specified position. | 升维操作

[Pytorch基础操作 —— 10. 改变张量的维度（升维、降维）](https://seagochen.blog.csdn.net/article/details/121042431)

### 维度变换

Function        | Description   | Detail
----------------|---------------|-----------
reshape         | Returns a tensor with the same data and number of elements as input, but with the specified shape. | 维度改变

[Pytorch基础操作 —— 11. 改变张量的维度（维度修改）](https://blog.csdn.net/poisonchry/article/details/121019784)


### 数据选取

Function        | Description   | Detail
----------------|---------------|-----------
index_select    | Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor. | 引索选取
masked_select   | Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor. | 掩码选取
take            | Returns a new tensor with the elements of input at the given indices. | 以1维的方式按引索选取
take_along_dim  | Selects values from input at the 1-dimensional indices from indices along the given dim. | 以某轴，按引索选取
gather          | Gathers values along an axis specified by dim. |  -
where           | Return a tensor of elements selected from either x or y, depending on condition. | -

[Pytorch基础操作 —— 12. 从张量中选取数据](https://blog.csdn.net/poisonchry/article/details/120837055)

### 缩略信息、共轭、非零测试


Function        | Description   | Detail
----------------|---------------|-----------
narrow          | Returns a new tensor that is a narrowed version of input tensor. | 返回缩略信息
conj            | Returns a view of input with a flipped conjugate bit. |  创建共轭张量
nonzero         | | 返回张量中非0的引索

[Pytorch基础操作 —— 13. 缩略信息、共轭、非零测试](https://seagochen.blog.csdn.net/article/details/120872195)

## 运算操作

### 加法、减法、乘法、除法

Function        | Description   | Detail
----------------|---------------|-----------
add             | Adds other, scaled by alpha, to input.   | 多个张量相加运算
sum             | Returns the sum of all elements in the input tensor. | 求和累加运算
sub             | Subtracts other, scaled by alpha, from input.   | 多个张量相减运算
prod            | Returns the product of all elements in the input tensor. | 以数值形式返回元素的连乘运算
subtract        | Alias for torch.sub(). | sub函数的别名
dot		        | Computes the dot product of two 1D tensors. |  两个向量的内积
inner           | Computes the dot product for 1D tensors. | 多个向量的内积
cross           | Returns the cross product of vectors in dimension dim of input and other. Computes the dot product of two 1D tensors.   | 计算两个向量，或多个向量的外积
mm              | Performs a matrix multiplication of the matrices input and mat2. | 矩阵叉乘，或矩阵与常数乘法
mul             | Multiplies input by other. | 矩阵元素乘法（可执行广播），或矩阵与常数乘法
multiply        | Alias for torch.mul(). | mul 函数的别名
matmul          | Matrix product of two tensors. | 根据输入的数据类型自动返回矩阵乘、向量叉乘、或矩阵向量乘法（不推荐）
div             | Divides each element of the input input by the corresponding element of other. | 矩阵元素除法运算（可执行广播）
divide          | Alias for torch.div(). | div的别名

[Pytorch基础操作 —— 14. 张量的加法、减法、乘法、除法运算](https://seagochen.blog.csdn.net/article/details/121543552)


### 极值

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

[Pytorch基础操作 —— 15. 极值](https://seagochen.blog.csdn.net/article/details/121581416)

### 累计运算

Function        | Description   | Detail
----------------|---------------|-----------
cummax          | Returns a namedtuple (values, indices) where values is the cumulative maximum of elements of input in the dimension dim. | 在数列中累计比较，找出每次比较结果的最大值和引索
cummin          | Returns a namedtuple (values, indices) where values is the cumulative minimum of elements of input in the dimension dim.  | 在数列中累计比较，找出每次比较结果的最小值和引索
cumsum          | Returns the cumulative sum of elements of input in the dimension dim. | 以累计形式返回张量内元算的连加运算
cumprod         | Returns the cumulative product of elements of input in the dimension dim. | 以累计形式返回张量内元算的连乘运算
logcumsumexp    | Returns the logarithm of the cumulative summation of the exponentiation of elements of input in the dimension dim.

[Pytorch基础操作 —— 16. 累计运算](https://seagochen.blog.csdn.net/article/details/121583778)

### 绝对值、取整

Function        | Description   | Detail
----------------|---------------|-----------
abs             | Computes the absolute value of each element in input. |
absolute        | Alias for torch.abs()  
ceil            | Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
round           | Returns a new tensor with each of the elements of input rounded to the closest integer.
floor           | Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.


### 指数与开方
Function        | Description   | Detail
----------------|---------------|-----------
pow             | Takes the power of each element in input with exponent and returns a tensor with the result.
sqrt            | Returns a new tensor with the square-root of the elements of input.
square          | Returns a new tensor with the square of the elements of input.
logdet          | Calculates log determinant of a square matrix or batches of square matrices.
exp             | Returns a new tensor with the exponential of the elements of the input tensor input.
exp2            | Alias for torch.special.exp2().
expm1           | Alias for torch.special.expm1().
lgamma          | Computes the natural logarithm of the absolute value of the gamma function on input.
log             | Returns a new tensor with the natural logarithm of the elements of input.
log10           | Returns a new tensor with the logarithm to the base 10 of the elements of input.
log1p           | Returns a new tensor with the natural logarithm of (1 + input).
log2            | Returns a new tensor with the logarithm to the base 2 of the elements of input.
float_power     | Raises input to the power of exponent, elementwise, in double precision.


### 均值、方差、协方差
Function        | Description   | Detail
----------------|---------------|-----------
gradient        | Estimates the gradient of a function g : \mathbb{R}^n \rightarrow \mathbb{R}g:R n  →R in one or more dimensions using the second-order accurate central differences method.
median          | Returns the median of the values in input.
mean            | Returns the mean value of all elements in the input tensor.
std             | If unbiased is True, Bessel’s correction will be used.
std_mean        | If unbiased is True, Bessel’s correction will be used to calculate the standard deviation.
var             | If unbiased is True, Bessel’s correction will be used.
var_mean        | If unbiased is True, Bessel’s correction will be used to calculate the variance.
histc           | Computes the histogram of a tensor.
histogram       | Computes a histogram of the values in a tensor.
cov             | Estimates the covariance matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.
diff            | Computes the n-th forward difference along the given dimension.
norm            | Returns the matrix norm or vector norm of a given tensor.


Function        | Description   | Detail
----------------|---------------|-----------
det             | Alias for torch.linalg.det()
slogdet         | Alias for torch.linalg.slogdet()
fix             | Alias for torch.trunc()


### 三角函数

Function        | Description   | Detail
----------------|---------------|-----------
acos            | Computes the inverse cosine of each element in input. |
arccos          | Alias for torch.acos(). |
acosh           |  Returns a new tensor with the inverse hyperbolic cosine of the elements of input. |
arccosh         | Alias for torch.acosh().
angle           | Computes the element-wise angle (in radians) of the given input tensor.
asin            | Returns a new tensor with the arcsine of the elements of input.
arcsin          | Alias for torch.asin().
asinh           | Returns a new tensor with the inverse hyperbolic sine of the elements of input.
arcsinh         | Alias for torch.asinh().
atan            | Returns a new tensor with the arctangent of the elements of input.
arctan          | Alias for torch.atan().
atanh           | Returns a new tensor with the inverse hyperbolic tangent of the elements of input.
arctanh         | Alias for torch.atanh().
atan2           | Element-wise arctangent of \text{input}_{i} / \text{other}_{i}input   with consideration of the quadrant.
cos             | Returns a new tensor with the cosine of the elements of input.
cosh            | Returns a new tensor with the hyperbolic cosine of the elements of input.


### 逻辑运算

Function        | Description   | Detail
----------------|---------------|-----------
le              | Computes \text{input} \leq \text{other}input≤other element-wise.
less_equal      | Alias for torch.le().
lt              | Computes \text{input} < \text{other}input<other element-wise.
less            | Alias for torch.lt().  
eq              | Computes element-wise equality
equal           | True if two tensors have the same size and elements, False otherwise.
ge              | Computes \text{input} \geq \text{other}input≥other element-wise.
greater_equal   | Alias for torch.ge().
gt              | Computes \text{input} > \text{other}input>other element-wise.
greater         | Alias for torch.gt().        
isclose         | Returns a new tensor with boolean elements representing if each element of input is “close” to the corresponding element of other.
isfinite        | Returns a new tensor with boolean elements representing if each element is finite or not.
isin            | Tests if each element of elements is in test_elements.
isinf           | Tests if each element of input is infinite (positive or negative infinity) or not.
isposinf        | Tests if each element of input is positive infinity or not.
isneginf        | Tests if each element of input is negative infinity or not.
isnan           | Returns a new tensor with boolean elements representing if each element of input is NaN or not.
isreal          | Returns a new tensor with boolean elements representing if each element of input is real-valued or not.


### 位运算

Function        | Description   | Detail
----------------|---------------|-----------
bitwise_not     | Computes the bitwise NOT of the given input tensor.
bitwise_and     | Computes the bitwise AND of input and other.
bitwise_or      | Computes the bitwise OR of input and other.
bitwise_xor     | Computes the bitwise XOR of input and other.
bitwise_left_shift  | Computes the left arithmetic shift of input by other bits.
bitwise_right_shift | Computes the right arithmetic shift of input by other bits.
logical_and     | Computes the element-wise logical AND of the given input tensors. |
logical_not     | Computes the element-wise logical NOT of the given input tensor. |
logical_or      | Computes the element-wise logical OR of the given input tensors. |
logical_xor     | Computes the element-wise logical XOR of the given input tensors. |




Function     | Description    | Detail
-------------|----------------|-----------
clamp | Clamps all elements in input into the range [ min, max ].
clip      | Alias for torch.clamp().
conj_physical | Computes the element-wise conjugate of the given input tensor.
copysign  |  Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.
deg2rad | Returns a new tensor with each of the elements of input converted from angles in degrees to radians.
digamma | Alias for torch.special.digamma().
erf | Alias for torch.special.erf().
erfc | Alias for torch.special.erfc().
erfinv | Alias for torch.special.erfinv().
fake_quantize_per_channel_affine | Returns a new tensor with the data in input fake quantized per channel using scale, zero_point, quant_min and quant_max, across the channel specified by axis.
fake_quantize_per_tensor_affine | Returns a new tensor with the data in input fake quantized using scale, zero_point, quant_min and quant_max.
floor_divide | 
fmod | Applies C++’s std::fmod for floating point tensors, and the modulus operation for integer tensors.
frac | Computes the fractional portion of each element in input.
frexp | Decomposes input into mantissa and exponent tensors such that \text{input} = \text{mantissa} \times 2^{\text{exponent}}input=mantissa×2 
imag | Returns a new tensor containing imaginary values of the self tensor.
ldexp | Multiplies input by 2**:attr:other.
lerp | Does a linear interpolation of two tensors start (given by input) and end based on a scalar or tensor weight and returns the resulting out tensor.
logaddexp | Logarithm of the sum of exponentiations of the inputs. |
logaddexp2 | Logarithm of the sum of exponentiations of the inputs in base-2. |
logit | Alias for torch.special.logit(). |
hypot | Given the legs of a right triangle, return its hypotenuse. |
i0 | Alias for torch.special.i0(). |
igamma | Alias for torch.special.gammainc().
igammac | Alias for torch.special.gammaincc().
mvlgamma | Alias for torch.special.multigammaln().
nan_to_num | Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
neg | Returns a new tensor with the negative of the elements of input.
negative | Alias for torch.neg()
nextafter | Return the next floating-point value after input towards other, elementwise.
polygamma | Alias for torch.special.polygamma().
positive | Returns input.
quantized_batch_norm | Applies batch normalization on a 4D (NCHW) quantized tensor.
quantized_max_pool1d | Applies a 1D max pooling over an input quantized tensor composed of several input planes.
quantized_max_pool2d | Applies a 2D max pooling over an input quantized tensor composed of several input planes.
rad2deg | Returns a new tensor with each of the elements of input converted from angles in radians to degrees.
real | Returns a new tensor containing real values of the self tensor.
reciprocal | Returns a new tensor with the reciprocal of the elements of input
remainder | Like torch.fmod() this applies C++’s std::fmod for floating point tensors and the modulus operation for integer tensors.
rsqrt | Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
sigmoid | Alias for torch.special.expit().
sign | Returns a new tensor with the signs of the elements of input.
sgn | This function is an extension of torch.sign() to complex tensors.
signbit | Tests if each element of input has its sign bit set (is less than zero) or not.
sin | Returns a new tensor with the sine of the elements of input.
sinc | Alias for torch.special.sinc().
sinh | Returns a new tensor with the hyperbolic sine of the elements of input.
tan | Returns a new tensor with the tangent of the elements of input.
tanh | Returns a new tensor with the hyperbolic tangent of the elements of input.
true_divide | Alias for torch.div() with rounding_mode=None.
trunc | Returns a new tensor with the truncated integer values of the elements of input.
xlogy | Alias for torch.special.xlogy().
all | Tests if all elements in input evaluate to True.
any | 
dist | Returns the p-norm of (input - other)
logsumexp | Returns the log of summed exponentials of each row of the input tensor in the given dimension dim.
nanmean | Computes the mean of all non-NaN elements along the specified dimensions.
nanmedian |Returns the median of the values in input, ignoring NaN values.
mode | Returns a namedtuple (values, indices) where values is the mode value of each row of the input tensor in the given dimension dim, i.e. a value which appears most often in that row, and indices is the index location of each mode value found.
nansum | Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.
quantile | Computes the q-th quantiles of each row of the input tensor along the dimension dim.
nanquantile | This is a variant of torch.quantile() that “ignores” NaN values, computing the quantiles q as if NaN values in input did not exist.
unique | Returns the unique elements of the input tensor.
unique_consecutive | Eliminates all but the first element from every consecutive group of equivalent elements.
count_nonzero | Counts the number of non-zero values in the tensor input along the given dim.
allclose | This function checks if all input and other satisfy the condition:
argsort | Returns the indices that sort a tensor along a given dimension in ascending order by value.
kthvalue | Returns a namedtuple (values, indices) where values is the k th smallest element of each row of the input tensor in the given dimension dim.
ne | Computes \text{input} \neq \text{other}input =other element-wise. 
not_equal | Alias for torch.ne().
sort | Sorts the elements of the input tensor along a given dimension in ascending order by value.
topk | Returns the k largest elements of the given input tensor along a given dimension.
msort | Sorts the elements of the input tensor along its first dimension in ascending order by value.


stft | Short-time Fourier transform (STFT).
istft | Inverse short time Fourier Transform.
bartlett_window | Bartlett window function.
blackman_window | Blackman window function.
hamming_window | Hamming window function.
hann_window | Hann window function.
kaiser_window | Computes the Kaiser window with window length window_length and shape parameter beta.

Other Operations
atleast_1d

Returns a 1-dimensional view of each input tensor with zero dimensions.

atleast_2d

Returns a 2-dimensional view of each input tensor with zero dimensions.

atleast_3d

Returns a 3-dimensional view of each input tensor with zero dimensions.

bincount

Count the frequency of each value in an array of non-negative ints.

block_diag

Create a block diagonal matrix from provided tensors.

broadcast_tensors

Broadcasts the given tensors according to Broadcasting semantics.

broadcast_to

Broadcasts input to the shape shape.

broadcast_shapes

Similar to broadcast_tensors() but for shapes.

bucketize | Returns the indices of the buckets to which each value in the input belongs, where the boundaries of the buckets are set by boundaries.
cartesian_prod | Do cartesian product of the given sequence of tensors.
cdist | Computes batched the p-norm distance between each pair of the two collections of row vectors.
clone | Returns a copy of input.
combinations | Compute combinations of length rr of the given tensor.
corrcoef| Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the input matrix, where rows are the variables and columns are the observations.
diag| If input is a vector (1-D tensor), then returns a 2-D square tensor
diag_embed| Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input.
diagflat| If input is a vector (1-D tensor), then returns a 2-D square tensor
diagonal| Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 appended as a dimension at the end of the shape.

einsum| Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention.
flatten| Flattens input by reshaping it into a one-dimensional tensor.
flip| Reverse the order of a n-D tensor along given axis in dims.
fliplr| Flip tensor in the left/right direction, returning a new tensor.
flipud| Flip tensor in the up/down direction, returning a new tensor.
kron | Computes the Kronecker product, denoted by \otimes⊗, of input and other.
rot90 | Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.

gcd | Computes the element-wise greatest common divisor (GCD) of input and other.
meshgrid | Creates grids of coordinates specified by the 1D inputs in attr:tensors.
lcm | Computes the element-wise least common multiple (LCM) of input and other.
ravel | Return a contiguous flattened tensor.
renorm | Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm
repeat_interleave | Repeat elements of a tensor.
roll | Roll the tensor along the given dimension(s).
searchsorted | Find the indices from the innermost dimension of sorted_sequence such that, if the corresponding values in values were inserted before the indices, the order of the corresponding innermost dimension within sorted_sequence would be preserved.
tensordot | Returns a contraction of a and b over multiple dimensions.
trace | Returns the sum of the elements of the diagonal of the input 2-D matrix.
tril | Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
tril_indices | Returns the indices of the lower triangular part of a row-by- col matrix in a 2-by-N Tensor, where the first row contains row coordinates of all indices and the second row contains column coordinates.
triu | Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
triu_indices | Returns the indices of the upper triangular part of a row by col matrix in a 2-by-N Tensor, where the first row contains row coordinates of all indices and the second row contains column coordinates.

vander | Generates a Vandermonde matrix.
view_as_real | Returns a view of input as a real tensor.
view_as_complex | Returns a view of input as a complex tensor.
resolve_conj | Returns a new tensor with materialized conjugation if input’s conjugate bit is set to True, else returns input.
resolve_neg | Returns a new tensor with materialized negation if input’s negative bit is set to True, else returns input.

BLAS and LAPACK Operations
addbmm | Performs a batch matrix-matrix product of matrices stored in batch1 and batch2, with a reduced add step (all matrix multiplications get accumulated along the first dimension).
addmm | Performs a matrix multiplication of the matrices mat1 and mat2.
addmv| Performs a matrix-vector product of the matrix mat and the vector vec.
addr | Performs the outer-product of vectors vec1 and vec2 and adds it to the matrix input.
baddbmm | Performs a batch matrix-matrix product of matrices in batch1 and batch2.
bmm | Performs a batch matrix-matrix product of matrices stored in input and mat2.
chain_matmul | Returns the matrix product of the NN 2-D tensors.
cholesky | Computes the Cholesky decomposition of a symmetric positive-definite matrix AA or for batches of symmetric positive-definite matrices.
cholesky_inverse | Computes the inverse of a symmetric positive-definite matrix AA using its Cholesky factor uu: returns matrix inv.
cholesky_solve | Solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix uu.
eig | Computes the eigenvalues and eigenvectors of a real square matrix.
geqrf | This is a low-level function for calling LAPACK’s geqrf directly.
ger | Alias of torch.outer().
inverse | Alias for torch.linalg.inv()
lstsq | Computes the solution to the least squares and least norm problems for a full rank matrix AA of size (m \times n)(m×n) and a matrix BB of size (m \times k)(m×k).
lu | Computes the LU factorization of a matrix or batches of matrices A.
lu_solve | Returns the LU solve of the linear system Ax = bAx=b using the partially pivoted LU factorization of A from torch.lu().
lu_unpack | Unpacks the data and pivots from a LU factorization of a tensor into tensors L and U and a permutation tensor P such that LU_data, LU_pivots = (P @ L @ U).lu().
matrix_power | Alias for torch.linalg.matrix_power()
matrix_rank | Returns the numerical rank of a 2-D tensor.
matrix_exp | Computes the matrix exponential of a square matrix or of each square matrix in a batch.
orgqr | Alias for torch.linalg.householder_product().
ormqr | Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.
outer | Outer product of input and vec2.
pinverse | Alias for torch.linalg.pinv()
qr | Computes the QR decomposition of a matrix or a batch of matrices input, and returns a namedtuple (Q, R) of tensors such that \text{input} = Q Rinput=QR with QQ being an orthogonal matrix or batch of orthogonal matrices and RR being an upper triangular matrix or batch of upper triangular matrices.
solve | This function returns the solution to the system of linear equations represented by AX = BAX=B and the LU factorization of A, in order as a namedtuple solution, LU.
svd | Computes the singular value decomposition of either a matrix or batch of matrices input.
svd_lowrank | Return the singular value decomposition (U, S, V) of a matrix, batches of matrices, or a sparse matrix AA such that A \approx U diag(S) V^TA≈Udiag(S)V 
T.
pca_lowrank | Performs linear Principal Component Analysis (PCA) on a low-rank matrix, batches of such matrices, or sparse matrix.
symeig | This function returns eigenvalues and eigenvectors of a real symmetric or complex Hermitian matrix input or a batch thereof, represented by a namedtuple (eigenvalues, eigenvectors).
lobpcg | Find the k largest (or smallest) eigenvalues and the corresponding eigenvectors of a symmetric positive defined generalized eigenvalue problem using matrix-free LOBPCG methods.
trapz | Alias for torch.trapezoid().
trapezoid | Computes the trapezoidal rule along dim.
cumulative_trapezoid | Cumulatively computes the trapezoidal rule along dim.
triangular_solve | Solves a system of equations with a triangular coefficient matrix AA and multiple right-hand sides bb.
vdot | Computes the dot product of two 1D tensors.

Utilities
compiled_with_cxx11_abi | Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1
result_type | Returns the torch.dtype that would result from performing an arithmetic operation on the provided input tensors.
can_cast | Determines if a type conversion is allowed under PyTorch casting rules described in the type promotion documentation.
promote_types | Returns the torch.dtype with the smallest size and scalar kind that is not smaller nor of lower kind than either type1 or type2.
use_deterministic_algorithms | Sets whether PyTorch operations must use “deterministic” algorithms.
are_deterministic_algorithms_enabled | Returns True if the global deterministic flag is turned on.
set_warn_always | When this flag is False (default) then some PyTorch warnings may only appear once per process.
is_warn_always_enabled | Returns True if the global warn_always flag is turned on.
_assert | A wrapper around Python’s assert which is symbolically traceable.