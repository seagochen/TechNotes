Arduino 提供了最基础的数学函数，能够处理一些简单的数学运算。

# 基础数学函数

## abs(x)

计算绝对值。

## constrain(x, a, b)
 限制输入X的范围，$a \leq x \leq b$ 如果x小于a，返回a；x如果大于b，返回b。
 

## map(value, fromLow, fromHigh, toLow, toHigh)

线性映射，把value从 $[fromLow, fromHigh]$ 线性映射到 [toLow, toHigh]。例如从0-255，映射到0-1023。也可以用于数值的翻转，例如从0-255，翻转到255-0。目前这个函数只支持整数的映射，小数或者分数的小数点部分会被直接舍弃，类似于：

```cpp
float pi = 3.14159;
int p = (int)pi;
```

## max(x, y)

比较两个数，返回最大的数。


## min(x, y)

比较两个数，返回最小的数。


## pow(base, exponent)

指数运算，例如求解$2^4$，即：

```cpp
pow(2, 4);
```

## sq(x)

求平方，$x^2$

## sqrt(x)

求开方，$\sqrt{x}$


# 三角函数

## cos(rad)
余弦函数，rad为角度。

## sin(rad)
正弦函数，rad为角度

## tan(rad)
正切函数，rad为角度