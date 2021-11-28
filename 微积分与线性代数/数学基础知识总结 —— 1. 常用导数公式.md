@[toc]

# 基本导数公式
| 原函数$f(x)$ | 导数$f'(x)$  |
|--|--|
| C (C为常数) | 0 |
| $x^n$ | $nx^{n-1}$ |
| $C^x$ | $C^xlnC$ (C为常数，且大于0) |
| $e^x$ (e为自然常数) | $e^x(lne) = e^x \cdot 1 = e^x$ |
| $log_cx$ | $\frac{log_ae}{x}$ |
| $ln x$ | $\frac{1}{x}$ |
| $sin x$ |  $cos x$ |
| $cos x$ | $-sin x$ |
| $tan x$ | $sec^2x = \frac{1}{cos^2 x}$ |
| $cot x$ | $-csc^2 x = -\frac{1}{sin^2 x}$ |

# 基本导数运算法则

## 加法运算
$$F'(x)=(f(x) + g(x))' = f'(x) + g'(x)$$

## 减法运算
$$F'(x)=(f(x) - g(x))' = f'(x) - g'(x)$$

## 乘法运算
$$F'(x)=(f(x) \times g(x))' = f'(x)g(x) + f(x)g'(x)$$

## 除法运算
$$F'(x) = \left \{ \frac{f(x)}{g(x)} \right \}' =  \left \{ \frac{f(x)'g(x) - f(x)g(x)'}{g^2(x)} \right \} $$

## 带有常数C的导数
$$F'(x) = (C \cdot f(x))' = C \cdot f(x)'$$

# 微分的四则运算
微分常见的表示符号有三种，在偏微分方程中，以$\partial$表示，在通常则是以$d$表示，某些教科书上也有以$diff(x)$进行表示，代表一种计算方法，$dx$表达的含义与通常$f(x)$是一样的，因为数学家比较懒的原因，$d(x)$就约定俗成的用$dx$进行表达了。

## 加减法计算
$$d(f(x) \pm g(x)) = d(f(x)) \pm d(g(x))$$

## 带有常数的微分

$$d(Cf(x)) = C \cdot d(f(x))$$

## 乘法计算
$$d(f(x)g(x)) = d(f(x))g(x) + f(x)d(g(x))$$

## 除法计算
$$ d \left \{ \frac{f(x)}{g(x)} \right \} = \left \{ \frac{df(x)g(x) - f(x)dg(x)}{g^2(x)} \right \}$$