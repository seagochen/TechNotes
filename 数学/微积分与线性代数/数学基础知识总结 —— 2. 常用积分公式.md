
@[toc]

微积分公式有很多，这里只是罗列常用的（主要是「同济高数」里常用到的部分）。一般来说，积分公式最终表现为

$$
\int \mathbf{u} dx = \mathbf{v} + C
$$

的形式。但是在最终计算最后一步时， $[\mathbf{v} + C] \left |_a^b \right . = v$ C往往会被消掉，所以可以在求积分时，可以不写C。

尽管我这里提供了相对完整的 [资料](https://download.csdn.net/download/poisonchry/40213023)，如果你是工程或科研需要，建议还是买一本中科大出版的《常用积分表》，或者利用在线积分计算工具来帮助你比较好。

另外很多稍复杂的微积分公式，多是从简单积分公式出发，然后依靠换元和链式法则求解，有兴趣的朋友可以自行尝试验证。


# 关于常数的积分

$$
\int a dx = ax
$$

# 关于 $\frac{1}{x}$ 的积分

$$
\int \frac{1}{x} dx = \ln |x|
$$

$$
\int \frac{1}{1 + x^2} dx = \arctan x
$$

$$
\int \frac{1}{ax + b} dx = \frac{1}{a} \ln |ax + b|
$$

$$
\int \frac{1}{x^2 + a^2} dx = \frac{1}{a} \arctan \frac{x}{a}
$$

$$
\int \frac{1}{x^2 - a^2} dx = \frac{1}{2a} \ln \left | \frac{x -a }{x + a} \right |
$$

$$
\int \frac{1}{(x + a)(x + b)} dx = \frac{1}{b - a} \ln \frac{a + x}{b + x}
$$

# 关于 $\frac{1}{\sqrt{x}}$ 的积分

$$
\int \frac{1}{\sqrt{1 - x^2}} dx = \arcsin x
$$

$$
\int \frac{1}{\sqrt{x^2 \pm a^2}} dx = \ln \left | x + \sqrt{x^2 \pm a^2} \right | 
$$

$$
\int \frac{1}{\sqrt{a^2 - x^2}} dx = \arcsin \frac{x}{a}
$$

# 关于 $x^n$ 的积分

$$
\int x^n dx = \frac{x^{(n+1)}}{n+1}
$$

$$
\int ax^{n} dx = \frac{ax^{(n+1)}}{n + 1}
$$

# 关于 $a^x$ 的积分

$$
\int a^x dx = \frac{a^x}{\ln a}
$$

$$
\int b^{ax} dx = \frac{b^{ax}}{a \ln b}
$$



# 关于 e 的积分

$$
\int e^x dx = e^x
$$

$$
\int x e^{x} dx = (x -1) e^x
$$

$$
\int e^{ax} dx = \frac{e^{ax}}{a}
$$

$$
\int x e^{ax} dx = (\frac{x}{a} - \frac{1}{a^2}) e^{ax}
$$

$$
\int x^n e^{ax} dx = \frac{x^n e^{ax}}{a} - \frac{n}{a} \int x^{(n-1)} e^{ax} dx
$$

# 关于 $x$ 的函数 $\mathbf{u}$、$\mathbf{v}$ 的积分

$$
\int \mathbf{u} d \mathbf{v} = \mathbf{uv} -  \int \mathbf{v} d \mathbf{u}
$$

$$
\int (\mathbf{u} + \mathbf{v}) dx = \int \mathbf{u} dx + \int \mathbf{v} dx
$$

# 关于三角函数的积分

## 正弦函数 $\sin$ 的积分
$$
\int \sin x dx = - \cos x
$$

$$
\int \sin ax dx = - \frac{1}{a} \cos ax
$$

$$
\int \sin^2 ax dx = \frac{x}{2} - \frac{\sin 2a x}{4a}
$$

## 余弦函数 $\cos$ 的积分
$$
\int \cos x dx = \sin x
$$

$$
\int \cos ax dx = \frac{1}{a} \sin ax
$$

$$
\int \cos^2 axdx = \frac{x}{2} + \frac{sin 2ax}{4a}
$$

## 正切函数 $\tan$ 的积分

$$
\int \tan x dx = - \ln |\cos x|
$$

## 余切函数 $\cot$ 的积分「$\tan^{-1}$」

$$
\int \cot x dx = \ln |\sin x|
$$

## 正割函数 $\sec$ 的积分「$\cos^{-1}$ 」
$$
\int \sec xdx = \ln |\sec x + \tan x|
$$

$$
\int \sec^2 x dx = \tan x 
$$

## 余割函数 $\csc$ 的积分「$\sin^{-1}$」

$$
\int \csc xdx = \ln |\csc x - \cot x|
$$

$$
\int \csc^2 x dx = -\cot x  
$$
















