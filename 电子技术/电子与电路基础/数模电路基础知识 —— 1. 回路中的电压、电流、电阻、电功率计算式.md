@[toc]

# 1. 电流 $I$、电压 $V$、电阻 $R$、电功率 $W$ 的计算公式

在 **纯直流电路** 中，电流、电压、功率有如下关系


$$
P = U \times I = I^2 \times R = \frac{U^2}{R} \\
U = I \times R
$$

在 **单相交流电路** 中，电流、电压、功率有如下关系

$$
P = U I \cos \phi
$$

其中，$\phi$ 是 [功率因数](https://baike.baidu.com/item/%E5%8A%9F%E7%8E%87%E5%9B%A0%E6%95%B0/1016945)。对于类似白织灯、电阻炉等电阻负荷的功率因数为1。而在对称三相交流电路中，不论负载的连接是哪种形式，对称三相负载的平均功率都是：

$$
P=\sqrt 3 \cdot UIcos \phi
$$

# 2. 串联电路和并联电路的功率、电压关系

# 2.1. 串联电路
> 在串联电路中电流相等，电压和功率等于各用电器之和

电流
$$
I_{circuit} = I_1 = I_2 = \cdots = I_n
$$

电压
$$
U_{circuit} = U_1 + U_2 + \cdots + U_n
$$

电阻
$$
R_{circuit} = R_1 + R_2 + \cdots + R_n
$$

电功率
$$
P_{circuit} = P_1 + P_2 + \cdots + P_n 
$$



# 2.2. 并联电路
> 在并联电路中电压相等，电流和功率等于各用电器之和

电压
$$
U_{circuit} = U_1 = U_2 = \cdots = U_n 
$$

电流
$$
I_{circuit} = I_1 + I_2 + \cdots + I_n 
$$

电阻

$$
R_{circuit} = \frac{R_1 R_2 \cdots R_n}{R_1 + R_2 + \cdots R_n} 
$$

电功率
$$
P_{circuit} = P_1 + P_2 + \cdots + P_n
$$