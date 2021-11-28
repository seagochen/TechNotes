@[toc]

# 3A、24V、500kHz、高效率同步降压变换器
MP2315S 是一款内置功率 MOSFET 的高效率同步整流降压开关变换器。它是 MP2315 的下一代产品。它提供了非常紧凑的解决方案，在宽输入范围内可实现 3A 连续输出电流，具有极好的负载和线性调整率。MP2315S 在输出负载范围内采用同步整流模式工作以达到高效率。电流控制模式提供快速瞬态响应，并使环路更易稳定。全方位保护功能包括过流保护（OCP）、过压保护（OVP）和过温关断保护。MP2315S 最大限度地减少了现有标准外部元器件的使用，采用 TSOT23-8 封装。

MP2315是一种 **高频同步整流降压开关模式转换器**，内置功率MOSFET。提供了一个非常紧凑的解决方案，以实现 **3A** 连续输出电流在一个广泛的输入电源范围内 **具有良好的负载和线路调节**。

# 主要功能 
简而言之，它是一种DC2DC的电源转换模块，可以提供最大3A的输出，具备快速的瞬态响应能力的一颗电源IC。根据在官网的介绍，这颗芯片的特性如下[^1]：

[^1]: https://www.monolithicpower.cn/cn/mp2315s.html

* 宽工作输入电压范围：4.5V 至 24V
* 3A 负载电流
* 110mΩ/55mΩ 低导通阻抗内部功率 MOSFET
* 低静态电流
* 高效同步工作模式
* 500kHz 固定开关频率
* AAM 节电模式
* 内部软启动
* 输出过压保护（OVP）
* 过流保护（OCP）自动恢复功能
* 过温关断保护
* 输出电压可调节低至 0.8V
* 采用 TSOT23-8 封装



**注意：**

**不过根据一些第三方的测试描述，这颗芯片如果按照设计的最大功率运作，会很快升温到100多度从而导致热关机保护，所以需要加上散热模块。**


# 主要参数
典型的MP2315S，包含8个引脚。从手册里功率曲线特性图可以看到，处理输入电压为24V-12V，输出5V/1A左右的时候可以达到最大的转换效率，所以朋友们在设计电路的时候可以注意这一点。

**过高负载会让这颗芯片发热严重，从而导触发热保护停机。所以想要处理更大的电压和电流，应考虑其他方案，或者加装散热模块。**


![在这里插入图片描述](https://img-blog.csdnimg.cn/3bb3a77103884db488369edf5102a2a6.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 关键性能指标

![在这里插入图片描述](https://img-blog.csdnimg.cn/5422fcc679254b7597ab9a61d51a87a2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 引脚定义

NAME | PIN | I/O | Description
---------|-------|-------|---------------
AAM	| 1  |  - | 	通过电阻引脚接地。且当负载较小时，MP2315S将被设置为非同步模式。如果接入VCC时，将迫使MP2315S进入CCM模式。
IN		| 2 | - | 		电源电压。推荐输入4.5V至24V的电压。需要用使用电容来对输入电压进行解耦，并且铺线时应该使用较宽的连线保证供电。
SW		| 3 | -  | 	开关输出，即供电输出，和供电一样，也应该使用较宽的连线保证输出品质。
GND	| 4 | - | 		GND接线
BST 	| 5 | - | 		Bootstrap，在SW和BST构成的回路之间应该连接一个电容和一个 10Ω 的电阻。这样可以基于SW产生的PWM信号上生成一个高频的开关电源。
EN/SYNC 	| 6 | - | 		启用或关闭输出，所以通常6脚会通过一个分压电容直接连到VCC或者IN上。
VCC 	| 7 | - | 		内部偏置供应，内部5.1VLDO输出。用0.1μF-0.22μF电容器解耦VCC。电容不应超过0.22μF。
FB 		| 8 |  - | 	反馈引脚，通常SW会通过分压电阻接入到FB引脚，从而调节输出电压。当FB电压低于400mV时，频率折叠比较器会降低振荡器频率，以防止短路故障条件下电流限制失控。


# 接线方法

## 输入6V~24V 输出5V/3A

![在这里插入图片描述](https://img-blog.csdnimg.cn/4c4bae69b5e347929c4d5f0cec9e3935.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 输入4.5V~24V 输出3.3V/3A

![在这里插入图片描述](https://img-blog.csdnimg.cn/880d7a56bd3d429380357eeb5fe2cf41.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 输入4.5V~24V 输出2.5V/3A

![在这里插入图片描述](https://img-blog.csdnimg.cn/92b06b0508f642458471ebb620f0bd19.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 输入4.5V~24V 输出1.8V/3A

![在这里插入图片描述](https://img-blog.csdnimg.cn/72c11c2fa81d487a9fbc81a6006b0f30.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 输入4.5V~24V 输出1.2V/3A
![在这里插入图片描述](https://img-blog.csdnimg.cn/94d9bf4925874f10b4e62ade6df8d34c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 输入4.5V~24V 输出1.05V/3A

![在这里插入图片描述](https://img-blog.csdnimg.cn/a601ed7803d24698b6425af3b3bb6343.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


可以看到，基本上芯片左侧的元器件并不需要多少修改，而我们重点需要关注L1，R1，R2，R6的数值。因为这三个主要负责调节反馈电压和输出的。

# 元件选取
以上是厂商给出的配置，如果你需要自己做一些调整，那么我们可以看看数据手册里元件选取这一部分章节。

## 设置电压输出
输出电压大小，由分压电阻决定。其计算公式如下：

$$
R_2 = \frac{R_1}{V_{out} / 0.791V - 1}
$$

并且官方强烈建议FB引脚的布线方式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/4745b11f90a54fc2bacdaaa9566775d2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
并且提供了一张常见电压输出的电阻表，对于常见的从5V-1.05V，都可以参考下表

![在这里插入图片描述](https://img-blog.csdnimg.cn/da8a6662faa74a249b1bae215f30459a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 选择电感L1

由于SW输出的是PWM方波，我们是通过控制方波占空比来调制电压，所以作为储能器件的电感的很大程度决定了我们生成电压的品质情况。官方的建议是：**对于大多数应用，建议使用直流额定电流至少比最大负载电流高25%的1µH到10µH电感器。为了效率最高，电感器的直流阻抗应小于20mΩ。**

其计算公式为：

$$
L_1 = \frac{V_{out} \times (V_{in} - V_{out})}{V_{in} \times \Delta I_L \times f}
$$

这里，$\Delta I_L$ 指纹波电流。选择电感器电流约为最大负载电流的40%。最大电感器峰值电流可用下式计算：

$$I_{L(MAX)} = I_{LOAD} + \frac{\Delta I_L}{2}$$

**官方还建议，在100mA以下的轻负载下，建议使用更高的电感以提高效率。**

## AAM电压设置

AAM电压用于设置从AAM到PWM的过渡点。选择它应提供效率、稳定性、涟漪和瞬态的最佳组合。如果AAM电压设置为较低，则稳定性和纹波提高，但AAM模式和瞬态效率降低。同样，如果AAM电压设置为高，则AAM模式和瞬态效率提高，但稳定性和纹波降低。通过将分压器电阻从VCC连接到GND来调整AAM阈值。请注意，在AAM上有一个6.7µA的电流源。

![在这里插入图片描述](https://img-blog.csdnimg.cn/8a10bf7d3d3d4f6083c06029f115b12f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

$V_{AAM}$ 电压也是可以计算的

$$
V_{AAM} = \frac{R_5 \times (VCC + 6.7uA \times R_4)}{R_4 + R_5}
$$

另外R5 的阻值不应该超过20K。

## 选择输入电容

降压转换器的输入电流不连续，因此需要电容器在保持直流输入电压的同时向降压变流器提供交流电流。使用低ESR电容器以获得最佳性能。由于X5R或X7R介质的温度系数低，强烈推荐具有X5R或X7R介质的陶瓷电容器。**对于大多数应用，一个22µF电容器就足够了。由于输入电容器（C1）吸收输入开关电流，因此它需要足够的纹波电流额定电流。** 输入电容中的均方根电流可以用方程来估计。

$$I_{C1} = I_{LOAD} \times \sqrt{\frac{V_{OUT}}{V_{IN}} \times (1 - V_{OUT} / V_{IN})}$$

手册中还提到，比较极端的情况就是输入电压为输出电压的2倍。即，$V_{IN} = 2V_{OUT}$，此时的电流公式可以简化为：

$$I_{C1} = \frac{I_{LOAD}}{2}$$

## 选择输出电容
尽管我们使用了电感调节了输出，但是如果缺少了电容，也会导致输出电压的不稳定。所以输出电容的作用就是稳定输出电压。**推荐使用陶瓷、钽或低ESR电解电容器。建议使用低ESR电容器，以保持输出电压纹波较低。** 输出电压纹波可以用方程来估计

$$
\Delta V_{OUT} = \frac{V_{OUT}}{f \times L_1} \times (1 - \frac{V_{OUT}}{V_{IN}}) \times (R_{ESR} + \frac{1}{8 \times f \times C_2})
$$

对于陶瓷电容器，开关频率上的阻抗由电容为主。输出电压的纹波主要是由电容引起的。为了简化，可以用公式估计输出电压纹波。

$$\Delta V_{OUT} = \frac{V_{OUT}}{8 \times f^2 \times L_1 \times C_2} \times (1 - \frac{V_{OUT}}{V_{IN}})$$

在钽或电解电容器的情况下，ESR在开关频率上主导阻抗。因此可以进一步简化为

$$\Delta V_{OUT} = \frac{V_{OUT}}{f \times L_1} \times (1 - \frac{V_{OUT}}{V_{IN}})  \times R_{ESR}$$

## 外部驱动二极管
外部引导二极管可以提高调节器的效率。外部BST二极管的适用条件为：

* Vout 在5V-3.3V之间
* 任务效率 $D = \frac{V_{OUT}}{V_{IN}} > 65\%$

此时可以用1N4148做外部二极管，但是我们为简化电路可以不设计这个部分。

![在这里插入图片描述](https://img-blog.csdnimg.cn/6179d0dfaaaf4a9aa437a56624c24fa4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



# 数据手册

更多细节，请参考官方数据手册：https://www.monolithicpower.cn/cn/mp2315s.html