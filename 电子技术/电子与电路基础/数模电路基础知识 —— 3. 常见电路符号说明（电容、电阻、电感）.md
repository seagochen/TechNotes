
@[toc]

# 1. 电容
电容是电路中的三大基本元件，其符号通常标识为 **C**，单位是 **法拉**。按电容的按材料和设计，一般有「无极性电容」、「有极性电容」、「可调电容」、「可变电容器」这几种不同类型。


符号 |  名称 |  说明
------|---------|---------
![在这里插入图片描述](https://img-blog.csdnimg.cn/9f2bd0511d004018b3514633b4abd150.gif#pic_center)  | 有极性电容（Polarised Capacitor） | 比较常见而且便宜的电容，电荷容量较大，电容是有极性的，当电极相反时能够承受一定的反向电压，但是设计电路时应该尽量避免出现这样的情况，或换上无极性电容。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/9b986353e50b4674a55713f6e92ac2ca.gif#pic_center) | 无极性电容 （Unpolarised Capacitor）| 比较常见但是相对较贵，电荷容量通常较小，电容引脚无极性限制。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/8597b1eeada842acb4049e55d9c4b08b.gif#pic_center) | 可调电容 (Variable Capacitor) | 不是很常见的电容，主要应用在模拟电路中，通常当作可调频滤波器，一个比较典型的应用是收音机。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/b5924babe87a402db05fdf6a22a1812c.gif#pic_center) | 可变电容器 (Trimmer Variable Capacitor) | 不是很常见的电容，主要应用在数电中，通常有三个引脚，其中一个是用电压调整电容量的引脚。


我们接下来看看比较常见的「无极性电容」和「有极性电容」都有哪些。

## 1.1. 无极性电容
无极性电容，在电路中不用担心电容的正反是否接错，只需要考虑电路的电压与电容的耐压值是否在合理区间。对于THT封装类型的无极性电容，常见的介质有陶瓷、水泥等。

![在这里插入图片描述](https://img-blog.csdnimg.cn/bef2d3c7079c47cfb88e4ceabf450d30.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
上面这图的右侧是有极性电容。而在集成电路中，用到的贴片电容可以有无极性的陶瓷电容，比如下面这个样子

![在这里插入图片描述](https://img-blog.csdnimg.cn/1f380484517c4bc4aea6e1a60156f191.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
无极性如果不考虑它的电容量，其实是非常理想的组件。但是对于需要大电容的电路来说，同等电容量的无极性电容通常更贵，而且也没有这种必要。

由于无极性电容的充放电速度非常快，它的这一特性正好可以满足高速数字电路的需要，所以在数字电路的设计中，它更多的出现在芯片附近，用作芯片信号的去耦或者旁路。


## 1.2. 有极性电容

有极性电容的应用相对无极性电容更为常见，这主要由两个原因导致，一个是有极性电容的电容量相对较大，二是成本相对较为低廉。在我们的电路设计中，最常见的就是这种铝电解电容。

![在这里插入图片描述](https://img-blog.csdnimg.cn/01ec6390a7114f6da87911c6e3b525a4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
由于极性电容，当电压反接时不是 **起火** 就是 **爆炸**，尤其对应大容量的电容反接时相当危险，所以使用极性电容首要的就是弄懂它的引脚定义。通常对于这种有塑料外壳包装的电解电容，会在对应地方标识 **「负引脚」**。

![在这里插入图片描述](https://img-blog.csdnimg.cn/174457c7b19c48dfba18a1223699e2dc.png#pic_center)


但是钽电容则是反过来的，它会标识 **「正引脚」**，知乎[^1]上有人给出了一张图片标识各种常见钽电容的电极

![在这里插入图片描述](https://img-blog.csdnimg.cn/21feb0ae4e7e42ec9b7da026e3a972b9.png#pic_center)
[^1]: https://www.zhihu.com/question/352398570

此外，如果对于THT封装的电容，也可以通过引脚的长短进行判断，通常遵循 **「长正短负」**。

![在这里插入图片描述](https://img-blog.csdnimg.cn/5548e81ec6374cdda05f083d0041ebb2.png#pic_center)


# 2. 电阻
电阻是基本电路中唯一的功耗元件，起到对电路的保护和分压作用，电路符号是 **R**，单位是 **欧姆**。另外根据其材料和使用方式不同，它还有「普通电阻」，「可变电阻器」，「电位计」，「预设变量电阻器」

符号 |  名称 |  说明
------|---------|---------
![在这里插入图片描述](https://img-blog.csdnimg.cn/f5f7448901be4a6fbcc192428df2ad18.gif#pic_center) | 电阻（Resistor） |  最常用的基本电阻形式
![在这里插入图片描述](https://img-blog.csdnimg.cn/dbd638b5b50e429e9b9cbc0a5fc260bc.gif#pic_center) | 可变电阻器（Rheostat Variable Resistor） | 一个变阻器通常两个触点，通常用于控制电流。用途包括控制灯的亮度或电机速度，并在定时电路中改变进入电容器的电荷流动速率。
![在这里插入图片描述](https://img-blog.csdnimg.cn/7785a7e8ce2d4064bd16b93477996e11.gif#pic_center) | 电位计（Potentiometer Variable Resistor） | 一个电位计有3个触点，通常用于控制电压。它可以这样用作传感器，将位置（控制主轴的角度）转换为电信号。
![在这里插入图片描述](https://img-blog.csdnimg.cn/f7924f54ff2146a1a5411b2b1722f23a.gif#pic_center) | 预设变量电阻器（Preset Variable Resistor） | 预设用小螺丝刀或类似工具操作。它被设计为设置当电路，然后离开没有进一步调整。预设比标准变阻便宜，因此有时用于项目以降低成本。

对于上面的2、3、4，在实际中经常混用，由于它们都可以通过旋钮或者螺丝刀调整电阻阻值，并在电路中可用于反馈电路，所以并没有特别大的区别。

## 2.1. 特殊电阻器

### 2.1.1. 电热丝或发热器

稍微一提的是，电阻作为功率器件，本身会消耗电路中的电能并发热，所以也有直接使用这一特性的特殊电阻——电热丝。通常它的电路符号是

符号 |  名称 |  说明
------|---------|---------
![在这里插入图片描述](https://img-blog.csdnimg.cn/02f6654db76c4790996d3428348e4422.gif#pic_center) | 加热器（Heater）| 单纯利用发热特性的特殊电阻。

不过在一些EDA，它也有使用下面这样符号进行表示的。


![在这里插入图片描述](https://img-blog.csdnimg.cn/257a9d407bed4e078c32da4cc27650a9.png#pic_center)
**不过需要知道一点的是，在美标中，该符号同时也是电阻的符号。**

它与电感有些相似，但电感符号是圆润的波浪线，而不是这种带尖刺的折线。除了这一类外，还有利用熔断制作的保险电阻器，以及掺杂了其他元素而制成的特殊半导体电阻器。

### 2.1.2. 保险电阻器
保险电阻：又叫熔断电阻器，在正常情况下起着电阻和保险丝的双重作用，当电路出现故障而使其功率超过额定功率时，它会像保险丝一样熔断使连接电路断开。保险丝电阻一般电阻值都小（0.33Ω~10KΩ），功率也较小。保险丝电阻器常用型号有：RF10型、RF111-5保险丝电阻器的符号型、RRD0910型、RRD0911型等。

![在这里插入图片描述](https://img-blog.csdnimg.cn/99458b9bef044659b793cf632839a3b7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_11,color_FFFFFF,t_70,g_se,x_16#pic_center)
以及在集成电路中比较常见的热熔断保险丝


![在这里插入图片描述](https://img-blog.csdnimg.cn/c26cbb5f4ca247c7bb411a473eadb581.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
至于它的电路符号，通常有下面几种样式

![在这里插入图片描述](https://img-blog.csdnimg.cn/01a4f9081a33413180e750d60d07c077.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

### 2.1.3. 敏感电阻器

敏感电阻器：是指其电阻值对于某种物理量（如温度、湿度、光照、电压、机械力、以及气体浓度等）具有敏感特性，当这些物理量发生变化时，敏感电阻的阻值就会随物理量变化而发生改变，呈现不同的电阻值。根据对不同物理量敏感，敏感电阻器可分为热敏、湿敏、光敏、压敏、力敏、磁敏和气敏等类型敏感电阻。敏感电阻器所用的材料几乎都是半导体材料，这类电阻器也称为半导体电阻器。

热敏电阻的阻值随温度变化而变化，温度升高阻为负温度系数（NTC）热敏电阻。应用较多的是负温度系数热敏电阻，又可分为普通型负温度系数热敏电阻；稳压型负温度系数热敏电阻；测温型负温度系数热敏电阻等。光敏电阻是电阻的阻值随入射光的强弱变化而改变，当入射光增强时，光敏电阻的减小，入射光减弱时电阻值增大[^2]。

[^2]: http://m.elecfans.com/article/1554618.html

![在这里插入图片描述](https://img-blog.csdnimg.cn/ee6760715e854bbd903246057653cfc0.png#pic_center)


由于掺杂稀土材料，使得这一类电阻器有很多种，难以全部罗列出来，通常我们在电路中会用到下面几种常用的：

**压敏电阻**

![在这里插入图片描述](https://img-blog.csdnimg.cn/a97bcb5379ca4b72a4c2c2d92103aec8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_11,color_FFFFFF,t_70,g_se,x_16#pic_center)



**光敏电阻**

![在这里插入图片描述](https://img-blog.csdnimg.cn/6a4c5318de1644f096482fcd1fe67883.png#pic_center)


**热敏电阻**


![在这里插入图片描述](https://img-blog.csdnimg.cn/c6e2574c5261445d9f6f1ebf773bd3a8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_11,color_FFFFFF,t_70,g_se,x_16#pic_center)




# 3. 电感

电感是另外一大储能元件，电路符号是 **L**，单位是 **亨**。电感是金属线圈缠绕所制作的元件，所以天然的会因电磁感应而产生与突变电势相反的电势，所以具备“**存储电势能**”，以及与电容正好相反的“**隔交通直**”的特性。


符号 |  名称 |  说明
------|---------|---------
![在这里插入图片描述](https://img-blog.csdnimg.cn/06d839098c7443a9a5cbfa206e6db82b.gif#pic_center) | 电感 （Inductor）| 电势能储能元件、滤波元件。

电感不存在极性，所以相对来说较为简单。在交流电路中，两股绕线不一的电感彼此靠近可以组成一个变压器。常见的电感有以下这些

![在这里插入图片描述](https://img-blog.csdnimg.cn/775a3e4e00d748e2ac2965d9872fdcff.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/add8c98fbd5d4217a30422193a15673f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_14,color_FFFFFF,t_70,g_se,x_16#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/10a4dde216a04d10ba1c13d7b0c033f4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
由于电感会产生较强的电磁场，对于一些精密元件的信号会造成干扰，所以在设计时应该把电感布置离这一类元件稍微远一点的地方，或者在比如精密芯片外面贴上一层金属外衣做电磁屏蔽处理。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e67e3759911b483cb02711ed1f4c65e2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


