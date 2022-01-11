
@[toc]

# 1. 二极管

普通的二极管的定义是正向导通，反向截止，和电阻器相似，由于设计物理结构不同，或掺杂稀土材料不同，在普通的二极管基础上，还衍生出了诸如光电二极管、齐纳二极管等特殊的二极管。

先说比较常见的几种：

符号 |  名称 |  说明
------|---------|---------
![在这里插入图片描述](https://img-blog.csdnimg.cn/84a3f4f6836b4862a78a7927162d6831.gif#pic_center) | 普通二极管（Diode） | 对电流能够单向截止，比如防止电流倒灌，电路保护，整流中比较常见。
![在这里插入图片描述](https://img-blog.csdnimg.cn/59b3b487ade8412389b7fc15e52f8e55.gif#pic_center) | 齐纳二极管（Zener Diode）| 又称稳压二极管，尽管它也具备普通二极管一样的单向导通能力，但是由于反接时能够在一定电压下出现崩溃效应，从而使供电电压能够维持在一定合理范围内，对电脑也可以起到保护作用。
![在这里插入图片描述](https://img-blog.csdnimg.cn/f9ff57905c6143e5b765d97d7e7c6a5e.gif#pic_center) | 发光二极管（Light Emitting Diode）| 与普通二极管相比，它的耐压值普遍不高，但是当通电时会发光。所以可作为照明、或者电路中工作状态指示灯使用。
![在这里插入图片描述](https://img-blog.csdnimg.cn/7dd359a1b8e14dcf8633ea60e5543633.gif#pic_center) | 光电二极管（Photodiode）| 它是一种能够将光根据使用方式，转换成电流或者电压信号的光探测器。常见的传统太阳能电池就是通过大面积的光电二极管来产生电能。

光电二极管相对来说比较特殊一些，它出现的位置通常会配合发光二极管，一起组成光耦合器，对电路起保护作用。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b250f170f3694d3890c6d376777db4bb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

下面就聊一些比较特殊，但很常见的二极管

## 1.1. TVS二极管
TVS管也叫 **瞬变电压抑制二极管**，它的作用和稳压二极管相似，只不过它的响应速度要超过稳压二极管，所以多用于精密电路的供电输入端，起到对浪涌的抑制作用。比方说对于下面这个电路，

![在这里插入图片描述](https://img-blog.csdnimg.cn/52af3c2b41434fa586256271566cdb62.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


**优点：**

* 响应时间快：相应时间可以达到ps级，是限压型浪涌保护器件中最快的。
* 结电容低：根据其制造工艺，大致可分为两种类型，高结电容TVS一般在几百-几千pF的数量级，低结电容TVS一般在几pF-几十pF的数量级。一般分立式TVS结电容较高，表贴式TVS结电容较低。在高频信号线路中，应主要选用较低结电容的TVS管。
* 非线性特性比压敏电阻好：当通过TVS管的过电流增大是，TVS管的钳位电压上升速度比压敏电阻慢，因此可以获得比压敏电阻更低的残压输出，将浪涌限制在较低的电压。
* 反向漏电流较低：一般为uA级别，几uA~几百uA之间。

**缺点：**
* 反向击穿电压较低：相对于压敏电阻和气体放电管及其它浪涌保护器件而言。
* 通流容量低：在限压型浪涌保护器件中是最小的，因此一般只能用于最末级的精细保护[^1]。

[^1]: https://blog.csdn.net/DIYfashaoyou/article/details/62486342

由于并没有强制规定TVS管和齐纳管的区别，所以多数情况下它们都公用相同的电路符号，因此区别它们的主要方法就是在电路原理图上注意型号的不同。另外TVS除了单极型结构外，还有双极型

![在这里插入图片描述](https://img-blog.csdnimg.cn/b95b3b5a3d7a455192b7ee41f8d94624.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

双极型TVS管的功能等于两个单极型TVS管串联一起。

## 1.2. 肖特基二极管（Schottky diode）

由于硅管都存在充电和放电时间，而普通二极管的充电时间过长导致它无法有效的在高速电路上起到快速状态切换。所以后来物理学家肖特基提出了一种适用在 **低压**、**高速** 电路中的二极管，并且取名叫肖特基二极管。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e343f5e50a52403a8307c35f1bc3ea3c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
肖特基二极管的电路符号和Zener二极管很相似，但注意它右侧的符号是个S。肖特基二极体的导通电压非常低。一般的二极管在电流流过时，会产生约 0.7-1.7 伏特的电压降，不过肖特基二极管的电压降只有 0.15-0.45 伏特，因此可以提升系统的效率。

一般二极管的反向恢复时间大约是数百nS，若是高速二极管则会低于一百 nS，肖特基二极管没有反向恢复时间，因此小信号的肖特基二极管切换时间约为数十 pS，特殊的大容量肖特基二极管切换时间也才数十 pS。由于一般二极管在反向恢复时间内会因反向电流而造成EMI噪声。

肖特基二极管可以立即切换，没有反向恢复时间及反相电流的问题[^2]。

[^2]: https://zh.wikipedia.org/wiki/%E8%82%96%E7%89%B9%E5%9F%BA%E4%BA%8C%E6%9E%81%E7%AE%A1

![在这里插入图片描述](https://img-blog.csdnimg.cn/a4838f4ca7e7476b9bd5194a3c046409.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 1.3. 隧道二极管（Tunnel Diode）

又称江崎二极管、穿隧效应二极管、穿隧二极管、透纳二极管，是一种可以高速切换的二极管，其切换速度可到达微波频率的范围，其原理是利用量子穿隧效应。

此种二极管是由高掺杂的PN接面所形成（空乏区通常只有10奈米宽），常用的材料包括锗、砷化镓等窄能隙的材料，由于高掺杂会产生晶格的破坏，使得能隙间的缺陷变多，加上窄能隙材料缩小量子穿隧的障碍，所以能够增加量子穿隧的电流。隧道二极管常用于频率转换器和侦测器上，由于隧道二极管的负微分电阻的特性，其也可应用于振荡器、放大器以及开关电路的迟滞[^3]。

[^3]: https://zh.wikipedia.org/wiki/%E9%9A%A7%E9%81%93%E4%BA%8C%E6%A5%B5%E9%AB%94

![在这里插入图片描述](https://img-blog.csdnimg.cn/efa5e7f29ce04db498058f12ffa4c058.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
其电路符号为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2bd3066e30be4c10bc6a7ceec855eb8e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 1.4. 变容二极管

变容二极管(Varactor Diodes)又称 "**可变电抗二极管**"，在电路中被当作可变电容使用。可在高频电路中用作自动调谐、调频、调相等。

其电路符号为

![在这里插入图片描述](https://img-blog.csdnimg.cn/f7beee8ea201429a9f5d60eb5cd7a553.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

那么，如何理解变容二极管是如何工作的：

首先，在不增加反向电压的时候，它等效于

![在这里插入图片描述](https://img-blog.csdnimg.cn/5187cdfbdb104af18372b3ec19dfffea.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

如果此时增加一点反偏电压时，它就会减少PN结中的电容量，等效于电容空隙增大[^4]
![在这里插入图片描述](https://img-blog.csdnimg.cn/3717e40dac934368bac0bd50f8e930bd.png#pic_center)

如果继续增大电压，“电容板空隙继续增大”

![在这里插入图片描述](https://img-blog.csdnimg.cn/95c9fc1abd1e4355824d39ea300289c1.png#pic_center)

这就是为什么  **变容二极管** 其实等效于可调电容的原因。

[^4]: https://baike.baidu.com/item/%E5%8F%98%E5%AE%B9%E4%BA%8C%E6%9E%81%E7%AE%A1/10864951
