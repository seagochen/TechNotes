@[toc]

# 0. 什么是三极管

说到三极管，它其实是一大类特制的晶体管。由于掺杂和微结构不同，比如NPN型、PNP型、N沟道型、P沟道型，所以往往根据它的结构和功能做了一个区分，比如 **「双极性晶体管（Bipolar Junction Transistor）」**、**「场效应晶体管 （Field Effect Transistor）」** 等不同子类。

在电路当中，广义的三极管通常被用于信号放大（电压、电流放大）、开关、稳压、信号调制等许多功能。常见实物形状一般如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/fd3a14ed5f1f4e11ac806416ba36af57.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 1. 双极性晶体管（BJT）

双极性晶体管是第一种量产的晶体管，它是由二种不同接面的二极管组成，其结构可分为二层N型半导体中间夹一层P型半导体的NPN晶体管，以及二层P型半导体中间夹一层N型半导体的PNP晶体管。因此会有二个PN结，分别是 **基极-发射结** 及 **基极-集电结**，中间隔着一层的半导体，即为基极[^1]。

[^1]: https://zh.wikipedia.org/wiki/%E6%99%B6%E4%BD%93%E7%AE%A1

![在这里插入图片描述](https://img-blog.csdnimg.cn/f8be23887c7446b793213c3b34729e98.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_13,color_FFFFFF,t_70,g_se,x_16#pic_center)

对于双极性晶体管常用的电路符号如下：

符号 |  名称 |  说明
------|---------|---------
![在这里插入图片描述](https://img-blog.csdnimg.cn/e40ca34122d74a2b939fa92c9b348783.gif#pic_center) | NPN型BJT管 | 晶体管放大电流，并可与其他组件一起使用，以构成放大器或开关电路。
![在这里插入图片描述](https://img-blog.csdnimg.cn/cbeaedb6e91e49eeafd94fc32ec630e7.gif#pic_center) | PNP型BJT管 | 晶体管放大电流，并可与其他组件一起使用，以构成放大器或开关电路。
![在这里插入图片描述](https://img-blog.csdnimg.cn/e2402244ba624a21bf1f9dc3652fe613.gif#pic_center) | NPN型光电晶体管（Phototransistor） | 比较特殊的一类晶体管，除了具备BJT管的基本特性外，它还有对外部光照强度的敏感性，具备随着光照强度改变电流大小的能力。


BJT管的三个引脚，分别是 **发射极（Emitter）**、**集电极（Collector）** 与 **基极（Base）**，关于BJT管的工作原理，我会放在随后的章节里进行介绍。这里先介绍一些比较特殊的一类BJT管，又称光电晶体管：

![在这里插入图片描述](https://img-blog.csdnimg.cn/40dcce70777d402cb3aa89edb03de187.png#pic_center)

由于它不需要在电路中接入基极，对基极的控制完全交由光电感应模块负责，所以这类晶体管的外观很像二极管，而它们的电路符号也通常表示成下面这样：
![在这里插入图片描述](https://img-blog.csdnimg.cn/9bb3d6f0eb8740ae88c0153302a102f8.png#pic_center)
多说一句，这类晶体管一般出现在光耦合组件里，它的一个电路符号和实体是下面这样，

![在这里插入图片描述](https://img-blog.csdnimg.cn/f27928e6b05148ddb4660afa778f3bf9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

即由发光二极管和光电晶体管组成的元件。


# 2. 场效应晶体管（FET）
另一大类晶体管便是**场效应晶体管**。它是利用电子（N沟道FET）或是空穴（P沟道FET）导通电流。与BJT的管对应三个引脚的叫法不同，场效应晶体管的三个引脚分别是 **栅极（Gate）**、**漏极（Drain）**、**源极（Source）** ，而对于 **金属氧化物半导体场效应管**，通常其内部还会集成一个 **基极（Base）**，但是对外引出的通常还是三个引脚的形式。

在场效应晶体管中，**源极** 与 **漏极** 的电流会流过连接彼此之间的 **沟道**，导通程度会依 **栅极** 和 **源极** 之间的电压产生的电场而定，因此可以利用电压控制源漏极电流，做为一个 **简单的开关**，或者做 **信号放大器**。

由于FET管对沟道的控制电压反应极为灵敏，而它们的价格通常也比BJT管更贵一些，所以通常信号放大的相关应用中更常见BJT管（例如运算放大器），而在高频数字信号的开关电路中，使用FET管。

FET管又根据制作工艺可以分为两种：分别是结型场效应管（Junction Gate Field-Effect Transistor）及绝缘栅极场效晶体管（Insulated Gate Field Effect Transistor）


[^2]: https://zh.wikipedia.org/wiki/%E6%99%B6%E4%BD%93%E7%AE%A1

## 2.1. 结型场效应管（JFET）

结型场效应管常见的有两种符号

符号 |  名称 
------|---------
![在这里插入图片描述](https://img-blog.csdnimg.cn/b074988f682949ec9c228ac0daae7816.png#pic_center) | P-沟道结型场效应管 （P-Channel JFET）
![在这里插入图片描述](https://img-blog.csdnimg.cn/81f241937efd4b319d433230c943534f.png#pic_center) | N-沟道结型场效应管 （N-Channel JEFT）

除此以外，可能还会见到这种符号

符号 |  名称 
------|---------
![在这里插入图片描述](https://img-blog.csdnimg.cn/082ec7bb29ff40dbbf11532a2d79842b.png#pic_center) | P-沟道JEFT |
![在这里插入图片描述](https://img-blog.csdnimg.cn/e056592c49254e2b9d7e0c0196b9a060.png#pic_center) | N-沟道JEFT |
功能上并没有什么区别，不过习惯上建议还是选择用带圈的符号。



## 2.2. 绝缘栅极场效晶体管（IGFET）
至于说绝缘栅极场效应晶体管来说，这里面最常见的是 **金属氧化物半导体场效应管 Metal-Oxide-Semiconductor Field-Effect Transistor（MOSFET）**，而对于MOS管来说，它可以有多种不同的符号，我们把JFET和MOSFET的符号放在一起进行对比记忆。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2ff82ee61a714e758c4a72ee213ec95b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
这里提到了**耗尽型** 与 **增强型** 两种，那么它有什么区别呢？

![在这里插入图片描述](https://img-blog.csdnimg.cn/fe36da5ea396476ebf93b36968bfca4c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_8,color_FFFFFF,t_70,g_se,x_16#pic_center)


简单的说，所谓 **增强型** 是通过“加厚”导电沟道的厚度来导通。栅极电压越低，则p型源、漏极的正离子就越靠近中间，n衬底的负离子就越远离栅极，栅极电压达到一个值，叫阀值或坎压时，由p型游离出来的正离子连在一起，形成通道，就是图示效果。因此，容易理解，栅极电压必须低到一定程度才能导通，电压越低，通道越厚，导通电阻越小。由于电场的强度与距离平方成正比，因此，电场强到一定程度之后，电压下降引起的沟道加厚就不明显了，也是因为n型负离子的“退让”是越来越难的。

耗尽型的是事先做出一个导通层，用栅极来加厚或者减薄来控制源漏的导通。但这种管子 **一般不生产**，在市面基本见不到。所以，大家平时说mos管，就默认是增强型的[^2]。

[^2]: https://www.eet-china.com/mp/a68769.html
