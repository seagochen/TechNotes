说实话，对于我这种朝三暮四，动不动就要开新坑的人来说，肯定很多人都烦死了。没办法，因为脑袋里的猴子有点多，一直做一件事的话，我反而很容易就弃坑。也就是所谓新鲜感，隔一段时间回过头来看看草稿箱里有哪些稿子还没有写的，有了兴趣就写一写，多多少少最后会完成一些系列的吧。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210504114411472.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)当然，开这个坑的目的是想要总结、整理最近一个半月的工作内容，也就是帮兄弟做Arduino软件层的资料。虽然说以后的工作内容不一定还会用到Arduino，不过万一用到了那我自己可以很方便的找到这些资料，然后快速入手做设计。

@[toc]

# 关于Arduino[^1]

Arduino 是一款便捷灵活、方便上手的开源电子原型平台，包含硬件（各种型号的arduino板）和软件（arduino IDE)。

## 可以做什么？

Arduino不仅仅是全球最流行的开源硬件，也是一个优秀的硬件开发平台，更是硬件开发的趋势。Arduino简单的开发方式使得开发者更关注创意与实现，更快的完成自己的项目开发，大大节约了学习的成本，缩短了开发的周期。

因为Arduino的种种优势，越来越多的专业硬件开发者已经或开始使用Arduino来开发他们的项目、产品；越来越多的软件开发者使用Arduino进入硬件、物联网等开发领域；大学里，自动化、软件，甚至艺术专业，也纷纷开展了Arduino相关课程。

## 为什么要使用？

用Arduino制作作品或者进行产品开发的优势是很明显的。

### 跨平台
Arduino IDE可以在Windows、Macintosh OSX、Linux三大主流操作系统上运行，而其他的大多数控制器只能在Windows上开发。

### 简单清晰的开发
Arduino IDE基于processing IDE开发。对于初学者来说，极易掌握，同时有着足够的灵活性。Arduino语言基于wiring语言开发，是对 AVRGCC库的二次封装，不需要太多的单片机基础、编程基础，简单学习后，你也可以快速的进行开发。

## 优势和特色

1、开放源代码的电路图设计，程序开发接口免费下载，也可依需求自己修改；
2、可以采用USB 接口供电，也可以外部供电，双向选择；
3、Arduino支持ISP 在线烧写，可以将新的“bootloader”固件烧入 ATmega168 或AT mega328芯片。有了bootloader之后，可以通过USB 更新程序；
4、可依据官方提供的Eagel格式PCB 和SCH 电路图，简化Arduino模组，完成独立运作的微处理控制。可简单地与传感器，各式各样的电子元件连接（红外线、超音波、热敏电阻、光敏电阻、伺服舵机…等）；
5、支持多种互动程序，如：Flash、Max/Msp 、VVVV 、C、Processing....等；
6、应用方面，利用Arduino，突破以往只能使用鼠标、键盘，CCD等输入的装置的互动内容，可以更简单地达成单人或多人游戏互动。

[^1]: https://zhuanlan.zhihu.com/p/72070458

## 总结

Arduino对于非微电子专业的人来说，是一种可以快速上手的开发板。主要编程语言为C/C++，经过测试，除了标准库、第三方库lib无法使用外，在不考虑主程序占用存储空间的前提下，是可以无缝使用C/C++的编程特性。

Arduino如果需要使用其他第三方库，例如i2c等进行模块间的数据交互，可以通过Arduino IDE中

```
Sketch/Include Library/
```

通过 **Add .ZIP library** 或者通过网络搜索 **Manage Libraries**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210504120110572.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
完成这一过程。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210504120312544.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

现在比较常见的支持Arduino的主控芯片， Mega328p / Mega2560 / Mega32u4 / stm32 / Esp8266 / esp32 等。这些芯片之间最主要的区别在于主频，支持不支持蓝牙，Flash，运存等差异。

所以与设计PC程序不同的是，在设计Arduino程序时，要尤其注意资源的消耗，因为这些芯片的可用空间真的非常小，可能有的只有2KB而已。


# Arduino的默认函数

由于Arduino舍弃了STL和C标准库，而集成了很多硬件I/O也就意味着对于熟悉软件设计的朋友来说，我们没法用
```cpp
	std::cout << "debug message" << std::endl;
```

这种类似的形式来做一些常见的Debug输出，取而代之的是通过串口协议，或者蓝牙协议等，把Arduino的设备信息状态返回给用户。

此外，由于芯片性能影响，不建议在Arduino平台上使用多线程技术，尽管有教材会告诉你如何处理多线程。但对于 Arduino 来说，它最准确的定位就是硬件设备的协调和控制。

如果有涉及到复杂编程的，应该考虑使用功能更为强大的Arm芯片，例如基于博通Arm的树莓派，甚至是x86的PC设备。

那么没有这些标准库函数，我们又该如何开发Arduino呢？

在这一章节里，我们将主要介绍Arduino的默认函数。

##  Digital I/O

Digital I/O 主要涉及到 Arduino 的针脚编程，控制包括但不限于LED的开关，舵机的转向等功能。 标准Arduino开发板的针脚定义图如下[^2]

**Arduino UNO**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210504133419218.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

而关于其他板子的详细介绍可以查看官网[《Arduino官网：Pin接口定义》](https://www.arduino.cc/reference/en/libraries/pin/)

Arduino中，可编程引脚可用3个函数，它们分别是：

```cpp
// 可从引脚中获取电势位，HIGH, LOW
digitalRead(pin);

// 可对引脚写入电势位，HIGH, LOW
digitalWrite(pin);

// 设置引脚为读(INPUT)、写(OUTPUT)
pinMode(pin, mode) 
```

例如：把引脚13的电势位设定与引脚7的电势位一致
```cpp
int ledPin = 13;  // LED connected to digital pin 13
int inPin = 7;    // pushbutton connected to digital pin 7
int val = 0;      // variable to store the read value

void setup() {
  pinMode(ledPin, OUTPUT);  // sets the digital pin 13 as output
  pinMode(inPin, INPUT);    // sets the digital pin 7 as input
}

void loop() {
  val = digitalRead(inPin);   // read the input pin
  digitalWrite(ledPin, val);  // sets the LED to the button's value
}
```

[^2]: https://blog.csdn.net/whatday/article/details/87271396

## Analog I/O 

顾名思义，Digital IO主要的是关于数字信号的读写，而Analog则是关于模拟信号的读写。它也有三个与之相关的函数：

通常情况下，Arduino的模拟信号带宽为10bit，可以用来表示 0-1023 之间的整数。对于大多数Arduino设备来说，其高电位通常是5V（也有3.3V），因此它的每个数值的表示关系为 5v / 1024 = 0.0049V，即4.9mV。
```cpp
// 用于读取模拟信号，通常为10bits带宽，可表示信号值范围从0-1023.
analogRead(pin);

// 用于读取模拟信号，通常为10bits带宽，可表示信号值范围从0-1023.
analogWrite(pin, value); 

// 配置用于模拟输入的基准电压/参考电压（即输入范围的最大值）
analogReference(MACRO); 
```

### 不同Arduino设备对应的模拟信号接口带宽表

BOARD | OPERATING VOLTAGE | USABLE PINS | MAX RESOLUTION
--------|-----------------------|---------------|---------------------
Uno            | 5 Volts				      | A0 - A5		    | 10 bits
Mini, Nano | 5 Volts                     | A0 - A7           | 10 bits
Mega, Mega2560, MegaADK | 5 Volts | A0 - A14 | 10 bits
Micro       | 5 Volts                   | A0 - A11 [1]          | 10 bits
Leonardo | 5 Volts                  | A0 - A11 [1]            | 10 bits
Zero         | 3.3 Volts               | A0 - A5              | 12 bits[2]
Due         | 3.3 Volts               | A0 - A11              | 12 bits[2]
MKR       | 3.3 Volts                | A0 - A6              |  12 bits[2]

* 1 - A0 - A5 被表示在板子上，A6 - A11 分别和4，6，8，9，10，12相对应
* 2 - 为了兼容性，这些板子都使用的是10bit带宽，但是你可以通过analogReadResolution调整它的分辨率


### analogReference 可使用参数说明

该函数是用来调整模拟接口的输出电压的。

#### Arduino AVR Boards (Uno, Mega, Leonardo, etc.)

DEFAULT:  针对5V的Arduino板，使用5V的电压；对于3.3V板，使用3.3V电压。

INTERNAL: 使用相对合适的中间电压，对于ATmega168 和 ATmega328P主控芯片的Arduino设备来说，使用1.1V；而对于ATmega32U4 和 ATmega8 则使用2.56V，（对于Arduino Mega 不适用）。

INTERNAL1V1：使用 1.1V，仅适用于Arduino Mega

INTERNAL2V56：使用2.56V，仅适用于Arduino Mega 

EXTERNAL：以AREF引脚（0至5V）的电压作为基准电压

####  Arduino SAMD Boards (Zero, etc.)

AR_DEFAULT: 使用默认的 3.3V

AR_INTERNAL: 使用 2.23V 

AR_INTERNAL1V0:使用 1.0V 

AR_INTERNAL1V65:使用 1.65V 

AR_INTERNAL2V23: 适用 2.23V

AR_EXTERNAL: 以AREF引脚的电压作为基准电压

#### Arduino megaAVR Boards (Uno WiFi Rev2)

DEFAULT: 使用默认的 0.55V

INTERNAL: 使用默认的 0.55V

VDD: Vdd of the ATmega4809. 5V on the Uno WiFi Rev2

INTERNAL0V55: 使用默认的 0.55V

INTERNAL1V1: 使用默认的 1.1V

INTERNAL1V5: 使用默认的 1.5V

INTERNAL2V5: 使用默认的 2.5V

INTERNAL4V3: 使用默认的 4.3V

EXTERNAL: 以AREF引脚（0至5V）的电压作为基准电压

#### Arduino SAM Boards (Due)

AR_DEFAULT: 使用3.3V，仅对Due有效。

AR_VDD:  使用3.3V

AR_INTERNAL: 使用0.6V

AR_INTERNAL1V2: 1.2 V reference (internal 0.6 V reference with 2x gain)

AR_INTERNAL2V4: 2.4 V reference (internal 0.6 V reference with 4x gain)


## 补充Analog I/O （Zero/Due/MKR family）
 针对Zero/Due/MKR系列，还有另外两个专用的模拟API接口

```cpp
// 读取模拟接口的带宽，默认为10bits，可传输的数据范围[0，1023]，如果带宽为12bits时，可传输数据范围为[0, 4095]，16bits时，可传输的数据范围为[0, 65535]
analogReadResolution(bits);

// 写入模拟接口的带宽，默认为10bits，可传输的数据范围[0，1023]，如果带宽为12bits时，可传输数据范围为[0, 4095]，16bits时，可传输的数据范围为[0, 65535]
analogWriteResolution(bits); 
```

* 这两个API可以在，Zero/Due/MKR系列/Nano33/Portenta上使用；
* 默认为10bits，可传输的数据范围[0，1023]，如果带宽为12bits时，可传输数据范围为[0, 4095]，16bits时，可传输的数据范围为[0, 65535]
* 由于 Portenta H7 可以支持16位，所以它传输的值可以为[0, 65535]

## 模拟信号可以做啥

众所周知，数字信号传输的只有两个信号，高电平和低电平。也就是0和1，那么在微电路上它能做的就是控制元器件的开关。但是对于喇叭、温控、光控、声控电路，舵机等设备的控制，仅仅只有[0, 1]是不够的。

假如我们需要产生一组正弦信号，去控制一个电机的运动，输出电压处于低位时，让电机缓慢运动，而到高电位时高速运动，处于 > 2.5 Volts 时，需要点击顺时针方向运动，而 < 2.5 Volts时让点击逆时针运动，那么就需要产生模拟信号。此外，使用模拟信号控制发光二极管，也可以获得呼吸灯的效果。

**此外，根据官方手册，1个PIN读写一次模拟信号的时间大概在100微秒，也就是说可以每秒读写1万次。**

但我相信，实际实验中肯定多少因为电路走线等问题，而会出现些许误差。怎么说呢，如果你要设计一些高速时序电路的话，可能要找找文章资料、或者跟做这行的前辈交流交流。

祝你好运🍀～！
