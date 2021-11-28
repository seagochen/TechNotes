@[toc]

# 主要功能
它可以用于单片机引脚扩展，可以将 8-bit 串行输入转化为并行输出。尽管我们可以用单片机的8个引脚各自驱动不同的用电器工作，但是如果需要控制大量的用电器时，有时候就可以用这一类的寄存器来扩展单片机的引脚。

除了VCC和GND引脚外，它至少还需要接入至少三个引脚，分别是时钟信号、输出激发、串行输入。

至于如何接入这颗芯片，我稍后放在文章里介绍。

# 主要参数
通过网络渠道，个人可以买到的74HC595，主要来自德州仪器（Texas Instruments），芯片名 **SN74HC595** ，主要封装形式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/6ae5170e7c564f9ca38c482236783fc9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
注意不同的封装形式，大小也是不一样的：
![在这里插入图片描述](https://img-blog.csdnimg.cn/39a03d9997b6459d8da027bd755364bd.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 工作参数

* 工作电压：2V ～ 6V
* 工作温度：-40C ～ 80C
* 工作电压：如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/7a0c6fd9e57746518c0c40af25246e3d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 引脚定义

NAME | PIN | I/O | Description
---------|-------|-------|---------------
GND   | 8  |  - | 接地
$\overline{OE}$ | 13 | I | 允许输出，低电平有效
$Q_{A}$ | 15 | O | 输出1
$Q_{B}$ | 1   | O | 输出2
$Q_{C}$ | 2   | O | 输出3
$Q_{D}$ | 3   | O | 输出4
$Q_{E}$ | 4   | O | 输出5
$Q_{F}$ | 5   | O | 输出6
$Q_{G}$ | 6   | O | 输出7
$Q_{H}$ | 7   | O | 输出8
$Q_{H'}$ | 9   | O | 串行输出
RCLK | 12 | I | 输出存储器锁存时钟线
SER  | 14 | I | 串行数据输入
SRCLK | 11 | I | 数据输入时钟线
$\overline{SRCLR}$ | 10 | I | 主复位（低电平有效）
VCC | 16 | - | VCC

主要引脚说明：

* 10脚: 低电平时将移位寄存器的数据清零。通常接到VCC防止数据清零。
* 11脚：上升沿时数据寄存器的数据移位。Q0->Q1->Q2-->Q3-->...-->Q7;下降沿移位寄存器数据不变。
* 12脚：上升沿时移位寄存器的数据进入数据存储寄存器，下降沿时存储寄存器数据不变。
* 13脚: 高电平时禁止输出(高阻态)。如果单片机的引脚不紧张，用一个引脚控制它，可以方便地产生闪烁和熄灭效果。

# 接线方法
简单应用，用74HC595做一个流水灯
![在这里插入图片描述](https://img-blog.csdnimg.cn/fe31cc4b3a8a4a2e8d1498830b7ee990.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
可以看到除了VCC和GND分别接地和供电外，我们实际上需要接入的，就是时钟信号，串行信号，以及信号输出信号。

如果接上Arduino设备，这个对应的代码就是这样的了。

~~~cpp
#define LATCH_PIN 4 // ST_CP
#define CLOCK_PIN 8 // SH_CP
#define DATA_PIN  2 // DS

void clock_signal(void)
{
  digitalWrite(CLOCK_PIN, HIGH);
  delayMicroseconds(500);
  digitalWrite(CLOCK_PIN, LOW);
  delayMicroseconds(500);  
}

void latch_enable(void)
{
  digitalWrite(LATCH_PIN, HIGH);
  delayMicroseconds(500);
  digitalWrite(LATCH_PIN, LOW);  
}

void send_data(unsigned int data)
{
  int i;
  unsigned hold;

  for (i = 0; i < 8; i++) {
    if ((data >> i) & (0x01)) 
      digitalWrite(DATA_PIN, HIGH);
    else
      digitalWrite(DATA_PIN, LOW);

    clock_signal();
  }
  latch_enable();
}


void setup()
{
  pinMode(LATCH_PIN, OUTPUT);
  pinMode(DATA_PIN, OUTPUT);
  pinMode(CLOCK_PIN, OUTPUT);

  digitalWrite(LATCH_PIN, LOW);
  digitalWrite(CLOCK_PIN, LOW);
  digitalWrite(DATA_PIN, LOW);
}

void loop()
{
  for (unsigned sig = 0b00000000; sig < 0b100000000; sig++) 
  {
    send_data(sig);
    delay(1000);
  }
}
~~~

有需要原始数据表的的朋友可以点这里[下载](https://download.csdn.net/download/poisonchry/36197277)

最终的效果大概就是这样 [《用74HC595 实现流水灯的最终效果》](https://www.bilibili.com/video/BV1rL411x7B7/)。


