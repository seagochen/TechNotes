@[toc]

如果想用Arduino播放音乐，或者声音，你就需要使用一些高级的IO功能。官方给出了数个高级IO编程的接口，我在这里把他们罗列出来，以方便使用。


# 播放不同频率的方波信号 tone()

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210513130429554.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70)

这个函数将会产生一连串不同频率的信号（也就是声音）给指定的针脚，可以在任意Arduino引脚上产生指定频率（50％占空比）的方波[^1]。可以定义时长，如果不定义时长，那么会持续播放旋律。对应针角可以连接到 压电式蜂鸣器（Piezo Buzzer），也可以连接到喇叭上。

[^1]: https://www.w3cschool.cn/arduino/arduino_tone_library.html

不过需要注意的是，同一时间内只能产生一个信号。如果在一个针角上发送多个不同频率的信号，会出现信号抢占的问题，导致其他tone()无效。

根据官方描述，使用tone函数将干扰引脚3和11（Mega以外的板上）上的PWM输出。此外，也无法产生低于31Hz的信号。

## 函数原型：
* tone(pin, frequency)
* tone(pin, frequency, duration)

## 参数说明：
* pin: 针脚号
* frequency: 信号频率，数据类型: unsigned int.
* duration: 持续时长 milliseconds (optional). 数据类型: unsigned long.

**需要注意的地方：** 如果打算在不同的针脚上发送信号，那么在发送信号前，为避免干扰，需要先 使用noTone停止之前的tone。


此外，在W3School的官网，我找到这样一个非常有意思的播放MIDI音乐的方式，这里是各个音高的频率定义。

```cpp
/*************************************************
* Public Constants
*************************************************/

#define NOTE_B0 31
#define NOTE_C1 33
#define NOTE_CS1 35
#define NOTE_D1 37
#define NOTE_DS1 39
#define NOTE_E1 41
#define NOTE_F1 44
#define NOTE_FS1 46
#define NOTE_G1 49
#define NOTE_GS1 52
#define NOTE_A1 55
#define NOTE_AS1 58
#define NOTE_B1 62
#define NOTE_C2 65
#define NOTE_CS2 69
#define NOTE_D2 73
#define NOTE_DS2 78
#define NOTE_E2 82
#define NOTE_F2 87
#define NOTE_FS2 93
#define NOTE_G2 98
#define NOTE_GS2 104
#define NOTE_A2 110
#define NOTE_AS2 117
#define NOTE_B2 123
#define NOTE_C3 131
#define NOTE_CS3 139
#define NOTE_D3 147
#define NOTE_DS3 156
#define NOTE_E3 165
#define NOTE_F3 175
#define NOTE_FS3 185
#define NOTE_G3 196
#define NOTE_GS3 208
#define NOTE_A3 220
#define NOTE_AS3 233
#define NOTE_B3 247
#define NOTE_C4 262
#define NOTE_CS4 277
#define NOTE_D4 294
#define NOTE_DS4 311
#define NOTE_E4 330
#define NOTE_F4 349
#define NOTE_FS4 370
#define NOTE_G4 392
#define NOTE_GS4 415
#define NOTE_A4 440
#define NOTE_AS4 466
#define NOTE_B4 494
#define NOTE_C5 523
#define NOTE_CS5 554
#define NOTE_D5 587
#define NOTE_DS5 622
#define NOTE_E5 659
#define NOTE_F5 698
#define NOTE_FS5 740
#define NOTE_G5 784
#define NOTE_GS5 831
#define NOTE_A5 880
#define NOTE_AS5 932
#define NOTE_B5 988
#define NOTE_C6 1047
#define NOTE_CS6 1109
#define NOTE_D6 1175
#define NOTE_DS6 1245
#define NOTE_E6 1319
#define NOTE_F6 1397
#define NOTE_FS6 1480
#define NOTE_G6 1568
#define NOTE_GS6 1661
#define NOTE_A6 1760
#define NOTE_AS6 1865
#define NOTE_B6 1976
#define NOTE_C7 2093
#define NOTE_CS7 2217
#define NOTE_D7 2349
#define NOTE_DS7 2489
#define NOTE_E7 2637
#define NOTE_F7 2794
#define NOTE_FS7 2960
#define NOTE_G7 3136
#define NOTE_GS7 3322
#define NOTE_A7 3520
#define NOTE_AS7 3729
#define NOTE_B7 3951
#define NOTE_C8 4186
#define NOTE_CS8 4435
#define NOTE_D8 4699
#define NOTE_DS8 4978
```

我在原来的示例代码基础上，做了一些修改，你可以自己执行看看是什么效果：

```cpp
int melody[] = { 
  NOTE_C4, NOTE_D4, NOTE_E4, NOTE_C4,
  NOTE_C4, NOTE_D4, NOTE_E4, NOTE_C4,
  NOTE_E4, NOTE_F4, NOTE_G4, 
  NOTE_E4, NOTE_F4, NOTE_G4,

  NOTE_G4, NOTE_A4, NOTE_G4, NOTE_F4,
  NOTE_E4, NOTE_C4,
  NOTE_G4, NOTE_A4, NOTE_G4, NOTE_F4,
  NOTE_E4, NOTE_C4,

  NOTE_D4, NOTE_G3, NOTE_C4, 0,
  NOTE_D4, NOTE_G3, NOTE_C4, 0,
 };


// note durations: 4 = quarter note,
// 8 = eighth note, etc.:
int noteDurations[] = {
  4, 4, 4, 4,  
  4, 4, 4, 4,  
  4, 4, 2, 
  4, 4, 2,
  
  4, 4, 4, 4,  
  2, 2,  
  4, 4, 4, 4,  
  2, 2, 

  4, 4, 4, 4,
  4, 4, 4, 4
 };


void setup() {
  // put your setup code here, to run once:
   // iterate over the notes of the melody:
   for (int thisNote = 0; thisNote < 34; thisNote++) {
      // to calculate the note duration, take one second
      // divided by the note type.
      //e.g. quarter note = 1000 / 4, eighth note = 1000/8, etc.
      int noteDuration = 1000/noteDurations[thisNote];
      tone(5, melody[thisNote],noteDuration);
      //pause for the note's duration plus 30 ms:
      delay(noteDuration +30);
   }

   noTone(5);
}

void loop() {
  // put your main code here, to run repeatedly:

}
```

注意一下你的喇叭连接的接口号，看看有什么有趣的。

# 停止播放信号 noTone()

这个函数是用来停止由tone所激发的方波信号。

## 函数原型：
noTone(pin)

## 参数说明：
pin: 针脚、管脚号


# 针脚测时 pulseIn()

根据官方描述，如果定义的值是高电平，当某针脚从低电平变成高电平的那一刻开始计时，一直到针脚重新从高电平变成低电平，返回此过程的微秒时间（microseconds），或者说是脉冲持续时间。

有什么用呢？

比方说，这种功能对于需要测距的时候，就可以派上用场。比如说：

超声波发射器向某一方向发射超声波，在发射的同时开始计时，超声波在空气中传播，途中碰到障碍物就立即返回来，超声波接收器收到反射波就立即停止计时。声波在空气中的传播速度为340m/s，根据计时器记录的时间t，就可以计算出发射点距障碍物的距离s，即：$s=340m/s \times t / 2$ 。[^2]

[^2]: https://blog.csdn.net/qq_31077649/article/details/72581968

需要注意的是，这个函数可用于测量的时间范围是  10微秒（1000 微秒 = 1 毫秒 = 1/1000秒），到3分钟。另外，根据官方描述，如果使用了超时参数，那么可以更快的返回结果

## 函数原型：
* pulseIn(pin, value)
* pulseIn(pin, value, timeout)

## 参数说明：
pin: 读取返回值信息的管脚号
value: 希望读取的脉冲信号类型： HIGH（高电平） or LOW（低电平）. 
timeout (optional): 等待获得脉冲信号开始的时间，默认是1秒。比方说当前的信号是低电平，希望读取的是高电平，如果在设定的等待时间里，检测到了高电平，则开始计时。

## 返回值

如果检测到了信号，返回信号持续的时间。如果超时，那么返回的数据是0.


```cpp
int pin = 7;
unsigned long duration;

void setup() {
  Serial.begin(9600);
  pinMode(pin, INPUT);
}

void loop() {
  duration = pulseIn(pin, HIGH);
  Serial.println(duration);
}
```

# pulseInLong()

根据官方描述，这个函数是一个对pulseIn来说，对擅长处理更长脉冲时间的函数。此外，对于检测中断的场景也比pulseIn更为得心应手。检测有效时常和pulseIn一样，但是对于长脉冲数据，精度会有问题。

所以对短脉冲信号的处理，可以只使用pulseIn，而对于长脉冲信号，推荐使用pulseInLong。由于参数基本一致，所以这里不做过多说明。


# shiftIn()

将一个字节的数据通过移位的方式逐位输入。数据可以从最高位（最左位）或从最低位（最右位）输入。在输入数据时，Arduino首先在时钟引脚输出高电平，然后通过数据输入引脚读取一位数据，读取结束后时钟引脚将被Arduino设置为低电平。

如果与Arduino进行数据通讯的设备是在时钟引脚脉冲信号上升沿发送数据，请确保在调用shiftIn()前，应先通过digitalWrite(clockPin, LOW)语句，将时钟引脚设置为LOW。这样做是为了确保数据读取准确无误。

以上介绍的方法使用软件实现数据输出操作。如果想要通过硬件方法输出数据，请参阅Arduino的SPI库函数。通过硬件方法输入数据更加快捷，但Arduino只有几个特定引脚可用于使用硬件方法输入数据[^3]。

[^3]: http://www.taichi-maker.com/homepage/reference-index/arduino-code-reference/shiftin

## 函数原型：
byte incoming = shiftIn(dataPin, clockPin, bitOrder)

## 参数说明：
* dataPin – 数据引脚，数据输入
* clockPin – 时钟引脚
* bitOrder – 移位顺序 ( 高位先入 或 低位先入)

## 返回值
从 dataPin  中读取的值，数据类型为 byte.

# shiftOut()

将一个字节的数据通过移位输出的方式逐位输出。数据可以从最高位（最左位）或从最低位（最右位）输出。在输出数据时，当一位数据写入数据输出引脚时，时钟引脚将输出脉冲信号，指示该位数据已被写入数据输出引脚等待读取。

如果读取数据的设备是在Arduino的时钟引脚脉冲信号上升沿读取Arduino的输出数据，请确保在调用shiftOut()输出数据前，应先通过digitalWrite(clockPin, LOW)语句，将时钟引脚设置为LOW。这样做是为了确保数据读取准确无误。

以上介绍的方法使用软件实现数据输出操作。如果想要通过硬件方法输出数据，请参阅Arduino的SPI库函数。通过硬件方法输出数据更加快捷，但Arduino只有几个特定引脚可用于使用硬件方法输出数据。


## 函数原型：
shiftOut(dataPin, clockPin, bitOrder, value)

## 参数说明：
dataPin – 数据引脚
clockPin – 时钟引脚
bitOrder – 移位顺序 ( 高位先出 或 低位先出)
val – 输出的数据

## 返回
无


