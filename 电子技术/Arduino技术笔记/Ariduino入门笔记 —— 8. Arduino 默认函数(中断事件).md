@[TOC]

# 一个恶心但是能说明一切的场景

对于控制系统来说，有一种很重要的业务场景，在Arduino的定义里，叫中断信号控制。

> 我们用大白话来举个栗子，假设你正在厕所LS，你的电话放在其他房间里，而此时你的老板、甲方或者女朋友突然给你打电话，如果此时你不去接这个电话，你有可能面临非常严重的后果。所以，即便你十分烦躁，也要提着裤子去接电话。只有解决了紧急的电话事件，你才能回到LS的任务进程中。

在实际的系统里存在大量的类似场景，比如用Arduino控制LED跑马灯，用一些针脚监控某些特别的模块，当脉冲信号产生的时候，就需要立即进行处理，就好像我上面提到的那个场景。那么，在Arduino中，一共提供了4个信号控制函数，我们一一道来。

# attachInterrupt()
还是那样，如果只看官方的说明，还是会云里雾里。这里简单的说一下，大多数跟IO相关的事务，大多都有在任务启动前需要用户进行调用的类似**bind**的函数。这里也是一样，如果要启动信号中断任务，也需要用 **attachInterrupt** 去启动事件监听机制。

当设置了之后，Arduino就启动了中断事件监听的功能，不过Arduino本身并不支持所有的Pin进行中断事件监听。于是官方给了如下一张表，表中表明了对于不同的平台，有哪些针脚可以使用：

Arduino平台 | 支持中断的针脚
-----------------|-----------------------
Uno, Nano, Mini, 其他以基于328的平台 | 2, 3
Uno WiFi Rev2., 全部的Nano | 所有的数字针脚
Mega, Mega2560, Mega ADK | 2, 3, 18, 19, 20, 21
Micro, Leonardo | 0, 1, 2, 3, 7
Zero | 除4号引脚外的所有数字针脚
MKR Family | 0, 1, 4, 5, 6, 7, 8, 9, A1, A2
Nano 33 IoT | 2, 3, 9, 10, 11, 13, A1, A5, A7
Nano 33 BLE, Nano 33 BLE Sense | 全部针脚
Due | 所有数字引脚

* Mega, Mega2560, Mega ADK， 20和21针脚，在启动**I2C**时候无法使用


在使用过程中，需要注意一些情况：

* 中断函数作用期间内，时间函数基本不起作用（例如，delay, millis，micros, 因为这些函数本身依赖于中断信号）
* delayMicroseconds 是唯一可以在中断函数中执行的时间函数
* 在函数中接收的串行数据可能会丢失
* 中断处理函数不接受任何传入任何参数，也不返回任何返回值
* 中断处理函数越精简越好
* 有多个中断处理函数时，Arduino一次只能处理一个，其他的只有在当前函数执行完毕后才会依次执行
* 主程序和中断程序之间只有通过全局变量进行传值，因此为确保这些全局变量的正常，需要使用 volatile 关键字进行修饰

## 函数原型

* attachInterrupt(digitalPinToInterrupt(pin), ISR, mode) (recommended)
* attachInterrupt(interrupt, ISR, mode) (not recommended)
* attachInterrupt(pin, ISR, mode) (Not recommended. Additionally, this syntax only works on Arduino SAMD Boards, Uno WiFi Rev2, Due, and 101.)

### 参数说明
* interrupt: 中断编号，在老式的版本中通常直接赋值中断编号，由于兼容性问题，不推荐使用第二、三种函数原型形式
* pin: 中断引脚号
* ISR: 中断服务程序名
* mode：中断模式

**中断模式(mode)有以下几种形式：**

* LOW 当引脚为低电平时触发中断服务程序
* CHANGE 当引脚电平发生变化时触发中断服务程序
* RISING 当引脚电平由低电平变为高电平时触发中断服务程序
* FALLING 当引脚电平由高电平变为低电平时触发中断服务程序

对于 Due, Zero and MKR1000 板来说有：

* HIGH 当引脚为高电平时触发中断服务程序

### 返回值
Nothing

```c
const byte ledPin = 13;
const byte interruptPin = 2;
volatile byte state = LOW;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(interruptPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(interruptPin), blink, CHANGE);
}

void loop() {
  digitalWrite(ledPin, state);
}

void blink() {
  state = !state;
}
```

## 关于中断编号

从官方的文档描述中，感觉是由于历次版本API的迭代更新，做的一个妥协。因为在早期版本中，是使用中断编号而不是引脚的方案，所以官方强烈推荐用户使用 **digitalPinToInterrupt(pin),** 这个函数。由于不同版本的Arduino设备，启动对应的**ISR**服务，它的中断号是不一样的，所以，它的作用相当于在不同版本的Arduino设备上做了一层翻译的封装。

因此对于一些老程序来说，就可以使用它就可以尽可能的做最小的改动工作。为了方便理解，官方给出了一个 digitalPinToInterrupt 中断编号和引脚号对应表，表中中间部分是引脚号，上方的列名是旧版本对应的中断编号，而左侧行名是各种不同的Arduino设备列表。


Board | Int.0 | Int.1 | Int.2 | Int.3 | Int.4 | Int.5
---------|-------|-------|--------|-------|--------|---------
Uno, Ethernet | 2 | 3
Mega2560 | 2 | 3 | 21 | 20 | 19 | 18
32u4 based | 3 | 2 | 0 | 1 |  7


# detachInterrupt()

有了 **attachInterrupt** 自然就有了 **detachInterrupt**, 其作用就跟 **unbind** 的原理是一样的，就是解除ISR服务，使中断任务失效，Arduino不再需要继续监听中断事件了。


## 函数原型
* detachInterrupt(digitalPinToInterrupt(pin)) (recommended)
* detachInterrupt(interrupt) (not recommended)
* detachInterrupt(pin) (Not recommended. Additionally, this syntax only works on Arduino SAMD Boards, Uno WiFi Rev2, Due, and 101.)


# interrupts()

如果你的程序希望在执行到某些阶段时暂停中断事件监听，你就需要使用 **noInterrupts** 而重新启用监听事件，则使用 **interrupts**

## 函数原型
* interrupts()
* 无参数无返回



# noInterrupts()
如果你的程序希望在执行到某些阶段时暂停中断事件监听，你就需要使用 **noInterrupts** 而重新启用监听事件，则使用 **interrupts**

## 函数原型
* noInterrupts()
* 无参数无返回

