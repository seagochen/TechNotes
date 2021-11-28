比特位操作，是编程中常用的一种技巧。Arduino也提供了这方面的API，以方便开发者使用。

@[toc]

# bit()

从描述上很难搞清楚这个函数有什么用，干什么的。这样解释一下，这是算比特位的。
举例来说：

> bit(1) = binary(10) = 2
> bit(2) = binary(100) = 4
> bit(3) = binary(1000) = 8
> ...

至于如何验证，你可以试一试执行以下程序看一看，如果你使用的是UNO板，那么一般默认UNO板上的#13引脚是板载LED灯。

另外，这个示例程序，最好的演示效果是，1秒闪8次。也就是bit(3)。

```cpp
void setup()
{
  pinMode(13, OUTPUT);
}

void loop()
{
    int bits = bit(2);
    for (int i = 0; i  < bits; i++) {
        digitalWrite(13, HIGH);
        delay(100);

        digitalWrite(13, LOW);
        delay(100);
    }

    delay(1000);
}
```

# bitClear()

同样，从官方的解释来看也是摸不着头脑。那不如我们直接写个程序跑跑看效果。

```cpp
void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  int x = 15;
  int n = 0;
  int res = bitClear(x, n);
  Serial.print(res, BIN); // print the output of bitClear(x,n)
  Serial.println("\n--------------");

  n = 1;
  res = bitClear(x, n);
  Serial.print(res, BIN); // print the output of bitClear(x,n)
  Serial.println("\n--------------");

  n = 2;
  res = bitClear(x, n);
  Serial.print(res, BIN); // print the output of bitClear(x,n)
  Serial.println("\n--------------");

  n = 3;
  res = bitClear(x, n);
  Serial.print(res, BIN); // print the output of bitClear(x,n)
  Serial.println("\n--------------");
}

void loop() {
}
```

官方说，这个函数负责给数位上写零，那么我们就要具体看看是怎么写零的。为了更好观察输出结果，我们提前使用到了串口指令。还不清楚什么叫串口的童鞋，没关系，你只需要记着编译完程序后，立即到Arduino IDE右上方点击这个标志，打开一个新的窗口。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210618105636123.png#pic_center)

然后不出意外，应该能得到这样一个输出
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021061810564419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
我把数据以2进制形式打印出来，所以可以更清晰易懂这个函数的使用方法。


# bitRead()
官方说法，是从数据中读取对应位置的比特值，那么具体怎么执行的，还是上代码比较直观。

```cpp
void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  int x = 13;
  int n = 0;
 
  for (int i = 0; i < 4; i++) {
    Serial.print(i);
    Serial.print(" --> ");

    int res = bitRead(x, i);
    Serial.print(res); // print the output of bitClear(x,n)
    Serial.println("\n--------------");
  }
}

void loop() {
}
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210618110421710.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
要知道，13的二进制值是 1101，所以你应该知道这个函数是怎么回事了。

# bitSet()
这个函数是把对应位置的比特值设置为1，因此对于13（1101）来说，如果我们让1位的比特值变成1，那么这个值就会变成15.


```cpp
void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  int x = 13;
  int n = 0;
 
   bitSet(x, 1);
   Serial.print(x, BIN);
}

void loop() {
}
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210618111621965.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
记住，这个函数不返回任何值，所以修改就会直接修改到原值。

# bitWrite()
这是一个写比特的函数，功能和 bitSet 很像，也一样不返回任何结果，直接对原数据进行修改。不过需要注意一点的是， bitSet是把指定的位写为1，而这个函数可以自定。

```cpp
void setup() {
  Serial.begin(9600);
  while (!Serial) {}  // wait for serial port to connect. Needed for native USB port only
  byte x = 0b10000000;  // the 0b prefix indicates a binary constant
  Serial.println(x, BIN); // 10000000
  bitWrite(x, 0, 1);  // write 1 to the least significant bit of x
  Serial.println(x, BIN); // 10000001
}

void loop() {}
```

# highByte()
从左往右取高位字节，我个人不推荐使用

# lowByte()
从右往左取低位字节，我个人不推荐使用。

我们这里直接上一段代码，这样可以更明白这两个函数是怎么使用的。

```cpp
void setup() {
  Serial.begin(9600);
  while (!Serial) {}  // wait for serial port to connect. Needed for native USB port only

  int x = 0b1110110011110000;  // the 0b prefix indicates a binary constant
  Serial.println(x, BIN);

  // high byte (from left)
  byte res = highByte(x);
  Serial.println(res, BIN);

  // low byte (from right)
  res = lowByte(x);
  Serial.println(res, BIN);
}

void loop() {}

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210618113047255.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)


