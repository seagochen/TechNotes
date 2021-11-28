@[toc]

# random()

生成伪随机数。

## 函数原型
* random(max)
* random(min, max)

## 参数
* min: 生成随机数的最小值，缺省/可选
* max: 生成随机数的最大值

## 返回
返回最小值到最大值-1，数据类型为long。没有填写最小值时，默认为0


```cpp
long randNumber;

void setup() {
  Serial.begin(9600);

  // if analog input pin 0 is unconnected, random analog
  // noise will cause the call to randomSeed() to generate
  // different seed numbers each time the sketch runs.
  // randomSeed() will then shuffle the random function.
  randomSeed(analogRead(0));
}

void loop() {
  // print a random number from 0 to 299
  randNumber = random(300);
  Serial.println(randNumber);

  // print a random number from 10 to 19
  randNumber = random(10, 20);
  Serial.println(randNumber);

  delay(50);
}

```

学过写随机数的朋友，应该知道计算机设备不能产生真正的随机数，这是因为随机数的产生本质是通过数学算法进行生成的，也就是说如果输入 $x$ 唯一且确定，那么产生的随机数输出 $f(x)$ 也一定是唯一且确定。

所以我们通常用计算机上的时钟脉冲信号作为产生随机数的种子。自然，Arduino这一类的嵌入式设备，也是需要种子的。而它的种子函数就是：

**randomSeed()**

但是我们还需要类似时间的脉冲信号，用来驱动种子。所以正如上面用例一样，使用 **analogRead()** 读取0号针角产生随机脉冲信号。另外就是要特别注意的一点，就是随机数的生成范围应该满足你代码的合理需求范围，毕竟嵌入式设备的资源十分有限，过大的范围并不能带来性能的提升，甚至可能带来你所不期望的异常。


# randomSeed()

随机数生成种子，每使用一次，会产生一组随机数数列。当你需要尽可能均衡的生成一组随机数数列，那么就可以使用这个函数。而如果你需要产生一组固定的随机数数列，那么可以给它一个固定值。


## 函数原型
randomSeed(seed)

## 参数
seed: 生成伪随机数数列的种子. 数据类型 unsigned long.