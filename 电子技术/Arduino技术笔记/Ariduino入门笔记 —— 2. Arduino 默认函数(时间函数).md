时间函数，是时序控制中非常重要的一环。比如说你要产生PWM信号用于控制舵机转向。那么就需要时间控制相关函数。它类似于C语言里的sleep函数和timeofday函数。对于Arduino来说也有专门的处理与时间相关的函数。

@[toc]

# delay(ms)

让设备暂停执行X毫秒


```cpp
int ledPin = 13;              // LED connected to digital pin 13

void setup() {
  pinMode(ledPin, OUTPUT);    // sets the digital pin as output
}

void loop() {
  digitalWrite(ledPin, HIGH); // sets the LED on
  delay(1000);                // waits for a second
  digitalWrite(ledPin, LOW);  // sets the LED off
  delay(1000);                // waits for a second
}
```

# delayMicroseconds(us)
让设备暂停执行X微秒。另外，根据官网的介绍，当前实际能执行到的微妙最大值为16383微秒。因此如果有大于这个时间，最好使用delay函数。

```cpp
int outPin = 8;               // digital pin 8

void setup() {
  pinMode(outPin, OUTPUT);    // sets the digital pin as output
}

void loop() {
  digitalWrite(outPin, HIGH); // sets the pin on
  delayMicroseconds(50);      // pauses for 50 microseconds
  digitalWrite(outPin, LOW);  // sets the pin off
  delayMicroseconds(50);      // pauses for 50 microseconds
}
```

# micros()

返回自Arduino板启动后，至今的微秒数。大约每70分钟左右，会跳转从0开始。对于Arduino  Portenta系列来说，返回的最小单位是1微秒，对于 16 MHz的Arduino设备来说，返回的最小单位都是4倍微妙时间，对于8MHz的Arduino设备来说，返回的是8倍微秒时间。

```cpp
unsigned long time;

void setup() {
  Serial.begin(9600);
}
void loop() {
  Serial.print("Time: ");
  time = micros();

  Serial.println(time); //prints time since program started
  delay(1000);          // wait a second so as not to send massive amounts of data
}
```

# millis()

返回自Arduino板启动后，至今的毫秒数。当运行时间超过50天以后，会跳转从0开始。

```cpp
unsigned long myTime;

void setup() {
  Serial.begin(9600);
}
void loop() {
  Serial.print("Time: ");
  myTime = millis();

  Serial.println(myTime); // prints time since program started
  delay(1000);          // wait a second so as not to send massive amounts of data
}
```


# 时间换算

1 second (秒) = 1000 milliseconds (毫秒) = $10^6$ microseconds (微秒) = $10^9$ nanoseconds （纳秒）
