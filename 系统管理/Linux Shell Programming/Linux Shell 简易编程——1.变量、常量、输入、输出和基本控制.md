> Linux Bash 是非常轻量化的脚本，在所有的Linux系统和Mac中都能完美支持。Win10自从有了Linux子系统后，也能在Windows 10上使用，其实是挺便利的一种工具。
> 它诞生的时间非常久，所以也导致了这个工具的语法十分古老。你说为什么不用类似Python一类的新时代的脚本语言做同样的事，最主要的原因是使用Python编程在处理系统这一层还是比较费劲和繁琐的。
> 通俗的一句话就是---------懒

> 我记得第一次看Shell的时候，还是差不多五六年前吧，忘了是叫什么名字，但记得是一本很厚的红皮书，看的叫一个头疼。尽管可以实现很多功能，例如正则匹配，文件搜索等。但在实际工作中，比较复杂的运算通常会交给程序来负责，而在文件系统这一层，主要应用在环境或资源的配置上。

> Shell平时用的并不多，不过关键的时候可以省去很多时间。所以我想了想，在结合一些资料的基础上做一个总结，虽然是为了我自己使用方便，不过也希望能帮到你。


主要参考资料：

https://www.runoob.com/linux/linux-shell-process-control.html
https://devconnected.com/how-to-check-if-file-or-directory-exists-in-bash/
https://linuxhandbook.com/bash-arguments/
https://ryanstutorials.net/bash-scripting-tutorial/bash-if-statements.php
https://bash.cyberciti.biz/guide/Pass_arguments_into_a_function
https://linuxize.com/post/bash-exit/
https://linuxize.com/post/bash-printf-command/
https://www.cyberciti.biz/faq/linux-bash-exit-status-set-exit-statusin-bash/
https://www.cyberciti.biz/faq/how-to-check-os-version-in-linux-command-line/


@[toc]

# 变量和常量

在很多脚本中，你能见到这样类似的代码行：

```shell
PREFIX_PATH=/usr/home/foo/folder
```

或者这种：

```shell
#!/bin/bash
result=`uname -a`
echo $result
```

这里的 PREFIX_PATH 和 result 类似其他编程语言中的常量和变量，只不过有所区别的是，Shell里并不强制定义，大小写通常随意。但从使用习惯上，我们通常以全大写的作为常量，并且在定义后通常不会去修改它的值，而小写的是变量。

在Shell中，使用之前定义的变量或者常量，通常要在名称前加上美元符号。

另外就是Shell中没有大小写检测，也没有类型强制要求，自己写脚本的时候注意命名规范即可了，至少你自己或者其他人能看懂。

# 输入和输出

## 使用echo进行输出

其中一个最关键的应用就是输入和输出。你能常见的一个范例就是使用echo进行输出，例如执行如下指令：

```shell
#!/bin/bash
echo "Hello World !"
```

执行后输出的结果是：

```
% bash test.sh 
Hello World !
```

## 使用printf进行输出

此外还有使用print进行输出

```shell
#!/bin/bash
printf "Hello, Shell\n"
```

## 参数式输入

shell作为一个脚本，通常会带着参数一并提交给系统进行处理，例如读取文件或者配置信息，然后启动程序这类操作。

```shell
#!/bin/bash
echo "param1: $1"
echo "param2: $2"
```

然后这样使用：

```
% bash test.sh hello world
param1: hello
param2: world
```

## 提示式输入

提示输入是令一种常见的输入方式，例如你写了一个脚本，需要在某个时候输入一个文件地址，并根据一些特殊的规则处理它的时候，就可以用到。

通常我们使用 **read** 命令来实现这一效果，例如：

```shell
#!/bin/bash
read -p "Enter Your Name: "  username
echo "Welcome $username!"
```


# 控制结构

## if... elif... else... 结构

所有程序的最基本流程控制结构，有选择执行才能称为程序，Shell也不例外，基本结构是这样的：

```shell
if condition_1
then
    commands_1
elif condition_2
then
	commands_2
else
    commands_3
fi
```

示例如下，例如比较两个数的大小，写成段落形式需要注意缩进的统一：

```shell
a=10
b=20
if [ $a == $b ]
then
   echo "a == b"
elif [ $a -gt $b ]
then
   echo "a > b"
elif [ $a -lt $b ]
then
   echo "a < b"
else
   echo "no valid condition"
fi
```

也可以写成这样子：

```shell
a=10
b=20
if [ $a == $b ]; then
   echo "a == b"
elif [ $a -gt $b ]; then
   echo "a > b"
elif [ $a -lt $b ]; then
   echo "a < b"
else
   echo "no valid condition"
fi
```

写成一行也可以，但是一定要注意加分号！

```shell
a=10
b=20
if [ $a == $b ]; then echo "a == b"; elif [ $a -gt $b ]; then echo "a > b"; elif [ $a -lt $b ]; then echo "a < b"; else echo "no valid condition"; fi
```

## for 结构

基本机构是这样的
```shell
for var in item1 item2 ... itemN
do
    commands
done
```

示例：

```shell
for loop in 1 2 3 4 5
do
    echo "The value is: $loop"
done
```
执行结果：

```
% bash test.sh
The value is: 1
The value is: 2
The value is: 3
The value is: 4
The value is: 5
```

同样，也可以写成一行，不影响执行结果，同样需要注意分号的使用！

```shell
#!/bin/bash
for loop in 1 2 3 4 5; do echo "The value is: $loop"; done
```

## while 结构

基本结构是这样的

```shell
while condition
do
    command
done
```

示例如下：

```shell
#!/bin/bash
num=1
while(( $num<=5 ))
do
    echo $int
    let "num++"
done
```


## 无限循环

类似其他语言中的while-true循环，

```swift
	while true {
		...
	}
```

虽然不常使用，但也有一定的应用场景。比如说测试远程服务器是否下线，需要每5秒发送一次ping的话，那么就可以使用这种无限循环了。



在shell中，它的语法格式是：

```shell
while :
do
    command
done
```

或者
```shell
while true
do
    command
done
```

亦或者
```shell
for (( ; ; ))
```

示例如下：
```shell
#!/bin/bash
while true
do
    echo "hello world"
done
```


## until 循环
与while循环相似，不是很常见的一种循环方式。它会循环执行一系列命令直至条件为 true 时停止。until 循环与 while 循环在处理方式上刚好相反。

一般 while 循环优于 until 循环，但在某些时候—也只是极少数情况下，until 循环更加有用。

```shell
until condition
do
    command
done
```
condition 一般为条件表达式，如果返回值为 false，则继续执行循环体内的语句，否则跳出循环。
以下实例我们使用 until 命令来输出 0 ~ 9 的数字：
实例
```shell
#!/bin/bash

a=0

until [ ! $a -lt 10 ]
do
   echo $a
   a=`expr $a + 1`
done
```


# 类switch选择执行

除了少数奇葩，大多数语言都提供了switch这种强力工具。我在查阅了资料后，令我惊奇的是Shell居然也提供了这种类似的语言特性。

它的基本语法结构是，**case ... esac**，然后它的每个 case 分支用右圆括号开始，用两个分号 ;; 表示 break，即执行结束，跳出整个 case ... esac 语句，esac（就是 case 反过来）作为结束标记。

可以用 case 语句匹配一个值与一个模式，如果匹配成功，执行相匹配的命令。语法格式如下：

```shell
case 比较值 in
条件1)
    command1
    command2
    ...
    commandN
    ;;
条件2）
    command1
    command2
    ...
    commandN
    ;;
esac
```

这里直接摘抄一下runoob中的内容：
> case 工作方式如上所示，取值后面必须为单词 in，每一模式必须以右括号结束。取值可以为变量或常数，匹配发现取值符合某一模式后，其间所有命令开始执行直至 ;;。
取值将检测匹配的每一个模式。一旦模式匹配，则执行完匹配模式相应命令后不再继续其他模式。如果无一匹配模式，使用星号 * 捕获该值，再执行后面的命令。

例子：
```shell
echo '输入 1 到 4 之间的数字:'
read -p "你输入的数字为: "  aNum
case $aNum in
    1)  echo '你选择了 1'
    ;;
    2)  echo '你选择了 2'
    ;;
    3)  echo '你选择了 3'
    ;;
    4)  echo '你选择了 4'
    ;;
    *)  echo '你没有输入 1 到 4 之间的数字'
    ;;
esac
```


# 中断循环指令

在大多数的编程语言中，都会有在满足某个条件后，跳出当前循环的语言特性，通常大家默认的是break和continue这两个关键字。对于Shell来说，这也是通用的，效果一样。

**break**

```shell
#!/bin/bash
while :
do
    echo -n "输入 1 到 5 之间的数字:"
    read aNum
    case $aNum in
        1|2|3|4|5) echo "你输入的数字为 $aNum!"
        ;;
        *) echo "你输入的数字不是 1 到 5 之间的! 游戏结束"
            break
        ;;
    esac
done
```

**continue**

```shell
#!/bin/bash
while :
do
    echo -n "输入 1 到 5 之间的数字: "
    read aNum
    case $aNum in
        1|2|3|4|5) echo "你输入的数字为 $aNum!"
        ;;
        *) echo "你输入的数字不是 1 到 5 之间的!"
            continue
            echo "游戏结束"
        ;;
    esac
done
```

# 测试比较

测试比较，通常在条件控制语句中出现，用于判断一个数的大小，文件存在与否，字符串中是否包含某些特定字符等。

## 数值大小的比较

命令 | 说明 | 举例 | 其他语言中的表示方式
----- | ------ | ----- | ------
-eq	| 等于则为真      | if [ $num -eq 10 ] | num == 10
-ne	| 不等于则为真  | if [ $num -ne 10 ] | num != 10
-gt	| 大于则为真      | if [ $num -gt 10 ] | num > 10
-ge | 大于等于则为真 | if [$num -ge 10 ] | num >= 10
-lt | 小于则为真 | if [ $num -lt 10 ] | num < 10
-le | 小于等于则为真 | if [ $num -le 10 ] | num <= 10


## 字符串测试

Runoob上提供的是比较简单的字符串比较，而对于比较复杂的，例如包含正则匹配、通配符匹配等操作，就没有怎么提及。

一个比较简单的例子：

```shell
#!/bin/bash
str="message"
if [ $str = "message" ]; then
    echo "true"
else
    echo "false"
fi
```

参数 | 说明 | 举例 
-----|-----|------
=	| 等于则为真 | if [ $str = "message" ]
!= | 不相等则为真 | if [ $str != "message" ]
-z | 判断 string 是否是空串 | if [ -z $str ]
-n | 判断 string 是否是非空串 | if [ -n $str ]

## 字符串长度测试

利用 **${#str}** 获取字符串长度

```shell
#!/bin/bash
str="message"
if [ ${#str} -lt 4 ]; then
    echo "true"
else
    echo "false"
fi
```

## 文件测试
参数 | 说明 | 举例
-----|------|------
-e |  如果文件存在则为真 | if [ -e $path ]
-r |  如果文件存在且可读则为真 | if [ -r $path ]
-w |  如果文件存在且可写则为真 | if [ -w $path ]
-x | 如果文件存在且可执行则为真 | if [ -x $path ]
-s |  如果文件存在且至少有一个字符则为真 | if [ -s $path ]
-d | 如果文件存在且为目录则为真 | if [ -d $path ]
-f | 如果文件存在且为普通文件则为真 | if [ -f $path ]
-c | 如果文件存在且为字符型特殊文件则为真 | if [ -c $path ]
-b | 如果文件存在且为块特殊文件则为真 | if [ -b $path ]