@[toc]

Arduino的字符处理函数相比传统的C/CPP库来说，十分羸弱，不过好过没有。

# isAlpha(thisChar)
该函数是判断输入的字符是否是字母，为真时返回true。


# isAlphaNumeric(thisChar)
该函数是判断输入的字符是否是字母或者数字，为真时返回true。


# isAscii(thisChar)

用于检查参数是否为ASCII 码字符，也就是判断c 的范围是否在0 到127 之间。为真时返回true。

# isControl(thisChar)
判断字符是否属于控制字符类别，为真时返回true

# isDigit(thisChar)
用来检测一个字符是否是十进制数字。

# isGraph(thisChar)
检查所传的字符是否有图形表示法。 带有图形表示法的字符是除了空白字符（比如 ' '）以外的所有可打印的字符。

# isHexadecimalDigit(thisChar)
检查所传字符是否是十六进制位数，为真时返回True。

# isLowerCase(thisChar)

检查所传字符是否是小写字母，为真时返回True。

# isPrintable(thisChar)
是否是可打印字符。


# isPunct(thisChar)
是否是标点符号。


# isSpace(thisChar)

是否是空格字符。



# isUpperCase(thisChar)

是否是大写字符。


# isWhitespace(thisChar)
是否是空白字符，包含TAB，回车等。