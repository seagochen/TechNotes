
在国内它被称为 **可控硅**，但是海外华人地区则普遍翻译为 **晶闸管**，它的英文名叫 **Thyristor**，是一种早期出现的用于控制电流单向导通的器件。与一般的二极管相比，它可以对导通电流进行控制。晶闸管具有以小电流（电压）控制大电流（电压）作用，并体积小、轻、功耗低、效率高、开关迅速等优点，广泛用于无触点开关、可控整流、逆变、调光、调压、调速等方面[^1]。

[^1]: https://zh.wikipedia.org/wiki/%E6%99%B6%E9%96%98%E7%AE%A1

其电路符号一般表示如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/509fc4376e4e414790a3daca8f3c5094.png#pic_center)

要想理解它的工作原理，可以看看它的等效电路：


![在这里插入图片描述](https://img-blog.csdnimg.cn/b0529cc9b51349eea9c0fc5f7f3e67ab.png#pic_center)

也就是两个三极管彼此相连，其运行方式可以跟EE专业里提到的通过三极管做的 **信号放大电路** 进行类似记忆

![在这里插入图片描述](https://img-blog.csdnimg.cn/dfadac77d5214c56907835cb6f4467e1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

其工作原理是当R4接受到高电平信号并对 Q1 Base 充电后，Q1导通；Q2则由于Q1导通后，其Base接地，于是Q2导通，在Output端输出12V电压信号。而当R4接受到低电平信号后，Q1 Base放电，Q1断开，导致Q2接高电压断开，于是Output接地后输出0V电压信号。


那么对于晶闸管来说，就不需要这么复杂的电路，我们只需要通过Gate控制导通与否，就能实现和上述设计相似的功能。