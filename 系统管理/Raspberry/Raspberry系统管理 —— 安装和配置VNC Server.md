@[toc]

# 什么是VNC Server

通常对于Linux来说，我们通常是把它们当作伺服器使用，很少有人会把Linux系统当作个人PC使用。那么在我们家里，如果使用低功耗的树莓派搭建家里的服务器使用时，除了SSH以外，有时候也会有需要使用远程桌面的时候。比方说，有时候像我们偶尔处理图形图像方面的程序时，不仅需要验证它是否在Windows平台上是否运行良好外，也需要看看它是否在Linux服务器上运作正常；又或者你给树莓派外接了一个摄像头，想用它低成本的监控家里的情况时，SSH是没法实时传回摄像头数据的，所以这个时候就需要VNC，它和Windows的远程桌面很像，但是要方便的多。

由于VNC服务并不是默认就启动的，所以我们需要先在树莓派上配置VNC服务。

# 如何启动树莓派的VNC服务

VNC的相关组件很多已经装在Raspberry-Desktop镜像文件中，我们要做的就是进入SSH，然后执行以下命令：
```bash
$ sudo raspi-config
```

它会打开一个配置界面

![在这里插入图片描述](https://img-blog.csdnimg.cn/3b226200db834b5ab86ce3e478009335.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)
VNC服务一般都在 **Interface Options**，所以我们把光标移动到 **Interface Options** 后，会看到下面的内容：

![在这里插入图片描述](https://img-blog.csdnimg.cn/ed34d1b076f44253ad36ccc15c67629a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)

可以看到有VNC，SSH，Camera等选项，这里我们只使用VNC

![在这里插入图片描述](https://img-blog.csdnimg.cn/eb62f3896b9349c78128b1d27328387e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)
选 **YES** 就行，然后就是等安装配置过程结束就行。

# 安装VNC Viewer

VNC Viewer 是VNC的客户端，如果电脑里没有想过的程序，可以通过这个地址下载VNC Viewer

> https://www.realvnc.com/en/connect/

安装过程很简单，我就不一一说明了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a78b9d72c3594089bf2ff1af23429913.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

连接过程就是把VNC Server的地址输入进去就好了，然后填写用户名和密码，如果不出问题，应该可以使用了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/358e9605bc3a446f83a5f5d172de3f54.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

如果你没法使用VNC，但是SSH又是可以连接到树莓派的话，那大概率你是缺少一个显示器。

# 没有显示器如何使用VNC

对于显卡来说，它在设备启动的Boot阶段，会监测是否有连接到显示器，如果没有连接到显示器，那么桌面通常不能正常显示。解决方案有两个，一个是找个显示器连上你的树莓派；如果没有多余的显示器，你也可以在某宝上，找个 “HDMI 诱骗器”，这样你就可以正常使用VNC服务了。
