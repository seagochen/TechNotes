@[toc]

# 安装前的一些说明

这文章里的安装方法，由于主要用到APT，所以在Ubuntu 18.04 版本上，对于OpenCV最多只支持到3.2。当然你可以考虑参考我之前的文章[《Ubuntu 18.04 最简安装 OpenCV 3.4.8 的方法》](https://blog.csdn.net/poisonchry/article/details/103610544)。

使用系统自带安装包的优点，在于可以十分方便而且安装过程可以变得非常简单。当然缺点就是没法使用一些最新的功能。不过对于3代的OpenCV来说，不同版本的最大区别，其实是变量参数名字的一些变化，大的变化并没有太多，以及修复一些bug之类的。所以如果你已经看了我之前的说明装了其他版本的OpenCV，并不需要在这里卸载，使用APT安装的，会在以后引用头文件和库文件的时候默认使用系统自带的。

但是如果要使用自己装的版本，那么在给GCC编译传递参数时，主动指明引用路径就好。又或者你用 PKG 文件来管理多版本的OpenCV的使用也是可以的，具体的一些方法我前面的文章，这里就不过多引述了。

# 安装 CUDA 10.02

我安装CUDA这一类框架时，一直遵循一个习惯，就是上一代技术的最后一个更新版，通常这意味着更少的错误。截止目前文章发布，Nvidia发布的CUDA最新版本是 11，而上一代最后一个更新是CUDA 10.02 相关的cuDNN也是8，这些可以在Nvidia开发者官网里找到下载链接，我就不在这里提供。

需要注意的是，如果你要安装cuDNN，那么应该同时下载DEV包和运行环境包，他们分别包含了cuDNN的开发头文件和库，和可执行文件，所以都是我们需要的。

CUDA的 RUN 包下载后，用 shell 进行加载，或者 chmod a+x 赋予运行权限后，执行在bash里执行。

accept 许可协议后，会出现一个选项框，通常包含 Nvidia驱动安装，CUDA工具库安装，CUDA Sample安装，Nvidia工具库安装。很多文章包括Nvidia官网都推荐你卸载系统自带的驱动，然后安装包里带的驱动，然后搞出一大堆麻烦事，其实如果**是 Ubuntu Desktop 版的，我们实际上可以通过安装扩展驱动来规避这一系列问题，而且我做CUDA编程主要都是进行Kernel级的，运行也良好，没看到有什么问题。**

## 关于显卡驱动的安装快速简易方法

扩展驱动的安装方法非常傻瓜，在 Ubuntu 的 Docker 界面里，选择 有地球图标的 **Software and Update** 然后在 **Additional Drivers**，如果你本身有N卡，那么过一会后，会在这个界面里看到一些Nvidia的开源显卡驱动。

这些显卡驱动早些年是由社区来做的，所以兼容性很差，但是这些年Nvidia自身也加入了这个开发，所以现在稳定性要好很多了。选择版本号最新的，我当前使用的版本号是440，然后确认后过一会就可以安装好了。

## 回到CUDA的安装过程
驱动安装的过程中，我们可以继续CUDA工具的安装，由于显卡驱动使用第三方，所以在安装选项里，把驱动关闭掉，只保留CUDA工具选项，最多再加上Nvidia显卡工具，CUDA Samples包，实际上你要图省事，只把安装驱动关掉也行。然后选择继续安装过程就行了。

驱动和CUDA工具库安装好后，一般会建议你重启电脑。如果你装了CUDA Sample 包，那么cd 到Sample的地址，不修改地址的情况下，一般会被默认安装到 /usr/local/cuda 这个地址里
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070912344454.png#pic_center)
你可以用 make 把全部的样例编译一遍，当然有些样例可能会失败，比如你系统里少装了一些什么库，也有可能提示缺少什么头文件，但不要紧，make -k 解决一切。

我是将全部样例包都安装在本地目录下，编译完成后可以得到一系列的可运行程序。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200709123816603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
然后 cd 到 bin 目录下，这里最关键的是看看 deviceQuery 有没有正常安装，然后执行这个可执行文件。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200709123930666.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
这表明全部安装成功，且运行良好。

# 剩下的安装过程
只要不在安装的过程中，修改相关路径，那么接下来的安装就会特别容易。

## 安装 caffe-gpu
caffe-gpu 是 Ubuntu 发布的caffe的GPU版本。我们安装这个，只需要执行以下命令：

```
	sudo apt install caffe-cuda libopencv-dev libatlas-base-dev
	sudo apt install libleveldb-dev libsnappy-dev libhdf5-serial-dev 
	sudo apt install libboost-all-dev libgflags-dev libgoogle-glog-dev 
	sudo apt install liblmdb-dev libprotobuf-dev protobuf-compiler
```

然后就是等待全部安装完毕......

==这些安装步骤能保证你的C/C++的代码能够正常的运行使用CUDA和Caffe，但是如果使用Python，有可能会遇到一些版本问题，而导致某些包没法使用，本身由于我本人的主要工作环境是C/C++。所以一些使用Python可能会出现的问题，请参考下面的建议==

# 题外话1：关于使用 pip 安装python包，和使用apt安装python包的区别。

使用上没什么大的区别，不过apt安装后，默认的包地址都被放在了

```
	/usr/lib/python3/dist-packages
```
或
```
	/usr/lib/python3.6/dist-packages
```
而使用pip的安装模式，python包都会被放在 **./local/lib/python** 目录下。

下面，所以如果发现了包依赖出现了版本冲突，或者功能冲突，你在确定具体，比如 caffe 的依赖包的具体要求后，可以尝试先分别用 sudo apt remove 和 pip uninstall 把冲突包，或者依赖错误包卸载掉。

然后用比如 pip3 install package_name==1.2.3 的方式，安装需要的相关包以及版本。而且本身caffe很久没有更新了，所以对于新版numpy的支持都有一些问题。我在一些测试工作上，以及很少用python写caffe的代码。

# 题外话2：关于Windows使用Tensorflow，提示cannot load cudart64_xxx.dll文件的解决方法
我的一部分测试开发测试工作是由Windows来做的，Tensorflow的安装本身并不复杂，只不过要用到GPU进行加速的时候，如果使用过新的CUDA，会出现找不到相关DLL的问题。解决方法倒是也简单，在CUDA的默认安装路径下，比如：

```
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin
```
找到和它同名的动态库文件，然后 ctrl+c/v 大法，把创建的副本命名为找不到的那个版本，比如提示：找不到 cudart64_101.dll，就把cudart64_110.dll 建一个副本后重名为 cudart64_101.dll 即可。

如果还报错，就是注意一下：环境PATH里，是否都加入了

* C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin
* C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64

本身不加入这些地址是没有问题的，因为CUDA安装后，会自动在系统变量里加入这个地址，但是还是报错的话，就手动在Path里增加这些地址吧。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200709135158496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

祝，好运！