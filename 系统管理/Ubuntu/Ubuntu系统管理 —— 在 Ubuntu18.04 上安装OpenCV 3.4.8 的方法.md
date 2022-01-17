@[toc]

# OpenCV

OpenCV很好用，但是很难装好，而且对于国内用户来说，由于某些特殊原因，安装过程总会爆出缺这缺那的错误。目前经过我的仔细研究后，找到了一条算是最简安装的方法，可以用最少的包，尽可能把OpenCV安装完成。

# 前期准备

首先，需要一个干净的Ubuntu 18.04系统，目前我还没有在其他版本上测试过，但是在18.04的系统上，无论虚拟机还是真机都成功的完成OpenCV的安装。如果你的系统安装过OpenCV或者其他，建议先把已经安装过的包尽可能删除掉，保证安装环境的干净，减少不必要的问题产生。

	$ sudo apt install build-essential

首先需要安装编译器，其次我们需要安装cmake/git/pkg-config等工具，但cmake稍微有点特别，请不要通过apt的仓库进行安装，最好选择从官网下载最新的cmake包，建议直接下载shell版本的，这样就只需要解压后，直接可用。

比如我下载的是cmake-3.16.1-Linux-x86_64.sh的版本，直接执行如下命令，就可以解压出最新版本的cmake

	$ sh cmake-3.16.1-Linux-x86_64.sh

将解压出的cmake，放在/opt目录下，然后用ln命令，创建对cmake的引用。

	$ sudo ln -s /opt/cmake/cmake-3.16.1-Linux-86_64/bin/cmake /usr/bin/cmake

然后执行apt，安装剩下的工具：

	$ sudo apt install pkg-config git 

然后就是准备OpenCV所需要的库：

	$ sudo apt install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev qt5-default libv4l-dev libeigen3-dev libavresample-dev

这些库已经经过测试，至少目前不存在无法安装的问题，至于以后就要看看官方的文档是怎么要求的了。

然后是一些可选库，建议也全部安装，因为涉及一些图片、视频压缩转码，或者对程序进行加速，所以还是需要的。

	$ sudo apt install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

# 下载OpenCV
直接从Github上下载相关代码

	$ git clone https://github.com/opencv/opencv
	$ git clone https://github.com/opencv/opencv_contrib

你如果用git下载困难，也可以直接下载对应的版本包，建议选择3.4.8版本，作为v3版本的OpenCV来说，这是目前最稳定的包，而v4版本的包，目前还处于新特性开发中，可能会带来一些不确定的问题。

下载完成后，分别cd到opencv和opencv_contrib目录下，用git checkout 3.4.8切换到版本3.4.8。

# CMAKE
接下来，在opencv的目录下，创建一个build目录，然后使用如下命令

	cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/opt/opencv/opencv.3.4.8 \
        -D WITH_TBB=ON \
        -D WITH_V4L=ON \
        -D WITH_QT=ON \
        -D WITH_OPENGL=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        ..

这些命令是目前测试下来最少依赖的，如果你需要其他什么特殊功能，再自行添加相关命令。

此外，cmake过程中，由于需要下载ippicv，还有其他一些编译过程中需要的文件，由于某原因的存在，可能无法下载。此时建议最好是翻墙后做再cmake，如果无法翻墙，那么你需要手动下载相关的包。对于ippicv还算好，这个我已经下载了，至于其他的tgg文件，你可能需要自行去下载，至于编译过程中缺少什么重要文件，在CMakeFiles文件夹里的CmakeError.log文件里，你也可以cmake的过程中用tail -f命令同步查看错误输出。

说回ippicv文件，你下载后，把它放在和opencv同级的目录里，然后cd到opencv/3rdparty/ippicv的ippicv.cmake文件，修改第47行的：

	"https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}/ippicv/"
为opencv所在目录级，比如你把ippicv文件下载到了/home/pee/downloads，那么就是：

	“file:///home/pee/downloads/”

# 最后
编译顺利通过后，用make命令编译后，再用sudo make install进行安装。

之后，你需要把opencv.pc文件拷贝到 **/usr/lib/x86_64-linux-gnu/pkgconfig/** 下，opencv.pc文件就在你cmake的**CMAKE_INSTALL_PREFIX**地址的 **lib/pkgconfig** 文件下。

然后你可以执行

	$ pkg-config --cflags --libs opencv

可以得到c编译所须的全部参数，至于那些库so文件怎么引导，你既可以通过设置bash的方式，也可以直接把**CMAKE_INSTALL_PREFIX**下的/lib文件地址添加到/etc/ld.so.conf文件中，比如：

	$ echo /opt/opencv/opencv.3.4.8/lib >> /etc/ld.so.conf

然后执行：

	$ sudo ldconfig -v

之后尝试编译一个简单的opencv代码就可以了

```c++
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main(int, char *argv[])
{
    Mat img;
        
    //读取原始图像
    img = imread(argv[1]);
    if (img.empty()) {
        //检查是否读取图像
        cout << "Error! Input image cannot be read...\n";
        return -1;
    }

    //创建窗口
    namedWindow(argv[1]);

    //显示图片
    imshow(argv[1], img);

    // 退出
    cout << "Press any key to exit...\n";
    waitKey(); // Wait for key press
    cvDestroyAllWindows();

    // return
    return 0;
}
```

执行如下命令：

	$ g++ source.cpp `pkg-config --cflags --libs opencv`

# 写在最后，关于Python
这个安装方法是不会产生任何Python可用的so的，我认为最方便的方法，是直接用pip3 install opencv-python，别人都给你整好了，何苦为难自己呢，如果你是用Python写OpenCV的童鞋，直接一句命令搞定得了。