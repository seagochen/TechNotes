Ubuntu 20.04 推出有一阵时间了。最近正好有一些项目需要在Linux系统上运行。所以我就自己做了一个虚拟机，试一试现在Ubuntu 20.04这一版本的OpenCV安装是否变得简单一些。

试过之后发现变得确实简单了，所以写在这里供有需要的朋友使用。

# 准备安装环境

~~~bash
	sudo apt install build-essential
	sudo apt install pkg-config
	sudo apt install cmake
~~~

这一步是必不可少的！

# 安装OpenCV4

现在Ubuntu 20.04 已经把最新一代的 OpenCV 集成到了DEB包里了，如果没有版本要求的朋友，可以直接如下命令安装最新的 OpenCV

~~~bash
	sudo apt install libopencv-dev
~~~

经过十几分钟之后，OpenCV 4 就安装好了。

# 编译程序

OpenCV4 与之前的 OpenCV3 最大的区别，我感觉目前还是在函数命名上做了适当的简化，所以为了验证你的系统是否具备了编译OpenCV的能力，可以编译下面这段代码

~~~cpp
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
    destroyAllWindows();

    // return
    return 0;
}
~~~

# PKGConfig

在 OpenCV4 成功安装到Linux系统后，会在 **/usr/lib/x86_64-linux-gnu/pkgconfig** 下建立一个新的名为 **opencv4.pc** 的文件，所以对于编译 OpenCV 来说，也可以简单的以下面这段命令来执行编译过程了。

~~~bash
	g++ opencv_sample.cpp `pkg-config --libs --cflags opencv4`
~~~

之后就是

~~~bash
	a.out /path/to/image.png
~~~

执行输出图像了。

这样一来，是不是发现现在Ubuntu可爱爆了呢？


----

如果这篇文章对你有帮助，点赞！收藏！
你的支持是我继续前进的动力！
Adios~！