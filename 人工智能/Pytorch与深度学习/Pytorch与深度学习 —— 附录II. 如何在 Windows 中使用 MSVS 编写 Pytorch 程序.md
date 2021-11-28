> 如果你想用宇宙第一好用的 C/CXX 编译器 Visual Studio 来安装部署 PyTorch 的 Windows 应用，但是不知道该怎么办好，那么这一篇也许能告诉你答案。

@[toc]

# 前期准备

## Visual Studio

尽管在 Windows 平台上也是可以通过部署CMake工程，来编译 Windows 版的 PyTorch 应用。但是我们放着宇宙第一好用的 Visual Studio 而使用其他的比如 CLion，Code::Blocks 这些工具，未免有些浪费了。毕竟现在Visual Studio都推出免费的社区版了，Debug又好用，安装配置项目也不麻烦，如果在Windows平台都还不使用Visual Studio写C代码，就像不用Eclipse写Java代码，不用PyCharm写Python代码一样，无聊的编程人的人生又要少几分色彩了。

如果在Windows平台还没装过Visual Studio的朋友，请一定通过以下链接去下载一个社区版的来玩一玩，你会感叹有钱的公司做的IDE就是不一样。

> https://visualstudio.microsoft.com/

![在这里插入图片描述](https://img-blog.csdnimg.cn/4e90eaf860b3485fb08bcc0e072c2251.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
如果你是个人开发者，选择Community就行，等待下载完成后，一路Next，大概半小时后就装好了。

## LibTorch

接下来我们需要LibTorch包，也就是PyTorch支持C/C++/Java开发的底层包，在官网可以直接下载。

> https://pytorch.org/

在当前1.9.0版本的时代，根据你的电脑具体配置情况下载对应的库包。如果你的PC支持CUDA运算，那么可以下载支持CUDA版本的库包。

![在这里插入图片描述](https://img-blog.csdnimg.cn/cd7ace4fb5a94aa49a7a5f9fcead8a60.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

不过为了普适性，比如笔记本只有核显，或者是AMD家显卡的朋友，那么就直接下载CPU的版本就好。

## 创建MSVC工程

安装好MSVS后，接下来需要创建一个MSVC工程，在Visual Studio的开始界面里，点击创建一个新工程。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b08ce4b6da7f4a6e89e0839292ed7c6f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_16,color_FFFFFF,t_70,g_se,x_16#pic_center)
然后在工程配置基础信息里，选择 Empty Project，并且决定好你的工程名字和工程代码存放路径。

![在这里插入图片描述](https://img-blog.csdnimg.cn/23c6cf8fd3084243a795266d78e80051.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b46016ef23834e639a4789d466eb8538.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
然后点击Create按钮后，就创建好了。

## 把LibTorch拷贝到工程目录下

工程创建完毕后，为了方便使用，最好把依赖库都拷贝到工程目录下。

![在这里插入图片描述](https://img-blog.csdnimg.cn/fd349d01737147bba6ca6b71d8f22dbf.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

这样我们在配置工程目录链接时，就可以使用VS的一些默认参数，比如

> $(SolutionDir)

而可以不必使用完整路径，并且如果你习惯把工程代码拷贝到U盘，随身携带，这样在新的电脑里打开工程的时候，也不必重新配置一遍全部参数。

接下来，介绍下配置工程的外部链接所需要的具体参数。

# 工程链接配置参数

在工程的工作界面里，你需要右键点击一下项目名称

![在这里插入图片描述](https://img-blog.csdnimg.cn/1103cbcb02d940368907c08048aa0251.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
 或者通过 Project/Properties
![在这里插入图片描述](https://img-blog.csdnimg.cn/eb603a2a39534d019bbd5bf9670c1900.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
打开工程属性配置界面

![在这里插入图片描述](https://img-blog.csdnimg.cn/4944c9653ef34168ad273b31849ddcf5.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
在 “VC++ Directories” 这个选项里，找到右侧的 “Include Directories” 和 “Library Directoies”。在这两项里，点击右侧一个下拉菜单，找到 “Edit” 选项。

![在这里插入图片描述](https://img-blog.csdnimg.cn/88b775d7ac374590a075f030c7ab0d71.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
请分别在 "Include Directories"  里，添加如下内容：

> $(SolutionDir)libtorch\include
> $(SolutionDir)libtorch\include\torch\csrc\api\include

在 “Library Directories” 里添加如下内容

> $(SolutionDir)libtorch\lib

编辑完成后，点击OK保存即可。

**需要注意一点的是，在配置工程平台信息（Platform）时，请选择x64平台，配置文件信息（Configuration）可以选择全平台通用（All Configurations）**

然后就是把工程生成文件类型，由默认的x86改为x64

![在这里插入图片描述](https://img-blog.csdnimg.cn/441063a0ce7c46839fa486d194989e71.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
这是因为LibTorch是x64版本的。

# 准备测试用例

接下来我们需要创建一个新的C++文件，并且写入下面的这段测试代码

~~~cpp
#include <torch/torch.h>
#include <iostream>

#pragma comment(lib, "asmjit.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "fbgemm.lib")
#pragma comment(lib, "torch_cpu.lib")

int main() {
	torch::Tensor tensor = torch::eye(3);
	std::cout << tensor << std::endl;
	return 0;
}
~~~

点击编译按钮后，就会生成可执行文件。


# 拷贝dll动态库
然后我们需要做的就是把 libTorch里全部的dll文件都拷贝到生成的可执行文件目录下。当然你也可以只拷贝这几个：

> asmjit.dll
> c10.dll
> fbgemm.dll
> libiomp5md.dll
> torch_cpu.dll

![在这里插入图片描述](https://img-blog.csdnimg.cn/e6439a0a5fdc4955b59b5376c25b59c6.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

最后再执行程序，就可以看到输出了。


![在这里插入图片描述](https://img-blog.csdnimg.cn/4e7fd2de691d4ffdad5e0d562d011de9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

----

如果你觉得这篇文章对你有用，点个赞再走呗！

