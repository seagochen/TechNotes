
@[toc]


在我的上一篇文章[《Pytorch与深度学习 —— 5.用全连接神经网络识别手写数字数据集MNIST》](https://seagochen.blog.csdn.net/article/details/119754293)里，示范了如何用Pytorch构建最基础的全连接训练网络，并且写了一个基于FNN的MNIST手写数字识别器。PyTorch由于面世的时间晚于Tensorflow，所以在设计之初就考虑过使用CUDA加速神经元网络的收敛，所以从CPU切换至GPU的过程就显得十分简单了。

考虑到这个系列纯粹面向新手，在这一章节里，我们将把这个用于MNIST数据集识别的FNN网络，稍微做一些改动，使得程序能运行在CUDA设备上。

# 什么是CUDA（Compute Unified Device Architecture）

CUDA 是Nvidia家基于自己的显卡并行处理管道所推出的一套可编程的并行计算框架技术，使得图形管线在同一时间对多边体、图形纹理的处理能力可以通用到一般的矩阵计算，所以当CUDA问世时，就注定了这种技术会受到工程与科学界的欢迎，并在之后很快的应用到了流体力学、生物仿真、材料学、核工业等行业中。

同一时期面试的并行框架技术，除了CUDA外，还有OpenCL和OpenMP，OpenCL是由AMD推动的与CUDA抗衡的框架技术，不过由于使用极其困难，所以一直很难流行起来。不过最近几年FPGA的发展，好像OpenCL又渐渐的回到了公众视野。

至于OpenMP，是由开源联盟主导的并行技术，不过与OpenCL和CUDA不同的是，OpenMP是针对CPU提出的一种标准，目前使用OpenMP制作的应用比较少，属于比较小众的技术。

# 准备CUDA设备

既然目前主流的并行技术框架是CUDA，这意味着你需要有一张Nvidia家生产的显卡。显卡的型号最好高于GeForce GTX 1650，这是因为1650以上的显卡采用了目前Nvidia最新的图灵架构的核心，可以最大程度上支持 CUDA 10 和 CUDA 11 的新技术......

| 显卡系列 | 具体型号 |
|------------------------|------------|
GeForce 16系列 | |
| | GeForce GTX 1650
| | GeForce GTX 1650 Super
| | GeForce GTX 1660
| | GeForce GTX 1660 Super
| | GeForce GTX 1660 Ti
GeForce 20系列 | |
| | GeForce RTX 2060
| | GeForce RTX 2060 Super
| | GeForce RTX 2070
| | GeForce RTX 2070 Super
| | GeForce RTX 2080
| | GeForce RTX 2080 Super
| | GeForce RTX 2080 Ti
| | Titan RTX
Nvidia Quadro | |
| | Quadro RTX 4000
| | Quadro RTX 5000
| | Quadro RTX 6000
| | Quadro RTX 8000
Nvidia Tesla | |
| | Tesla T4

如果经费有限，没有条件升级显卡的朋友，只要你的显卡高于GTX  650 基本也可以满足学习深度框架的需要，如果到了需要去做实验或者项目，那就想办法怂恿你的老板给你换一台好一点的台式机吧。


# 准备CUDA环境

对于PyTorch，截止我这篇文章，官方推荐使用CUDA10的技术，也支持CUDA11，你可以选择两者之一的版本进行安装，我个人比较推荐个人选择CUDA 10安装，你可以在官网找到安装包下载：

> https://developer.nvidia.com/zh-cn/cuda-toolkit

如果是Linux用户，可以参考我这篇文章的介绍自行安装 CUDA 10 [《Ubuntu 18.04 环境配置之 CUDA 10.02 + Caffe GPU + OpenCV 3.2 最简安装方式》](https://seagochen.blog.csdn.net/article/details/107225281)。

此外，无论TS还是Torch，除了安装CUDA外，还需要安装cuDNN，你可以在官网找到对应的下载：

> https://developer.nvidia.com/zh-cn/cudnn

为了验证你的CUDA是否准备完毕，建议你最好把 Nvidia CUDA Samples 编译一遍，最起码要把里面的一个名为 deviceQuery 的工具自行编译完成。这样你才能通过运行deviceQuery直到CUDA是否顺利安装成功。

![在这里插入图片描述](https://img-blog.csdnimg.cn/84e59b962c7b43c89a5b3c37cb689c0c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_30,color_FFFFFF,t_70,g_se,x_16#pic_center)


因为我个人平时还有做一些图形学以及CUDA编程的工作，所以我安装的是11的版本，如果你的deviceQuery正常编译后，并且CUDA也是顺利安装成功的，那么运行这个测试工具就应该可以输出你的CUDA设备信息。

## 对于MAC用户怎么装CUDA

没什么好办法，除非你用雷电接口外接一个外置显卡，或者多花几万买一台Mac工作站。当然，如果你能忍受兼容性问题，也可以考虑黑苹果的方案。

## Linux用户安装CUDA总是失败

这是比较容易遇到的问题，一个比较变通的方案是看看软件库是否提供了CUDA，直接使用软件库进行安装要容易许多；如果一定要在Linux环境下安装官方CUDA工具，你可以考虑采用第三方兼容驱动先把Nvidia显卡的驱动打上去，然后安装CUDA的时候，把官方驱动关闭，仅安装库和工具。我目前测试下来这种成功率是最高的，而且使用过程没有遇到什么大问题。说到底，驱动干的事基本上就是沟通显卡和操作系统的数据通道，并且给CUDA工具库提供必要的渲染管线接口，兼容的驱动基本上已经把日常用到的部分全部做好了，官方提供的那部分特性，很多人的显卡用坏的那一天也不一定会用得到。

## Windows提示某某DLL无法找到

Windows对于动态库的访问地址，默认是在

> C:\Windows\System32
> C:\Windows\SysWOW64 (如果你是64位系统的话)

以及程序运行时的exe文件所在目录下，所以你可以有两种方法，一种是把需要DLL拷贝到系统目录里，又或者拷贝到Python的执行目录里，我的Windows系统安装的是CUDA11，而Torch在之前仅支持到CUDA10，所以在当时运行的时候提示缺少了几个CUDA的动态库文件，我就是这样处理的。当然，现在Torch已经支持到CUDA11了，所以这个问题也不存在了。

## Anaconda 如何配置Torch和CUDA支持

Anaconda 确实是比较好用的Python环境控制器，不过对于相关开发环境的准备很多文献和教材说的太恶心了，而且极其不好弄，我不推荐用其他人说的方法。

其实有非常简单的方法准备好相关开发环境。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f9609cb151d84a0a9e9c869d959fb270.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_34,color_FFFFFF,t_70,g_se,x_16#pic_center)

你可以通过Environment下面的Create按钮创建一个新的开发环境，比如叫它torch。等Anaconda准备完毕后，点击新环境右边的那个三角形。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a17a39bc25a54c69815d19d12a8fbfe0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_34,color_FFFFFF,t_70,g_se,x_16#pic_center)
选择Open Terminal，打开一个命名控制器。

如果你安装的是CUDA10.2，执行以下命令：

> pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

如果你安装的是CUDA11，执行以下命令

> pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

等待一定时间后就装好了，非常快捷。然后剩下的就是配置好你的IDE开发环境，比如PyCharm，或者Visual Studio Code，就可以使用了。

# 修改代码

我们现在来看看需要修改什么东西才能让程序在CUDA设备上顺利运行。首先要有一个概念，CUDA设备仅通过PCI-E通道与主机设备进行通信，PCI E通道，是一种高速串行总线技术，由于CUDA设备的时钟周期和主机CPU之间的时钟周期不统一，而GPU的时钟频率很多情况下比CPU要高，所以这决定了CPU无法和GPU进行实时的数据交互。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a92c735450604eeab90f17b5f5a4415b.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_23,color_FFFFFF,t_70,g_se,x_16#pic_center)

所以为了保证性能，高速PCI设备通常会配置有自己专用的运存。PCI E设备和CPU之间，更多情况下彼此之间仅做数据交换工作。这个结构，你可以把整个计算机想象成一家大型的设备加工厂。设备加工厂里有专门负责批量生产设备的车间（GPU），和专门负责设计产品的研发部（CPU）。研发部的工作就是设计好一个产品的参数后，交给车间负责人，然后由车间负责人全权负责包括原材料采购、生产工艺等过程，并且把最终产品生产出来。

那么连接车间和研发部的物理通道，比如高速电梯，就类似于PCI E通道了。


![在这里插入图片描述](https://img-blog.csdnimg.cn/b0b111c208294d9db84eb335a5335b00.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)

在以前的老架构，主板上的南桥芯片负责显卡等PCI设备之间的数据通信任务，当然现在最新的架构，南桥和北桥芯片都被集成到了CPU中，由CPU统一进行调度和控制了。尽管如此，但是CPU和GPU之间的运行，还是相对彼此独立的。

所以，这意味着说，如果我们的程序需要利用好GPU单元，那就需要在开始的时候确定清楚哪些模块是要放到GPU上进行计算的。

一般来说包含如下内容：

* 利用Pytorch构建的网络模型（必须）
* 训练和比对用的基础数据（必须）
* 测试用的数据（选择）

既然这样我们就清楚程序该怎么改了，首先，自然是网络：

~~~python
    # full neural network model
    cpu_model = FullyNeuralNetwork()
    gpu_model = cpu_model.cuda()
~~~


然后就是训练过程中的数据

~~~python
    for batch_idx, data in enumerate(train_loader, 0):

        # convert data to GPU
        cpu_inputs, cpu_target = data
        gpu_inputs = cpu_inputs.cuda()
        gpu_target = cpu_target.cuda()

		...
		
        # forward, backward, update
        gpu_outputs = gpu_model(gpu_inputs)
        gpu_loss = criterion(gpu_outputs, gpu_target)
        gpu_loss.backward()
        optimizer.step()

		...
~~~


如果测试过程可以放在CPU上进行比对，不过也可以放在GPU上，如果是GPU上，就是这样的：


~~~python
     for images, labels in test_loader:
         # convert data to gpu
         gpu_images = images.cuda()

         # test
         gpu_outputs = gpu_model(gpu_images)
         _, gpu_predicated = torch.max(gpu_outputs.data, dim=1)

         # count the accuracy
         total += labels.size(0)
         predicated = gpu_predicated.cpu()
         correct += (predicated == labels).sum().item()
~~~

只是需要注意一个问题，如果GPU计算出的数据要回到CPU上，一定要做一个数据拷贝的过程。

那么最后完整的程序是怎样的呢：

~~~python
import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# global definitions
BATCH_SIZE = 64
MNIST_PATH = "../Data/MNIST"

# transform sequential
transform = transforms.Compose([
    transforms.ToTensor(),
    #                     mean       std
    transforms.Normalize((0.1307,), (0.3081,))
])

# training dataset
train_dataset = datasets.MNIST(root=MNIST_PATH,
                               train=True,
                               download=True,
                               transform=transform)
# training loader
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=BATCH_SIZE)

# test dataset
test_dataset = datasets.MNIST(root=MNIST_PATH,
                              train=False,
                              download=True,
                              transform=transform)
# test loader
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=BATCH_SIZE)


class FullyNeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # layer definitions
        self.layer_1 = torch.nn.Linear(784, 512)   # 28 x 28 = 784 pixels as input
        self.layer_2 = torch.nn.Linear(512, 256)
        self.layer_3 = torch.nn.Linear(256, 128)
        self.layer_4 = torch.nn.Linear(128, 64)
        self.layer_5 = torch.nn.Linear(64, 10)

    def forward(self, data):
        # transform the image view
        x = data.view(-1, 784)

        # do forward calculation
        x = functional.relu(self.layer_1(x))
        x = functional.relu(self.layer_2(x))
        x = functional.relu(self.layer_3(x))
        x = functional.relu(self.layer_4(x))
        x = self.layer_5(x)

        # return results
        return x


def train(epoch, model, criterion, optimizer):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):

        # convert data to GPU
        inputs, target = data
        inputs = inputs.cuda()
        target = target.cuda()

        # clear gradients
        optimizer.zero_grad()

        # forward, backward, update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.cpu().item()
        if batch_idx % 300 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, running_loss / 300))
            running_loss = 0.0


def test(model):
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:
            # convert data to gpu
            images = images.cuda()

            # test
            outputs = model(images)
            _, predicated = torch.max(outputs.data, dim=1)

            # count the accuracy
            total += labels.size(0)
            predicated = predicated.cpu()
            correct += (predicated == labels).sum().item()

    print("Accuracy on test set: %d %%" % (100 * correct / total))


if __name__ == "__main__":

    # full neural network model
    cpu_model = FullyNeuralNetwork()
    gpu_model = cpu_model.cuda()

    # LOSS function
    criterion = torch.nn.CrossEntropyLoss()

    # parameters optimizer
    # stochastic gradient descent
    optimizer = optim.SGD(gpu_model.parameters(), lr=0.1, momentum=0.5)

    # training and do gradient descent calculation
    for epoch in range(5):
        # training data
        train(epoch, gpu_model, criterion, optimizer)

        # test model
        test(gpu_model)
~~~


当然，这个过程在我看来还有很多可以优化的空间，如果你有兴趣的话，可以试一试。