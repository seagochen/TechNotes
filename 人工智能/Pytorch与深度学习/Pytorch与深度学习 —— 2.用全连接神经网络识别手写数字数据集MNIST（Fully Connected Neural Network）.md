@[toc]

我们使用全连接，来构建第一个神经元网络。这是迈向深度学习的第一步，之后的卷积神经网络、循环神经网络等，都是在全连接神经网络的基础上发展而来的。它的基本机构如下，由一个输入层、一个或一个以上的隐藏层，即一个输出层组成。

![在这里插入图片描述](https://img-blog.csdnimg.cn/18da44b839b94e05b64341c792647338.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
说起全连接网络能做什么呢，由于“结构不可知（structure agnostic）”，也就是说，不需要对输入做出某种假设，换句话说，模型不关心用户输入的是视频、图像还是一般数据（例如股票交易价格的历史数据），都可以使用这种结构处理问题。

所以，全连接网络模型也可以算是深度学习里的万金油，只不过与所有万金油一样，能够处理全部问题，也意味着不能精通全部问题。作为“通用逼近器”，自然会在某些问题上表现不如人意，体现在比方说收敛速度过慢、精度过低等。所以在之后的一些列 Deep Learning 方面的研究，多是对特定问题，提出的改进模型。

所以，从这个意义上说，掌握了全连接网络模型，你将拥有处理大量简单、基本问题的能力。

# 准备数据集

在上一章节里，我给出了建立一个深度学习模型所需要的四步骤，你所需要做的就是像做填空题一样，把缺少的部分补充进来。

首先我们要准备的自然是数据了，这里我们使用很多教科书都会提到的MNIST数据集。

![在这里插入图片描述](https://img-blog.csdnimg.cn/bd150fbd6d2c41748125ca1f5b60d59f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

MNIST 数据集是非常有意思的开源数据集，是一组早期研究机器学习而收集一组手写数字的数据集。经过标准化后，以单字长宽28像素灰度值的形式建立的数据集（28 x 28, 0-255）。对于目前主流的数学框架，都提供了接入、下载、加载MNIST的封装方法，通常可以直接调用使用。

不过你也可以自己去网上下载数据集本尊，然后自己写代码解析数据。不过我本人已经把数据集传到CSDN的资源分享，你可以点击这个链接直接[下载](https://download.csdn.net/download/poisonchry/21451625?spm=1001.2014.3001.5501)。

不过你也可以直接访问老杨[^1] 的网站，[下载](http://yann.lecun.com/exdb/mnist)这个数据集。

[^1]: Yann André LeCun, 目前仍在世的深度学习教父之一，关于老杨头的生平有兴趣且英文好的童鞋可以去看看 https://en.wikipedia.org/wiki/Yann_LeCun

不过如果你是自己下载这个数据集，那么你需要自己解析这个代码。我在 www.monitor1379.com 找到一位原作者是monitor1379的朋友写的解析代码，并且在他基础上做了一点修改，然后根据代码协议可以开源，所以我就把代码贴在这里，有需要的请遵守开源协议后使用。

~~~python
# encoding: utf-8
"""
@author: monitor1379
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: mnist_decoder.py
@time: 2016/8/16 20:03

对MNIST手写数字数据文件转换为bmp图片文件格式。
数据集下载地址为http://yann.lecun.com/exdb/mnist。
相关格式转换见官网以及代码注释。

========================
关于IDX文件格式的解析规则：
========================
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""

import struct

import matplotlib.pyplot as plt
import numpy as np

# 训练集文件
train_images_idx3_ubyte_file = './Data/MNIST/train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = './Data/MNIST/train-labels-idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = './Data/MNIST/t10k-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = './Data/MNIST/t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    print('done')


if __name__ == '__main__':
    run()

~~~

## 导入工具库

除了直接下原数据集的，我们还可以使用PyTorch提供的工具直接使用：

~~~python
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
~~~

先说说 DataLoader，Torch提供的数据加载器（数据遍历器），我们可以根据需要把数据集分为训练集和测试集，并且提供把原始数据集打乱的功能。

然后是datasets，这部分内容属于torchvision这个模组，你可以根据自己需要决定要不要装torchvision，在torchvision中的datasets模块，提供了大量主流的图像开源数据集。对于NLP相关的朋友，你可以使用torchnlp，关于这方面的详细介绍参考[官网](https://pytorchnlp.readthedocs.io/en/latest/index.html)。

接下来就是transforms，这个模块提供了基础的图像剪切、拉伸、归一化等操作，并且提供了可以把字节数据转化为特定标准的Tensor，便于我们在以Tensor为基础的网络模型中使用。

## 准备数据

0-255的字节数据显然是不能直接用在Torch网络中的，如果使用第二种方法加载数据集，那么就需要先定义一个把字节数据转化成Tensor的方法，并按照一定标准归一化数据。

~~~python
transform = transforms.Compose([
    transforms.ToTensor(),
    #                     mean       std
    transforms.Normalize((0.1307,), (0.3081,))
])
~~~

我们使用compose函数创造一个操作指令序列，第一步是先将PLI或者Numpy数据转化成Tensor，所以 **transforms.ToTensor()**，然后我们使用归一化指令，把数据以正态形式重新进行归一化。这样做的目的可以把图片中存在的噪音，或者一些与被检测物体无关的背景弱化掉，更好的突出主体[^2]。

[^2]: 【CV】图像标准化与归一化，https://blog.csdn.net/xylin1012/article/details/81217988

然后，我们接下来分别准备测试集和训练集的数据。

~~~python
train_dataset = datasets.MNIST(root='dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=8)
~~~

**datasets.MNIST** 就是告诉torchvision，在这个项目里我要使用MNIST这个数据集，root指示的是数据集存储位置；

download设置为true，就是告诉程序我们需要下载数据集，但是如果数据集已经下载过则不用下载；

train设置为True，告诉程序本次的对象是训练集，datasets 会自动的从数据集里随机的抽取足够数量的数据作为训练集使用，通常采用8-2分配，即训练集会抽取数据集中80%的样本；

transform，是对这些数据的转化工作，我们在前面已经定义过了。

关于MNIST的详细说明，可以[参考Torch官网](https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html)。然后就是测试集……

~~~python
test_dataset = datasets.MNIST(root='dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)

test_loader = DataLoader(test_dataset,
                         shuffle=True,
                         batch_size=8)
~~~

**DataLoader** 作为数据加载器，有很多参数，详细的说明请[参考官网](https://pytorch.org/docs/stable/data.html)。我们在这里主要用到这几个：

分别是数据来源，对于测试集的 Dataloader 来说就是 test_dataset；

然后就是数据是否要打乱，为了最大程度检验准确性，我们令 shuffle=True；

最后就是 batch_size，一个网络如果一次一次的送数据，训练过程必然十分缓慢，所以我们自然会考虑一次性多送一些数据给设备使用。

由于这部分涉及一些优化工作，如果有需要，你可以参考一下[官网的文章](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。


# 准备网络模型

![在这里插入图片描述](https://img-blog.csdnimg.cn/2a4aaaf875ad463898d8e855a659b39d.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

构建网络层，不是越多越好，也不是越少越好。由于全连接层随着层数的增加，能够识别和分析的数据特征就越多，极端情况下甚至会学习到噪音，所以我们需要设置合理的层数。不过对于这个MNIST例子来说，多少层的问题都不大，这里我们试着建立一个五层的结构看看。

~~~python
class FullyNeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # layer definitions
        self.layer_1 = torch.nn.Linear(784, 512)   # 28 x 28 = 784 as input
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
~~~

我们让特征从最初的784 ($28 \times 28 \rightarrow 1 \times 784$) 然后在每一层的计算中逐步缩小维度，到最后输出10个特征权重（分别对应 0 - 9）。

然后在forward函数里，除了最后一层，在其他层我们分别在输出的部分使用了激活函数（relu），关于激活函数的定义可以在我[前面的文章](https://seagochen.blog.csdn.net/article/details/118526467)里找到说明。



# 定义训练过程

训练过程，我们需要考虑的是针对我们的问题，选择什么样的损失函数和优化器更为合适。这里由于是分类问题，所以应该使用交叉熵，而优化器我们使用随机梯度下降就可以了。

~~~python
    # LOSS function
    criterion = torch.nn.CrossEntropyLoss()

    # parameters optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5) # stochastic gradient descent
~~~

动量设置一个0.5，是为了避免优化器落入局部最优点（鞍点[^3]），而错过全局最优点。学习率为0.1。

[^3]: 【深度学习】鞍点，https://blog.csdn.net/baidu_27643275/article/details/79250537

接下来就是定义学习过程：

~~~python
def train(epoch, model):

	# running loss
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward, backward, update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, running_loss / 100))
            running_loss = 0.0
~~~

# 测试过程

训练结束后，我们要做一个测试，看看效果如何。

~~~python
def test(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicated = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicated == labels).sum().item()

    print("Accuracy on test set: %d %%" % (100 * correct / total))
~~~

这一步做了什么工作呢，首先我们从test数据集拿出测试的图片和对应标签。然后从每一次的计算结果里，选出最有可能的部分（预测的结果），然后把预测的结果和标签进行比对，然后得出结果的准确率。

# 完整的程序

现在我们来看看把每一部分都整合起来，看看完整的程序是什么样的。

~~~python
import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# global definitions
BATCH_SIZE = 100
MNIST_PATH = "../../../Data/MNIST"

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
        inputs, target = data
        optimizer.zero_grad()

        # forward, backward, update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, running_loss / 100))
            running_loss = 0.0


def test(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicated = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicated == labels).sum().item()

    print("Accuracy on test set: %d %%" % (100 * correct / total))


if __name__ == "__main__":

    # full neural network model
    model = FullyNeuralNetwork()

    # LOSS function
    criterion = torch.nn.CrossEntropyLoss()

    # parameters optimizer
    # stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

    # training and do gradient descent calculation
    for epoch in range(5):
        # training data
        train(epoch, model, criterion, optimizer)

        # test model
        test(model)
~~~

最后这个模型的准确度大概在95%-97%，还不错哟。
