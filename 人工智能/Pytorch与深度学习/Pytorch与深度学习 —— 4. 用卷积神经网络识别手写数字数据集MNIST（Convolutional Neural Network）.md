@[toc]

# 为什么改进FNN

在我们掌握了基本的FNN网络后，现在我们来看看FNN的改进型CNN技术吧。FNN（Fully Neural Network）、FC（Fully Connected Network）或者FN（Fully Network）虽然可以处理图片、音频等不同形式的数据，并且具有很好的通用性，但是它的本质是一种全尺度的矩阵运算。

一来，计算量偏大，二来对原始数据的信噪比要求极高，换句话说就是原始数据中如果有太多的噪音，会在很大程度上影响网络收敛的结果和准确性。

所以研究人员就针对这个问题，提出一个想法。我能不能在数据输入到FN前，先对原始信号进行滤波呢？答案当然是可以的，无论对图像数据还是音频数据，只要提到滤波，就会想到经典的傅里叶算法。不过傅里叶算法比较消耗计算资源，而针对傅里叶算法的改进型，有小波、还有卷积，能在一定程度上满足我们的需要。

至于为什么是卷积而不是小波，尽管小波擅长对时域信号的处理，而且擅长滤除噪音和提取关键信号。而卷积，更擅长对空间信号的处理，而且在滤除低频信号的同时，能够保留有效的高频特征信号。这对于特征敏感的人工神经元来说极为重要，至于说小波能不能用在神经元上，我觉得得依据具体得情况分析，说不定在某些领域里卷积的表现反而不如小波。

# 卷积神经元的基本结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/b8efb7c1eba74d07a2afe8e351e476fe.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

这算是一个范式，大多数的卷积神经网络是由上图这样一个结构构成的。通常顺序是：**卷积层1 -> 池化层1 -> 卷积层2 -> 池化层2 -> 全连接层1 -> 全连接层1 -> 再通过一个 sigmoid 类函数输出最后的概率结果**。

CNN网络的基本结构大体上已经固定，目前针对CNN网络的论文，也就是在上述模型的基础上增加或删除一些结构。所以对于MNIST数据集，我们用比较常见的这个结构来做做看，是不是精度上能有所提高。


# 修改FNN代码
我们接下来将针对FNN的MNIST识别器进行修改，不知道FNN代码的童鞋请移步到我的前一篇文章里看看[《Pytorch与深度学习 —— 5.用全连接神经网络识别手写数字数据集MNIST》](https://blog.csdn.net/poisonchry/article/details/119754293?spm=1001.2014.3001.5501)。


## 修改网络模型
这是最重要的一步，原先的FNN网络层代码定义是这样的：

~~~python
    def __init__(self):
        super().__init__()

        # layer definitions
        self.layer_1 = torch.nn.Linear(784, 512)
        self.layer_2 = torch.nn.Linear(512, 256)
        self.layer_3 = torch.nn.Linear(256, 128)
        self.layer_4 = torch.nn.Linear(128, 64)
        self.layer_5 = torch.nn.Linear(64, 10)
~~~

通过组合多个不同维度层级的全连接网络，来分析和处理输入和输出，不过针对CNN来说，我们并不需要这么多的全连接层，此外要把需要卷积层等部分添加上去：

~~~python
class ConvolutionalNeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()

        """
        Data (64, 1, 28, 28) 
            --Cov1(5, 5)--> (64, 10, 24, 24) 
            --Pool(2, 2)--> (64, 10, 12, 12)
            --Cov1(5, 5)--> (64, 20, 8, 8)
            --Pool(2, 2)--> (64, 20, 4, 4)
            --Flatten/FC--> (64, 320)
            --FC(128, 10)-> (64, 10)
        """

        # layer definitions
        self.conv_1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv_2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.fc_3 = torch.nn.Linear(320, 128)
        self.fc_4 = torch.nn.Linear(128, 10)

        # tools / max pooling
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, data):
        # obtain the batch size
        batch_size = data.size(0)

        # do convolutional computations
        data = functional.relu(self.pool(self.conv_1(data)))
        data = functional.relu(self.pool(self.conv_2(data)))

        # transform the tensor to (batch_size, 320)
        data = data.view(batch_size, -1)

        # FNN
        data = functional.relu(self.fc_3(data))
        data = self.fc_4(data)

        # return results
        return data
~~~

至于其他的内容基本可以不用修改，秉承拿来主义我们直接使用，仅仅修改模型后我们居然可以得到精度为98%，这比FNN时的精度95%要高出不少。可以说非常好用。

这里面可能问题比较大的就是什么是卷积，如果你确实不明白什么是卷积，可以看看我的这两篇文章有个大致的了解。

[《卷积——1. 关于卷积的基本概念》](https://seagochen.blog.csdn.net/article/details/114535427)
[《卷积——2. 一些常用于图像的卷积核与应用》](https://seagochen.blog.csdn.net/article/details/114647076)

然后我们来聊聊算法里的提到的池化。

# 池化算法
就是在工程当中，我们经常会遇到一个问题，就是需要处理的数据过多，但是我们确实又不都需要处理这些，所以就会考虑以某种形式，产生出这些数据的“代表”。这个过程呢，通常被称为采样（Sampling）。

采样的算法有很多，而在CNN网络中，使用到了其中一种采样技术，就成为最大池化算法。它的含义是把原始样本按比例进行划分成多个等大小的区域，从每一块中选取最大值，即：
![在这里插入图片描述](https://img-blog.csdnimg.cn/50e1128ad9de42c7b608944c5b7b7628.jpg#pic_center)

在红色块中选取了最大值20，在黄色块中选取了30…… 很明显，这种算法会丢失掉原始数据中的一些关键信息。如果数据分布遵循泊松分布，显然这种算法会导致采样后的样本均值向右侧整体移动。从统计学的角度看，这是一种极为糟糕的采样算法。

但是CNN中为什么会采用这种有偏见的（ｂｉａｓ）的采样算法？你可以观察一下使用特征提取的卷积。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210311004620314.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

经过卷积处理后的图片，会去除图片中低通纹理信息，而保留作为高通数据的边界信息。所以对这样的数据做最大池化算法时，在一定程度上就类似于图片压缩。

所以此时，如果通过最大池化后，很多为0的区域就可以得到压缩，而我们所关心的仅仅是比如一个4x4像素的范围内有无特征信息（特征值是多少）。所以在经过一次池化（核大小为2x2）压缩后，我们会得到一副原图的特征浓缩图（长宽为原图的1/2）。

这就是为什么此时最大池化算法，从概率上说并不会丢失太多的特征信息，而且能加速我们网络的收敛过程。