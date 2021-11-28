


@[toc]

这是Pytorch里非常重要的工具类，它的主要工作就是提供给使用者一个工厂化的数据接入方法。在 **torchvision**  这个包里，就使用它对接了诸多开源数据集，方便使用者调用。


# 使用官方的数据集 Torch Vision
如果要做某方面的训练，我们先来看看有什么开源数据集是我们可以直接使用的。如果需要详细了解这个资料库的信息，请访问[官方主页](https://pytorch.org/vision/stable/index.html)。

我们先来看看官方提供的补充包 torchvision 里提供了哪些东西。

## torchvision.datasets

这里罗列了你可以直接使用，并且已经完成格式转换后的开源数据集。

* Caltech
	> 行车记录仪照片数据集
	> 相关地址:  http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

* CelebA
	> 人脸数据集
	> 相关地址：https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
	
* CIFAR
	> 物体分类数据集
	>相关地址：https://www.cs.toronto.edu/~kriz/cifar.html
	
* Cityscapes
	> 城市景观数据集
	> 相关地址：https://www.cityscapes-dataset.com/
	
* COCO
	> 物体识别、分类数据集
	> 相关地址：https://cocodataset.org
	
* EMNIST
	> 手写数字及字母数据集，比较适合新手学习使用
	> 相关地址：https://www.nist.gov/itl/products-and-services/emnist-dataset
	
* FakeData
	> 随机RGB图像
	
* Fashion-MNIST
	> 10类由衣服、鞋子组成的数据集，比较适合新手学习使用。
	> 相关地址：https://www.kaggle.com/zalando-research/fashionmnist
	
* Flickr
	> 标准的基于文字描述的图像数据集。
	> 相关地址：https://www.kaggle.com/hsankesara/flickr-image-dataset
	
* HMDB51
	> 包含51类各种动作视频数据集。
	> 相关地址：https://www.kaggle.com/fengqianpang/hmdb51
	
* ImageNet
	> 物体识别数据集。
	> 相关地址：https://www.image-net.org/
	
* Kinetics-400
	> 人类动作数据集。
	> 相关地址：https://deepmind.com/research/open-source/kinetics
	
* KITTI
	> 自动驾驶相关数据集。
	> 相关地址：http://www.cvlibs.net/datasets/kitti/
	
* KMNIST
	> 日语手写平假名数据集。
	> 相关地址：http://codh.rois.ac.jp/kmnist/index.html.en
	
* LSUN
	>  场景分类数据集，10种不同类型的场景图片。
	> 相关地址：https://www.yf.io/p/lsun
	
* MNIST
	> 手写数字集
	> 相关地址：http://yann.lecun.com/exdb/mnist/
	
* Omniglot
	> 5050 个不同字母的手写字符数据集。
	> 相关地址：https://www.omniglot.com/
	
* PhotoTour
	> 世界知名景点建筑照片数据集。
	> 相关地址：http://phototour.cs.washington.edu/
	
* Places365
	> 场景分类集。
	> 相关地址：http://places2.csail.mit.edu/
	
* QMNIST
	> 对MNIST数据集重构后的手写数字数据集。
	> 相关地址：https://paperswithcode.com/dataset/qmnist
	
* SBD
	> 语义边界数据集。
	> 相关地址：https://paperswithcode.com/dataset/sbd
	
* SBU
	> 人体动作数据集。
	> 相关地址：https://paperswithcode.com/dataset/sbu-captions-dataset
	
* SEMEION
	> 手写数字集.
	> 相关地址：https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit
	
* STL10
	> 图片分类数据集
	> 相关地址：https://cs.stanford.edu/~acoates/stl10/
	
* SVHN
	> 谷歌街景门牌号
	> 相关地址：https://paperswithcode.com/dataset/svhn
	
* UCF101
	> 人体动作数据集。
	> 相关地址：https://paperswithcode.com/dataset/ucf101
	
* USPS
	> 手写邮邮政数据集。
	> 相关地址：https://paperswithcode.com/dataset/usps
	
* VOC
	> 目标检测数据集。
	> 相关地址：https://paperswithcode.com/dataset/pascal-voc
	
* WIDERFace
	> 人脸识别数据集。
	> 相关地址：http://shuoyang1213.me/WIDERFACE/
	
## torchvision.io
如果你想使用一些自定义的数据集，比如自定义的图片或者视频训练你的网络，可能你会需要用到 torchvision.io

## torchvision.models
在models这个包内，提供了很多已经公开的网络模型，所以你可以直接，或者做一些修改后使用。另外现在经常提到的预训练这个技术，也很喜欢使用这里提供的公开模型。所以对于想进一步了解

## torchvision.ops
这是一个和图像拼接、剪切、转换等操作相关的包，如果有需要对图像批量处理，可以使用这个包里提供的工具。

## torchvision.transforms
这个包提供了很多把普通照片转换为Tensor的方法，我是比较喜欢自己实现类似方法，毕竟可以自由的控制各种图片的大小尺度的。

需要注意的是，如果自己实现图片到Tensor的转换。如果从图片输入到模型时，最好对图片做个归一化的转换，把所有模型中涉及到的数据，以 [0, 1] 区间的浮点数的形式进行表示，直接把图片不加处理的输入网络，会导致网络收敛、泛化能力大大减弱。

## torchvision.utils
对于物体识别来说，经常要用到的一个工具性质的功能，就是在找出的物体上绘制区域框，或者是所谓的 bounding box，这个包里就提供了这样的工具。当然你也可以自己用 OpenCV 或者 OpenGL 来做同样的事情。
