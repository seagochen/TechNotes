@[toc]

就像其他科学框架那样, torch 的基本数据单元是张量 **(Tensor)**。 和数学与物理概念里的张量有所区别的是, torch等相关框架的张量, 更多的是类似于矩阵, 向量这样的概念. 如果你学习过C++, 就类似于Vector的概念. 

它是一块一块的基础数据集合, 与Matlab中的矩阵概念类似. 在 torch 中 tensor 不仅提供了数据存储的功能, 还附带了矩阵, 向量等的计算支持, 在进行大量数据计算中是非常便利的.


![在这里插入图片描述](https://img-blog.csdnimg.cn/00f88ea092fe43e2b0f8be0d3a11ab74.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)

它既可以提供 1x1 的常量(scalar), 也可以提供 1xN 的向量(Vector), 还可以提供 NxM 的矩阵(Matrix), 以及多个长宽相等的矩阵组合在一起的组(Batch).

> 如何理解组的概念?
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/754a6598473f4d1e86f93d722f8ff058.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
>
> 对于一张彩色照片来说,  通常最少包含RGB三个色道(Channel)的数据, 分别用来记录红色分量, 绿色分量和蓝色分量, 而每个分量的像素数是一样的, 像素组成的矩阵大小也是一样的. 那么一张彩色的照片就可以理解为 "一组包含了三个通道" 的矩阵.


正如各种科学框架都需要定义通道输入顺序一样, OpenCV的默认通道规则是BGR, 而OpenGL是RGB, 以Tensorflow为代表的AI框架也有通道顺序的定义规则. 这是非常需要我们使用者注意的地方.


## 张量的通道规则

### 针对于图像分析时的通道规则

对于做图形图像识别相关工作的朋友来说, 我们一般用到4个通道去定义数据. 这4个通道分别是:

* N ---- Batch(es), 批/组
* C ---- Channel(s), 图片颜色通道数
* H ---- Height, 图片高
* W ---- Width, 图片宽

#### Caffe 的通道顺序

> 通常定义为 NCHW

#### Tensorflow 的通道顺序

> 通常定义为 NHWC, 也支持NCHW。

#### Pytorch 的通道顺序

> 通常定义为 NCHW

#### Numpy 的通道顺序

稍微有些不太一样

> 通常定义为 HWC


为什么通道顺序定义会这样呢, 主要是因为各种框架开发时间先后不一致，以及在设计时候的理念不一致导致。比方说，最早一批的深度学习框架，例如 **Tensorflow**，因为刚推出时，基于GPU的计算还不那么普及，而基于CPU的 **OpenMP** 是当时常用的并行处理框架技术，所以用**NCHW**格式对缓存利用率更高。而后面几种框架, 因为支持了Nvidia cuDNN, 使用GPU加速时，**NHWC**这种格式效率和速度更快.

在网络设计过程中，一定要充分考虑两种格式，最好能灵活切换，在 GPU 上训练时使用 **NCHW** 格式，在 CPU 上做预测时使用 **NHWC** 格式。

### 针对自然语言分析时的通道规则

自然语言和图像处理，是深度学习框架最常见的两类问题，但是自然语言处理所采用的规则稍微与处理图像时的顺序不一样。

通常为 **(L, B, F)** ，以处理单词来说即：

**L: Sequence**

即这个单词包含有多少个字符，对于单词 ‘abc’ 来说，它包含3个字符；

**B: Batch**

一个Batch对应一个单词，比如上面提到的 ‘abc’；

**F: Features**

即每一个字符所对应的 One-Hot 编码。


# Tensor 的基本属性

关于张量来说，最重要的有三种基本属性，其一是数据的维度、其二是数据的类型，其三是数据所在的数据设备上。

## 获取张量基本属性
~~~python
# 获取数据维度
print(f"Shape of tensor: {tensor.shape}")

# 获取数据类型
print(f"Datatype of tensor: {tensor.dtype}")

# 获取数据所在的设备类型，CPU或GPU数据
print(f"Device tensor is stored on: {tensor.device}")
~~~

输出结果：

```shell
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```
## 获取张量维度还可以使用 size 命令
除此以外，我们还可以使用size来获取张量的维度，这也是代码里常见的用法，不过我比较推荐使用shape，因为如果我们想要修改张量的维度，可以直接对应使用 **reshape** 命令。

~~~python
tensor.size()
~~~