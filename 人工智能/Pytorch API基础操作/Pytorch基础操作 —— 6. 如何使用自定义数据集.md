
@[toc]

# 自定义数据集

开源数据集比较适合用来测试算法有效性，但是如果真要商用或者自用，以及某些特别的项目，比方说输电线巡检，金属X光探伤，甚至医疗X光片分析这一类针对特定行业时，就需要使用自己自定义的数据集了。

所以，在这一章节里，要介绍给大家的是如何使用自定数据集的方法。

# Step 1. 熟悉你的数据集

你要熟悉自己要使用的数据集，我们的数据集一定是由原始数据和标签这两部分构成的。通过给模型喂数据，模型生成预测，然后预测与标签进行比对计算Loss后，再通过导数更新网络权重这一过程。

即：

$$\mathbb{D_{out}} = Net(\mathbb{D_{in}}) $$

## 有数据就要有标签

标签可以是被检测物体在图片上的几何图形坐标，也可以是某种经过处理后的图片（比如说劣化后的图片作为数据，原始高清图片作为标签），也可以是某种序列和某种序列的对应（比如说，I'm a human 作为数据集输入，标签是 je suis un humain）。

也就是说，训练网络的数据，不是说我随便找一组数据喂给网络，但是不告诉网络我想得到什么信息，让它自己猜。

![在这里插入图片描述](https://img-blog.csdnimg.cn/ef2825b6b2bf4d17ac25608d14e4589c.png#pic_center)
人尚且听到这个答案会气的抽出82年的鞋底想打人，那机器由于没有目标更是不知道应该如何进行收敛。所以一定要确保数据和标签是一一对应的，有数据就一定要有标签！

## 数据大小、维度一定要一样

其次，当下的主流神经网络基本都是流水线作业，这意味着它没法处理大小、纬度不一致的数据，所以喂数据给网络前，你需要把数据的大小、维度等弄成一致的。

以YOLO为例（如果对YOLO感兴趣的童鞋，可以关注我的博客，我会在之后不久告诉你如何从0构建一个属于自己的YOLO网络），它能处理的图像数据是 448 x 448 x 3，如果你喂数据大小超过或不足这个尺寸，会导致网络训练过程中报错。

所以，无论你拉伸、压缩，还是填0，你都应该确保数据喂给网络前都应该符合一样的大小。

## 归一化
归一化是我们在做网络时需要特别注意的地方，在我本人的博客 [Pytorch与深度学习 —— 2.用全连接神经网络识别手写数字数据集MNIST](https://blog.csdn.net/poisonchry/article/details/119754293) 里我虽然简略的带过了归一化这个步骤，但是归一化用的好的话，能提高你的预测精确度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/29bbd14a26e44d95a43da1daa0073519.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

以QMNIST为例，通常情况下，原始的数据都是这样，如果是图片的话，通常以 [0, 255] 灰度值进行表示。尽管我们也可以直接把这样的数据直接喂给网络，但是图片中的信噪比过低，以及由于 [0, 255] 的跳跃过大，会导致一定程度上网络收敛精度下降。

归一化不会导致图片信息的丢失，但是我们可以把数据从比如 [0, 255] 压缩到 [0, 1] 时，这样经过比如sigmoid时，就有很好的表现了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e239d5e30b2b45f6ab29ecf117661eee.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_19,color_FFFFFF,t_70,g_se,x_16#pic_center)
观察一下sigmoid函数，通常对数据在 [-2, 2] 有比较好的敏感性，对于大于这个区间的数据则表现不敏感。试想一下，如果你的网络要处理一个输入为 $x:=500$ 的数，经过线性层处理后，输出的 $y:=100$。

你希望通过sigmoid函数调整线性输出，结果 $\sigma(100) \approx 1$ 那基本等于什么都没做。

归一化的方法有很多，不一一举例，很多归一化函数的和gamma计算有相近的地方，关于这部分内容，你可以参考我之前写过的 [数字图像学笔记——4. 直方图计算、线性变换、对数变换、Gamma变换](https://seagochen.blog.csdn.net/article/details/110204387)。当然，如果你自己不知道用什么函数做归一化，也可以粗暴的使用线性方法，也就是让所有值同时除以元素中最大值，这个计算在小学课本里应该有教过，我就不重复了。

$$\overline{D_i} = \frac{D_i}{D_{max}}$$


# Step 2. 确定如何加载你的数据集
我在上一章里介绍过 [Pytorch基础操作 —— 5. 标准化数据集接口 Dataset 与开源数据集](https://seagochen.blog.csdn.net/article/details/120628595) 官方所提供的所有开源数据集都是通过一个名叫 Dataset 接入到PyTorch的，你也可以照葫芦画瓢。

这个类可以通过如下方法导入：

```python
from torch.utils.data import Dataset
```

然后你需要填充的就是下面这两部分

```python
class MyDataset(Dataset):

    def __init__(self, ...):
		# TODO

    def __getitem__(self, idx):
        # TODO
```

\__init__ 方法和 \__getitem__ 都需要自己来定，分别是确定数据如何加载的，以及数据被访问到的时候用什么形式返回给调用者数据和标签。

比方说，你要实现自定义图片集的加载，**通过某种方法处理了图片数据后**，大概就可以这样实现了

```python
class MyDataset(Dataset):

    def __init__(self, images, labs):
		if len(images) != len(labs):
			print("Data and labels are not the same size")
			exit(1)

		self.data = []
		self.labels = []

		for img, l in zip(images, labs):
			self.data.append(img)
			self.labels.append(l)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

...

dataset = MyDataset(images, labels)
```

当然，实现方法并不一定就是上面这个样子是唯一标准答案，你可以自定义很多东西，这个类只是个接口而已。

# 使用 DataLoader 批量加载数据

当你完成了数据的基本封装后，就可以使用 DataLoader 批量的加载自己的数据集了。DataLoader 不需要你重构，这就是为什么我推荐你使用Dataset来封装你的数据。使用 DataLoader + Dataset 就可以给定义每次训练，喂给网络多少数据了。

导入DataLoader 的方法
```python
from torch.utils.data import DataLoader
```

如何把DataLoader和Dataset关联起来

```python
# 一次处理数据10个
BATCH_SIZE = 10

# 准备数据集
dataset = MyDataset(images, labels)

# 把数据集装载到DataLoader里
train_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
```

在需要训练和测试的地方：

```python

for idx, (data, labels) in enumerate(train_loader, 0):

	# 训练网络
	pred = model(data)

	# 计算损失
	loss = criterion(pred, labels)

	# 反向传播
	loss.backward()

	# 更新参数
	optimizer.step()

	# 清空梯度
	optimizer.zero_grad()
```

## 需要注意的地方
### 确保参与计算的都是Tensor
在data和label参与网络计算前，都应该确保它们已经转化成了Tensor的格式，否则会报错。

### 确保参与计算前数据已经全部Normalized
为了得到更好的结果，所有的数据在参与计算前，应该都 Normalized 到 [0, 1] 区间里，Tensor中应该不存在超过1的数。

### DataLoader 有可能加载到不足Batch的数据
因为数据不会恰好按照你预计的批量加载到网络中，比如你一次想加载10条数据，但总数据有1173条，那么一定在最后一次DataLoader加载时，只会加载到3条数据，所以在网络的forward这步计算里，应该动态的计算维度


比如对于输入数据维度为 $(N,C, W, H)$ 时：
```python
class MyNetwork(Module):

	...

	def forward(self, x):
		# 可以在参与计算前先记录一下数据的维度
		batch = x.size()[0]

		...

		# 最后维度重建时就可以这样
		output = output.view(batch, -1)
		return output
```

但是对于NLP来说，一般会遇到 $(L, N, F)$ 这种维度和 $(N, L, F)$ 这两种维度，在处理前最好先确定在哪些步骤里使用第一个维度，哪些步骤里使用第二个维度，并做必要的转化准备。

