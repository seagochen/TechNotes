@[toc]

在深度学习相关论文中，经常可以看到一堆“生物学”方面的名词，例如 **backbone（脊梁骨、脊椎）**，**head（脑袋）**等。那么它们是什么意思呢，这篇文章我们就来做深度学习方面的词汇扫盲。

# Ablation Study / 部分切除学习

> removing features from your model one by one to see how much each one individually contributes to the performance. Common to see in research papers about new model architectures that contain many novel contributions.

一项一项地从模型中删除特征，以查看每个特征对性能的贡献。 这是一种用来分析新模型和功能的常用手段。

# Accuracy / 精确度
> proportion of "correct" vs "incorrect" predictions a model makes. Common in classification models that have a single correct answer (vs object detection where there is a gradient from "perfect" to  "pretty close" to "completely wrong".) Often terms such as "top-5 accuracy" are used which means "how much of the time was the correct answer in the model's top 5 most confident predictions?" Top-1 accuracy and Top-3 accuracy are also common.

模型做出的“正确”与“不正确”预测的比例。 在具有单一正确答案的分类模型中很常见（与对象检测相比，其中存在从“完美”到“非常接近”到“完全错误”的梯度）。

# Activation / 激活函数

> The equation of a neural network cell that transforms data as it passes through the network. See activation function.

对网络输出进行非线性转换的一类函数，见[激活函数](https://seagochen.blog.csdn.net/article/details/120960751)。

# Anchor Box / 锚箱，边界框
> common in object detection models to help predict the location of bounding boxes.

在目标检测应用中，用来标定物体所在位置的边界框。

# Annotation / 标记
> the "answer key" for each image. Annotations are markup placed on an image (bounding boxes for object detection, polygons or a segmentation map for segmentation) to teach the model the ground truth.

目标检测时，用来标记被检测出的物体的名字或答案。


# Annotation Format / 标记格式

> the particular way of encoding an annotation. There are many ways to describe a bounding box's size and position (JSON, XML, TXT, etc) and to delineate which annotation goes with which image.

通常指的是标记的存储格式，大多数标记以TXT、XML、JSON格式存储，开发者需要对标记格式进行解析后才能使用标记。

# Annotation Group / 标记组
> describes what types of object you are identifying. For example, "Chess Pieces" or "Vehicles". Classes (eg "rook", "pawn") are members of an annotation group.

这个术语不太常用，更多会用Categories或者Class，它表示标记所属的组别，例如“人类”，“鸟”，“车”等。

# Architecture / 架构
> a specific neural network layout (layers, neurons, blocks, etc). These often come in multiple sizes whose design is similar except for the number of parameters. For example, EfficientDet ranges from D0 (smallest) to D7 (largest).

描述神经网络结构的术语，特定的神经网络布局（层、神经元、块等）。 这些通常有多种尺寸，除了参数数量外，它们的设计相似。 例如，EfficientDet 的范围从 D0（最小）到 D7（最大）。

# AUC / 曲线下面积
> Area Under the Curve. An evaluation metric for the efficacy of a prediction system that is trading off precision at the expense of recall. The precision recall curve is downward sloping as a predictions algorithms confidence is decreased, to allow more, but less precise predictions.

模型有效性的一种评价指标。由PR曲线得到，它用来预测系统有效性的评估指标，以牺牲召回率为代价来权衡精度。 随着预测算法置信度的降低，精确召回曲线向下倾斜，以允许更多但不太精确的预测。

# Augmentation / 增加训练集
> creating more training examples by distorting your input images so your model doesn't overfit on specific training examples. For example, you may flip, rotate, blur, or add noise.

通过扭曲输入图像来创建更多训练示例，这样您的模型就不会在特定训练示例上过度拟合。 例如，您可以翻转、旋转、模糊或添加噪声。

# AutoML
> one-click to train models that optimize themselves (usually hosted in the cloud). They can be a good starting point, a good baseline, and in some cases a "it just works" solution vs tuning your own models.

更多的是一种概念，同时也是一种自动化的机器学习工具。它内置了大量通用模块，可以允许用户仅输入数据和标签，通过查找找出最合适的模型，并自动化地学习，使得机器学习任务可以在无人工干预的情况下即可被使用。

当前提供的AutoML大多是收费的，且价格不菲。


# Backbone / 主干网络
> an object detection model is made up of three parts, a head, a neck, and a backbone. The backbone is the “base” classification model that the object detection model is based on.

这个主干网络大多时候指的是提取特征的网络，其作用就是提取图片中的信息，共后面的网络使用。

这些网络经常使用的是resnet、VGG等，而不是我们自己设计的网络，因为这些网络已经证明了在分类等问题上的特征提取能力是很强的。在用这些网络作为backbone的时候，都是直接加载官方已经训练好的模型参数，后面接着我们自己的网络。让网络的这两个部分同时进行训练，因为加载的backbone模型已经具有提取特征的能力了，在我们的训练过程中，会对他进行微调，使得其更适合于我们自己的任务。

# Backprop / Back propagation / 反向传播
> Back propagation is the way that neural networks improve themselves. For each batch of training data, they do a “forward pass” through the network and then find the direction of the “gradient” of each neuron in each layer from the end working backwards and adjust it a little bit in the direction that most reduces the loss function. Over millions of iterations, they get better bit by bit and this is how they “learn” to fit the training data.

网络用来更新参数权重的过程，关于反向传播的具体细节，可以参考我的[文章](https://seagochen.blog.csdn.net/article/details/118082114)。

# Bag of Freebies / 免费赠品
> a collection of augmentation techniques that have been shown to improve performance regardless of model architecture. YOLOv4 and YOLOv5 have built these techniques into their training pipeline to improve performance over YOLOv3 without dramatically changing the model architecture.

一组增强技术已被证明可以提高性能，而不管模型架构如何。 YOLOv4 和 YOLOv5 已将这些技术构建到他们的训练管道中，以在不显着改变模型架构的情况下提高 YOLOv3 的性能。

# Batch Inference / 批量推理
> making predictions on many frames at once to take advantage of the GPU’s ability to perform parallel operations. This can help improve performance if you are doing offline (as opposed to real-time) prediction. It increases throughput (but not FPS).

利用 GPU 执行并行操作的能力一次对多帧进行预测。 如果您进行离线（而不是实时）预测，这有助于提高性能。 它增加了吞吐量（但不是 FPS）。

# Batch Size / 批大小
> the number of images your model is training on in each step. This is a hyperparameter you can adjust. There are pros (faster training) and cons (increased memory usage) to increasing the batch size. It can also affect the model’s overall accuracy (and there is a bit of an art to choosing a good batch size as it is dependent on a number of factors). You may want to experiment with larger or smaller batch sizes.


模型在每一步中训练的数据数量。 这是可以调整的超参数。 增加批量大小有优点（更快的训练）和缺点（增加内存使用量）。 它还可以影响模型的整体准确性（并且选择一个好的批量大小需要一些技巧，因为它取决于许多因素）。 

# BCCD / 血细胞计数和检测数据集
> blood cell count and detection dataset. A set of blood cell images taken under a microscope that we commonly used for experimentation.

血细胞计数和检测数据集，在显微镜下拍摄的一组血细胞图像。

# Black Box / 黑盒
> a system that makes it hard to peek behind the curtain to understand what is going on. Neural networks are often described as black boxes because it can be hard to explain “why” they are making a particular prediction. Model explainability is currently a hot topic and field of study.

一个很难窥视幕后了解正在发生的事情的系统。 神经网络通常被描述为黑盒，因为很难解释它们“为什么”做出特定预测。 模型可解释性是当前的热门话题和研究领域。

# Block / 块
>  to simplify their description and creation, many computer vision models are composed of various “blocks” which describe a set of inter-connected neurons. You can think of them a bit like LEGO bricks; they interoperate with each other and various configurations of blocks make up a layer (and many layers make up a model).

为了简化它们的描述和创建，许多计算机视觉模型由描述一组相互连接的神经元的各种“块”组成。 你可以把它们想象成乐高积木； 它们彼此互操作，块的各种配置构成一个层（许多层构成一个模型）。

# Bounding Box / 边界框
> a rectangular region of an image containing an object. Commonly described by its min/max x/y positions or a center point (x/y) and its width and height (w/h) along with its class label.

包含对象的图像的矩形区域。 通常由其最小/最大 x/y 位置或中心点 (x/y) 及其宽度和高度 (w/h) 及其类别标签来描述。

# Channel / 通道
> images are composed of one or more channels. A channel has one value for each pixel in the image. A grayscale image may have one channel describing the brightness of each pixel. A color image may have three channels (one for red, green, and blue or hue, saturation, lightness respectively). A fourth channel is sometimes used for depth or transparency.

图像由一个或多个通道组成。 一个通道对于图像中的每个像素都有一个值。 灰度图像可能有一个通道来描述每个像素的亮度。 彩色图像可能具有三个通道（一个分别用于红色、绿色和蓝色或色调、饱和度、亮度）。 第四个通道有时用于深度或透明度。

# Checkpoint / 检查点
> a point-in-time snapshot of your model’s weights. Oftentimes you will capture a checkpoint at the end of each epoch so you can go back to it later if your model’s performance degrades because it starts to overfit.

模型权重的时间点快照。 通常，您会在每个 epoch 结束时捕获一个检查点，以便在您的模型由于开始过度拟合而性能下降时可以返回到它。

# Class / 类别
> a type of thing to be identified. For example, a model identifying pieces on a Chess board might have the following classes: white-pawn, black-pawn, white-rook, black-rook, white-knight, black-knight, white-bishop, black-bishop, white-queen, black-queen, white-king, black-king. The Annotation Group in this instance would be “Chess Pieces”.

一种待识别的事物。 例如，识别棋盘上棋子的模型可能具有以下类别：白棋、黑棋、白车、黑车、白骑士、黑骑士、白象、黑象、白 -皇后，黑皇后，白王，黑王。 在这种情况下，注释组将是“棋子”。

# Class Balance / 类别平衡
> the relative distribution between the number of examples of each class. Models generally perform better if there is a relatively even number of examples for each class. If there are too few of a particular class, that class is “under-represented”. If there are many more instances of a particular class, that class is “over-represented”.

每个类的示例数量之间的相对分布。 如果每个类的示例数量相对偶数，则模型通常会表现得更好。 如果某个特定类别的人数太少，则该类别“代表性不足”。 如果某个特定类有更多实例，则该类被“过度代表”。

# Classification / 分类任务
> a type of computer vision task that aims to determine only whether a certain class is present in an image (but not its location).

一种计算机视觉任务，旨在仅确定图像中是否存在某个类别（而不是其位置）。

# COCO
> the Microsoft Common Objects in Context dataset contains over 2 million images in 80 classes (ranging from “person” to “handbag” to “sink”). MS COCO is a standard dataset used to benchmark different models to compare their performance. Its JSON annotation format has also become commonly used for other datasets.

Microsoft Common Objects in Context 数据集包含 80 个类（从“人”到“手提包”再到“水槽”）中超过 200 万张图像。 MS COCO 是一个标准数据集，用于对不同模型进行基准测试以比较其性能。 其 JSON 注释格式也已普遍用于其他数据集。

# Colab / Google Colaboratory
> Google Colaboratory is a free platform that provides hosted Jupyter Notebooks connected to free GPUs.

Google Colaboratory 是一个免费平台，可提供连接到免费 GPU 的托管 Jupyter Notebook。

# Computer Vision / 机器视觉
> the field pertaining to making sense of imagery. Images are just a collection of pixel values; with computer vision we can take those pixels and gain understanding of what they represent.

与图像有关的研究领域。我们可以通过机器视觉，获取图像像素背后的含义（例如图片测量深度信息、机械物件的损伤等）。

# Confidence / 置信度
>  A model is inherently statistical. Along with its prediction, it also outputs a confidence value that quantifies how “sure” it is that its prediction is correct.

模型本质上是统计的。 除了预测之外，它还输出一个置信值，用于量化其预测正确的“确定性”。

# Confidence Threshold / 置信度阈值
> we often discard predictions that fall below a certain bar. This bar is the confidence threshold.

我们经常丢弃低于某个条的预测。 该条是置信度阈值。

# Converge / 收敛
> over time we hope our models get closer and closer to a hypothetical “most accurate” set of weights. The march towards this maximum performance is called converging. The opposite of convergence is divergence, where a model gets off track and gets worse and worse over time.

随着时间的推移，我们希望我们的模型越来越接近假设的“最准确”权重集。 迈向这种目标的过程称为收敛。 收敛的反面是发散，其中模型偏离轨道并且随着时间的推移变得越来越糟。

# Convert / 转换
> taking annotations or images in one format and translating them into another format. Each model requires input in a specific format; if our data is not already in that format we need to convert it with a custom script or a tool like Roboflow.

以一种格式获取注释或图像，然后将它们翻译成另一种格式。 每个模型都需要特定格式的输入； 如果我们的数据还不是那种格式，我们需要使用自定义脚本或 Roboflow 之类的工具对其进行转换。

# Convolution / 卷积
> a convolution is a type of block that helps a model learn information about relationships between nearby pixels.

一种特殊的数学矩阵工具，可以通过卷积提取或加强被卷积数据的某些特征。

# Convolutional Neural Network (CNN, ConvNet) / 卷积神经网络

> the most common type of network used in computer vision. By combining many convolutional layers, it can learn about more and more complex concepts. The early layers learn about things like horizontal, vertical, and diagonal lines and blocks of similar colors, the middle layers learn about combinations of those features like textures and corners, and the final layers learn to combine those features into identifying higher level concepts like “ears” and “clocks”.

计算机视觉中最常见的网络类型。 通过组合许多卷积层，它可以学习越来越复杂的概念。 早期层学习水平、垂直和对角线以及相似颜色的块等内容，中间层学习纹理和角落等特征的组合，最后一层学习将这些特征组合成识别更高级别的概念，例如“ 耳朵”和“时钟”。

# CoreML 
> A proprietary format used to encode weights for Apple devices that takes advantage of the hardware accelerated neural engine present on iPhone and iPad devices.

应用于Apple设备的框架技术，它利用了 iPhone 和 iPad 设备上的硬件加速神经引擎。

# CreateML 
>  A no-code training tool created by Apple that will train machine learning models and export to CoreML. It supports classification and object detection along with several types of non computer-vision models (such as sound, activity, and text classification).

由 Apple 创建的无代码训练工具，可训练机器学习模型并导出到 CoreML。 它支持分类和对象检测以及多种类型的非计算机视觉模型（例如声音、活动和文本分类）。

# Cross Validation / 交叉验证
在机器学习中常用的一种评价模型准确度的训练、验证方法，它的一种扩展叫K-折交叉验证。

# CUDA
> NVIDIA’s method of creating general-purpose GPU-optimized code. This is how we are able to use GPU devices originally designed for 3d games to accelerate neural networks.

NVIDIA 创建通用 GPU 优化代码的方法。 这就是我们能够使用最初为 3d 游戏设计的 GPU 设备来加速神经网络的方式。

# CuDNN
> NVIDIA’s CUDA Deep Neural Network library is a set of tools built on top of CUDA pertaining specifically to efficiently running neural networks on the GPU.

NVIDIA 的 CUDA 深度神经网络库是一组建立在 CUDA 之上的工具，专门用于在 GPU 上高效运行神经网络。

# Custom Dataset / 自定义数据集
> a set of images and annotations pertaining to a domain specific problem. In contrast to a research benchmark dataset like COCO or Pascal VOC.

# Darknet
> A C-based neural network framework created and popularized by PJ Reddie, the inventor of the YOLO family of object detection models.

由 YOLO 系列对象检测模型的发明者 Reddie 创建并推广的基于 C 的神经网络框架，目前基本没人使用。

# Data / 数据
> information of any kind. It could be images, text, sound, or tabular.

在深度学习中，可以是图片、文字、声音或者其他形式的数据。

# Dataset / 数据集
> a collection of data and a ground truth of outputs that you use to train a machine learning model by example. For object detection this would be your set of images (data) and annotations (ground truth) that you would like your model to learn to predict.

在深度学习中，指有真实值的数据的集合。对于图像和视频方面的公开数据集，可以参考这篇[文章内容](https://seagochen.blog.csdn.net/article/details/120628595)。

# Deploy / 部署
> taking the results of a trained model and using them to do inference on real world data. This could mean hosting a model on a server or installing it to an edge device.

获取经过训练的模型的结果并使用它们对现实世界的数据进行推理。 这可能意味着在服务器上托管模型或将其安装到边缘设备。

# Differentiable / 可微的
> in order for backprop to work, all of the operations that the neural network performs must be able to have their derivative calculated in order to determine the gradient.

为了让反向传播起作用，神经网络执行的所有操作都必须能够计算它们的导数以确定梯度。

# Distributed / 分布式
> spread across multiple devices. Distributed training usually means using multiple GPUs (often located on separate physical machines) to train your model.

分布在多个设备上。 分布式训练通常意味着使用多个 GPU（通常位于不同的物理机器上）来训练您的模型。

# Domain Specific / 特定领域
> problems or techniques that are not generally applicable. For example, if you’re trying to detect tumors in X-Rays, anything that has to do with the cancer biology is domain-specific because it wouldn’t apply to someone working on measuring traffic flows via satellite imagery.

不普遍适用的问题或技术。 例如，如果在 X 射线中检测肿瘤，任何与癌症生物学有关的事情都是特定领域的。

# Early Stopping / 提前停止
> detecting when your model has reached peak performance and terminating the training job prior to “completion.” There are a number of heuristics you can use to determine your model has reached a local maximum; stopping early can prevent overfitting and save you from wasting time and compute resources.

检测您的模型何时达到最佳性能并在“完成”之前终止训练作业。 您可以使用多种启发式方法来确定您的模型已达到局部最大值； 提前停止可以防止过度拟合并避免浪费时间和计算资源。

# Edge Deployment / 边缘部署
> deploying to a device that will make predictions without uploading the data to a central server over the Internet. This could be an iPhone or Android device, a Raspberry Pi, a NVIDIA Jetson, a robot, or even a full computer with a GPU located on-site.

部署到无需通过 Internet 将数据上传到中央服务器即可进行预测的设备。 这可能是 iPhone 或 Android 设备、Raspberry Pi、NVIDIA Jetson、机器人，甚至是一台带有 GPU 的完整计算机。

# EMA / 指数移动平均
> exponential moving average. Helps to smooth noisy inputs.

指数移动平均，有助于平滑嘈杂的输入。


# Embedding / 嵌入层

嵌入层（Embedding Layer）是使用在模型第一层的网络层，其目的是将所有索引标号映射到致密的低维向量中。比方说使用One-hot 方法编码的向量会很高维也很稀疏。而使用嵌入层，则会把One-Hot为代表的稀疏矩阵转换为数个关键特征表示的稠密矩阵，从而减少数据维度。

对于比如“deep”，如果使用One-Hot进行编码，如果词典数量多达2000个，那么表示 “deep”
的稀疏向量的长度可能为2000，其中仅有1个有效数据。而经过嵌入层优化后，由于提取了词典的关键特征，使得它可采用极少的比如7-8个与关键特征来表示同一个单词，用类似 [.32, .02, .48, .21, .56, .15] 的向量进行表达。

# Epochs / 代
>  the number of times to run through your training data.

运行训练数据的循环次数。

# EXIF
> metadata attached to images (for example, orientation, GPS data, information about the capture device, shutter speed, f-stop, etc).

附加到图像的元数据（例如，方向、GPS 数据、有关捕获设备的信息、快门速度、光圈等）。

# Export / 导出
> In Roboflow, an export is a serialized version of a dataset that can be downloaded.

# F1
> A measure of efficacy of a prediction system. F1 is a combination of recall (guessing enough times) with precision (guessing correctly when the system does guess). High F1 means guessing correctly when there is a guess to be made.

预测系统有效性的度量。 F1 是召回率（猜测足够次数）和精确度（当系统猜测时正确猜测）的组合。 高 F1 意味着在有猜测时正确猜测。

# False Negative / 漏测率
> when your model fails to predict an object that is actually present.

本应该被检测到的样本，但是没有检测到。

# False Positive / 误检率
> when you model predicts that an object is present when it actually isn’t.

不存在的样本被检测到了。

# Family / 族
> a set of models that are all related to each other. For example, the YOLO family of models follow a lineage from YOLOv1 all the way to YOLOv5. The core concepts of the models are all the same but they have had new techniques and improvements bolted on as time has progressed.

一组相互关联的模型。 例如，YOLO 系列模型遵循从 YOLOv1 一直到 YOLOv5 的谱系。 这些模型的核心概念都是相同的，但随着时间的推移，它们有了新的技术和改进。

# FastAI 
> A library built on top of PyTorch for rapid prototyping and experimentation. There is a companion course that teaches the fundamentals of machine learning.

一个建立在 PyTorch 之上的库，用于快速原型设计和实验。 有一门配套课程教授机器学习的基础知识。

# Feature / 特征
> a derived property of your data that is learned by your model. For example, a set of convolutions may learn how to recognize zigzag lines in images. Zigzag lines are then a learned feature.

模型学习到的数据的派生属性。 例如，一组卷积可以学习如何识别图像中的锯齿线。 之字形线则是学习到的特征。

# Feature Fusion / 特征融合
> Combining derivative data features in a neural network.

在神经网络中结合衍生数据特征。

# Feature Pyramid Network (FPN) / 特征金字塔网络
> A basic feature fusion strategy in object detectors. Combines convolutional neural network features sequentially.

目标检测器中的基本特征融合策略。 按顺序组合卷积神经网络特征。

# Filter Null / 空滤除
> removing some proportion of null examples from your dataset so that your model doesn't learn to optimize its loss function by predicting “null” too often.

从您的数据集中删除一定比例的空示例，以便您的模型不会通过过于频繁地预测“空”来学习优化其损失函数。

# FLIR 
> forward looking infrared. Infrared measures the heat of objects in the infrared spectrum rather than the color of an object in the visual spectrum. Models can be trained on infrared images as well as visual.

前视红外线。 红外线测量红外光谱中物体的热量，而不是可见光谱中物体的颜色。 模型可以在红外图像和视觉图像上进行训练。

# FLOPS / 每秒浮点运算
> floating point operations per second (used as a measure of computing power). For example, you may see a GPU purport to do 8 TFLOPS which menas 8 trillion floating point operations per second.

每秒浮点运算（用作计算能力的度量）。 例如，您可能会看到一个 GPU 声称要执行 8 TFLOPS，这意味着每秒执行 8 万亿次浮点运算。

# FP8 
> 8-bit floating point. (Also known as quarter-precision.) Reducing the precision of your model can improve its speed and accuracy and can also take advantage of features of newer GPUs like tensor cores.

8比特浮点数

# FP16
> 16-bit floating point. (Also known as half-precision.)

16比特浮点数

# FPS / 帧率
> frames per second. In real-time inference, this is the measure of how many sequential inference operations a model can perform. A higher number means a faster model.

每秒帧数。 在实时推理中，这是衡量模型可以执行多少顺序推理操作的指标。 更高的数字意味着更快的模型。

# Framework / 框架
> Deep learning frameworks implement neural network concepts. Some are designed for training and inference - TensorFlow, PyTorch, FastAI, etc. And others are designed particularly for speedy inference - OpenVino, TensorRT, etc.

深度学习框架实现了神经网络概念。 有些是为训练和推理而设计的——TensorFlow、PyTorch、FastAI 等。还有一些是专门为快速推理而设计的——OpenVino、TensorRT 等。

# GAN Synthesis / 对抗生成
> using a generative adversarial network to create more training data.

使用生成对抗网络来创建更多的训练数据。

# Generalize / 泛化
> the ability of a model to make accurate predictions on input data it has never seen before.

模型对从未见过的输入数据进行准确预测的能力。

# Generate / 生成
> in Roboflow, generating images means processing them into their final form (including preprocessing and augmenting them).

通常指最终生成的数据，比如图像的预处理、和图像增强

# GPU / 图形处理单元 / 显卡
> graphics processing unit. Originally developed for use with 3d games, they’re very good at performing matrix operations which happen to be the foundation of neural networks. Training on a GPU lets your model’s calculations run in parallel which is vastly faster than the serial operations a CPU performs (for the subset of operations they are capable of).

图形处理单元。 最初开发用于 3d 游戏，它们非常擅长执行矩阵运算，而这恰好是神经网络的基础。 在 GPU 上训练让您的模型计算并行运行，这比 CPU 执行的串行操作（对于它们能够执行的操作子集）快得多。

# GPU Memory / GPU内存
> the amount of information your GPU can fit on it. A bigger GPU will be able to process more information in parallel which means it can support bigger models (or bigger batch sizes) without running out of memory. If you run out of GPU memory it will crash your program.

您的 GPU 可以容纳的信息量。 更大的 GPU 将能够并行处理更多信息，这意味着它可以支持更大的模型（或更大的批量）而不会耗尽内存。 如果你的 GPU 内存用完了，你的程序就会崩溃。

# Gradient / 梯度
> neural networks use gradient descent to improve bit by bit. The gradient is a set of directions calculated (by taking the derivative of the loss function) that will most improve predictions. By taking a short step in the direction of the gradient and then recalculating the gradient and repeating the process, a neural network can improve its performance over the course of training.

神经网络使用梯度下降来一点一点地改进。 梯度是一组计算的方向（通过获取损失函数的导数），最能改善预测。 通过在梯度方向上一小步，然后重新计算梯度并重复该过程，神经网络可以在训练过程中提高其性能。

# Ground Truth / 真实值 / 地面真相
> the “answer key” for your dataset. This is how you judge how well your model is doing and calculate the loss function we use for gradient descent. It’s also what we use to calculate our metrics. Having a good ground truth is extremely important. Your model will learn to predict based on the ground truth you give it to replicate.

最早用于遥感专业，指的是通过卫星或飞行器扫描土地，获得的测量数据。现在指的是数据的标签值，也是真实值，实际的值。

# Head / 头
> The portion of an object detector where prediction is made. The head consumes features produced in the neck of the object detector.

进行预测的对象检测器部分。 头部消耗在物体检测器颈部产生的特征。在[上一章](https://blog.csdn.net/poisonchry/article/details/122368271)里，backbone指的是主干网络，也就是神经元网络中主要负责特征提取的部分，而Head主要是负责消化主干网络中提取的特征，然后生成最终检测结果的部分，例如分类问题、回归问题等。

# Health Check / （数据）状态检测
> a set of tools in Roboflow that help you understand the composition of your dataset (eg size, dimensions, class balance, etc).

在处理训练网络前，对数据的状态进行检测，譬如大小、维度、类型平衡等。

# Hold Out Set / 测试集
> another name for “test set” -- the part of your dataset you reserve for after training is complete to check how well your model generalizes.

测试集的另外一种别称，通常我们使用 test set。

# Hosted Dataset / 托管数据集
> Robofow stores your dataset in the cloud (secured by your API keys) so that you can access it from whichever machine you are training on.

在服务器或者远程设备上托管的数据集。

# Hosted Model / 托管模型
>  A set of trained weights located in the cloud that you can receive predictions from via an API. (As opposed to an edge-deployed model.)

在服务器或者远程设备上托管的模型。

# Hyperparameter / 超参数
> the levers by which you can tune your model during training. These include things like learning rate and batch size. You can experiment with changing hyperparameters to see which ones perform best with a given model for your dataset.

您可以在训练期间调整模型的杠杆。 这些包括学习率和批量大小等内容。 您可以尝试更改超参数，以查看哪些超参数对数据集的给定模型表现最佳。

# Inference / 推理
> making predictions using the weights you save after training your model.

使用您在训练模型后保存的权重进行预测。

# IoU / 交并比
> intersection over union (also abbreviated I/U). A metric by which you can measure how well an object detection model performs. Calculated by taking the amount of the predicted bounding box that overlaps with the ground truth bounding box divided by the total area of both bounding boxes.

联合的交集（也缩写为 I/U）。 您可以用来衡量对象检测模型执行情况的指标。 通过将预测的边界框与真实边界框重叠的数量除以两个边界框的总面积来计算。关于交并比的详细内容可以查看这篇[文章](https://seagochen.blog.csdn.net/article/details/122134314)。

# Jetson 
> an edge computing device created by NVIDIA that includes an onboard GPU.

由 NVIDIA 创建的边缘计算设备，包括板载 GPU。目前某宝价大概在2千左右，由于显存一般只有2Gb，所以通常当作推理设备使用。另外一个相对便宜的类似方案，有FPGA（前提你懂FPGA专用的语言），或者英特尔的神经元计算棒，这篇文章截止时，已经出到了第二代，价格大概在500左右。

用户根据自己需要创建的数据集

# Keypoint Detection / 关键点检测
> a type of computer vision model that predicts points (as opposed to boxes in object detection). Oftentimes keypoint detection is used for human pose estimation or finger tracking where only the position of an object, not its size, matters.

在机器视觉方向，关键点检测主要指的是人体姿态、手指跟踪的关键节点。

# Label / 标签
> the class of a specific object in your dataset. In classification, this is the entirety of the prediction. In object detection, it is the non-spatial component of the bounding box.

数据集中特定对象的类。

# Layer / 网络层
> layers are made up of neurons (and, more commonly, blocks of neurons). Deep neural networks are composed of several layers. The neurons in each layer are connected to neurons in one or more other layers. Adding layers makes a network “deeper”. As a network gets deeper it becomes more complex which gives it more predictive power (but also makes it harder to train as it exponentially increases the solution space).

层由神经元（更常见的是神经元块）组成。 深度神经网络由多层组成。 每一层的神经元都连接到一个或多个其他层的神经元。 添加层使网络“更深”。

随着网络变得越来越深，它变得越来越复杂，这赋予了它更多的预测能力（但也使得它更难训练，因为它以指数方式增加了解决方案空间）。

# Learning Rate / 学习率
> a hyperparameter that defines the size of the steps along the gradient that you take after each batch during training. Often the learning rate will change over the course of training (this is called having a “cyclical learning rate”. If your learning rate is too small, your model will converge very slowly. If it’s too large, it might lead your model’s weights to explode and your model to diverge.

一个超参数，它定义了训练期间每个批次后沿梯度采取的步长大小。 通常学习率会在训练过程中发生变化（这被称为“循环学习率”。如果你的学习率太小，你的模型会很慢地收敛。如果太大，它可能会导致你的模型的权重爆炸。

# LiDAR
> laser imaging detection and ranging. This is a device that uses lasers to Detect depth. Built into many self-driving cars and now included in the iPad Pro for constructing a 3d world map for use in augmented reality.

激光成像探测和测距。

这是一种使用激光检测深度的设备。内置于许多自动驾驶汽车中，现在包含在 iPad Pro 中，用于构建用于增强现实的 3D 世界地图。

# Localization / 定位
> identifying where in an image an object resides. This is the part of object detection and keypoint detection that gives the x/y coordinates (as opposed to the class label).

识别对象在图像中的位置。 这是对象检测和关键点检测的一部分，提供 x/y 坐标（与类标签相反）。

# Loss Function / 损失函数
> A differentiable calculation of “how far off” a prediction is. This is used to calculate the gradient and in turn steer which direction your model steps at each iteration of your training loop. The output of the loss function is called the “loss” and is usually calculated on the training set and validation set separately (called “training loss” and “validation loss” respectively). The lower this value, the more accurate the model’s predictions were.

预测“离我们有多远”的可微分计算。 这用于计算梯度，然后在训练循环的每次迭代中引导模型的步进方向。 损失函数的输出称为“损失”，通常在训练集和验证集上分别计算（分别称为“训练损失”和“验证损失”）。 该值越低，模型的预测就越准确。

# Machine Learning / 机器学习

机器学习在近30多年已发展为一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、计算复杂性理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与推断统计学联系尤为密切，也被称为统计学习理论。


# mAP / 平均精度
> mean average precision. A metric to judge how well your object detection model is performing. We have a whole post breaking this down.

平均精度。 用于判断对象检测模型执行情况的指标。 

# Memory Footprint / 内存占用
> how much space in memory a model takes. This is largely a function of the number of parameters in your model and your batch size. You want to make sure this fits inside of your GPU memory.

模型占用多少内存空间。 这在很大程度上取决于模型中的参数数量和批量大小。

# Metadata / 元数据
> ancillary information stored about your data. For example, the date and time it was collected. Often stored as EXIF.

存储的有关您的数据的辅助信息。 例如，收集它的日期和时间。 通常存储为 EXIF。

# Metrics / 指标
> Evaluation metrics are used to assess the performance of a machine learning system.

评估指标用于评估机器学习系统的性能。

# Mixed Precision / 混合精度
> using both full precision and half precision floating point numbers during training. This has been shown to increase speed without degrading performance.

在训练期间同时使用全精度和半精度浮点数。 

# Mobile Deployment / 移动部署
> deploying to an edge device like a mobile phone. Considerations like battery usage and heat dissipation come into play.

部署到手机等边缘设备。 

# Model / 模型
> a specific incarnation of an architecture. A model has a defined input size and layout for its weights file. For example, YOLOv5s is the smallest version of YOLOv5 which is an architecture in the YOLO family.

架构的特定化身。 模型具有为其权重文件定义的输入大小和布局。 例如，YOLOv5s 是 YOLOv5 的最小版本，它是 YOLO 家族中的一种架构。

# Model Configuration / 模型配置
> used to adjust an architecture into a specific model and set its hyperparameters.

用于将架构调整为特定模型并设置其超参数。

# Model Size / 模型大小
> the number of parameters (or neurons) a model has. This can also be measured in terms of the size of the weights file on disk.

模型具有的参数（或神经元）的数量。 这也可以根据磁盘上权重文件的大小来衡量。

# Model Zoo 
> a collection of model architectures (and sometimes pre-trained model weights) available for download.

可供下载的模型架构（有时是预训练的模型权重）的集合。

# Mosaic 
> an advanced augmentation that combines multiple images from your training set that has been shown to improve object detection training performance.

一种高级增强功能，它结合了训练集中的多个图像，已被证明可以提高对象检测训练性能。

# ncnn
由腾讯主导的基于C的神经网络框架，与Tensorflow，Pytorch 这类框架不同的是，它是专门针对手机、嵌入式设备上轻量级框架。

# Neck / 脖子
> The portion of an object detection model that forms features from the base convolutional neural network backbone.

对象检测模型的一部分，它从基础卷积神经网络主干中形成特征。

# Neural Architecture Search / 神经架构搜索
> automatically trying many variations of model layouts and hyperparameters to find the optimal configuration.

自动尝试模型布局和超参数的许多变体以找到最佳配置。

# Neuron / 神经元
> also known as a parameter, a neuron or perceptron is a mathematical functions takes several inputs and outputs, multiplies them together with its weights (which change over time as the network learns) and outputs a single value which is then fed into other neurons as one of their inputs.

也称为参数，神经元或感知器是一个数学函数，它接受多个输入和输出，将它们与其权重（随着网络学习而随时间变化）相乘，并输出单个值，然后将其作为一个值输入其他神经元 他们的投入。

# NMS / 非最大抑制
> non maximum suppression.

在给出的推测的所有可能性中，仅保留最大的可能性。这种技术常见于目标检测相关应用中。


# Non-Destructive / 非破坏性
> an operation that can be reversed without losing information. Roboflow’s preprocessing and augmentation steps are non-destructive because they do not overwrite the source values.

可以在不丢失信息的情况下反转的操作。

# Normalization / 标准化、归一化
> standardizing data inputs based on a distribution.

基于数据分布情况进行标准化的一种手段。比如对于0-255的像素数据，可以对每一个像素点的除以255，使得数据落入$[0, 1]$之间，这样做的主要目的在于避免处理时、例如在梯度下降算法出现梯度爆炸的情况，或加快模型收敛速度。

# nvidia-smi 
> a tool that can be used to inspect the state of the GPU on machines with an NVIDIA graphics card. You can use this command’s output to determine how much of the GPU memory is being consumed at a given point in time, for example.

一种可用于检查配备 NVIDIA 显卡的机器上的 GPU 状态的工具。 例如，可以使用此命令的输出来确定在给定时间点消耗了多少 GPU 内存。

# NVIDIA Container Toolkit 
> a helper library to assist in creating Docker containers that can access their host machine’s GPU.

一个帮助程序库，用于帮助创建可以访问其主机 GPU 的 Docker 容器。另外就是目前的很多GPU租赁服务，底层多是采用这种技术。

# Null Annotation / 空注释
> a purposely empty annotation that can help teach your model that an object is not always present.

一个故意空的注释，可以帮助你的模型知道对象并不总是存在。

# Object Detection / 目标检测
> a category of computer vision models that both classify and localize objects with a rectangular bounding box.

一类计算机视觉模型，使用矩形边界框对对象进行分类和定位。

# Occlusion / 遮挡
> when an object is partially obscured behind another object it is “occluded” by that object. It is important to simulate occlusion so your model isn’t overly dependent on one distinctive feature to identify things (eg by occluding a cat’s ears you force it also to learn about its paws and tail rather than relying solely on the ears to identify it which helps if the cat’s head ends up being hidden by a chair in a real world situation).

当一个物体在另一个物体后面被部分遮挡时，它就会被那个物体“遮挡”。 模拟遮挡很重要，这样您的模型就不会过度依赖一个独特的特征来识别事物（例如，通过遮挡猫的耳朵，您会迫使它也了解它的爪子和尾巴，而不是仅仅依靠耳朵来识别它 如果猫的头在现实世界的情况下最终被椅子隐藏起来会有所帮助）。

# Offline Prediction / 离线预测
> the opposite of “real-time” prediction, this is when the model does not have a hard limit on how quickly it needs to return an answer. An example would be indexing a user’s photo library to do search. This task can wait to be performed while the user is sleeping and the device is not otherwise in use.

与“实时”预测相反，这是当模型对返回答案所需的速度没有硬性限制时。 一个例子是索引用户的照片库来进行搜索。 该任务可以在用户睡眠且设备未在其他情况下使用时等待执行。

# ONNX 
> a cross-platform, cross-framework serialization format for model weights. By converting your weights to ONNX you can simplify the dependencies required to deploy it into production (oftentimes converting to ONNX is a necessary step in deploying to the edge and may be an intermediary step in converting weights to another format).

模型权重的跨平台、跨框架序列化格式。 通过将权重转换为 ONNX，您可以简化将其部署到生产中所需的依赖项（通常转换为 ONNX 是部署到边缘的必要步骤，并且可能是将权重转换为另一种格式的中间步骤）。

# Ontology / 本体论
> the categorization and hierarchy of your classes. As your project grows it becomes more and more important to standardize on common nomenclature conventions for your team.

类的分类和层次结构。 随着您的项目的发展，为您的团队标准化常见的命名约定变得越来越重要。

# One-Hot / 独热向量

中文翻译为「独热向量」，它是一种用稀疏向量来表示数据的方法，通常用于文本分析。比如对于有词典 ['a', 'b', 'c', 'd']，由于它有4个单词，所以会使用至少四个数字表示词典中的单词。例如对于 a 来说，它可能会被表示为 [1, 0, 0, 0]，对于d而言，则表示为 [0, 0, 0, 1]。

在数据很少的时候使用独热向量这种稀疏向量形式会减少编码难度，并且能一定程度上加速模型对文本的处理速度。但是当单词数超过一定大小，比如日常常用汉字数（4000多字）的情况下，就不应该再使用独热向量这种形式对文本进行编码了。

# OpenCV 
> an “traditional” computer vision framework popularized before deep learning became ubiquitous. It excels at doing things like detecting edges, image sticking and object tracking. In recent years it has also started to expand into newer machine learning powered computer vision techniques as well.

在深度学习无处不在之前，一种“传统”的计算机视觉框架已经普及。 它擅长做边缘检测、图像残留和对象跟踪等事情。 近年来，它也开始扩展到更新的机器学习驱动的计算机视觉技术。

# OpenVINO 
> Intel’s inference framework. Designed for speedy inference on CPU and GPU devices.

英特尔的推理框架。 专为在 CPU 和 GPU 设备上进行快速推理而设计。

# Output / 输出
> the result of a process. The output of training is a set of weights. The output of inference is a prediction.

一个过程的结果。 训练的输出是一组权重。 推理的输出是预测。

# Outsourced Labeling / 外包标签
> paying people to annotate and/or label your images. There are several companies specializing in taking on this task. It’s most effective when little domain expertise is required to determine the correct annotation (and difficult to do in cases like plant disease detection where an expert is needed to provide an accurate ground truth).

付钱给人们注释和/或标记您的图像。 有一些公司专门从事这项工作。 当需要很少的领域专业知识来确定正确的注释时（并且在植物病害检测等需要专家提供准确的基本事实的情况下很难做到），这是最有效的。

# Overfitting / 过拟合
> if your model starts memorizing specific training examples to such an extent that it starts degrading its performance on the validation set. Tactics to counteract overfitting include collecting more training data, augmentation, and regularization.

如果您的模型开始记住特定的训练示例，以至于它开始降低其在验证集上的性能。 对抗过度拟合的策略包括收集更多的训练数据、增强和正则化。

# PaddlePaddle 
> a deep learning framework developed by Baidu.

百度开发的深度学习框架。

# Parameters / 网络权重
> the number of weights. There is one parameter for each connection between two neurons in the network. Each one is stored as a floating point number and is adjusted during each backpropagation step during training.

权重的数量。 网络中两个神经元之间的每个连接都有一个参数。 每个都存储为浮点数，并在训练期间的每个反向传播步骤中进行调整。

# Pascal VOC 
> the Visual Object Classes was an early benchmark dataset. It has largely been replaced with newer datasets like COCO in the literature but its XML annotation format has become widely used by other datasets, labeling tools, and models.

Visual Object Classes 是早期的基准数据集。 在文献中它已在很大程度上被更新的数据集（如 COCO）所取代，但其 XML 注释格式已被其他数据集、标签工具和模型广泛使用。

# Performance / 性能
> how fast and accurate your model is.

你的模型有多快和多准确。

# Platform / 平台
> a computer vision platform is a (usually cloud-hosted) meta-tool that ties into various other tools to manage all (or part of) your pipeline.

计算机视觉平台是一种（通常是云托管的）元工具，它与各种其他工具相关联，以管理您的全部（或部分）管道。

# Pipeline / 管道
> your computer vision pipeline is the process of going from raw images to a prediction. Usually this encompasses collecting images, annotation, data inspection and quality assurance, transformation, preprocessing and augmentation, training, evaluation, deployment, inference (and then repeating the cycle to improve the predictions).

您的计算机视觉管道是从原始图像到预测的过程。 通常这包括收集图像、注释、数据检查和质量保证、转换、预处理和增强、训练、评估、部署、推理（然后重复循环以改进预测）。

# Polygon / 多边形
> a (usually non-rectangular) region defining an object with more detail than a rectangular bounding box. Polygon annotations can be used to train segmentation models or to enhance performance of object-detection models by enabling a more accurate bounding box to be maintained after augmentation.

一个（通常是非矩形的）区域，它定义了一个比矩形边界框更详细的对象。 多边形注释可用于训练分割模型或通过在增强后保持更准确的边界框来提高对象检测模型的性能。

# Precision / 准确度
> A measure of how precise a model is at prediction time. True positives divided by all positives that have been guessed.

衡量模型在预测时的精确程度。 真阳性除以所有猜测的阳性。

# Prediction / 预测
> an attempt by a model to replicate the ground truth. A prediction usually contains a confidence value for each class.

模型尝试复制基本事实。 预测通常包含每个类别的置信度值。

# Preprocessing / 预处理
> deterministic steps performed to all images (training, validation, testing, and production) prior to feeding them into the model.

在将图像输入模型之前，对所有图像（训练、验证、测试和生产）执行确定性步骤。

# Pretrained Model / 预训练模型
> a model that has already been trained on another dataset. Many things it learns will be broadly applicable to images in other datasets (for example, finding lines, corners, and patterns of colors). Pre-training on a large dataset like COCO can reduce the number of custom images you need to obtain satisfactory results.

已经在另一个数据集上训练过的模型。 它学到的许多东西将广泛适用于其他数据集中的图像（例如，寻找线条、角落和颜色图案）。 在 COCO 等大型数据集上进行预训练可以减少获得满意结果所需的自定义图像数量。

# Production / 生产环境
> the deployment environment where the model will run in the wild on real-world images (as opposed to the testing environment where the model is developed).

模型将在真实世界图像上运行的部署环境（与开发模型的测试环境相反）。

# PyTorch 
> a popular open source deep learning framework developed by Facebook. It has a focus on accelerating the path from research prototyping to production deployment.

由Facebook主导的深度学习框架，也是我现在学习和研究使用的主要框架，同时也是当前主流的两大深度学习框架（另外一个是tensorflow）。

它支持动态图，由于从原型到实践非常容易部署和实施都，在个人、中小型企业、学院中得到追捧，但是由于Pytorch在端计算方面的先天劣势，目前大型工业应用上仍然是Tensorflow的天下。

此外，腾讯主推的ncnn，和百度主推的PaddlePaddle，它们都是轻量级的框架。在目前一些互联网级应用，以及一些App里也能见到它们的身影。

# Realtime / 实时
> when a model needs to be run in a specified amount of time, for example in a mobile augmented reality application it needs to provide its predictions in less time than the desired frame rate so that it can keep up with the incoming images. Doing batch prediction to take advantage of parallel processing doesn’t help here because as soon as the next image comes in the previous prediction is no longer as relevant.

当模型需要在指定的时间内运行时，例如在移动增强现实应用程序中，它需要在比所需帧速率更短的时间内提供预测，以便跟上传入的图像。 进行批量预测以利用并行处理在这里没有帮助，因为一旦下一张图像出现在之前的预测中，就不再那么相关了。

举个例子来说，对于自动驾驶来说，模型能否实时的处理数据比其他指标更重要。车辆前方的行人，对向车辆的突然变道，甚至路上突然出现的路障，都需要模型在第一时间内处理并预警。而实时性，是很多端应用所要求的。

# Recall / 召回率
> A measure of performance for a prediction system. Recall is used to assess whether a prediction system is guessing enough. True positives / All possible true positives.

召回率又叫[查全率](https://seagochen.blog.csdn.net/article/details/122182996)，指实际检测出的样本，与应该查出的样本比值。

# Region Attribute / 区域属性
> additional properties beyond the class name and location that can be added to bounding boxes and polygons in some annotation tools. They can be thought of as metadata at the object (rather than the image) level.

在某些注释工具中可以添加到边界框和多边形的类名称和位置之外的其他属性。 它们可以被认为是对象（而不是图像）级别的元数据。

# Regression / 回归
> a model that predicts one or multiple real numbers (such as an age, year, or pixel location) where “how close” a prediction is to the ground truth is measurable (as opposed to classification where the prediction is either right or wrong).

预测一个或多个实数（例如年龄、年份或像素位置）的模型，其中预测与基本事实的“接近程度”是可测量的（与预测正确或错误的分类相反）。

# Regularization / 正则化
> A technique to reduce bias in machine learning models. Machine learning models have a tendency to overfit to training data. Regularization introduces a penalty for heavily weighting features, forcing a machine learning system to formulate a flexible algorithm.

一种减少机器学习模型偏差的技术。 机器学习模型倾向于过度拟合训练数据。 正则化，会对重权较大的特征引入惩罚，例如使用正态分布函数，使得低权重的特征也能被模型识别并处理，最终得到相对灵活的模型。

# Remap / 重映射
> changing class composition after the annotation task. Sometimes your ontology needs to evolve as you learn more about the problem and your process matures.

用某种特定的方法，把注释后的描述信息转换为另外一种格式或描述方式，它借鉴了数学上的“映射”这个概念。

# Resolution / 分辨率
> the number of pixels in an image (defined by its width multiplied by its height). The standard unit of measure is megapixels (millions of pixels).

一张数字图片，除了长、宽、颜色通道这些属性外，还有单位面积内，包含的有效像素数（通常这个单位是一英寸）。比方说定义一英寸的有效像素为 80 x 60，当照片尺寸为（4英寸 x 3.2英寸），此时该照片的分辨率即为 320 x 192 = 61440 的分辨率。显然，这个分辨率越高照片越清晰，但是也会带来文件体积过大的问题。


# Runtime Environment / 运行环境
> Where machine learning code is being executed. CPU, GPU, VPU (Vision Processing Unit), or TPU.

执行机器学习代码的地方。 可以是CPU、GPU、VPU（视觉处理单元）或 TPU（张量处理单元，例如Google的TPU）。

# SageMaker 
> Amazon AWS’ machine learning platform encompassing tools for outsourced annotation, model training, and deployment.

Amazon AWS 的机器学习平台，包括用于注释、模型训练和部署的工具。

# Segmentation / 分割
> a type of model that classifies each individual pixel used when the exact outlines of objects are needed.

一种模型，用于对需要精确对象轮廓时使用的每个像素进行分类。

# Self Adversarial Training / 自我对抗训练
> a technique where the model strategically starves itself of the information it is most reliant on to force itself to learn other ways to make predictions. For example, if it detects it is mostly relying on cat ears to identify cats it will turn off the parts of the input feeding into those neurons to force it to also learn other ways to identify cats (like its paws and tail).

一种模型训练技术，通常我们通过给模型喂数据完成机器学习任务，但是后来人发现这样做同时会给模型带来“偏见”，所以尝试让模型有策略地关闭对一些数据的依赖，学习其他方法方法进行预测。例如，如果它检测到它主要依靠猫耳朵来识别猫，那么我们就会让它会关闭这些神经元，以迫使它也学习其他识别猫的方法（比如它的爪子和尾巴）。

# Session / 对话
> A TensorFlow Session allocates resources for a machine to execute a TensorFlow defined neural network graph structure. Sessions are deprecated in TensorFlow 2.

已弃用的技术，指的是 TensorFlow 为机器分配资源以执行 TensorFlow 定义的神经网络图结构。在 TensorFlow 2 中已弃用会话技术。

# Split / 数据分离
> segregating subsets of your data and delineating them for different purposes. Usually we create three splits: train (given to your model to mimic), valid (used for evaluation during training), and test (held back until the very end to determine how well your model will generalize).

把数据分离为不同的部分，并用于不同的目的。比如，通常我们创建三个拆分：训练（训练模型以模仿）、有效（用于训练期间的评估）和测试（直到最后才确定模型的泛化程度）。

# SSD / 单发检测器
> single shot detector. A model that only does a single pass to both localize and classify objects. The YOLO family famously plays off of this concept in its name: You Only Look Once.

单发检测器。 一种模型，只执行一次即可对对象进行定位和分类。

# State of the Art / 最先进的
> a model that is currently top of its class, performing better on the benchmark dataset than any other previously known model.

目前是同类产品中的佼佼者，在基准数据集上的表现优于任何其他先前已知的模型。

# Subjective / 主观性
> opposite of objective; performance that is observed intuitively but not necessarily able to be measured. For example, in language modeling, it’s common for models’ metrics to be similar but for one of their outputs to be subjectively better as judged by a human reader.

与目标相反； 直观地观察到但不一定能够测量的性能。 例如，在语言建模中，不同的模型输出仅从指标看是相似的，但它们当中可能只有一两个的输出，在人类的主观感觉上更合适。

# Synthetic Data / 数据合成
> images that are created rather than collected. There are several strategies for creating more training data including using 3D models, GAN synthesis, and context augmentation.

通常指，创建而不是收集的图像。有几种策略可以创建更多的训练数据，包括使用 3D 模型、GAN 合成和上下文增强，当然还有翻转、缩放等技巧。使用数据合成技术，主要的目的在于增强模型的泛化能力。

# Tensor / 张量
> a (possibly multi-dimensional) array of numbers of a given type with a defined size. Because they have a defined size and shape it makes it possible to optimize and parallelize operations on them with hardware accelerators.

具有定义大小的给定类型的数字（可能是多维）数组。 因为它们具有定义的大小和形状，所以可以使用硬件加速器优化和并行化它们的操作。

# Tensor Core / 张量核
> NVIDIA’s brand name for the part of their GPUs that is specifically optimized for deep learning (and especially mixed-precision neural networks).

NVIDIA 的品牌名称是其 GPU 专为深度学习（尤其是混合精度神经网络）优化的部分。

# Tensorboard 
> a tool used to track and visualize training metrics including graphs of common statistics like loss and mean average precision originally developed for Tensorflow but now compatible with other frameworks like PyTorch.

一种用于跟踪和可视化训练指标的工具，包括最初为 Tensorflow 开发的损失和平均平均精度等常见统计数据图表，但现在与 PyTorch 等其他框架兼容。

# Tensorflow 
> Google’s popular open source deep learning framework.

谷歌流行的开源深度学习框架。

# Tensorflow Lite 
> model serialization for Tensorflow models to optimize them to run on mobile and edge devices.

Tensorflow 模型的模型序列化以优化它们以在移动和边缘设备上运行。

# TensorRT 
> NVIDIA’s framework agnostic inference optimization tooling. Helps to optimize models for deployment on NVIDIA-powered edge-devices.

NVIDIA 的与框架无关的推理优化工具。 帮助优化模型以部署在 NVIDIA 驱动的边缘设备上。

# Test Set Bleed / 测试集出血
> an issue that occurs when data from your test set leaks into your training set. This is bad because it defeats the purpose of your hold out set; you no longer have any way of judging how well your model will generalize to predict on images it has never seen before.

它指的是测试集中的数据泄漏到训练集中时的问题。这会带来一个很严重的问题，就是模型在测试集上表现良好，而在真实环境上的泛化能力表现很差，而且极难被发现。


# TFJS 
> Tooling that enable (some) Tensorflow-trained models to perform inference in the web browser with Javascript, WebAssembly, and WebGPU.

使（某些）经过 Tensorflow 训练的模型能够使用 Javascript、WebAssembly 和 WebGPU 在 Web 浏览器中执行推理的工具。

# TFRecord 
> a binary data format compatible with Tensorflow. In the object detection API all of the images and annotations are stored in a single file.

与 Tensorflow 兼容的二进制数据格式。 在对象检测 API 中，所有图像和注释都存储在一个文件中。

# Tile / 块
> splitting an image into a grid of smaller images and running them through a model independently to boost the effective resolution. This can be an effective strategy to improve model accuracy while still fitting into the available memory (at the expense of having to run the model several times per image).

将图像分割成较小图像的网格，并通过模型独立运行它们以提高有效分辨率。 这可能是一种提高模型准确性的有效策略，同时仍然适合可用内存（代价是每个图像必须运行模型多次）。

# TPU / 张量处理单元
> Tensor Processing Unit. Google’s hardware accelerator for performing operations on tensors. It is much faster than a GPU for some workloads. Most often they are run on Google Cloud or Google Colab but there is also an edge-TPU that can be deployed in the field.

张量处理单元。 用于对张量执行操作的 Google 硬件加速器。 对于某些工作负载，它比 GPU 快得多。 大多数情况下，它们在 Google Cloud 或 Google Colab 上运行，但也有一个可以在现场部署的边缘 TPU。

# Tradeoff / 权衡
> when two competing concerns pull you in opposite directions. For example, there is often a tradeoff between speed and accuracy (where there is a continuum from fast and inaccurate to slow and accurate with a continuum of valid choices in between depending on the needs of your specific problem).

大多数问题都会遇到所谓的“不可能三角”，在机器学习模型中也会遇到类似性能、精确度、检出率等关键指标的掣肘。它使得你不得不在各种指标中选择你最关心的一两个指标，并优化模型。类似问题有，精确度高的模型，运行速度慢；运行速度快的模型，体积过大而难以用在嵌入式设备等。

# Train / 训练
> the process iteratively of adjusting your model’s parameters to converge on the weights that optimally mimic the training data.

迭代调整模型参数以收敛到最佳模拟训练数据的权重的过程。

# Transfer Learning / 迁移学习
> using pre-trained weights to bootstrap your model’s learning. You are “transferring” the knowledge learned on another dataset and then “fine-tuning” it to learn about your new domain.

使用预训练的权重来引导模型的学习。比如利用训练好的模型，通过微调使它适用于新的问题空间。


# Tune / 调整
> adjusting hyperparameters to find the optimal settings to get the best model.

调整超参数以找到最佳设置以获得最佳模型。

# Two Stage Detector / 两级探测器
> a category of (typically older generation) object detection models that first localize, then classify. As opposed to single shot detectors which do both tasks in one pass.

一类（通常是老一代）对象检测模型，首先定位，然后分类。 与一次性完成两项任务的单次检测器相反。

# Validate / 证实
> during the training process of a neural network, the validation set is used to assess how well the model is generalizing. These examples are not used to calculate the gradient; they are the ones used to calculate your metrics and see how well they are improving over time.

在神经网络的训练过程中，验证集用于评估模型的泛化程度。 这些示例不用于计算梯度； 它们用于计算您的指标并查看它们随着时间的推移而改进的程度。

# VPU / 视频处理单元

Video Processing Unit，视频处理单元,是一种全新的视频处理平台核心引擎，具有硬解码功能以及减少CPU负荷的能力。另外，VPU可以减少服务器负载和网络带宽的消耗。VPU由ATI提出，用于区别于传统GPU（Graph Process Unit，图形处理单元）。图形处理单元又包括视频处理单元、外视频模块和后处理模块这三个主要模块。

# Weights / 权重
> the parameters of your model that neurons use to determine whether or not to fire. The optimal values are learned via backpropagation during training and can then be serialized and deployed for inference.

神经元用来确定是否触发的模型参数。 最佳值是在训练期间通过反向传播学习的，然后可以序列化和部署以进行推理。

# Workflow / 工作流程
> The process you follow. This will be some combination of manual steps, custom code, and third party tools in any number of environments. A computer vision platform can help you set up an optimal workflow.

业务的流程、工作的流程。这将是任意数量环境中手动步骤、自定义代码和第三方工具的某种组合。

# YAML 
> a markup language originally invented by Yahoo that is now commonly used as a format for configuration files (notably in YOLOv5's YAML configuration).

一种最初由 Yahoo 发明的标记语言，现在通常用作配置文件的格式（特别是在 YOLOv5 的 YAML 配置中）。

# YOLO 
> You Only Look Once, a family of single-shot learning object detection models providing state of the art results for object detection as of fall 2020.

You Only Look Once，一系列单次学习对象检测模型，截至 2020 年秋季，为目标检测领域最先进的技术。


----

参考内容

[1]  https://blog.roboflow.com/glossary/