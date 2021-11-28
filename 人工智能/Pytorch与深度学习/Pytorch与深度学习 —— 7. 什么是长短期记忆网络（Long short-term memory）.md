> 在上一篇文章里，给大家介绍了如何使用RNN网络，并且做了一个极为简单的文字序列转化。尽管我们发现RNN存在的重大缺陷，但是RNN网络所展示的可能性依然令人振奋，所以我们在这一章里，我将给大家介绍一种对RNN网络的改进型——长短期记忆网络（Long short term memory），并且这一次我们将正式完成一个比较完整的工程，一起来看看怎么实现吧。


@[toc]

# 什么是LSTM网络
老规矩，先从数学公式入手来理解这个网络模型是怎么工作的。首先在LSTM网络中负责处理输入的有四个函数与之相关的函数

$$\begin{matrix} 
i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) & e.q. 1 \\
f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) & e.q. 2 \\
g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) & e.q. 3 \\
o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) & e.q. 4 \\
\end{matrix}$$

以及两个负责处理隐藏层的线性函数，如下：

$$\begin{matrix}
c_t = f \odot c_{(t-1)} + i \odot g & e.q. 5 \\
h_t = o \odot \tanh(c_t) & e.q.6
\end{matrix}$$

这一共6个线性公式，看着脑袋都大了有木有。但是也并不复杂，首先还是要明确之前在讨论RNN网络我们是怎么分析它的数学原理的。首先，上面6个式子的 $W$ 一直都表示的是权重，$x$ 则表示当前序列的第 $i$ 个元素的输入，例如上一篇文章里，**a** 对于 **apple** 这个字符串序列的输入一样。

如果是处理序列文字，就像 **Hello** 对于 **Hello World** 这样的序列一样，只不过如果是要处理单词的话，就需要对单词进行整体编码，以及录入足够进行分析的样本数据（这不在本章节的讨论范围内，但是你在学习完这部分的内容后，可以很轻易的扩展出类似的应用）。

我们现在来分析公式 1，2，3，4，可以看到针对输入 $x$ 的处理，隐藏层都参与了计算，只不过每次计算时我们都采取不同的权重以及sigmoid家族函数，用于生成不同的参数 **I, F, G, O**，对应于下图，就是图中表示的黑色部分指向的  **0，1，2， 3**

![在这里插入图片描述](https://img-blog.csdnimg.cn/3b42726c066a41dda46ff5c530eb2d30.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
然后最终生成的隐藏层 H 是由 O 和 $C_t$ 通过哈达玛积[^1]得到，而  $C_t$ 则是由 **I，F，G** 以及上一次计算得到的 **$C_{(t-1)}$** 彼此之间通过哈达玛积而得到。


[^1]: https://seagochen.blog.csdn.net/article/details/114661205

这里由输入 x 分别通过sigmoid函数生成的 **I，F，G，O**，被定义为 输入门（Input Gate），遗忘门（Forget Gate），Cell门（Cell Gate）以及输出门（Output Gate）。

我个人认为这些门没啥必要去死记硬背，除非你要考试或者其他什么用途。现在我们来深入探讨这些线性运算公式它实际上在数理层面上起到的作用是什么。

# LSTM网络背后的数理逻辑

我们最关心的肯定是输入 X 以及输出 H，H的表达公式为：

$$h_t = o \odot \tanh(c_t) $$

换一个表达方式就是

$$h_t = O(x) \odot \tanh(c_t)$$

$O(x)$ 就是所谓的输出门了，可以看到输入X通过权重可以直接影响到输出 $h_t$，接下来就需要解析一下 $\tan(c_t)$。

与C有关的计算公式为：

$$c_t = f \odot c_{(t-1)} + i \odot g$$

我们也给它们分别换个符号进行表示

$$c_t = F(x) \odot c_{(t-1)} +  G(x) \odot  I(x) $$ 

哈达玛积在很多时候充当参数权重或者掩码计算，所以可以看到，F(x) 其实负责对上一次的数据参与到下一次计算提供了一个类似信号增益或者衰减的功能（如果 $F(x)$ 的元素都 > 1的时候，之前的计算数据就会极大的保留进来，反之之前的数据就会逐渐衰减，直至消失）。

$I(x)$ 与 $G(x)$ 共同控制输入，而且注意 $G(x)$ 是由 tanh 函数控制的，这两个线性层的计算结果再通过哈达玛积之后，就能对输入的数据进行有效调节（主要是调节输入x的各权重的贡献情况）。

我们再把上面那个式子进一步修改一下，就得到这样的一个非常直观的表达式了：

$$c_t = \begin{bmatrix}
F(x) & G(x)
\end{bmatrix} \begin{bmatrix}
c_{(t-1)} \\
I(x)
\end{bmatrix} \rightarrow \mathbb{Mask} \cdot \mathbb{Input}$$


相信到这里，你应该能容易想象和理解整个LSTM网络是怎么运作的了。接下来，我们准备使用 LSTM 来实现我在前一篇文章里提到的内容，如何进行姓名分类，首先我们来看看 LSTM 需要用到的主要参数，以及参数说明。

**总而言之，言而总之，我们发现**
* **通过 Cell Gate（G门）控制输入数据对于当前计算节点的数值贡献；**
* **通过Forget Gate（F门）控制上一次计算结果对于当前计算节点的数值贡献；**
* **然后把上一次计算结果和本次输入数据进行加和后得到新的 c_t;**
* **通过Output Gate（O门）和tan(c_t)输出新的隐藏层，并且把新的隐藏层和c_t传入下一次计算中。**


# LSTM 主要参数说明

LSTM 的参数和 LSTMCell 的参数很相似，最大的区别就是在于Cell指的是单个计算单元，例如RNNCell 对于 RNN 来说，它可以处理每一次计算过程的输出，而RNN则会把序列一次性处理完后再把计算结果交还给用户。LSTM也是一样的，自然也有可以支持单步计算的 LSTMCell ，我们都可以使用，不过在这一部分里，我们把目光放到 LSTM 这种结构上面。

要想使用好LSTM，我们需要重点关注以下几个输入参数和输出参数，而关于LSTM的全部完整说明可以[参考官网](https://pytorch.org/docs/master/generated/torch.nn.LSTM.html)。

## 主要输入参数
主要输入四个参数

 数据参数 | 数据维度 | 数据说明
-------------|--------------|-------------
input_size | $(L, N, H_{in})$ | 对于LSTM，我们可以直接输入一个数据序列，例如 'apple'
hidden_size | $(D * num\_layers, N, H_{out})$ | 隐藏层，主要负责记忆上一次的计算结果；**如果bidirectional = True时， D = 2；如果bidirectional = False时，D=1.** 
num_layers | (integer) | 网络通常默认为1层结构，如果有需要可以把多个LSTM连接起来，使得上一个的输出成为下一个网络的输入
batch_first  | (bool) | **最好不要动这个参数**，默认为False，这样输入的参数维度就是$(L,N,H)$，但是为True时，维度就要变成$(N, L, H)$了，这对我们在后续的数据处理中会增加很多不必要的混乱。


### 什么是 num_layers
通常，这个参数为1，表明网络结构里只有一层循环层参与计算。但是当num_layers为其他数字时，比如为3时，就意味着原来的循环层变成了三层

![在这里插入图片描述](https://img-blog.csdnimg.cn/79a7b3478b6c45c58c6936d990aad3bb.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

由于RNN网络本质是由线性层实现的，循环网络层数越多意味着精度越高。但是随之而来的是会带来计算性能的大幅下降。所以如果没有特别需求，一般我们使用默认的1层结构就行了。


## 输出参数

这里LSTM输出的参数是三个

 数据参数 | 数据维度 | 数据说明
-------------|--------------|-------------
output | (L, N, D * H_{out}) | 它会把每一次计算过程中产生的 $h_t$ 都记录到张量中，并一起交回给用户。这个过程跟上一章里用 RNNCell 实现文字序列转化是一样的。
h_n | $(D * num\_layers, N, H_{out})$ | 它会返回给用户最后一次的隐藏层计算结果，如果直接使用LSTM，通常用不到这个参数。
c_n | $(D * num\_layers, N, H_{cell})$ | 返回cell的最后一次计算结果，如果直接使用LSTM，通常用不到这个参数。


## 使用示例

一个由LSTMl实现的简单计算过程就是下面这个样子的。

~~~python
# model = nn.LSTM(input_size, hidden_size, num_layers)
rnn = nn.LSTM(10, 20, 2)

# input data (sequence_size, batch_size, input_size)
input = torch.randn(5, 3, 10)

# hidden data (D * num_layers, N, out_size)
h0 = torch.randn(2, 3, 20)

# cell data (D * num_layers, N, out_size)
c0 = torch.randn(2, 3, 20)

# forward computation
output, (hn, cn) = rnn(input, (h0, c0))
~~~



