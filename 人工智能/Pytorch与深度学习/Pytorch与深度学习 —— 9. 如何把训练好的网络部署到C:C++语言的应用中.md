> 这应该是Pytorch入门教程的最后一章，在前面的章节部分，给大家介绍了深度学习框架里最常见的三种神经网络，全连接网络、卷积网络以及循环网络。现在我们稍微多探讨一点，假设我们训练了一个足够精度的网络，我们接下来一定会考虑如何把这个网络应用到具体的开发中，这也就是所谓的前沿部署工作。

@[toc]

# 正题开始前先随便闲聊一下
## 带有智能的系统是如何构建的

如果一个系统包含硬件和软件部分，硬件的数据采集和反馈与业务逻辑之间是物理上分割的，那么通常包含硬件的部分叫「下位机」，而负责逻辑处理以及给硬件下达驱动指令的叫「上位机」。

而硬件与软件以垂直形式直接部署在一套系统里，比方说类似手机这种，那么硬件这一层通常被称为「底层硬件」和「上层软件」。

而如果我们的系统没有硬件这一层，或者说它是运行跑在服务器上的，那么从形式上讲它就是个最直接的纯软件结构，当然对应的也有纯硬件结构。针对于纯软件结构，与用户进行交互的部分，比如以控制界面出现的，就是我们通常所说的「前端」，而背后的负责数据处理的这部分业务逻辑，就是所说的「后端」了。

有智能的系统，通常是运行在软件这一层，然后偏后端这部分，有时会覆盖数据层，有时又作为较数据层要高一阶，和其他模块一起集成在整个业务系统中。

![在这里插入图片描述](https://img-blog.csdnimg.cn/c263d630aa7e4e6893cc49536c61356d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_11,color_FFFFFF,t_70,g_se,x_16#pic_center)

日常工作中，AI或者说广义的算法模块，占总体工作量其实非常少，以我个人的经验来看大概10%左右的工作量，但是从投入的开发资源占用，和投入的总体时间成本来说，感觉可以达到项目的近 1/3 的比例。

所以，我所了解的目前行业内的总体情况是，对于大多数中小企业来说，他们的算法或者AI部分，更愿意采用第三方方案，比如百度的语音助手、阿里和华为的人脸识别框架等方案，通过类似调用API，传入图像和语音数据的形式来给自己的产品进行智能赋能。

在一定程度上算是给在学习这个方向的朋友泼冷水吧，所以如果真想以后从事软件设计工作，尤其是偏向算法设计这块的，你可能要做好随时因为项目或者任务调整去负责一般性的日常开发任务。

当然了，有调用API的自然也有研发API的。在国内，目前一定程度上，AI的研发主力还是集中在以华为、腾讯、百度、旷视、阿里的相关核心团队上。

所以，你如果对于这个行业有兴趣的话，不妨提前做好一些调查和准备。


## 为什么需要前沿计算（Frontier Computation）
聊完了一个智能系统是怎么构建的，以及AI模块处于软件系统中的哪个位置的关系后，就来聊一个相对比较有意思的新名词----前沿计算。

前沿计算，好像正式作为一个概念出现在4年前吧，它实际上说的是我们把算法模块向嵌入式或者客户端方向去前置。比如说，就像把算法前置到手机里。


![在这里插入图片描述](https://img-blog.csdnimg.cn/ba8fd499d22741bb8c3d1f832c874b44.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_11,color_FFFFFF,t_70,g_se,x_16#pic_center)


之所以提出这种新的架构模型，主要是因为算法模块不仅消耗大量的计算资源，也消耗大量带宽。比方说语音、图像、GPS信息、实时运动状态等：这些数据如果都通过4G/5G网络传输回数据中心的话，不仅会带来严重的延迟问题，还带来严重的丢包，且挤占企业的计算资源。

所以企业希望的是，通过把算法模块前置到设备中，直接计算出结果后，再返回给计算中心计算结果，这样就可以有效降低资源占用度。

那么既然提到了前沿计算，就需要我们的框架支持前沿计算。总体来说，PyTorch经过这么些年的发展，尽管目前来看好像还没有给出支持ARM的方案。但是对于传统x64 系统来说，项目组已给出 支持C/C++和Java的LibTorch。

它可以让你用最小代价把深度学习技术部署到比如说，服务器上运行的物理网服务、或者游戏这些应用方案里。也就是说，现在可以满足大部分的应用方案需求，除了智能硬件设备这方面可能还是要考虑使用传统的ncnn或者TensorFlow了。

尽管PyTorch项目组已经开展针对ARM的支持，但是在短期内，至少一两年时间内应该还看不到ARM的库问世。那是不是说学PyTorch没有用了呢。

其实目前整体来看，国内的智慧应用项目更多的还是以服务器/WEB为代表的互联网、内网项目。而互联网、内网项目所依赖的设备，基本清一色是运行在x86/x64架构的服务器上。所以，并不存在学PyTorch没法用在这些项目上，唯一的问题就是项目负责人选用方案时，会更多考虑用Tensorflow还是PyTorch了。


## 选用Tensorflow还是PyTorch

其实没啥好坏之分，Tensorflow面世较早，成功的应用部署也更多，社区也相对成熟，对于求稳的团队来说，可以考虑Tensorflow，不过相对来说Tensorflow的入门门槛较高，使用起来也相对较为麻烦。

不过对于个人开发者和小团队来说，可以使用更容易开发和使用的PyTorch，用更短的开发周期部署自己的AI应用。

当然也可以使用大公司团队提供的API，只要你团队的经费足够，也能达到不错的效果。

# 接下来才是正篇
无论是部署到服务器，还是本地应用，如果只让模型运行在本地都是不行的，我们还是需要让模型能够真正的运行在软件系统中，比如 C/C++ 或者 Java。鉴于C/C++ 有更多的应用场景，所以我们接下来好好聊聊Python模型怎么转到C/C++的应用里。

> 相关阅读
> [Pytorch与深度学习 —— 11. 使用 LSTM 做文字分类预测之 RNN 提高篇](https://seagochen.blog.csdn.net/article/details/120358980)
> [Pytorch与深度学习 —— 附录I. 如何在 Ubuntu 中使用 C/C++ 语言编写 Pytorch 程序](https://seagochen.blog.csdn.net/article/details/120318273)
> [Pytorch与深度学习 —— 附录II. 如何在 Windows 中使用 MSVS 编写 Pytorch 程序](https://seagochen.blog.csdn.net/article/details/120346716)

## Step 1. 保存 PyTorch 模型

我们要让PyTorch模型能够被LibTorch使用，需要把模型转换成序列化后的模型，并且以Torch Script的形式存储起来，并且习惯上我们保存的torch script文件后缀以.pt为主。

~~~python
	# define a neural network module
	model = DefinedNeuralNetworkModule

    # converting to Torch Script via Annotation
    serialized_model = torch.jit.script(model)

    # save the torch script for C++
    serialized_model.save("LSTM_Surname_Classfication_CPU_89acc.pt")
~~~

如果你定义的模型能够顺利通过转序后保存为torch script文件，那么你就直接跳过 Step 2.

## Step 2. 修改 PyTorch 模型

我们之前编写的LSTM模型的代码是这样的

~~~python
class LSTMModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, sequence_size, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.output_size = output_size

        self.batch_size = batch_size

        # lstm layer
        self.cell = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers)

        # linear layer for output
        self.linear = torch.nn.Linear(sequence_size * hidden_size, self.output_size)

    def forward(self, input_x):
        """
        forward computation

        @param input_x, tensor of shape (L, N, H_in)
        @return tensor of shape (N, H_out)
        """

        # get dimension from input_x
        _, batch, features = input_x.size()

        # hidden, tensor of shape (D * num_layers, N, H_hidden)
        hidden = self.init_zeros(batch)

        # cell, tensor of shape (D * num_layers, N, H_hidden)
        cell = self.init_zeros(batch)

        # output tensor (L, N, D * H_hidden)
        output, _ = self.cell(input_x, (hidden, cell))

        # convert the shape of output to (N, L * H_hidden)
        hidden = convert_hidden_shape(output, batch)

        # (N, L * H_hidden) to (N, H_out)
        output = self.linear(hidden)

        return output

    def init_zeros(self, batch_size=0, hidden_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        if hidden_size == 0:
            hidden_size = self.hidden_size

        return torch.zeros(self.num_layers, batch_size, hidden_size)
~~~

但是这个模型不能直接转成PT文件。之所以转换失败，其中一个原因是PT里的网络模型每个函数入口应该使用 torch.Tensor 类型，其他的如int，string都不能通过编译。所以不得不把 init_zeros 等写回到 forward 函数里。

为了方便省事，我把很多函数重新写回模型里，你也可以这样做。


~~~python
import torch
import torch.nn.functional as functional


class LSTMModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, sequence_size, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.output_size = output_size

        self.batch_size = batch_size

        # lstm layer
        self.cell = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers)

        # linear layer for output
        self.linear = torch.nn.Linear(sequence_size * hidden_size, self.output_size)

    def forward(self, input_x):
        """
        forward computation
        @param input_x, tensor of shape (L, N, H_in)
        @return tensor of shape (N, H_out)
        """

        # get dimension from input_x
        _, batch, features = input_x.size()

        # hidden, tensor of shape (D * num_layers, N, H_hidden)
        # hidden = self.init_zeros(batch)
        hidden = torch.zeros(self.num_layers, batch, self.hidden_size)

        # cell, tensor of shape (D * num_layers, N, H_hidden)
        # cell = self.init_zeros(batch)
        cell = torch.zeros(self.num_layers, batch, self.hidden_size)

        # output tensor (L, N, D * H_hidden)
        output, _ = self.cell(input_x, (hidden, cell))

        # convert the shape of output to (N, L * H_hidden)
        # hidden = self.convert_hidden_shape(output, torch.tensor(batch))
        tensor_list = []
        for i in range(batch):
            ts = output[:, i, :].reshape(1, -1)
            tensor_list.append(ts)

        final_output = torch.cat(tensor_list)

        # (N, L * H_hidden) to (N, H_out)
        output = self.linear(final_output)
        output = functional.relu(output)

        return output
~~~

除了不好看，后期维护比较麻烦外，也不存在什么大问题。当然，TorchScript 是另外一项技术，你也可以参考相关文档，来深度了解这个技术细节，而展开讨论 TorchScript 完全够格再写上一两篇技术博文了，故不在这里进行讨论。

> 英文资料： https://pytorch.org/docs/stable/jit.html
> 中文资料：https://pytorch.apachecn.org/docs/1.4/29.html


## Step 3. 选择合适的LibTorch包

接下来，你需要确定你的应用准备部署到什么平台上，并以什么形式进行展现。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1fd6df74b7aa4b538a4e5678c00f08d9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

如果你准备以Windows本地应用的形式集成模型数据，那么就需要PyTorch提供了Debug和Release包，这两个包虽然都是 **64位** 形式，但是如果在加载模型、本地运算的时候，Debug包只能在Debug模式下使用，而Release包则只能在Release形式下使用，所以你可能需要同时下载两个包，并在自己的工程里，定义如下宏

~~~cpp

#if _DEBUG
#pragma comment(lib, "xxxxd.lib")
#else 
#pragma comment(lib, "xxxx.lib")
#endif
~~~

而如果计划部署在诸如Linux服务器上，那么需要考虑的就是你的 GNU C 是不是高于 C++11 标准的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2d7430174abe4e988dfc02abf4cfe9bb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

如果是新版本的GNU C编译器，那么用以上两个版本都可以，如果是低版本的请使用第一个链接下载（其实这主要是根据你代码是否要使用新特性决定，如果没有用到新特性的代码，就请下载第一个链接吧）。

下载之后验证配置是否正确，请根据具体需要参考这两篇文章。

> [Pytorch与深度学习 —— 附录I. 如何在 Ubuntu 中使用 C/C++ 语言编写 Pytorch 程序](https://seagochen.blog.csdn.net/article/details/120318273)
> [Pytorch与深度学习 —— 附录II. 如何在 Windows 中使用 MSVS 编写 Pytorch 程序](https://seagochen.blog.csdn.net/article/details/120346716)


## Step 4. 准备代码移植

模型能否正确跑起来，很大程度上在于你给模型喂的数据格式是否符合你训练时定义的大小、数据表达形式。在上一篇LSTM识别姓名分类任务的文章里，已经用Python实现了很多数据的预处理过程，但是当你把模型迁移到C上面的时候，就意味着很多工作要重新准备一遍。

所以比方说 One-Hot-Vector 的表示方式就要重写成C的形式，不过好在这些过程都不复杂。

~~~cpp
int OHVCoder::letter_to_index(char letter) {
    for (int i = 0; i < strlen(all_characters); i++) {
        if (letter == all_characters[i]) return i;
    }
    return -1;
}

void OHVCoder::line_to_one_hot_tensor
(torch::Tensor& out, std::string line, int max_padding)
{
    if (max_padding >= line.size()) {
        int features = static_cast<int>(strlen(all_characters));
        out = torch::zeros({ max_padding, features });
    }
    else {
        int linelen = static_cast<int>(line.size());
        int features = static_cast<int>(strlen(all_characters));
        out = torch::zeros({ linelen, features });
    }

    for (int idx = 0; idx < line.size(); idx++) {
        char letter = line[idx];
        out[idx][letter_to_index(letter)] = 1;
    }
}
~~~

还比如说，根据NLP的Tensor结构，把输入的姓名转化为（L, N, H）的形式


~~~cpp
void NLPBatchCreator::concatenate_tensors
(torch::Tensor& out)
{
	std::vector<torch::Tensor> list;
	list.push_back(out);
	torch::TensorList tensorlist = torch::TensorList(list);
	out = torch::pad_sequence(tensorlist);
}

void NLPBatchCreator::to_one_hot_based_tensor
(torch::Tensor& out, std::string surname, int padding)
{
	OHVCoder::line_to_one_hot_tensor(out, surname, padding);
	concatenate_tensors(out);
	//std::cout << out.sizes() << std::endl;
}
~~~

## Step. 5 准备主程序

基础工作准备妥当后，就可以在C代码里正式加载模型了。


~~~cpp

void test(torch::jit::script::Module& module, std::string name)
{
    // no gradient
    torch::NoGradGuard no_grad;

    // convert string to tensor
    torch::Tensor ts;
    NLPBatchCreator::to_one_hot_based_tensor(ts, name);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(ts);

    // forward computation
    auto probabilities = module.forward(inputs).toTensor();

    // print out predication
    auto tp = torch::max(probabilities, 1); // dim = 1
    auto max_id = std::get<1>(tp).item().toInt();

    std::cout << "Is this name " << lang[max_id] << "?" << std::endl;

    // print out each probabilities
    lang.lang_with_perhaps(probabilities);
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model." << std::endl;
        return -1;
    }

    // main cycle
    std::string input;
    while (true) {
        std::cout << "Name or exit?" << std::endl;
        std::cin >> input;

        if (input == "exit") {
            break;
        }

        // test input
        test(module, input);
    }

    std::cout << "Adios~!" << std::endl;
    return 0;
}
~~~

这里面没什么可以说太多的，基本上都写了有注释，所以你应该能比较容易理解这个主程序的。需要说明的是，Tensor 类型的对象不像C++ STL 容器技术那样可以在函数释放的时候赋值给对象（类似的也有OpenCV的Mat）。

~~~cpp
torch::Tensor yield() {
	return torch::zeros({2, 2});
}

...
auto ts = yield(); // 这样会失败
...
~~~

所以要实现赋值操作就只能依靠传递指针，或者通过引用来修改对象，我个人比较推荐C++新人用引用的方式来赋值。

~~~cpp
void yield(torch::Tensor& in) {
	in = torch::zeros({2, 2});
}

...
torch::Tensor ts;
yield(ts); // 这样可行
...
~~~

# 特别感谢
这个系列的文章很多有参考来自河北工业大学智能学院的**刘老师**所做的基础讲解——[《PyTorch深度学习实践》完结合集](https://www.bilibili.com/video/BV1Y7411d7Ys)，我在这位老师的基础工作上，结合官方的用例和说明，补充了很多在视频中没有提到的基础知识部分，并且最终完成了这个Pytorch 基础知识章节的写作。有兴趣的同学也可以先去看看刘老师所做的视频合集，也许你也会有不一样的收获。


# 打个小广告

目前在这个博客所有公开的文章里涉及到的代码，都可以在我的个人仓库里找到对应实现代码，算法主体部分主要是用Python写的，少部分测试用例是用MSVC写的，基本上都经过了测试能够正常运行，有需要的朋友可以通过以下链接访问我的仓库。

如果有什么问题，也可以在仓库的ISSUE里提出，我看到后如果有时间会回复的。

> https://github.com/seagochen/AlgorithmLearning.git