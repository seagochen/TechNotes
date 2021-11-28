> 在前面的章节里，已经给大家介绍了什么是RNN网络的进阶型——LSTM网络的基本知识，如果不清楚的同学请移步到[《Pytorch与深度学习 —— 10. 什么是长短期记忆网络》](https://seagochen.blog.csdn.net/article/details/120091116)。在[《Pytorch与深度学习 —— 9. 使用 RNNCell 做文字序列的转化之 RNN 入门篇》](https://seagochen.blog.csdn.net/article/details/120091116) 这篇文章里，我提前做了一些简单的铺垫，例如独热向量等基础知识后，现在我们就正式开始回答在介绍RNN网络模型一开始便提到的姓名分类问题。

@[toc]

# 回顾一下问题

我们现在有这样的一组数据集，它是按照拉丁文字进行拼写的来自不同国家的常见姓氏，如果打开这个数据集，可以发现它大概是这样

| Input | Output |
|--------|-----------|
| Abbas | English |
| Addams | English |
| Brooks | English |
| Muirchertach | Irish |
| Neil | Irish |
| Ha | Korean |
| ... | ... |

数据集我已经放在了CSDN的下载里，如果有需要的同学也可以自己去[下载](https://download.csdn.net/download/poisonchry/21736458)。

在我们这个应用中，我们要考虑的是，当输入一个新的姓名后，比如 'Abbas' 后，我们的程序能否判断出它是一个英语姓氏。

# 读取数据

回顾问题后，现在我们要来做一个读取数据的简单程序，把在文本里的姓氏，按照所在 **{ 语言 : [姓氏] }** 这种字典-列表的形式导入到程序里。

```python
# ASCII codes
all_letters = string.ascii_letters + " .,;'"

def find_files(path): return glob.glob(path)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

def load_data(path):
    for filename in find_files(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        language_list.append(category)
        lines = read_lines(filename)
        names_dictionary[category] = lines
```

我们主要依靠的是这几段代码，它们的作用就是从文本中依次读取每一个名字，然后把名字由UTF-8 转码成ASCII，然后以存储在前面我提到过的 **{语言: [姓氏]}** 这样的字典-列表结构中。


# 对文本进行编码
为了让程序能够理解数据集，我们需要对这些字符串数据进行一定程度的编码。One-Hot-Vector 我在前面的文章里已经解释过了，所以在这里不做过多的重复。

这里只做一些必要的补充性介绍。

## 使用 One-Hot-Vector 编码姓名

我们已经通过如下的代码，创建出了一个新的用于编码的字符序列，这个序列包括一些特定的符号（在西班牙语、葡萄牙语等传统拉丁语族国家才有的重音符号）。

> all_letters = string.ascii_letters + " .,;'"

比方说我们要编码一个名叫 'abc' 的姓名，那么每一个字符对应一个长度为57的One-Hot向量。然后按照顺序进行输出后，结果应该是

~~~bash
tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]]])
~~~

由 ‘abc’ 所转化成的用于网络输入的张量维度就是 (3, 1, 57)，也就是我在前面章节介绍过的对于NLP方向来说，pytorch接受的数据默认维度即：

$$(L, N, H_{in}) \approx (Sequence, Batch, Features)$$

**Batch**

一个Batch对应一个单词，比如上面提到的 'abc'；

**Sequence**

即这个单词包含有多少个字符，对于单词 'abc' 来说，它包含3个字符；

**Features**

即每一个字符所对应的 One-Hot 编码。


**需要注意的地方**

这里存在一个新手很容易犯的问题，就是对于我们的数据来说，有单词 'Muirchertach' 长度为12， 也有单词像 'Ha' 这样只有长度为2的，对于可执行并行计算的神经网络来说，如果数据长度不统一，那么不仅导致网络处理效率会降低，而且我们实际处理起来也特别麻烦。所以我们需要对这些数据做一个类似 padding 的操作。

### 对数据进行填充

这个概念在很多方面都有提到或者广泛使用，如果是第一次学习计算机或者数据分析的朋友，填充简而言之就是下面这个列表表示的意思。

原始列表 | 填充后至8位长
-------------|----------
[‘a’, 'b', 'c'] | [‘a’, 'b', 'c', 0, 0, 0, 0, 0]
['h', 'e', 'l', 'l', 'o'] | ['h', 'e', 'l', 'l', 'o', 0, 0, 0]

理解这些基本概念后，我们就可以使用代码实现这个过程。

~~~python
def line_to_one_hot_tensor(line, max_padding=0):
    """
    Turn a line into a one-hot based tensor (character, one-hot-vector)
    """
    if max_padding >= len(line):
        tensor = torch.zeros(max_padding, len(all_letters))
    else:
        tensor = torch.zeros(len(line), len(all_letters))

    for idx, letter in enumerate(line):
        tensor[idx][_letter_to_index(letter)] = 1
    return tensor
~~~

这个代码片段会把比如 'abc' 这样的单词转换为 (Sequence, Features) 这样结构的独热向量表示的张量。之所以没有直接转换为 （L, N, H）这样的结构，是因为我们在读取数据的时候可能一次性要读取很多个不同的单词，所以得到的单词组，比如 ['Abbas', 'Addams', ...] 这样的数组，就可以通过下面这段代码，再转换成 （L、N、H）的结构了。

~~~python
from torch.nn.utils.rnn import pad_sequence

def concatenate_tensors(tensor_list):
    return pad_sequence(tensor_list)

def to_one_hot_based_tensor(surnames: list, padding=20):
    tensors = []
    for name in surnames:
        tensor = line_to_one_hot_tensor(name, padding)
        tensors.append(tensor)

    return concatenate_tensors(tensors)
~~~

## 使用序列编码语言

我们提到过，通过神经元网络输出的结果，其实是个概率。比如通过网络输入的Features是这样的一组 One-Hot 向量

$$\overrightarrow{Input} = \left\{ \begin{matrix}
(1, 0, 0, 0) \\
(1, 0, 0, 0)\\
(0, 1, 0, 0) \\
(0,0,1,0)
\end{matrix} \right.$$

经过我们的网络处理后，输出的 Output 是对应的每一个标签的可能性：

$$\overrightarrow{Output} = \left\{ \begin{matrix}
label.1 & 0.15 \\
label.2 & 0.35 \\
label.3 & 0.1 \\
label.4 & 0.4
\end{matrix} \right.$$

然后经过比如交叉熵进行比对的时候，我们告诉这个网络输出的值其实应该是 label 2， 你给出的 label 4 是错误的，所以网络会根据我们告诉它的情况，执行反馈计算的时候调整 label 2 和 label 4的权重。

因此，对于我们这个例子来说，我们就需要把 ['English', 'Irish', 'French', ...] 这一类字符标签，转化成 [0, 1, 2, 3, 4, 5, ....] 这样的形式，所以其实还是挺简单的。

~~~python

def line_to_index(line: str, data_list: list):
    """
    Turn a line into an index based from dataset
    """
    return data_list.index(line)
    
def to_lang_list_tensor(lang_list: list, languages: list):
    """
    lang_list: 每一个姓名所对应的语言
    languages: 语言所在的列表
    """
    indices = []
    for lang in lang_list:
        index = line_to_index(lang, languages)
        indices.append(index)

    return torch.tensor(indices)
~~~

所以我们用到了这样两段极为简单的代码，帮助我们转化标签。

# 数据加载

数据加载这里，由于我们使用的是自己的数据，所以没法直接用 Pytorch 提供的 DataLoader，但是我们可以重载名为Dataset的类。

~~~python
from torch.utils.data import Dataset

class MyNameDataset(Dataset):

    def __init__(self, dict_data: dict):
        self.x_data = []
        self.y_data = []
        self.languages = []

        for lang, names in dict_data.items():
            for name in names:
                self.x_data.append(name)
                self.y_data.append(lang)

            self.languages.append(lang)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]
~~~

我们把 **{'语言': [姓名]}** 这个字典-列表类型输入这个重载类后，再加载到DataLoader里，就可以根据需要输出我们想要的 **{标签：姓名}** 姓名对了。

比如我们通过DataLoader，让它一次性抓取10条数据，输出的结果就是这样的

> [('Durdin', 'Guliev', 'Palmer', 'Gerhard', 'Timpe', 'Jelvakov', 'Seighin', 'Neverov', 'Babayants', 'Robishaw'), ('Russian', 'Russian', 'English', 'German', 'Czech', 'Russian', 'Irish', 'Russian', 'Russian', 'English')]


# 编写LSTM网络模型

我们的这个模型比较简单，用到了一层LSTM作为主要的数据处理，以及一层线性层做最终的输出。

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

这里需要注意的是经过LSTM计算后的网络，输出的Output维度是

$$(L, N, D * H_{out})$$

由于我们使用的是单向，所以D=1，最终输出的维度是

$$(L, N, H_{out})$$

但是线性层能接受的输入维度是

$$(N, H_{in})$$

这意味着我们要把LSTM网络输出的结构转化成线性层可接受的维度， 即

$$(L, N, H_{out}) \rightarrow (N, H_{in}) = ( N, L * H_{out})$$

这里我提供一个比较笨的转化方法，你可以在学会LSTM之后对这部分进行修改。

~~~
def convert_hidden_shape(hidden, batch_size):
    tensor_list = []

    for i in range(batch_size):
        ts = hidden[:, i, :].reshape(1, -1)
        tensor_list.append(ts)

    ts = torch.cat(tensor_list)
    return ts
~~~

另外就是需要注意下，维度转换的时候，一定要注意保证数据的正确性和完整性，否则会影响最终的输出。

# 把上面的内容拼接起来

这部分就是例行公事了，创建网络对象、选择合适的损失函数、合适的优化函数，然后制定训练和测试过程。

**主要过程**

~~~python
    # define a model
    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        sequence_size=SEQUENCE_SIZE,
        batch_size=BATCH_SIZE)

    # convert to cuda model
    # model = model.cuda()

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # majorized function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training and testing process
    for epoch in range(10):

        # load dataset
        languages, train_loader, test_loader = load_dataset()

        # training
        train(epoch, model, languages, optimizer, criterion, train_loader)

        # testing
        test(model, languages, test_loader)  
~~~

这里稍微玩了点小花招，我们在每一次训练网络的过程中，都会重新随机的加载一次数据，这样保证每次训练集和测试集都有所不同，能够更好的评估和修正模型的准确度。

然后分别就是训练过程和测试过程的代码了

**训练过程**

~~~python
def train(epoch, model, lang, optimizer, criterion, train_loader):
    running_loss = 0

    for idx, data in enumerate(train_loader, 0):

        # convert data
        input_x, label_y = convert_data(data, lang)

        # clear the gradients
        optimizer.zero_grad()

        # forward computation
        predicate_y = model(input_x)

        # loss computation
        loss = criterion(predicate_y, label_y)

        # backward propagation
        loss.backward()

        # update network parameters
        optimizer.step()

        # print loss
        running_loss += loss.item()
        if idx % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, idx, running_loss / 100))
            running_loss = 0
~~~


**测试过程**

~~~python
def test(model, lang, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader, 0):
            # convert data
            input_x, label_y = convert_data(data, lang)

            # predicate
            predicate_y = model(input_x)

            # check output
            _, predicated = torch.max(predicate_y.data, dim=1)
            total += label_y.size(0)
            correct += (predicated == label_y).sum().item()

    print("Accuracy on test set: %d %%" % (100 * correct / total))
~~~

**实验输出结果**

如果程序一切正常，输出的结果就是这样的

~~~bash
C:\Users\seago\anaconda3\envs\algorithm\python.exe D:/Repositories/AlgorithmLearning/Neurons/RecurrentNeuralNetwork/LSTM/LSTMMain.py
[0,     0] loss: 0.029
[0,   100] loss: 2.005
[0,   200] loss: 1.582
[0,   300] loss: 1.567
[0,   400] loss: 1.463
[0,   500] loss: 1.341
[0,   600] loss: 1.443
[0,   700] loss: 1.287
[0,   800] loss: 1.347
[0,   900] loss: 1.254
[0,  1000] loss: 1.293
[0,  1100] loss: 1.288
[0,  1200] loss: 1.253
[0,  1300] loss: 1.084
[0,  1400] loss: 1.024
Accuracy on test set: 72 %
[1,     0] loss: 0.008
[1,   100] loss: 0.933
[1,   200] loss: 0.921
[1,   300] loss: 0.881
[1,   400] loss: 0.942
[1,   500] loss: 0.851
[1,   600] loss: 0.979
[1,   700] loss: 0.850
[1,   800] loss: 0.810
[1,   900] loss: 0.814
[1,  1000] loss: 0.822
[1,  1100] loss: 0.746
[1,  1200] loss: 0.850
[1,  1300] loss: 0.804
[1,  1400] loss: 0.799
[1,  1500] loss: 0.770
Accuracy on test set: 79 %
[2,     0] loss: 0.002
[2,   100] loss: 0.721
[2,   200] loss: 0.716
[2,   300] loss: 0.656
[2,   400] loss: 0.585
[2,   500] loss: 0.667
[2,   600] loss: 0.669
[2,   700] loss: 0.705
[2,   800] loss: 0.605
[2,   900] loss: 0.652
[2,  1000] loss: 0.632
[2,  1100] loss: 0.621
[2,  1200] loss: 0.652
[2,  1300] loss: 0.628
[2,  1400] loss: 0.656
[2,  1500] loss: 0.636
Accuracy on test set: 82 %
[3,     0] loss: 0.021
[3,   100] loss: 0.512
[3,   200] loss: 0.532
[3,   300] loss: 0.563
[3,   400] loss: 0.625
[3,   500] loss: 0.533
[3,   600] loss: 0.611
[3,   700] loss: 0.554
[3,   800] loss: 0.546
[3,   900] loss: 0.487
[3,  1000] loss: 0.566
[3,  1100] loss: 0.586
[3,  1200] loss: 0.496
[3,  1300] loss: 0.501
[3,  1400] loss: 0.570
[3,  1500] loss: 0.588
Accuracy on test set: 83 %
[4,     0] loss: 0.003
[4,   100] loss: 0.456
[4,   200] loss: 0.532
[4,   300] loss: 0.497
[4,   400] loss: 0.460
[4,   500] loss: 0.458
[4,   600] loss: 0.484
[4,   700] loss: 0.464
[4,   800] loss: 0.475
[4,   900] loss: 0.444
[4,  1000] loss: 0.470
[4,  1100] loss: 0.518
[4,  1200] loss: 0.522
[4,  1300] loss: 0.497
[4,  1400] loss: 0.520
[4,  1500] loss: 0.545
Accuracy on test set: 84 %
[5,     0] loss: 0.007
[5,   100] loss: 0.442
[5,   200] loss: 0.480
[5,   300] loss: 0.420
[5,   400] loss: 0.464
[5,   500] loss: 0.470
[5,   600] loss: 0.454
[5,   700] loss: 0.484
[5,   800] loss: 0.448
[5,   900] loss: 0.479
[5,  1000] loss: 0.474
[5,  1100] loss: 0.511
[5,  1200] loss: 0.414
[5,  1300] loss: 0.462
[5,  1400] loss: 0.395
Accuracy on test set: 86 %
[6,     0] loss: 0.008
[6,   100] loss: 0.396
[6,   200] loss: 0.406
[6,   300] loss: 0.363
[6,   400] loss: 0.406
[6,   500] loss: 0.415
[6,   600] loss: 0.402
[6,   700] loss: 0.382
[6,   800] loss: 0.444
[6,   900] loss: 0.438
[6,  1000] loss: 0.410
[6,  1100] loss: 0.409
[6,  1200] loss: 0.402
[6,  1300] loss: 0.424
[6,  1400] loss: 0.479
[6,  1500] loss: 0.404
Accuracy on test set: 87 %
[7,     0] loss: 0.001
[7,   100] loss: 0.312
[7,   200] loss: 0.357
[7,   300] loss: 0.354
[7,   400] loss: 0.402
[7,   500] loss: 0.363
[7,   600] loss: 0.411
[7,   700] loss: 0.358
[7,   800] loss: 0.342
[7,   900] loss: 0.331
[7,  1000] loss: 0.448
[7,  1100] loss: 0.408
[7,  1200] loss: 0.376
[7,  1300] loss: 0.381
[7,  1400] loss: 0.422
[7,  1500] loss: 0.443
Accuracy on test set: 88 %
[8,     0] loss: 0.001
[8,   100] loss: 0.351
[8,   200] loss: 0.362
[8,   300] loss: 0.348
[8,   400] loss: 0.275
[8,   500] loss: 0.349
[8,   600] loss: 0.396
[8,   700] loss: 0.339
[8,   800] loss: 0.349
[8,   900] loss: 0.374
[8,  1000] loss: 0.406
[8,  1100] loss: 0.390
[8,  1200] loss: 0.348
[8,  1300] loss: 0.368
[8,  1400] loss: 0.391
[8,  1500] loss: 0.383
Accuracy on test set: 88 %
[9,     0] loss: 0.004
[9,   100] loss: 0.295
[9,   200] loss: 0.305
[9,   300] loss: 0.294
[9,   400] loss: 0.341
[9,   500] loss: 0.330
[9,   600] loss: 0.341
[9,   700] loss: 0.337
[9,   800] loss: 0.388
[9,   900] loss: 0.406
[9,  1000] loss: 0.369
[9,  1100] loss: 0.371
[9,  1200] loss: 0.367
[9,  1300] loss: 0.353
[9,  1400] loss: 0.377
[9,  1500] loss: 0.376
Accuracy on test set: 89 %

Process finished with exit code 0

~~~



# 测试一下结果

可以看到整体的实验结果还算满意，大概有89%左右的准确度。不过如果想进一步提高准确度就会有点问题。其中一个重要的原因就是因为数据本身带有bias，如果那有兴趣统计一下词条数，就会发现样本总体数量不完全均衡。

~~~bash
Arabic (2000)        train: 1498 (74.90%)   test: 502 (25.10%)   
Chinese (268)        train: 210 (78.36%)    test: 58 (21.64%)    
Czech (519)          train: 394 (75.92%)    test: 125 (24.08%)   
Dutch (297)          train: 222 (74.75%)    test: 75 (25.25%)    
English (3668)       train: 2747 (74.89%)   test: 921 (25.11%)   
French (277)         train: 211 (76.17%)    test: 66 (23.83%)    
German (724)         train: 536 (74.03%)    test: 188 (25.97%)   
Greek (203)          train: 147 (72.41%)    test: 56 (27.59%)    
Irish (232)          train: 182 (78.45%)    test: 50 (21.55%)    
Italian (709)        train: 527 (74.33%)    test: 182 (25.67%)   
Japanese (991)       train: 760 (76.69%)    test: 231 (23.31%)   
Korean (94)          train: 74 (78.72%)     test: 20 (21.28%)    
Polish (139)         train: 111 (79.86%)    test: 28 (20.14%)    
Portuguese (74)      train: 53 (71.62%)     test: 21 (28.38%)    
Russian (9408)       train: 7083 (75.29%)   test: 2325 (24.71%)  
Scottish (100)       train: 78 (78.00%)     test: 22 (22.00%)    
Spanish (298)        train: 227 (76.17%)    test: 71 (23.83%)    
Vietnamese (73)      train: 51 (69.86%)     test: 22 (30.14%)    

Process finished with exit code 0
~~~

为了验证测试结果，我稍微把源码做了一点修改，通过输入姓名后，然后给出各种语言的推测概率

~~~bash
Name? Abalov
My predicate is  Russian
Arabic              : 0.0
Chinese             : 0.0
Czech               : 0.0
Dutch               : 0.0
English             : 0.0
French              : 0.0
German              : 0.0
Greek               : 0.0
Irish               : 0.0
Italian             : 0.0
Japanese            : 0.0
Korean              : 0.0
Polish              : 0.0
Portuguese          : 0.0
Russian             : 4.1e+01
Scottish            : 0.0
Spanish             : 0.0
Vietnamese          : 0.0
Name? Tuma
My predicate is  Arabic
Arabic              : 1.7e+01
Chinese             : 0.0
Czech               : 0.0
Dutch               : 0.0
English             : 1.4
French              : 0.0
German              : 0.0
Greek               : 0.0
Irish               : 0.0
Italian             : 0.48
Japanese            : 0.0
Korean              : 0.0
Polish              : 0.0
Portuguese          : 0.0
Russian             : 9.9
Scottish            : 0.0
Spanish             : 2.5
Vietnamese          : 0.0
Name? Burden
My predicate is  English
Arabic              : 0.0
Chinese             : 0.0
Czech               : 0.0
Dutch               : 0.0
English             : 9.8
French              : 0.73
German              : 0.25
Greek               : 0.0
Irish               : 3.2
Italian             : 0.0
Japanese            : 0.0
Korean              : 0.0
Polish              : 0.0
Portuguese          : 0.0
Russian             : 0.0
Scottish            : 0.0
Spanish             : 0.0
Vietnamese          : 0.0
Name? 
~~~

基本上对于名字数量占大头的英语、阿拉伯语和俄语姓名，总体比较准确，但是对于其他少数语言，韩语、越南语、汉语就表现就差强人意，有可能时条数低于300的好像都不怎么准确。

~~~bash
Name? Long
My predicate is  English
Arabic              : 0.0
Chinese             : 0.0
Czech               : 0.0
Dutch               : 0.0
English             : 3.0
French              : 0.0
German              : 1.2
Greek               : 0.0
Irish               : 0.0
Italian             : 0.0
Japanese            : 0.0
Korean              : 0.0
Polish              : 0.0
Portuguese          : 0.0
Russian             : 0.0
Scottish            : 0.0
Spanish             : 0.0
Vietnamese          : 0.0
Name? Yu
My predicate is  Arabic
Arabic              : 0.0
Chinese             : 0.0
Czech               : 0.0
Dutch               : 0.0
English             : 0.0
French              : 0.0
German              : 0.0
Greek               : 0.0
Irish               : 0.0
Italian             : 0.0
Japanese            : 0.0
Korean              : 0.0
Polish              : 0.0
Portuguese          : 0.0
Russian             : 0.0
Scottish            : 0.0
Spanish             : 0.0
Vietnamese          : 0.0
Name? Blanc
My predicate is  French
Arabic              : 0.0
Chinese             : 0.0
Czech               : 0.0
Dutch               : 0.0
English             : 3.7
French              : 5.6
German              : 0.0
Greek               : 0.0
Irish               : 1.5
Italian             : 0.0
Japanese            : 0.0
Korean              : 0.0
Polish              : 0.0
Portuguese          : 0.0
Russian             : 0.0
Scottish            : 0.0
Spanish             : 0.0
Vietnamese          : 0.0
Name? Bran
My predicate is  German
Arabic              : 0.0
Chinese             : 0.0
Czech               : 0.0
Dutch               : 0.0
English             : 0.1
French              : 0.0
German              : 3.6
Greek               : 0.0
Irish               : 3.0
Italian             : 0.0
Japanese            : 0.0
Korean              : 0.0
Polish              : 0.0
Portuguese          : 0.0
Russian             : 0.0
Scottish            : 0.0
Spanish             : 0.0
Vietnamese          : 0.0
Name? 
~~~

# 保存训练好的模型

如果你希望把训练好的模型保存起来，并且期待能否把模型应用到一般的应用程序里，比如说C程序里，那么就可以把模型和参数都保存起来。

**保存模型和数据**

文件的后缀名没啥强制要求，我比较喜欢叫ptm，因为是 pytorch model 的简写，你也可以自己定义个喜欢的后缀名。

~~~python
    # finally save the model
    torch.save(model, "LSTM_Surname_Classfication.ptm")
~~~

----

在下一章节里，我将给大家演示如何使用C程序加载训练好的模型。

欢迎关注我的博客~ 

Adios~~

# 参考资料
* 《NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN》， Sean Robertson，https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html