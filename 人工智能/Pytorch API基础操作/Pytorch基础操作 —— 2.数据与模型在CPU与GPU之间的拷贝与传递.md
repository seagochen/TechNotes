
@[toc]

# GPU 与 CPU 的运算对比

首先不是所有的电脑都有GPU，我们这里的GPU要强调，必须是 **Nvidia** 家的显卡，所以你无论是Intel的独显，还是AMD家的独显，都没法使用到以下的特性加速你的计算过程，那就更不要提什么核显这种了。

GPU相对CPU来说更擅长科学计算，这是因为GPU舍弃，或大大简化了CPU需要负担的复杂任务执行的Control单元，而同时有更多负责加减乘除运算的ALU单元。

![在这里插入图片描述](https://img-blog.csdnimg.cn/0840d18e391d44abb771c81d85a7365b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

为了更直观对比两者在计算性能上的差异，我们做一个实验。

~~~python
import torch
import time


with torch.no_grad():
    # 程序计时开始
    time_start = time.time()

    tensor1 = torch.randn(100, 1000, 1000)
    tensor2 = torch.randn(100, 1000, 1000)

    result = tensor1 * tensor2
    for i in range(1000):
        result = result + tensor1 * tensor2

    # 程序片段后插入以下两行
    time_end = time.time()
    print('Time cost on CPU = %fs' % (time_end - time_start))


    # 程序计时开始
    time_start = time.time()

    tensor1 = torch.randn(100, 1000, 1000).cuda()
    tensor2 = torch.randn(100, 1000, 1000).cuda()

    result = tensor1 * tensor2
    for i in range(1000):
        result = result + tensor1 * tensor2

    # 程序片段后插入以下两行
    time_end = time.time()
    print('Time cost on GPU = %fs' % (time_end - time_start))
~~~

这是一个程序计时，我们模拟的是两个高维度的张量数据的计算，这种运算规模在亿级以上。使用CPU的计算结果是：

~~~bash
Time cost on CPU = 178.318684s
~~~

而到了GPU上，则表现为

~~~bash
Time cost on GPU = 4.024427s
~~~

我现在使用的设备主要是Dell G7，主要的配置是

![在这里插入图片描述](https://img-blog.csdnimg.cn/d23b5e485be74076b07187277c7162dd.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)

可以看到GPU其实是一颗很羸弱的1060，但是同样维度的张量计算却比CPU块了不止一倍。这就是GPU的恐怖计算能力。


# 张量或模型所在的设备位置
我们在创建了张量，或者网络模型后，有时候会好奇这些模型、数据存储在哪里，所以可以通过下面这条命令来查看。

~~~python
tensor.device
~~~

这对于模型也是一样的。

~~~python
next(model.parameters()).device
~~~

我们来看看怎么用的

~~~python
import torch


class UserDefinedModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 30)

    def forward(self, data):
        return self.linear(data)


tensor1 = torch.randn(100, 1000, 1000)
tensor2 = tensor1.cuda()

cpu_model = UserDefinedModel()
gpu_model = UserDefinedModel().cuda()

print("tensor on", tensor1.device)
print("tensor on", tensor2.device)

print("model on", next(cpu_model.parameters()).device)
print("model on", next(gpu_model.parameters()).device)
~~~

输出结果：

~~~bash
tensor on cpu
tensor on cuda:0
model on cpu
model on cuda:0
~~~

# 检查自己的设备是否支持CUDA

现在你可能会问，我有什么办法可以知道自己的设备是否支持CUDA呢？你可以执行下面这个命令来获得自己的设备支持情况。

~~~python
import torch

if torch.cuda.is_available():
	print("CUDA available")
else:
	print("CPU only")
~~~

# 把数据或模型从CPU转到GPU上

由于 torch 同时支持GPU与CPU计算, 这使得它的应用性可以更广泛的覆盖到不同的设备上. 你可以根据自己设备的特点, 来决定如何更好的使用 torch。

目前对于科学计算来说，Nvidia家的显卡能更好的支持科学计算, 所以自然而然的, 当需要消耗大量资源的矩阵计算出现时, 最好是把这类计算全部放到GPU上更为合算. 所以我们可以通过以下命令，把数据写入到GPU的运存上

拷贝方法很简单，可以这样做：

**CPU $\rightarrow$ GPU**
```python
if torch.cuda.is_available():
  tensor = tensor.cuda()
```

你也可以这样：

~~~python
import torch

# 检查支持什么设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 把数据拷贝到适合的设备上
tensor1 = torch.randn(10, 10).to(device)
print(tensor1.device)
~~~

相对来说，上面的方法相对比较温和一些

# 把数据或模型拷贝到多张GPU上

如果你有多张GPU，通常是渲染工作站，有多过一张的CUDA卡，那么也可以通过指定的形式，把数据传拷贝到指定的 GPU 设备的运存上

```python
	tensor_cuda_0 = tensor.to('cuda:0')
	tensor_cuda_1 = tensor.to('cuda:1')
	tensor_cuda_2 = tensor.to('cuda:2')
	...
```

也可以使用上面提到过的device方法

~~~python
import torch

# 检查支持什么设备
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device3 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device4 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
...

tensor_cuda_0 = tensor.to(device1)
tensor_cuda_1 = tensor.to(device2)
tensor_cuda_2 = tensor.to(device3)
...
~~~

或者直接用.cuda()

~~~python
tensor1 = torch.randn(10, 10).cuda()
~~~

这个指令会把数据传送到默认的CUDA设备上，如果有多块设备要制定，可以这样

~~~python
tensor1 = torch.randn(10, 10).cuda(0)
tensor2 = torch.randn(10, 10).cuda(1)
tensor3 = torch.randn(10, 10).cuda(2)
...
~~~

# 把数据或模型拷贝到回CPU上


如果想要把数据从 GPU 拷贝到回 CPU 的运存上，那么就执行如下命令：

```python
	tensor = tensor.to('cpu')
```

除此以外，目前的torch还支持使用 cpu() ，以函数的形式进行拷贝。

```python
	tensor = tensor.cpu()
```