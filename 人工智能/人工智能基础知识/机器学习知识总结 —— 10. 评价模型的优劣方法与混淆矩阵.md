@[toc]

# 1. 如何评价一个模型的好坏

评价一个机器学习模型的好坏，通常需要一个具体的量化指标。在展开我们后面的内容前，我们先考虑这样一个场景。

假设我们有三类预测目标，但是我们做了可能有十种不同的模型，现在怎么评判某个模型优于另外一个模型呢？回答这个问题之前，我们可以先把预测和目标做成一张表，然后把计算结果填到这张表里面，于是对于某模型A：

![在这里插入图片描述](https://img-blog.csdnimg.cn/400e776b0a4c45cd9189644aece06d40.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
从左往右分别是1，2，3；从上往下分别是A，B，C；列头表示目标结果，也就是（target）；而行表示的是预测情况，也就是（prediction）。

对于模型A，它认为属于类别 **I** 的一共有8个，类别 **II** 的有9个，类别是 **III** 的有8个。我们来统计一下预测准确率（这时统计列的情况），我们来看看：

* 预测为类别 **I**，预测准确率为 $6/8 =75\%$
* 预测为类别 **II**，预测准确率为 $6/9 =66.67\%$
* 预测为类别 **III**，预测准确率为 $8/8 = 100\%$

这是我们从竖直方向，也就是预测准确率得出的评价指标；我们再观察这个矩阵，会发现有一些目标本来应该被预测出的，但是模型没有识别到，也就是漏检了，那么如果把漏检情况考虑进去，于是这个指标又会发生改变（这时我们要统计行的情况）：

* 实际属于类别 **I**，但实际检出率为 $6/8 = 75\%$
* 实际属于类别 **II**，但实际检出率为 $6/7 = 85.7\%$
* 实际属于类别 **III**，但实际检出率为 $8/10 = 80\%$

然后，我们可以把上面这个表简化成 0-1 分布的形式，也就是只有真和假的一张表，于是得到了所谓的「混淆矩阵」。

![在这里插入图片描述](https://img-blog.csdnimg.cn/53cb8aaf76664047a9022c32ae85fe96.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
对于这个只有二分类的混淆矩阵，它的行，和我们上面的表一样，是预测情况；列表示的实际情况。我们分别用 True(预测正确), False(预测错误), Positive(真值), Negative(假值) 表示它的预测情况。

# 2. 与「混淆矩阵」有关的几个评价指标

正如上面的例子所展示的，评价一个模型的检测情况，可以分别通过「准确率」，「查出率」两个指标，尽管在「数理学」里相关的评价指标还有几个，但是在机器学习领域，知道这两个就足够了。

## 2.1. 查准率/准确率
在很多机器学习相关书籍里，预测正确的「准确率」，被翻译为「查准率」，它的计算公式如下：

$$
P = \frac{TP}{FP + TP}
$$

P 对应的是 Precision Rate。

## 2.2. 查全率/召回率/查出率

另外一个常用的是所谓的「查全率」，我更喜欢叫它「检出率」，英文对应的是 Recall Rate。

$$R = \frac{TP}{TP + FN}$$

接下来我们将引入一个重要的评估曲线——「PR曲线」

# 3. 什么是PR曲线

PR曲线是一种评价方式，「准确率」与「检出率」往往在统计学里好似一对跷跷板的两头。模型很难得出「检出率」和「准确率」都非常高的情况。于是，我们可以把这两个比率做成一张表，于是可以得出一个「PR曲线」

![在这里插入图片描述](https://img-blog.csdnimg.cn/e3899a37892e45a5952bcd8bdb8169f4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


我们可以把不同的模型，跑同样的数据，然后把数据对不同样本的「准确率」、「检出率」绘制出来，就可以得到这样的曲线。通常情况下，曲线越靠外面的，说明模型越好，但是并非什么时候都是这样的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/409572dee55b453a88896b2bbaf3ab49.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_13,color_FFFFFF,t_70,g_se,x_16#pic_center)

当模型A和B的曲线出现这种相交时，我们无法得出模型A优于模型B，反之依然。