> 我鸽了很久，终于有点时间来填这个梯度下降算法的坑。

@[toc]

[《梯度下降算法——1. 什么是梯度下降》](https://blog.csdn.net/poisonchry/article/details/116401539?spm=1001.2014.3001.5502) 在上一章里，介绍了什么是梯度下降算法，如果还有点懵逼，那么你就应该好好看一看这篇文章了。看完后，你应该能理解什么是梯度下降算法了。

# 数据准备

首先，我们需要原始的数据。我们用一个比较复杂的，但是在我其他文章里反复用到的正态分布核去生成我们需要的原始数据，并设置它的$\mu$和$\sigma$分别是3和5，这样我们可以得到这样一个非标准的正态分布曲线的x和y值：

```python
def gaussian_noise_kernel(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210531160935988.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
生成了数据点还不够，我们先把数据存起来，使用CSV格式进行存储。关于CSV的操作，你可以查一些网上公开的资料，相信能很容易理解的。

**存储数据**
```python
def save_pts(x_pts, y_pts, filename="Data/data.csv"):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for item in zip(x_pts, y_pts):
            csv_writer.writerow(item)
```

**读取数据**

```python
def load_pts(filename="Data/data.csv"):
    x_pts = []
    y_pts = []
    with open(filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            x_pts.append(float(row[0]))
            y_pts.append(float(row[1]))
    return x_pts, y_pts
```

到目前为止，我们已经顺利的准备好测试数据。现在我们的问题转换成，我们有如下一组数据，我们如何求解它是由什么函数，或者已知某函数$f(x)$，求解其中的函数参数。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210602112404851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
由于我们会利用到计算机能够大规模重复计算的能力，所以我们会采用一个准暴力的方法去求解函数的参数。但是我们需要一个判断方法，以验证我们暴力求解的数值是否与期望相近。

这里引入一个新的概念叫 **评价函数(损失函数)** ，用来指导我们程序。它类似于如下一个过程：

```mermaid
flowchat
st=>start: Start
e=>end: End
op=>operation: 计算新的参数
cond=>condition: 符合期望？

st->op->cond
cond(yes)->e
cond(no)->op
```

# 评价函数（损失函数）

**评价函数(损失函数)**, 可以多种多样，在这里，我们选取均方差为我们的评价函数：

$$
cost(x) = \frac{1}{N} \sum (\hat{y} - y)^2
$$

其中$\hat{y}$，是我们使用了新参数产生的数值，我们将每次产生的新数值都要和原函数的数值进行比对，以此确定误差有多大，并确定改进方向。

因此，我们可以在这里做一些简单的计算，并把每次生成的均方差和对应的 $\mu$ 和 $\sigma$， 绘制到一张图上，看看生成的效果如何。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210602210427475.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
我们再换几个角度观察这个三维图形

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021060221100752.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021060221105834.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3BvaXNvbmNocnk=,size_16,color_FFFFFF,t_70#pic_center)可以发现在我们这个平面里，有个梯度最低点，这也就是全局最优点，是我们求解的目标值。

# 梯度下降

想象一个问题，如果一个待求解的函数问题包含若干个变量，如果一个变量有1000种可能解，那么这个问题的计算量就会轻易的变成 $O(n) = 1000^n$ 也就是一个 $O(n) = m^n$ 问题，那么计算量就会成为海量，也就是所谓的**维度爆炸**，这显然不是我们追求的。

从性价比来说，如果我们有一种办法能告知我们的程序，让它朝着某一个特定方向缩小计算范围，而那个范围里有可能存在着全局最优解，那么这将大大节省我们的计算量和花费的时间。

而这种思路就是所谓的 **梯度下降**。想象在一个平面上放置一个金属小球，在不施加外力的情况下，它会通常自动滚停在这个平面上的最低点，梯度下降算法也是一样的思路，只不过我们用计算机模拟了这一个过程。

如果设计算法的话，那么这个思路大致就是这样：

1. 随机的给定程序一组**初始参数**，然后给定一个固定的**参数变化率（在深度学习领域又称为学习率）**，以及**迭代次数**
2. 程序使用输入的参数，计算与比对数据之间的损失，如果**loss > 0**，且**迭代次数> 0**，则执行第三步；否则执行第四步；
3. 对输入参数加上/减去变化参数，并与数据对比损失，如果损失减小，则更新变化率，并重复第二步；
4. 返回给用户收敛的最终参数。

OK，那么我们用代码实现以上思路：

```python
def cost(y_hats, y_origin):
    return np.sum((y_hats - y_origin) ** 2) / len(y_hats)


def update(mu, sigma, x, y, cost_val, learning_rate):
    # update mu
    y_hat = gaussian_noise_kernel(x, mu - learning_rate, sigma)
    y_hat_cost = cost(y_hat, y)

    if cost_val > y_hat_cost:
        mu = mu - learning_rate
    else:
        mu = mu + learning_rate

    # update sigma
    y_hat = gaussian_noise_kernel(x, mu, sigma - learning_rate)
    y_hat_cost = cost(y_hat, y)

    if cost_val > y_hat_cost:
        sigma = sigma - learning_rate
    else:
        sigma = sigma + learning_rate

    return mu, sigma


if __name__ == "__main__":
    x, y = load_data()
    mu = 100
    sigma = 100
    learning_rate = 1
    iterations = 100

    while iterations > 0:
        predicated_y = gaussian_noise_kernel(x, mu, sigma)
        cost_val = cost(predicated_y, y)

        if cost_val > 0:
            mu, sigma = update(mu, sigma, x, y, cost_val, learning_rate)

        iterations = iterations - 1

    print("mu is", mu, "sigma is", sigma)
```

输出的结果是多少呢：

```
mu is 5 sigma is 3
```

Ok， 得到了我们想要的最终结果。

# 后记

当然，你可能在其他书上看到实现方法不是这样的，你可能会看到别人使用了梯度函数去逼近结果。但是我这里提供的是一种比较简单，而且实用的程序化求解的方法。并且这些不同思路的梯度下降，本质上是一样的，希望你能理解这背后蕴含的伟大数学思想。
