@[toc]

# 1. 什么是 IoU

在深度学习的相关任务中，尤其当涉及到目标识别这一类的任务时，总能在论文或博客中看到或者听到 IoU，那么 IoU 指的是什么，它又是如何计算的呢？

IoU 的全称是「Intersection of Union」对应的中文是「交并比」，也就是交集与并集的比。我们来看看示例图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/5abf3954c3984f07a0dd7fedbb88be64.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
它表示的是我们的检测区域与目标区域的重合程度，所以自然它的取值范围在 $[0, 1]$。

# 2. 什么是「边界框（bounding box）」 
在目标检测为方向的应用中，和「IoU」相关的还有一个概念「边界框（bounding box）」。我不太确定这个概念最早出自哪里。不过对于目标检测为目的的机器学习应用来说，「bounding box」具体表达的含义是物体在「识别空间」中的范围，也就是说我们可以画一个框把需要识别的物体 “框” 起来。

![在这里插入图片描述](https://img-blog.csdnimg.cn/dbe437c60400478f9ce75af8614e2ceb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQWtpIFVud3ppaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)
比如说对上面的这张图里，需要识别的有三个物体，分别用红色、黄色、绿色的方框给“框起来”了，我们告诉模型“框起来”的物体，以及物体坐标是模型需要学习的。而模型经过学习后，当我们输入图片时，模型应该给出一个推测，也就是它认为在照片中哪些地方有物体，以及物体的范围。由此，模型推测的结果和目标之间就会出现一个误差，而评价这个误差程度的方法就是IoU。

# 3. 如何计算IoU
通常计算过程分为两部分，第一部分先计算出目标框和预测框的交集。我们假设这个框的坐标表示方式为

$$
B_{coord} = (x_{left\_top}, \ y_{left\_top}, \  x_{right\_bottom}, \ y_{right\_bottom})
$$

那么对于预测框 $B_{pred}$，对 $B_{true}$ 的交并比就可以这样计算

```python
# coordinations of predication box
lt_x0, lt_y0, rb_x0, rb_y0 = bbox_pred

# coordinations of target box
lt_x1, lt_y1, rb_x1, rb_y1 = bbox_true

# sort the coordinations of x and y
sorted_x = sort(lt_x0, rb_x0, lt_x1, rb_x1)
sorted_y = sort(lt_y0, rb_y0, lt_y1, rb_y1)

# compute the intersection area
intersection_area = (sorted_x[2] - sorted_x[1]) * (sorted_y[2] - sorted_y[1])

# compute the union area
union_area = (sorted_x[3] - sorted_x[0]) * (sorted_y[3] - sorted_y[0]) - intersection_area

# compute the iou
iou = intersection_area / (union_area + 1e-7)
```

最后这一步是为了避免出现除0的错误。这并没有结束，因为你需要知道一个问题，数据的范围只有可能存在 $[0, 1]$，所以所有超出0和1的值都要被归置为0，另外模型有可能出现两个框间距很小，但是没有相交的情况，而通过以上代码也能有一个IoU，这是不正确的。

所以我们需要做进一步的处理，判断一下两个框是否发生了相交的情况，这里可以用到一些几何的知识，也就是IoU存在的合理情况，有且只有封闭的几何图形出现相切以及相交的情况，于是就可以有

```python
# compute radius 
radius_pred_x = (rb_x0 - lt_x0) / 2.0
radius_true_x = (rb_x1 - lt_x1) / 2.0

# center x of the predication and target box
center_pred_x = (rb_x0 + lt_x0) / 2.0
center_true_x = (rb_x1 + lt_x1) / 2.0

# intersection detect
if abs(center_true_x - center_pred_x) > (radius_pred_x + radius_true_x):
	return True # the two boxes not intersected together
else:
	return False # the two boxes intersected together
```

这样我们就可以得到正确的IoU了。