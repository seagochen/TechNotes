@[toc]

在某些时候，我们可能需要在图像数据被处理后保存结果。对于 OpenCV 来说，我们需要保存的主要有两种数据，一种是图片，还有一种是视频。在这个章节里，我们先来探讨如何读区和保存图片数据。

# 读取图片数据

这个功能，可以用到下面这个函数。

```cpp
Mat cv::imread(const String & filename, int flags = IMREAD_COLOR)	
```
## 参数说明
* filename，读取的文件名，或文件的绝对、相对路径；
* flags，文件读取时以什么方式读取，默认为彩色。

关于可用的flags选项，可以看下面这个表
![在这里插入图片描述](https://img-blog.csdnimg.cn/631834d2d0c447ceaf02575b19c17ea4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
**请注意：** 不同版本的OpenCV，在参数选项这块稍微有些不一样，如果你的IDE支持代码补全，可以用关键字的方式找到对应的FLAG，如果不支持这个功能，那么有可能你还是需要到官网上查看对应版本的描述信息，以上内容目前都是 **OpenCV 4.5.6** 的内容。

## 支持格式
OpenCV里自带了很多开源的，或免费使用的解码库，所以可以支持如下格式的数据读取。
* Windows bitmaps - *.bmp, *.dib (always supported)
* JPEG files - *.jpeg, *.jpg, *.jpe (see the Note section)
* JPEG 2000 files - *.jp2 (see the Note section)
* Portable Network Graphics - *.png (see the Note section)
* WebP - *.webp (see the Note section)
* Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
* PFM files - *.pfm (see the Note section)
* Sun rasters - *.sr, *.ras (always supported)
* TIFF files - *.tiff, *.tif (see the Note section)
* OpenEXR Image files - *.exr (see the Note section)
* Radiance HDR - *.hdr, *.pic (always supported)
* Raster and Vector geospatial data supported by GDAL (see the Note section)

# 保存图片数据

当某个图片我们处理完毕后，可以使用如下函数保存结果。

```cpp
bool cv::imwrite(const String & filename, InputArray img, 
	const std::vector<int> &params = std::vector< int >())	
```

## 参数说明
* filename，保存的文件名，或文件的绝对、相对路径；
* img，Mat 矩阵数据，处理后的图片数据，如果Mat要以图片形式进行保存，请确保它的底层数据格式为 uint8，通常经过一些列计算后，数据会被升格为 float32 的浮点型，如果就这样直接存起来，会导致报错或精度丢失；
* params，图片存储辅助参数，一般不会用到，可以用来指定图片的大小、存储格式、采样率等信息。

具体的params信息，可以看下面这个表

![在这里插入图片描述](https://img-blog.csdnimg.cn/e3cd0eaa639440aea6d785f9123a1006.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
如果你需要使用参数，参数的使用方式是这样的：

```cpp
	std::vector<int> params;
	// set jpeg quality to 100
	params.push_back(IMWRITE_JPEG_QUALITY);
	params.push_back(100);
	// set luma quality to 10
	params.push_back(IMWRITE_JPEG_LUMA_QUALITY);
	params.push_back(10);
	// ...

	// save image
	imwrite("foobar.jpg", mat_image, params);
```

# 用例

```cpp
#include <opencv2/imgcodecs.hpp>
using namespace cv;
using namespace std;

static void paintAlphaMat(Mat &mat)
{
    CV_Assert(mat.channels() == 4);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            Vec4b& bgra = mat.at<Vec4b>(i, j);
            bgra[0] = UCHAR_MAX; // Blue
            bgra[1] = saturate_cast<uchar>((float (mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX); // Green
            bgra[2] = saturate_cast<uchar>((float (mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX); // Red
            bgra[3] = saturate_cast<uchar>(0.5 * (bgra[1] + bgra[2])); // Alpha
        }
    }
}
int main()
{
    Mat mat(480, 640, CV_8UC4); // Create a matrix with alpha channel
    paintAlphaMat(mat);
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    bool result = false;
    try
    {
        result = imwrite("alpha.png", mat, compression_params);
    }
    catch (const cv::Exception& ex)
    {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
    if (result)
        printf("Saved PNG file with alpha data.\n");
    else
        printf("ERROR: Can't save PNG file.\n");
    vector<Mat> imgs;
    imgs.push_back(mat);
    imgs.push_back(~mat);
    imgs.push_back(mat(Rect(0, 0, mat.cols / 2, mat.rows / 2)));
    imwrite("test.tiff", imgs);
    printf("Multiple files saved in test.tiff\n");
    return result ? 0 : 1;
}
```