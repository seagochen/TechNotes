@[toc]

在[《OpenCV API使用笔记 —— 1. 如何打开摄像头或视频文件》](https://blog.csdn.net/poisonchry/article/details/120435439) 介绍过使用「VideoCapture」类，可以打开摄像头或视频文件，如果数据经过处理后，我们希望保存这些数据时，又该怎么做呢。

# 写入图片帧信息

我们这里主要用到一个名为 **VideoWriter** 的类，它可以帮助我们达成以上目标。现在来看看「VideoWriter」类的原型：

```cpp
cv::VideoWriter::VideoWriter()；

cv::VideoWriter::VideoWriter(
const String & 	filename,
int 	fourcc,
double 	fps,
Size 	frameSize,
bool 	isColor = true 
)；


cv::VideoWriter::VideoWriter(
const String & 	filename,
int 	apiPreference,
int 	fourcc,
double 	fps,
Size 	frameSize,
bool 	isColor = true 
)；
```

对于C来说，可以使用上面三个构造函数中的任意一个，而我个人比较偏好使用第二个构造函数。以上参数名分别表示如下含义：

* filename，保存的视频文件名
* apiPreference，可以用来指定保存视频时使用的解码器，可以使用「cv::CAP_FFMPEG」、「cv::CAP_GSTREAMER」等；如果使用第一个或第二个构造函数，那么当程序运行在Linux系统时，它会默认使用FFMPEG，Windows时使用FFMPEG或VFW，如果是MacOS时，则使用QTKit。
* fourcc，视频压缩指令，使用4个字节进行表示。例如「VideoWriter::fourcc('P','I','M','1')」使用 MPEG-1 codec, 「VideoWriter::fourcc('M','J','P','G')」使用 motion-jpeg codec. 关于codec的相关指令，可以在 [codecs](https://www.fourcc.org/codecs.php) 里找到详细列表。
* fps，用于指定视频的帧率
* frameSize，用于指定视频的帧大小
* isColor，默认以彩色模式处理数据，如果设置为false时，它将以灰白模式处理数据。


# C/CPP示例
```cpp
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat)
#include <opencv2/videoio.hpp>  // Video write
using namespace std;
using namespace cv;

static void help()
{
    cout
        << "------------------------------------------------------------------------------" << endl
        << "This program shows how to write video files."                                   << endl
        << "You can extract the R or G or B color channel of the input video."              << endl
        << "Usage:"                                                                         << endl
        << "./video-write <input_video_name> [ R | G | B] [Y | N]"                          << endl
        << "------------------------------------------------------------------------------" << endl
        << endl;
}

int main(int argc, char *argv[])
{
    help();
    if (argc != 4)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }

    const string source      = argv[1];           // the source file name
    const bool askOutputType = argv[3][0] =='Y';  // If false it will use the inputs codec type

    VideoCapture inputVideo(source);              // Open input
    if (!inputVideo.isOpened())
    {
        cout  << "Could not open the input video: " << source << endl;
        return -1;
    }

    string::size_type pAt = source.find_last_of('.');                  // Find extension point
    const string NAME = source.substr(0, pAt) + argv[2][0] + ".avi";   // Form the new name with container
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    Size S = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter outputVideo;                                        // Open the output
    if (askOutputType)
        outputVideo.open(NAME, ex=-1, inputVideo.get(CAP_PROP_FPS), S, true);
    else
        outputVideo.open(NAME, ex, inputVideo.get(CAP_PROP_FPS), S, true);
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << source << endl;
        return -1;
    }

    cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
         << " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
    cout << "Input codec type: " << EXT << endl;
    int channel = 2; // Select the channel to save
    switch(argv[2][0])
    {
    case 'R' : channel = 2; break;
    case 'G' : channel = 1; break;
    case 'B' : channel = 0; break;
    }

    Mat src, res;
    vector<Mat> spl;
    for(;;) //Show the image captured in the window and repeat
    {
        inputVideo >> src;              // read
        if (src.empty()) break;         // check if at end
        split(src, spl);                // process - extract only the correct channel
        for (int i =0; i < 3; ++i)
            if (i != channel)
                spl[i] = Mat::zeros(S, spl[0].type());
       merge(spl, res);
       //outputVideo.write(res); //save or
       outputVideo << res;
    }
    cout << "Finished writing" << endl;
    return 0;
}
```

上面的内容相对比较复杂，不过主要步骤如下：

```cpp
// 创建output
VideoWriter outputVideo;

// 指定参数
if (askOutputType)
   	outputVideo.open(NAME, ex=-1, inputVideo.get(CAP_PROP_FPS), S, true);
else
   	outputVideo.open(NAME, ex, inputVideo.get(CAP_PROP_FPS), S, true);

// 写入数据
outputVideo << res;
```

如果对于Python来说，上述执行步骤就可以变得更简单了

# Python示例

```python
import cv2

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 15
print(f"width: {width}, height: {height}, fps: {fps}")

fourcc = cv2.VideoWriter_fourcc(*"DIVX")
writer = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

	cv2.imshow("Frame", frame)
    writer.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cap.release()
```