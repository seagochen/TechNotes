@[toc]

# 函数说明

实现这个功能，主要依靠OpenCV的 **VideoCapture** ，它可以用于打开摄像头或者视频，如果你使用笔记本，并且带有摄像头的话，可以使用 0 打开摄像头。如果给定视频位置，则打开视频。

```cpp
VideoCapture ()
```

默认函数，通常我们不使用这个来构造VideoCapture对象。

```cpp
VideoCapture (const String &filename, int apiPreference=CAP_ANY)
```
用来打开视频，也可以用来打开IP摄像头。其中 **apiPreference** 可以用来强制 OpenCV 使用某种方法对数据进行解码，可以用参数包括
* cv::CAP_FFMPEG
* cv::CAP_IMAGES
* cv::CAP_DSHOW.

```cpp
VideoCapture (int index, int apiPreference=CAP_ANY)
```

index指示的是打开什么视频设备，通常我们令index为0，打开默认的摄像头。



# 使用Python打开摄像头或者视频

```python
    video = cv2.VideoCapture(0)
    cv2.namedWindow("frame")

    while video.isOpened():
        success, frame = video.read()
        
        if success and cv2.waitKey(1) & 0xFF != ord('q'):
            cv2.imshow('frame', frame)
        else:
            break
            
    cv2.destroyAllWindows()
    video.release()
```

# 使用C++打开摄像头或者视频

~~~cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    cv::VideoCapture video("sample.mp4");
    cv::namedWindow("frame");

    cv::Mat frame;
    while(video.isOpened()) {
        bool success = video.read(frame);

        if (success && cv::waitKey(1) != 'q') {
            cv::imshow("frame", frame);
        } else {
            break;
        }
    }


    cv::destroyAllWindows();
    video.release();

    return -1;
}
~~~

# 展现效果

![在这里插入图片描述](https://img-blog.csdnimg.cn/8b6ecc1033794625bd526e6c1567cc59.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5omT56CB55qE6Zi_6YCa,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
