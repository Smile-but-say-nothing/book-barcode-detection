# Book-Barcode-Detection
A automatic method using OpenCV to detect book barcode

## Requirement
- OpenCV 3.4.1
- Python 3.6
- Pc & Camera
## Performance
Because the positioning of the barcode is detected first in the code by the black frame outside the barcode, the detection performane is better when the barcode has a frame. By virture of this, the future direction of improvement lies in how to identify and detect barcode without a special positioning mark like black frame. Moreover, how to speed up the method(FPS is low).

**Example 1**

![Example 1](https://img-blog.csdnimg.cn/8a933cf952874535855667582cfacbbc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ruh6IS45YaZ552A5pq06LqB,size_20,color_FFFFFF,t_70,g_se,x_16)

**Example 2**

![Example 2](https://img-blog.csdnimg.cn/a9c93f6c48d7483984f7870c41b1bde5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5ruh6IS45YaZ552A5pq06LqB,size_20,color_FFFFFF,t_70,g_se,x_16)
