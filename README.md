# everybodyDanceNow_sample

based on pix2pixHD

step1：训练数据准备

    Target video：
    1、手机拍摄，以下是要求

        1)	分辨率1080p，>15min
        2)	摄像头保持不动，垂直地面
        3)	帧率设为120fps，减少拖影
        4)	背景最好不要有变化，例如阳光，倒影
        5)	单独一人
        6)	衣服尽量减少褶皱，紧身衣最好
        7)	如果是长发，需固定束好
        8)	选定一个中心位置，然后围绕中心位置乱动，注意收集侧脸和背影的数据，不要有表情
        9)	可以做和目标舞蹈相近的动作
        10)	要求动作足够丰富，摇头晃脑够乱才行
        11)	注意留小段无人的场景，方便截取留作背景

    2、做裁剪，拼接，窗口截取分辨率为1920*1024适应网络的输入
    工具使用ffmpeg，几个命令：
    从视频中截一段出来：
    ffmpeg -i input.wmv -ss 00:00:30.0 -c copy -to 00:01:10.0 output.wmv

    3、截取视频中一张站立的姿势的图像作为姿态对齐使用，截取一张无人的图像作为背景使用。
    至此target video准备完毕
    
    Source video：
    1. 单人
    2. 高清，确保关键点提取的准确，特别是手部
    3. 摄像头固定
    和target video不同，不需要做烦琐的预处理，只需要一张正立姿态的图片用到后面的姿态对齐中

step2：姿态检测，openPose，保存json关键点文件

step3：util/smooth_points.py
主要是将训练视频产生的关键点的json文件转成txt文件，后面用C++读json比较麻烦，所以转成txt文件。

step4:姿态对齐, util/transPose.py
测试的时候，将source video中的关键点姿态根据target video中的姿态进行对齐，同样生成txt文件。

step5: user_code/smooth.cpp
通过3、4步中产生的关键点的txt文件，将火柴图画出来，这样做的好处是所有的视频都只需要检测一次，检测的过程非常慢，画图比较快。
需要放在openpose官方工程目录下，/openpose/examples/user_code 路径下面。

step6: train.py

step7: test.py

step8: util/image2video.py

result: https://zhuanlan.zhihu.com/p/47591419

https://zhuanlan.zhihu.com/p/55134122
