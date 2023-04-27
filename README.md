# Self-driving Car Simulator

This project is a final project of the [AIO-2022 cource](https://www.facebook.com/aivietnam.edu.vn) about self-driving car simulator built in the Unity environment. It uses YOLOv8 Nano for traffic light detection and PIDNet for lane segmentation. The car is controlled using a PID controller rule-based on the traffic light and segmented image.

## Introduction

### Detection model
YOLOv8 Nano is one of the models in the YOLOv8 family of object detection models from Ultralytics. It is the fastest and smallest model in the family, while still providing state-of-the-art performance. YOLOv8 builds on the success of previous versions of YOLO, introducing new features and improvements for enhanced performance, flexibility, and efficiency. It supports a full range of vision AI tasks, including detection, segmentation, pose estimation, tracking, and classification12.

### Segmentation model
PIDNet is a real-time semantic segmentation network inspired by PID controllers. It is a novel three-branch network architecture that contains three branches to parse detailed, context and boundary information respectively. The additional boundary branch is introduced to mimic the PID controller architecture and remedy the overshoot issue of previous models. PIDNet achieves the best trade-off between inference speed and accuracy and surpasses all existing models with similar inference speed on the Cityscapes and CamVid datasets34.

### Controller
A PID controller (Proportional Integral Derivative controller) is a control loop mechanism that is widely used in industrial control systems and other applications requiring continuously modulated control. It continuously calculates an error value as the difference between a desired setpoint and a measured process variable, and applies a correction based on proportional, integral, and derivative terms (denoted P, I, and D respectively). The controller attempts to minimize the error over time by adjustment of a control variable to a new value determined by a weighted sum of the control terms1.

## Usage
Step 1: Download weights and place them in the ```pretrain``` directory

Step 2: Start Unity

Step 3: Run the model:

```
python PIDv8.py
```


## Demo

[![Watch the video](https://i.imgur.com/vKb2F1B.png)](https://youtu.be/-5h1aEelFdw)

## Contributing

Contributions to this project are welcome! If you have any ideas or suggestions for improvements, please feel free to open an issue or submit a pull request.


## References
* [YOLOv8](https://github.com/ultralytics/ultralytics)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [PIDNet](https://github.com/XuJiacong/PIDNet)
