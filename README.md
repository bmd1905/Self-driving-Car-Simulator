# Self-driving Car Simulator

This project is a final project of the [AIO-2022 cource](https://www.facebook.com/aivietnam.edu.vn) about self-driving car simulator built in the Unity environment. It uses YOLOv8 Nano for traffic light detection and PIDNet for lane segmentation. The car is controlled using a PID controller rule-based on the traffic sign and segmented image.

## Introduction
### Pipeline
<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/236652354-843e9a41-3289-435c-be5a-fee681d38f2f.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="600" height="400" />
</p>

### Detection task
We compared difference models on number of parameters, mAP@.5, and mAP@.5:.95, this is our result:

<div align="center">

| Model | Params (M) | mAP@.5 | mAP@.5:.95 |
|---|---|---|---|
| YOLOv5m | 21.2 | 0.993 | 0.861 |
| YOLOv6n | 4.7 | **0.996** | 0.856 |
| YOLOv7 | 6 | 0.991 | 0.835 |
| YOLOv8n | **3.2** | 0.992 | **0.887** |

</div>

Based on our results, YOLOv8 Nano has been adopted for this project. 

These are some example images that were predicted using YOLOv8 Nano. The model was successful in predicting small objects, which is significant for the control task.
<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/236683285-79e0f75c-a199-4b30-98de-2ca0a1ef8be2.png" width="700" height="500" />
</p>

### Segmentation task
PIDNet is a real-time semantic segmentation network inspired by PID controllers. It is a novel three-branch network architecture that contains three branches to parse detailed, context and boundary information respectively. The additional boundary branch is introduced to mimic the PID controller architecture and remedy the overshoot issue of previous models. PIDNet achieves the best trade-off between inference speed and accuracy and surpasses all existing models with similar inference speed on the Cityscapes and CamVid datasets34.

<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/236684096-ce02c2b3-5b9c-48da-a712-9d0d0e6bb5e5.png" width="700" height="300" />
</p>



### Controller
A PID controller (Proportional Integral Derivative controller) is a control loop mechanism that is widely used in industrial control systems and other applications requiring continuously modulated control. It continuously calculates an error value as the difference between a desired setpoint and a measured process variable, and applies a correction based on proportional, integral, and derivative terms (denoted P, I, and D respectively). The controller attempts to minimize the error over time by adjustment of a control variable to a new value determined by a weighted sum of the control terms1.

<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/236684583-6f31d6ff-80eb-44c4-99ee-0df2c42a4f10.png" width="500" height="350" />
</p>


#### Pseudocode for Control

```
Find A and B
C = (A+B)/2     (midpoint)
error ≡ IC = HI-HC
angle = PID(error,  p,  i,  d)
speed = f(angle)


*OH: hyper-parameter (e.g., ⅓ or ¼ of image height)
```

#### Pseudocode for Handle Traffic Sign
```
# Pre-turning
for each detected_class in yolo_outputs:
	if detected_class in traffic_signs:
		majority_classes.add(detected_class)

		if len(majority_classes) = 10:
			majority_class = find_majority(majority_classes)
			turning = True
```

```
# Turning
while turning_counter <= max_turning_counter:
	switch case for majority_class:
		angle = constant
		speed = constant
		
		turning_counter += 1
```

## Usage
Step 1: Download weights and place them in the ```pretrain``` directory

Step 2: Start Unity

Step 3: Run the script:

```
python main.py
```


## Demo

<video src="https://www.youtube.com/watch?v=gZ3nPZWp-eE&ab_channel=BuiMinhDuc" controls="controls" style="max-width: 730px;">
</video>

Video demo [here](https://youtu.be/gZ3nPZWp-eE).

## Contributing

Contributions to this project are welcome! If you have any ideas or suggestions for improvements, please feel free to open an issue or submit a pull request.


## References
* [YOLOv8](https://github.com/ultralytics/ultralytics)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [PIDNet](https://github.com/XuJiacong/PIDNet)
