import cv2
import numpy as np
from numpy import random
import time
from pathlib import Path
import os
import sys
import torch

from ultralytics import YOLO

from tools.custom import LandDetect
from tools.controller import Controller
from utils.opt import ModelConfig, ControlConfig
from utils.socket import create_socket, send_state, recv_state


# Target output size
output_size = (640, 380)


def main(s, config, controller, yolo, land_detector):
    try:
        cnt_fps = 0
        t_pre = 0

        # Mask segmented image
        mask_lr = False
        mask_l = False
        mask_r = False
        mask_t = False

        # Counter for speed up after turning
        reset_counter = 0

        while True:
            """
            Input:
                image: the image returned from the car
                current_speed: the current speed of the car
                current_angle: the current steering angle of the car
            You must use these input values to calculate and assign the steering angle and speed of the car to 2 variables:
            Control variables: sendBack_angle, sendBack_Speed
                where:
                sendBack_angle (steering angle)
                sendBack_Speed (speed control)
            """
                
            try:
                # message_getState = bytes("0", "utf-8")
                # s.sendall(message_getState)
                # s.settimeout(0.1)  # Set a timeout of 0.1 second
                # state_date = s.recv(100)

                # config.current_speed, config.current_angle = state_date.decode(
                #     "utf-8"
                # ).split(' ')
                # message = bytes(f"1 {config.sendBack_angle} {config.sendBack_Speed}", "utf-8")
                # s.sendall(message)
                config.current_speed, config.current_angle = send_state(s, config)
                # s.settimeout(0.2)  # Set a timeout of 0.2 second
                # data = s.recv(100000)
                data = recv_state(s)
            except Exception as er:
                pass

            

            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                    ), -1
                )
                image_ = image.copy()
                # ============================================================ PIDNet
                segmented_image = land_detector.reference(
                    image, output_size, mask_lr, mask_l, mask_r, mask_t)

                # ============================================================ YOLO
                # Resize the image to the desired dimensions
                image = cv2.resize(image, (640, 384))

                with torch.no_grad():
                    yolo_output = yolo(image)[0]

                # ============================================================ Controller
                angle, speed, next_step, mask_l, mask_r = controller.control(segmented_image=segmented_image,
                                                                             yolo_output=yolo_output)

                # Control when turing
                if next_step:
                    print("Next step")
                    print("Angle:", angle)
                    print("Speed:", speed)
                    config.update(-angle, speed)

                    reset_counter = 1

                # Default control
                else:
                    error = controller.calc_error(segmented_image)
                    angle = controller.PID(error, p=0.2, i=0.0, d=0.02)

                    # Speed up after turning (in 35 frames)
                    if reset_counter >= 1 and reset_counter < 35:
                        speed = 300
                        reset_counter += 1
                    elif reset_counter == 35:
                        reset_counter = 0
                        speed = 300 
                    else:
                        speed = controller.calc_speed(angle)

                        if float(config.current_speed) > 44.5:
                            speed = 15

                    print("Error:", error)
                    print("Angle:", angle)
                    print("Speed:", speed)

                    config.update(-angle, speed)

                # ============================================================ Show image
                # Just for visualize
                # segmented_image = cv2.resize(
                #     segmented_image, (336, 200), interpolation=cv2.INTER_NEAREST)
                # yolo_output = yolo_output.plot()

                # cv2.imshow("IMG_goc", yolo_output)
                # cv2.imshow("IMG", segmented_image)
                # cv2.waitKey(5)
                # ============================================================ Calculate FPS
                if cnt_fps >= 90:
                    t_cur = time.time()
                    fps = (cnt_fps + 1)/(t_cur - t_pre)
                    t_pre = t_cur
                    print('FPS: {:.2f}\r\n'.format(fps))
                    cnt_fps = 0

                cnt_fps += 1

            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()

if __name__ == "__main__":
    # Create socket
    s = create_socket()

    # Config model
    config_model = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'

    # Config control
    config_control = ControlConfig()

    # Controller
    controller = Controller()

    # Load YOLOv8
    yolo = YOLO(config_model.weights_yolo)

    # Load the YOLOv8 ONNX model using OpenCV's dnn module
    # net = cv2.dnn.readNet('pretrain/yolov8-best.onnx')
    # # Set the preferred backend and target for running inference
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load PIDNet
    land_detector = LandDetect('pidnet-s', os.path.join(config_model.weights_lane))

    main(s, config_control, controller, yolo, land_detector)

    