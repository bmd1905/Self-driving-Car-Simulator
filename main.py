import cv2
import numpy as np
import time
import os
import torch

from ultralytics import YOLO

from tools.custom import LandDetect
from tools.controller import Controller
from utils.config import ModelConfig, ControlConfig
from utils.socket import create_socket


def main(s, config, config_model, controller, yolo, land_detector):
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
                # Send signal to the server.
                message_getState = bytes("0", "utf-8")
                s.sendall(message_getState)

                # Set a timeout of 0.1 seconds.
                s.settimeout(0.1)

                # Receive 100 bytes of data from the server.
                state_date = s.recv(100)

                # Decode the received data from UTF-8.
                config.current_speed, config.current_angle = state_date.decode("utf-8").split(" ")

            except Exception as er:
                pass

            # Send sendBack_angle and sendBack_Speed to the server.
            message = bytes(f"1 {config.sendBack_angle} {config.sendBack_Speed}", "utf-8")
            s.sendall(message)

            # Set a timeout of 0.5 seconds.
            s.settimeout(0.5)

            # Receive 100000 bytes of data from the server.
            data = s.recv(100000)


            try:
                # Decode image in byte type recieved from server
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                    ), -1
                )

                # ============================================================ PIDNet
                segmented_image = land_detector.reference(
                    image, config_model.segmented_output_size, mask_lr, mask_l, mask_r, mask_t)

                # ============================================================ YOLO
                # Resize the image to the desired dimensions
                imaeg = cv2.resize(image, (640, 384))

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
                # Resize image if it's too small for you to see
                # segmented_image = cv2.resize(
                #     segmented_image, (336, 200), interpolation=cv2.INTER_NEAREST)

                
                yolo_output = yolo_output.plot()

                if config_model.view_seg:
                    cv2.imshow("segmented image", segmented_image)

                if config_model.view_first_view:
                    cv2.imshow("first view image", yolo_output)
                
                cv2.waitKey(1)
                # ============================================================ Calculate FPS
                if cnt_fps >= 90:
                    t_cur = time.time()
                    fps = (cnt_fps + 1)/(t_cur - t_pre)
                    t_pre = t_cur
                    print('FPS: {:.2f}\r\n'.format(fps))
                    cnt_fps = 0

                cnt_fps += 1

            except Exception as er:
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

    main(s, config_control, config_model, controller, yolo, land_detector)

    