import socket
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
# from controller import Calc_error, PID, calc_speed, calc_error_with_right_lane
from controller import Controller
from opt import Options
# from utils.utils import find_majority

root = Path(__file__).parent.absolute()

# Control config
global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0


def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed


# Target output size
output_size = (640, 380)

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321

# Connect to the server on local computer
s.connect(('127.0.0.1', PORT))

if __name__ == "__main__":
    # Config
    opt = Options()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'

    # Controller
    controller = Controller()

    # Load YOLOv8
    yolo = YOLO(opt.weights_yolo)

    # Load the YOLOv8 ONNX model using OpenCV's dnn module
    # net = cv2.dnn.readNet('pretrain/yolov8-best.onnx')
    # # Set the preferred backend and target for running inference
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load PIDNet
    land_detector = LandDetect('pidnet-s', os.path.join(opt.weights_lane))

    try:
        cnt_fps = 0
        t_pre = 0

        # Mask segmented image
        mask_lr = False
        mask_l = False
        mask_r = False
        mask_t = False

        # Check for reset frame
        majority_class_check = ""

        reset_counter = 0

        while True:
            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: (âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """

            try:
                message_getState = bytes("0", "utf-8")
                s.sendall(message_getState)
                s.settimeout(0.1)  # set a timeout of 1 second
                state_date = s.recv(100)

                current_speed, current_angle = state_date.decode(
                    "utf-8"
                ).split(' ')
            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            s.settimeout(0.2)  # set a timeout of 1 second
            data = s.recv(100000)

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
                    Control(-angle, speed)

                    reset_counter = 1

                # Default control
                else:
                    error = controller.calc_error(segmented_image)
                    angle = controller.PID(error, p=0.2, i=0.0, d=0.02)

                    if reset_counter >= 1 and reset_counter < 35:
                        speed = 300
                        reset_counter += 1
                    elif reset_counter == 35:
                        reset_counter = 0
                        speed = 300
                    else:
                        speed = controller.calc_speed(angle)

                        if float(current_speed) > 44.5:
                            speed = 15

                    print("Error:", error)
                    print("Angle:", angle)
                    print("Speed:", speed)

                    Control(-angle, speed)

                # ============================================================ Show image
                # Just for visualize
                # segmented_image = cv2.resize(
                #     segmented_image, (336, 200), interpolation=cv2.INTER_NEAREST)
                # yolo_output = yolo_output.plot()

                # cv2.imshow("IMG_goc", yolo_output)
                # cv2.imshow("IMG", segmented_image)
                # cv2.waitKey(5)
                # # ============================================================ Calculate FPS
                if cnt_fps >= 90:
                    t_cur = time.time()
                    fps = (cnt_fps + 1)/(t_cur - t_pre)
                    t_pre = t_cur
                    print('FPS: {:.2f}\r\n'.format(fps))
                    cnt_fps = 0

                cnt_fps += 1

                # time.sleep(0.001)
            except Exception as er:
                print(er)
                pass

    finally:
        print('closing socket')
        s.close()
