import socket
import numpy as np
from numpy import random
import cv2
import time
from pathlib import Path
import os
import torch
import sys

from tools.custom import LandDetect
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized, TracedModel
from controller import Calc_error, PID, calc_speed
from utils.plots import plot_one_box
from opt import Options

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
    opt.weights_yolo = 'pretrain/my-yolov7-tiny-best-final.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'

    # Load YOLOv7
    yolo = attempt_load(opt.weights_yolo, map_location=device)
    stride = int(yolo.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    if True:
        yolo = TracedModel(yolo, device, opt.img_size)

    # Get names and colors
    names = yolo.module.names if hasattr(yolo, 'module') else yolo.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Load PIDNet
    land_detector = LandDetect('pidnet-s', os.path.join(opt.weights_lane))

    try:
        cnt_fps = 0
        t_pre = 0
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
                s.settimeout(1) # set a timeout of 1 second
                state_date = s.recv(100)
            
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                    ).split(' ')

            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            s.settimeout(1) # set a timeout of 1 second
            data = s.recv(100000)

            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                        ), -1
                    )
                # ============================================================ PIDNet
                rs_image = land_detector.reference(image, output_size)

                # ============================================================ YOLO 
                # Resize the image to the desired dimensions
                image = cv2.resize(image, (640, 384))
                image_ = image.copy()

                # We want this size: [1, 3, 384, 640]
                yolo_input = image.transpose(
                    (2, 0, 1)).copy()  # -> [3, 384, 640]
                yolo_input = torch.from_numpy(yolo_input).to(device)
                yolo_input = yolo_input.half() if half else yolo_input.float()  # uint8 to fp16/32
                yolo_input /= 255.0  # 0 - 255 to 0.0 - 1.0

                if yolo_input.ndimension() == 3:
                    yolo_input = yolo_input.unsqueeze(0)  # -> [1, 3, 384, 640]

                # Warmup
                if device.type != 'cpu' and (old_img_b != yolo_input.shape[0] or old_img_h != yolo_input.shape[2] or old_img_w != yolo_input.shape[3]):
                    old_img_b = yolo_input.shape[0]
                    old_img_h = yolo_input.shape[2]
                    old_img_w = yolo_input.shape[3]
                    for i in range(3):
                        yolo(yolo_input)[0]

                with torch.no_grad():
                    yolo_output = yolo(yolo_input)[0]

                t2 = time_synchronized()
                # Apply NMS
                yolo_output = non_max_suppression(
                    yolo_output, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                t3 = time_synchronized()
                # Process detections
                for i, det in enumerate(yolo_output):  # Detections per image
                    # Normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            yolo_input.shape[2:], det[:, :4], image_.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # Create bouding box and save in im0 variable
                            plot_one_box(xyxy, image_, label=label,
                                         color=colors[int(cls)], line_thickness=3)
                    else:
                        continue

                # ============================================================ Controller
                # Option 1
                # angle = PID(error, p=0.48, i=0.02, d=0.18)
                # height 12, clip 125
                # waitkey 5

                # Option 2 (not successed)
                # angle = PID(error, p=0.20, i=0.0, d=0.20)
                # height 12
                # waitkey 50

                error = Calc_error(rs_image)
                angle = PID(error, p=0.48, i=0.02, d=0.18)
                speed = calc_speed(angle)

                Control(-angle, speed)
                # ============================================================ Show image
                # rs_image = cv2.resize(rs_image, output_size, interpolation=cv2.INTER_NEAREST) # For visualize
                cv2.imshow("IMG_goc", image_)
                cv2.imshow("IMG", rs_image)
                cv2.waitKey(1)
                # ============================================================ Calculate FPS
                if cnt_fps >= 30:
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
