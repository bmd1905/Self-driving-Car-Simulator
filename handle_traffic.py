import numpy as np
import cv2
import time
from tools.custom import LandDetect


class Controller():
    def __init__(self, s, land_detector):
        self.error_arr = np.zeros(5)
        self.error_sp = np.zeros(5)

        self.pre_t = time.time()
        self.pre_t_spd = time.time()

        self.output_size = (640, 380)

        self.sendBack_angle = 0
        self.sendBack_Speed = 0

        self.turning_counter = 0
        self.angle_turning = 0

        self.majority_class = ""

        self.s = s

        self.land_detector = land_detector

        # Mask
        self.mask_lr = False
        self.mask_l = False
        self.mask_r = False
        self.mask_t = False

        self.sum_left_corner = 0
        self.sum_right_corner = 0
        self.sum_top_corner = 0

    def __call__(self, majority_class, rs_image, boxes):
        # Calculate areas from bouding boxe
        areas = (boxes[2] - boxes[0]) * \
            (boxes[3] - boxes[1])

        # Calculate area of left, right, and top corner
        self.sum_left_corner = np.sum(rs_image[:25, :25, 0])
        self.sum_right_corner = np.sum(
            rs_image[:25, -25:, 0])
        self.sum_top_corner = np.sum(
            rs_image[:25, 67:92, 0])

        # ======================================================= Control
        # For turn right
        if areas > 3500.0 and majority_class == 'turn_right':
            self.handle_turn_right()

        # For turn left
        if areas > 1100.0 and majority_class == 'turn_left':
            self.handle_turn_left()

        # For straight
        if areas > 600.0 and majority_class == 'straight':
            self.handle_straight()

        # For no turn left
        if areas >= 4000.0 and majority_class == 'no_turn_left':
            self.handle_no_turn_left()

        # For no turn right
        if areas >= 4000.0 and majority_class == 'no_turn_right':
            self.handle_no_turn_right()

        # For no straight
        if areas > 2500.0 and majority_class == 'no_straight':
            self.handle_no_straight()

    def calc_error(self, image):
        arr = []
        height = 12
        lineRow = image[height, :]
        for x, y in enumerate(lineRow):
            if y[0] == 100:
                arr.append(x)

        if len(arr) > 0:
            center = int((min(arr) + max(arr))/2)
            error = int(image.shape[1]/2) - center
            return error

    def calc_error_with_right_lane(self, image):
        arr = []
        height = 12
        lineRow = image[height, :]
        for x, y in enumerate(lineRow):
            if y[0] == 100:
                arr.append(x)

        if len(arr) > 0:
            # center = int((min(arr) + max(arr))/2)
            center_right_lane = int((min(arr) + max(arr)*3)/4)
            error = int(image.shape[1]/2) - center_right_lane
            return error
        else:
            return 0

    def PID(self, error, p, i, d):  # 0.43,0,0.02
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error*p
        delta_t = time.time() - self.pre_t
        self.pre_t = time.time()
        D = (error-self.error_arr[1])/delta_t*d
        I = np.sum(self.error_arr)*delta_t*i
        angle = P + I + D

        angle = int(angle)

        if abs(angle) > 25:
            angle = np.sign(angle)*25

        return int(angle)

    def calc_speed(self, angle):
        if abs(angle) < 10:
            speed = 60
        elif 10 <= abs(angle) <= 20:
            speed = 40
        else:
            speed = 30
        return speed

    def control(angle, speed):
        global sendBack_angle, sendBack_Speed
        sendBack_angle = angle
        sendBack_Speed = speed

    def handle_turn_left(self):
        angle = 8

    def handle_turn_right(self):
        angle = -9

    def handle_no_turn_left(self):
        if self.sum_right_corner > self.sum_top_corner/2:  # Turn right
            angle = -0.0001
        else:
            angle = 1e-5  # Keep straight

    def handle_no_turn_right(self):
        if self.sum_left_corner > self.sum_top_corner/5:  # Turn left
            angle = 10
        else:
            angle = 1e-5  # Straight

    def handle_straight(self):
        self.mask_lr = True

        for i in range(200):
            print("Straight", i)
            image = self.send_and_receive_data()

            rs_image = self.land_detector.reference(
                image, self.output_size, self.mask_lr, self.mask_l, self.mask_r, self.mask_t)

            error = self.calc_error_with_right_lane(rs_image)
            angle = self.PID(error, p=0.20, i=0.0, d=0.08)
            speed = self.calc_speed(angle)

            self.sendBack_angle = angle
            self.sendBack_Speed = speed

            cv2.imshow("IMG", rs_image)

        self.mask_lr = False

    def handle_no_straight(self):
        if self.sum_left_corner > self.sum_right_corner:
            # Turn left
            angle = 9
        else:
            # Turn right
            angle = -8

    def send_and_receive_data(self):
        while True:
            try:
                message_getState = bytes("0", "utf-8")
                self.s.sendall(message_getState)
                self.s.settimeout(0.1)
                state_date = self.s.recv(100)

            except Exception as er:
                print(er)
                pass

            message = bytes(
                f"1 {self.sendBack_angle} {self.sendBack_Speed}", "utf-8")
            self.s.sendall(message)
            self.s.settimeout(0.2)
            data = self.s.recv(100000)

            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                    ), -1
                )
            except:
                continue

            return image


# import numpy as np
# import time


# class Controller():
#     def __init__(self, s, land_detector):
#         self.error_arr = np.zeros(5)
#         self.error_sp = np.zeros(5)

#         self.pre_t = time.time()
#         self.pre_t_spd = time.time()

#     def calc_error(self, image):
#         arr = []
#         height = 12
#         lineRow = image[height, :]
#         for x, y in enumerate(lineRow):
#             if y[0] == 100:
#                 arr.append(x)

#         if len(arr) > 0:
#             center_right_lane = int((min(arr) + max(arr)*2.6)/3.6)
#             error = int(image.shape[1]/2) - center_right_lane
#             return error

#     def PID(self, error, p, i, d):
#         global pre_t
#         global error_arr
#         error_arr[1:] = error_arr[0:-1]
#         error_arr[0] = error
#         P = error*p
#         delta_t = time.time() - pre_t
#         pre_t = time.time()
#         D = (error-error_arr[1])/delta_t*d
#         I = np.sum(error_arr)*delta_t*i
#         angle = P + I + D

#         if abs(angle) > 25:
#             angle = np.sign(angle)*25

#         return int(angle)

#     def calc_speed(self, angle):
#         if abs(angle) < 10:
#             speed = 60
#         elif 10 <= abs(angle) <= 20:
#             speed = 30
#         else:
#             speed = 30
#         return speed
