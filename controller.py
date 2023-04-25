import numpy as np
import time

from utils.utils import find_majority


class Controller():
    def __init__(self):
        self.error_arr = np.zeros(5)
        self.error_sp = np.zeros(5)

        self.pre_t = time.time()
        self.pre_t_spd = time.time()

        self.sendBack_angle = 0
        self.sendBack_speed = 0

        self.traffic_lights = ['turn_right', 'turn_left',
                               'straight', 'no_turn_left', 'no_turn_right', 'no_straight']

        self.class_names = ['no', 'turn_right', 'straight', 'no_turn_left',
                            'no_turn_right', 'no_straight', 'car', 'unknown', 'turn_left']

        self.stored_class_names = []  # Stored class for finding majority class

        self.majority_class = ""
        self.start_cal_area = False
        self.turning_counter = 0
        self.angle_turning = 0

        self.sum_left_corner = 0
        self.sum_right_corner = 0
        self.sum_top_corner = 0

        self.mask_l = False
        self.mask_r = False
        self.mask_lr = False
        self.mask_t = False

        self.next_step = False
        self.is_turning = False

        self.reset_counter = 0

    def reset(self):
        # Reset
        self.turning_counter = 0
        self.majority_class = ""
        self.start_cal_area = False
        self.stored_class_names = []

        self.mask_lr = False
        self.mask_l = False
        self.mask_r = False
        self.mask_t = False

        self.stored_class_names = []  # Stored class for finding majority class

        self.majority_class = ""
        self.start_cal_area = False
        self.turning_counter = 0
        self.angle_turning = 0

        self.next_step = False
        self.is_turning = False

        self.reset_counter = 0

    def control(self, segmented_image, yolo_output):

        # Reset in 90 frames
        # and (not self.start_cal_area or not self.is_turning):
        # and not self.start_cal_area and not self.is_turning:
        if self.majority_class == "turn_left":
            # Increase counter for controlling
            self.reset_counter += 1
            if self.reset_counter >= 100:
                self.reset()
                print("Reset"*200)

        # Calculate area of left, right, and top corner of the segmented image
        self.sum_left_corner = np.sum(segmented_image[:25, :25, 0])
        self.sum_right_corner = np.sum(
            segmented_image[:25, -25:, 0])
        self.sum_top_corner = np.sum(
            segmented_image[:25, 67:92, 0])

        print("Is calculate areas:", self.start_cal_area)
        print("Is turning:", self.is_turning)

        if self.start_cal_area:
            self.calc_areas(segmented_image, yolo_output)

        elif self.is_turning:
            self.handle_turning()

        # Get class from yolo output for adding to stored classes list
        elif len(self.stored_class_names) < 9:
            preds = yolo_output.boxes.data.numpy()  # List of (bouding_box, conf, class_id)

            for pred in preds:
                class_id = int(pred[-1])
                if self.class_names[class_id] in self.traffic_lights:
                    self.stored_class_names.append(self.class_names[class_id])

                # Making stored class more robust because of the wakeness of YOLO model
                if self.class_names[class_id] == 'straight':
                    self.stored_class_names.remove('turn_left')
                    self.stored_class_names.remove('turn_right')

                # if self.class_names[class_id] == 'no_straight':
                #     self.stored_class_names.extend(['no_straight']*2)

                if self.class_names[class_id] == 'turn_right':
                    self.stored_class_names.extend(['turn_right']*2)
                    # self.stored_class_names.remove('turn_left')
                    # self.stored_class_names.remove('straight')

                if self.class_names[class_id] == 'turn_left':
                    self.stored_class_names.extend(['turn_left'])

                # if self.class_names[class_id] == 'no_turn_left':
                #     self.stored_class_names.extend(['no_turn_left']*3)

                if self.class_names[class_id] == 'no_turn_right':
                    self.stored_class_names.extend(['no_turn_right'])
                    # self.stored_class_names.remove('straight')

        # Starting to find majority class
        elif len(self.stored_class_names) >= 9:  # 9 is a hyperparameter
            # Get the majority class
            self.majority_class = find_majority(
                self.stored_class_names)[0]  # Returned in set type

            # Assign for checking
            # majority_class_check = majority_class

            # Start calculate areas
            self.start_cal_area = True

        return self.sendBack_angle, self.sendBack_speed, self.next_step, self.mask_l, self.mask_r

    def handle_turning(self):
        print("Handle Turning")
        # Default config
        speed = 0
        angle = 0

        # Check turning counter
        if self.turning_counter < 30:  # self.turning_counter >= 1 and
            print('Turning Counter:', self.turning_counter)

            match self.majority_class:
                case 'turn_left':
                    speed = 30
                    if self.turning_counter <= 12:
                        angle = 2
                    elif self.turning_counter >= 13 and self.turning_counter <= 24:
                        angle = self.angle_turning
                    elif self.turning_counter == 26:
                        self.turning_counter = 30
                    elif self.turning_counter >= 25:
                        angle = 0

                case 'turn_right':
                    speed = 1
                    if self.turning_counter <= 8:
                        angle = 2
                    elif self.turning_counter == 20:
                        self.turning_counter = 30
                    elif self.turning_counter > 8 and self.turning_counter < 15:
                        angle = self.angle_turning
                    elif self.turning_counter >= 15 and self.turning_counter <= 24:
                        angle = 0

                case 'no_turn_left':
                    # speed = 40
                    if self.turning_counter <= 10:
                        angle = 2
                    elif self.turning_counter == 27:
                        self.turning_counter = 30
                    elif self.turning_counter > 10:
                        angle = -90

                case 'no_turn_right':
                    speed = 30
                    if self.turning_counter <= 9:
                        angle = 1
                    elif self.turning_counter > 9 and self.turning_counter <= 19:
                        angle = self.angle_turning
                    elif self.turning_counter == 20:
                        self.turning_counter = 30

                case 'straight':
                    if self.turning_counter <= 17:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = 30

                case 'no_straight':
                    speed = 10
                    if self.turning_counter <= 9:
                        angle = 2
                    elif self.turning_counter == 25:
                        self.turning_counter = 30
                    else:
                        angle = self.angle_turning

            if speed == 0:
                speed = 70
            if angle == 0:
                angle = self.angle_turning

            # Set send back values
            self.sendBack_angle = angle
            self.sendBack_speed = speed

            self.turning_counter += 1

            self.next_step = True

        elif self.turning_counter >= 30:
            print("Reset"*1000)

            # Reset
            self.reset()

    def handle_areas(self, areas, segmented_image):
        print("Handle Areas:", areas)
        # print("Sum of top corner:", self.sum_top_corner)
        # print("Sum of top left corner:", self.sum_left_corner)

        if areas > 600.0 and self.majority_class == 'turn_right':
            # Set angle and error turning
            self.angle_turning = -15

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

        if areas > 550.0 and self.majority_class == 'turn_left':
            # Set angle and error turning
            self.angle_turning = 13

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

        if areas >= 500.0 and self.majority_class == 'no_turn_left':
            if self.sum_right_corner > self.sum_top_corner/2:  # Turn right3
                angle = -90
            else:
                angle = 1e-5  # Straight

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

            # Set global angle
            self.angle_turning = angle

        # ============ Test
        if self.majority_class == 'no_turn_right':
            self.mask_r = True

        self.sum_left_corner = np.sum(segmented_image[:12, :12, 0])
        self.sum_top_corner = np.sum(segmented_image[:12, 67:92, 0])
        # ============
        print("self.sum_left_corner", self.sum_left_corner)
        print("self.sum_top_corner", self.sum_top_corner)

        if areas >= 500.0 and self.majority_class == 'no_turn_right':
            if (self.sum_left_corner > 2_000 and self.sum_top_corner < 17_500) \
                    or (self.sum_left_corner < 9_000 and self.sum_top_corner < 9_000):
                angle = 30  # Left
            else:
                angle = 0  # Straight
                self.mask_l = True
                self.mask_r = True

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

            # Set global angle
            self.angle_turning = angle

        if areas > 600.0 and self.majority_class == 'straight':
            # Set global angle
            self.angle_turning = 0

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

            self.mask_l = True
            self.mask_r = True

        if areas > 500.0 and self.majority_class == 'no_straight':
            if self.sum_left_corner > self.sum_right_corner*4:
                # Turn left
                angle = 10
            else:
                # Turn right
                angle = -17

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

            # Set angle and error turning
            self.angle_turning = angle

    def calc_areas(self, segmented_image, yolo_output):
        # Testing
        print("Calculating areas!")
        print("Majority class:", self.majority_class)

        preds = yolo_output.boxes.data.numpy()  # List of (bouding_box, conf, class_id)

        try:
            for pred in preds:
                class_id = int(pred[-1])
                if self.class_names[class_id] == self.majority_class:
                    # Get boxes
                    boxes = pred[:4]

                    # Calculate areas from bouding boxe
                    areas = (boxes[2] - boxes[0]) * \
                        (boxes[3] - boxes[1])

                    self.handle_areas(areas, segmented_image)

                    # print("self.start_cal_area:", self.start_cal_area)

                    break

        except Exception as e:
            print(e)
            pass

    def calc_error(self, image):
        arr = []
        height = 12
        lineRow = image[height, :]
        for x, y in enumerate(lineRow):
            if y[0] == 100:
                arr.append(x)

        if len(arr) > 0:
            center_right_lane = int((min(arr) + max(arr)*2.5)/3.5)
            error = int(image.shape[1]/2) - center_right_lane
            return error
        else:
            return 0

    def PID(self, error, p, i, d):
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error*p
        delta_t = time.time() - self.pre_t
        self.pre_t = time.time()
        D = (error-self.error_arr[1])/delta_t*d
        I = np.sum(self.error_arr)*delta_t*i
        angle = P + I + D

        if abs(angle) > 25:
            angle = np.sign(angle)*25

        return int(angle)

    def calc_speed(self, angle):
        if abs(angle) < 10:
            speed = 65
        elif 10 <= abs(angle) <= 20:
            speed = 30
        else:
            speed = 30
        return speed
