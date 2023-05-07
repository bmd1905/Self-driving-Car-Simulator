import numpy as np
import time

from utils.utils import find_majority


class Controller():
    def __init__(self):
        # Initialize variables for PID control and traffic signs detection
        self.error_arr = np.zeros(5)       # Array to store the error values for PID control
        self.error_sp = np.zeros(5)        # Array to store the speed error values for PID control

        self.pre_t = time.time()           # Store the current time for steering control
        self.pre_t_spd = time.time()       # Store the current time for speed control

        self.sendBack_angle = 0            # Initialize the steering angle to 0
        self.sendBack_speed = 0            # Initialize the speed to 0

        # List of all the possible traffic light labels
        self.traffic_lights = ['turn_right', 'turn_left', 'straight', 'no_turn_left', 'no_turn_right', 'no_straight']
        
        # List of all the possible object detection labels
        self.class_names = ['no', 'turn_right', 'straight', 'no_turn_left', 'no_turn_right', 'no_straight', 'car', 'unknown', 'turn_left']
        

        self.stored_class_names = []       # List to store the detected labels for finding majority class

        self.majority_class = ""           # Initialize the majority class to empty
        self.start_cal_area = False        # Flag to start calculating the area for turning
        self.turning_counter = 0           # Counter to track the number of turning frames
        self.angle_turning = 0             # Angle to turn the car

        self.sum_left_corner = 0           # Sum of the pixel values in the left corner of the image
        self.sum_right_corner = 0          # Sum of the pixel values in the right corner of the image
        self.sum_top_corner = 0            # Sum of the pixel values in the top corner of the image

        self.mask_l = False                # Flag to indicate mask left image
        self.mask_r = False                # Flag to indicate mask right image
        self.mask_lr = False               # Flag to indicate mask leftn and right image
        self.mask_t = False                # Flag to indicate mask top image

        self.next_step = False             # Flag to indicate the next step of the car when turning
        self.is_turning = False            # Flag to indicate if the car is currently turning

        self.reset_counter = 0             # Counter to track the number of frames since the last reset

        self.is_turn_left = False          # Flag to indicate if the car is turning left
        self.is_turn_right = False         # Flag to indicate if the car is turning right
        self.is_straight = False           # Flag to indicate if the car is going straight

        self.is_no_turn_right_case_1 = False    # Flag to indicate if there is a "no turn right" sign in case 1
        self.is_no_turn_right_case_2 = False    # Flag for case 2
        self.is_no_turn_right_case_3 = False    # Flag for case 3
        self.is_no_turn_right_case_4 = False    # Flag for case 4

        self.is_turn_left_case_1 = False        # Flag to indicate if there is a "turn left" sign in case 1
        self.is_turn_left_case_2 = False        # Flag for case 2

    def reset(self):
        # Reset all values to default values
        self.turning_counter = 0
        self.majority_class = ""
        self.start_cal_area = False
        self.stored_class_names = []

        self.mask_lr = False
        self.mask_l = False
        self.mask_r = False
        self.mask_t = False

        self.majority_class = ""
        self.start_cal_area = False
        self.turning_counter = 0
        self.angle_turning = 0

        self.next_step = False
        self.is_turning = False

        self.reset_counter = 0

        self.is_turn_left = False
        self.is_turn_right = False
        self.is_straight = False

        self.is_no_turn_right_case_1 = False
        self.is_no_turn_right_case_2 = False
        self.is_no_turn_right_case_3 = False
        self.is_no_turn_right_case_4 = False

        self.is_turn_left_case_1 = False
        self.is_turn_left_case_2 = False

    def control(self, segmented_image, yolo_output):
        if self.majority_class == "turn_left":
            # Increase counter for controlling
            self.reset_counter += 1
            if self.reset_counter >= 100:
                self.reset()
                print("Reset"*200)

        # Calculate area of left, right, and top corner of the segmented image
        self.sum_right_corner = np.sum(
            segmented_image[:25, -25:, 0])
        self.sum_left_corner = np.sum(segmented_image[:12, :12, 0])
        self.sum_top_corner = np.sum(segmented_image[:12, 67:92, 0])

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

                if self.class_names[class_id] == 'turn_right':
                    self.stored_class_names.extend(['turn_right']*2)

                if self.class_names[class_id] == 'turn_left':
                    self.stored_class_names.extend(['turn_left'])

                if self.class_names[class_id] == 'no_turn_right':
                    self.stored_class_names.extend(['no_turn_right'])

        # Starting to find majority class
        elif len(self.stored_class_names) >= 9:  # 9 is a hyperparameter
            # Get the majority class
            self.majority_class = find_majority(
                self.stored_class_names)[0]  # Returned in set type

            # Start calculate areas
            self.start_cal_area = True

        return self.sendBack_angle, self.sendBack_speed, self.next_step, self.mask_l, self.mask_r

    def handle_turning(self):
        print("Handle Turning")
        # Default config
        speed = 0
        angle = 0

        # Check turning counter
        MAX_COUNTER = 40
        if self.turning_counter < MAX_COUNTER:
            print('Turning Counter:', self.turning_counter)

            match self.majority_class:
                case 'turn_left':
                    if self.is_turn_left_case_1:
                        speed = -1
                        if self.turning_counter <= 18: # Hard
                            angle = 1
                        elif self.turning_counter > 18 and self.turning_counter <= 31:
                            angle = self.angle_turning
                        else:
                            self.turning_counter = MAX_COUNTER
                    elif self.is_turn_left_case_2:
                        speed = -2
                        if self.turning_counter <= 17:
                            angle = -1
                        elif self.turning_counter > 17 and self.turning_counter <= 35:
                            angle = self.angle_turning
                        else:
                            self.turning_counter = MAX_COUNTER

                case 'turn_right':
                    speed = -5
                    if self.turning_counter <= 8:
                        angle = 2
                    elif self.turning_counter > 8 and self.turning_counter < 26:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER

                case 'no_turn_left':
                    speed = -3
                    if self.turning_counter <= 9:
                        angle = 2
                    elif self.turning_counter > 9 and self.turning_counter <= 28:
                        angle = -33
                    else:
                        self.turning_counter = MAX_COUNTER
                        
                case 'no_turn_right':
                    if self.is_no_turn_right_case_1:    # Left hard
                        speed = -2
                        if self.turning_counter <= 17:
                            angle = 5
                        elif self.turning_counter > 17 and self.turning_counter <= 27:
                            angle = self.angle_turning
                        else:
                            self.turning_counter = MAX_COUNTER
                    elif self.is_no_turn_right_case_2:  # Left
                        speed = -2
                        if self.turning_counter <= 13:
                            angle = -1
                        elif self.turning_counter > 13 and self.turning_counter <= 30:
                            angle = self.angle_turning
                        else:
                            self.turning_counter = MAX_COUNTER
                    elif self.is_no_turn_right_case_3: # Straight (left of map)
                        speed = -2
                        if self.turning_counter <= 8:
                            angle = 0
                        elif self.turning_counter > 8 and self.turning_counter <= 15:
                            angle = self.angle_turning
                        else:
                            self.turning_counter = MAX_COUNTER
                    else:   # Straight: self.is_no_turn_right_case_4
                        speed = -1
                        if self.turning_counter <= 13:
                            angle = 0
                        # elif self.turning_counter > 8 and self.turning_counter <= 15:
                        #     angle = self.angle_turning
                        else:
                            self.turning_counter = MAX_COUNTER

                case 'straight':
                    speed = -1
                    if self.turning_counter <= 25:
                        angle = self.angle_turning
                    else:
                        self.turning_counter = MAX_COUNTER

                case 'no_straight':
                    if self.is_turn_left: # Left
                        speed = -2
                        if self.turning_counter <= 17:
                            angle = 0
                        elif self.turning_counter > 17 and self.turning_counter <= 35:
                            angle = self.angle_turning
                        else:
                            self.turning_counter = MAX_COUNTER
                    else: # Right
                        speed = -4
                        if self.turning_counter <= 9:
                            angle = 0
                        elif self.turning_counter > 9 and self.turning_counter <= 25:
                            angle = self.angle_turning
                        else:
                            self.turning_counter = MAX_COUNTER
                            
            # Set default speed
            if speed == 0:
                speed = 70

            # Set send back values
            self.sendBack_angle = angle
            self.sendBack_speed = speed

            # Increase the counter by 1
            self.turning_counter += 1

            # Send back to not use PID calculate again when turning
            self.next_step = True

        elif self.turning_counter >= MAX_COUNTER:
            # Reset after turning
            self.reset()

    def handle_areas(self, areas, segmented_image):
        print("Handle Areas:", areas)

        # ============ Test
        # self.sum_left_corner = np.sum(segmented_image[:12, :12, 0])
        # self.sum_top_corner = np.sum(segmented_image[:12, 67:92, 0])
        # ============

        print("self.sum_left_corner", self.sum_left_corner)
        print("self.sum_top_corner", self.sum_top_corner)
        print("self.sum_right_corner", self.sum_right_corner)

        if areas > 600.0 and self.majority_class == 'turn_right':
            # Set angle and error turning
            self.angle_turning = -45

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

        if areas > 550.0 and self.majority_class == 'turn_left':
            if self.sum_top_corner  > 17_000:
                self.is_turn_left_case_1 = True
            else:
                self.is_turn_left_case_2 = True

            self.angle_turning = 26

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

        if areas >= 500.0 and self.majority_class == 'no_turn_left':
            if self.sum_right_corner > self.sum_top_corner/2:  # Turn right3
                angle = -30
            else:
                angle = 1e-5  # Straight

            # Start turning and stop cal areas
            self.is_turning = True
            self.start_cal_area = False

            # Set global angle
            self.angle_turning = angle

        if areas >= 500.0 and self.majority_class == 'no_turn_right':
            if (self.sum_left_corner > 2_000 and self.sum_top_corner < 17_500):# \
                #    or (self.sum_left_corner < 9_000 and self.sum_top_corner < 9_000):
                
                if self.sum_top_corner < 3_000: # Hard
                    self.is_no_turn_right_case_1 = True
                else:
                    self.is_no_turn_right_case_2 = True

                angle = 30  # Left
            
            elif self.sum_top_corner > 18_000:
                self.is_no_turn_right_case_3 = True

                angle = 0  # Straight
                self.mask_l = True
                self.mask_r = True

            else:
                self.is_no_turn_right_case_4 = True

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
                angle = 22
                self.is_turn_left = True
            else:
                # Turn right
                angle = -24
                self.is_turn_right = True

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
        """
        Calculates the error between the center of the right lane and the center of the image.

        Args:
        image: A NumPy array representing the image.

        Returns:
        The error between the center of the right lane and the center of the image.
        """

        arr = []
        height = 12
        lineRow = image[height, :]
        for x, y in enumerate(lineRow):
            if y[0] == 100:
                arr.append(x)

        if len(arr) > 0:
            center_right_lane = int((min(arr) + max(arr)*2.5)/3.5)
            error = int(image.shape[1]/2) - center_right_lane
            return error*1.3
        else:
            return 0

    def PID(self, error, p, i, d):
        """
        Calculates the PID output for the specified error.

        Args:
        error: The error value.
        p: The proportional gain.
        i: The integral gain.
        d: The derivative gain.

        Returns:
        The PID output.
        """
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
        """
        Calculates the speed of the car based on the steering angle.

        Args:
        angle: The steering angle.

        Returns:
        The speed of the car.
        """
        if abs(angle) < 10:
            speed = 70
        elif 10 <= abs(angle) <= 20:
            speed = 1
        else:
            speed = 1
        return speed
