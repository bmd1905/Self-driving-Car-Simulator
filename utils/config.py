class ModelConfig:
    def __init__(self):
        self.weights_yolo = 'pretrain/yolov8-best.pt'
        self.weights_lane = 'pretrain/PIDNet.pt'

        self.img_size = 640
        self.conf_thres = 0.7
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False


class ControlConfig:
    def __init__(self):
        self.sendBack_angle = 0
        self.sendBack_Speed = 0
        self.current_speed = 0
        self.current_angle = 0

    def update(self, angle, speed):
        self.sendBack_angle = angle
        self.sendBack_Speed = speed
