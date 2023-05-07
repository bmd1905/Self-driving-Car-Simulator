# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import tools._init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(0, 0, 0), (100, 10, 200), (20, 200, 100)]


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (
        k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    return model


class LandDetect:
    def __init__(self, pidNetType: str, path_to_pretrained_model: str):
        model = models.pidnet.get_pred_model(pidNetType, len(color_map))
        # self.model = load_pretrained(model, path_to_pretrained_model).cuda()
        self.model = load_pretrained(model, path_to_pretrained_model)
        self.model.eval()

    def reference(self, img, output_size: tuple, mask_lr=False, mask_l=False, mask_r=False, mask_t=False):
        with torch.no_grad():
            # Mask background
            img = img[125:, :, :]

            img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_AREA)

            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0)  # .cuda()
            pred = self.model(img)
            pred = F.interpolate(pred, size=img.size()
                                 [-2:], mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:, :, j][pred == i] = color_map[i][j]

            # Mask image after segmentation step
            if mask_lr:
                sv_img[:, 90:, :] = [0, 0, 0]
                sv_img[:, :60, :] = [0, 0, 0]
            if mask_r:
                sv_img[:, 90:, :] = [0, 0, 0]
            if mask_l:
                sv_img[:, :60, :] = [0, 0, 0]

        # return cv2.resize(sv_img, output_size, interpolation=cv2.INTER_NEAREST)
        return sv_img
