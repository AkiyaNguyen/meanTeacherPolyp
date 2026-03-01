## use to make sure that all elements in dataset are transformed logically and consistently


import torch
import torchvision.transforms.functional as F
import scipy.ndimage
import random
from PIL import Image
import numpy as np
import cv2
import numbers
from torchvision import transforms

class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, data):
        return {key: F.to_tensor(data[key]) for key in data.keys()}

class Resize(object):

    def __init__(self, size):
        self.size = size
    def __call__(self, data):
        return {key: F.resize(data[key], self.size) for key in data.keys()}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if random.random() < self.p:
            return {key: F.hflip(data[key]) for key in data.keys()}
        return data


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if random.random() < self.p:
            return {key: F.vflip(data[key]) for key in data.keys()}
        return data


class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
    @staticmethod
    def get_params(degrees):

        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, data):
        if random.random() < 0.5:
            angle = self.get_params(self.degrees)
            return {key: F.rotate(data[key], angle, self.resample, self.expand, self.center) for key in data.keys()}
        return data


class RandomZoom(object):
    def __init__(self, zoom=(0.8, 1.2)):
        self.min, self.max = zoom[0], zoom[1]
    def __call__(self, data):
        if random.random() < 0.5:
            for item in data.keys():
                img_format = 'L' if item == 'label' else 'RGB'
                img = np.array(data[item])
                zoom = random.uniform(self.min, self.max)
                img = clipped_zoom(img, zoom)
                img = Image.fromarray(img.astype('uint8'), img_format)
                data[item] = img

        return data


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]


    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)


    if zoom_factor < 1:


        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)

    elif zoom_factor > 1:

        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        zoom_in = scipy.ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)


        if zoom_in.shape[0] >= h:
            zoom_top = (zoom_in.shape[0] - h) // 2
            sh = h
            out_top = 0
            oh = h
        else:
            zoom_top = 0
            sh = zoom_in.shape[0]
            out_top = (h - zoom_in.shape[0]) // 2
            oh = zoom_in.shape[0]
        if zoom_in.shape[1] >= w:
            zoom_left = (zoom_in.shape[1] - w) // 2
            sw = w
            out_left = 0
            ow = w
        else:
            zoom_left = 0
            sw = zoom_in.shape[1]
            out_left = (w - zoom_in.shape[1]) // 2
            ow = zoom_in.shape[1]

        out = np.zeros_like(img)
        out[out_top:out_top + oh, out_left:out_left + ow] = zoom_in[zoom_top:zoom_top + sh, zoom_left:zoom_left + sw]


    else:
        out = img
    return out

class Normalization(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        normalize = transforms.Normalize(self.mean, self.std)
        return {key: normalize(sample[key]) if key != 'label' else sample[key] for key in sample.keys()}
