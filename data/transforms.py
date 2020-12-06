import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
# ===============================sequence tranforms============================

class Compose_seq(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sequence):
        for t in self.transforms:
            sequence = t(sequence)
        return sequence

class RandomHorizontallyFlip_seq(object):
    def __call__(self, sequence):
        if random.random() < 0.5:
            return [item.transpose(Image.FLIP_LEFT_RIGHT) for item in sequence]
        return sequence

class RandomCrop_seq(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = (size[0], size[1])
        self.padding = padding

    def __call__(self, sequence):
        if self.padding > 0:
            sequence = [ImageOps.expand(item, border=self.padding, fill=0) for item in sequence]
        w, h = sequence[0].size
        tw, th = self.size
        if w == tw and h == th:
            return sequence
        if w < tw or h < th:
            return [item.resize((tw, th), Image.BILINEAR) for item in sequence]
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [item.crop((x1, y1, x1 + tw, y1 + th)) for item in sequence]

class Scale_seq(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = (size[0], size[1])

    def __call__(self, sequence):
        return [item.resize((self.size[0], self.size[1]), Image.BILINEAR) for item in sequence]

class CenterCrop_seq(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sequence):
        w, h = sequence[0].size
        tw, th = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [item.crop((x1, y1, x1 + tw, y1 + th)) for item in sequence]

class Crop_seq(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, sequence, x1, y1):
        tw, th = self.size
        return [item.crop((x1, y1, x1 + tw, y1 + th)) for item in sequence]

class HorizontallyFlip_seq(object):
    def __call__(self, sequence, flip):
        if flip:
            return [item.transpose(Image.FLIP_LEFT_RIGHT) for item in sequence]
        return sequence

# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class LabelToTensor(object):
    def __call__(self, tensor):
        tensor = torch.from_numpy(np.array(tensor))
        if len(tensor.shape) != 3:
            tensor = tensor.unsqueeze(0)
        return torch.from_numpy(np.array(tensor))

class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        tensor = tensor*self.para
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor==1:
            return img
        tmp = np.array(img.resize((w/self.factor, h/self.factor), Image.BICUBIC))*self.factor*self.factor
        img = Image.fromarray(tmp)
        return img


