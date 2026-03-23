import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class Transforms(object):
    def __init__(self, size=256):
        self.size = size
    
    def __call__(self, _data):
        img1, img2,  cd_label = _data['img1'], _data['img2'], _data['cd_label']

        if random.random() < 0.5:
            img1_ = img1
            img1 = img2
            img2 = img1_

        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            cd_label = TF.hflip(cd_label)

        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            cd_label = TF.vflip(cd_label)

        if random.random() < 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            cd_label = TF.rotate(cd_label, angle)

        if random.random() < 0.3:
            sigma = random.uniform(0.1, 2.0)
            kernel_size = random.choice([3, 5])
            img1 = TF.gaussian_blur(img1, kernel_size=kernel_size, sigma=sigma)
            img2 = TF.gaussian_blur(img2, kernel_size=kernel_size, sigma=sigma)

        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=(self.size, self.size)).get_params(img=img1, scale=[0.333, 1.0],
                                                                                  ratio=[0.75, 1.333])
            img1 = TF.resized_crop(img1, i, j, h, w, size=(self.size, self.size), interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resized_crop(img2, i, j, h, w, size=(self.size, self.size), interpolation=InterpolationMode.BILINEAR)
            cd_label = TF.resized_crop(cd_label, i, j, h, w, size=(self.size, self.size), interpolation=InterpolationMode.NEAREST)
        else:
            # 强制 resize 到统一尺寸
            img1 = TF.resize(img1, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resize(img2, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
            cd_label = TF.resize(cd_label, (self.size, self.size), interpolation=InterpolationMode.NEAREST)

        return {'img1': img1, 'img2': img2, 'cd_label': cd_label}


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string





