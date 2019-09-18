import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask, flow = sample['image'], sample['label'], sample['flow']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            flow = ImageOps.expand(flow, border=self.padding, fill=0)


        assert img.size == mask.size
        assert img.size == flow.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask,
                    'flow': flow}
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            flow = flow.resize((tw, th), Image.BILINEAR)
            return {'image': img,
                    'label': mask,
                    'flow': flow}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        flow = flow.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'flow': flow}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        label = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        flow = flow.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            flow = flow.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask,
                'flow': flow}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        flow = np.array(sample['flow']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        flow /= 255.0
        flow -= self.mean
        flow /= self.std

        return {'image': img,
                'label': mask,
                'flow': flow}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        flow = np.array(sample['flow']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        flow = torch.from_numpy(flow).float()
        mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': mask,
                'flow': flow}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']

        assert img.size == mask.size
        assert img.size == flow.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        flow = flow.resize(self.size, Image.BILINEAR)

        return {'image': img,
                'label': mask,
                'flow': flow}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask,
                    'flow': flow}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        flow = flow.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                flow = flow.crop((x1, y1, x1 + w, y1 + h))

                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)
                flow = flow.resize((self.size, self.size), Image.BILINEAR)

                return {'image': img,
                        'label': mask,
                        'flow': flow}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        flow = flow.rotate(rotate_degree, Image.BILINEAR)


        return {'image': img,
                'label': mask,
                'flow': flow}

class RandomRotateOrthogonal(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']

        rotate_degree = random.randint(0, 3) * 90
        if rotate_degree > 0:
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            flow = flow.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'label': mask,
                'flow': flow}

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        w = int(random.uniform(0.8, 2.5) * img.size[0])
        h = int(random.uniform(0.8, 2.5) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        flow = flow.resize((w, h), Image.BILINEAR)

        sample = {'image': img, 'label': mask, 'flow': flow}

        return self.crop(self.scale(sample))

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size


        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        flow = flow.resize((w, h), Image.BILINEAR)

        return {'image': img, 'label': mask, 'flow': flow}