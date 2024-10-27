# # from https://github.com/facebookresearch/barlowtwins/blob/main/main.py

# import random
# import numpy as np
# from PIL import Image, ImageOps, ImageFilter

# import torchvision.transforms.v2 as transforms


# CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
# CIFAR_STD = np.array([0.2023, 0.1994, 0.2010])

# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
# IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# class GaussianBlur(object):
#     def __init__(self, p):
#         self.p = p

#     def __call__(self, img):
#         if random.random() < self.p:
#             sigma = random.random() * 1.9 + 0.1
#             return img.filter(ImageFilter.GaussianBlur(sigma))
#         else:
#             return img


# class Solarization(object):
#     def __init__(self, p):
#         self.p = p

#     def __call__(self, img):
#         if random.random() < self.p:
#             return ImageOps.solarize(img)
#         else:
#             return img


# class PositivePairSampler:
#     def __init__(self, dataset="CIFAR10"):

#         if "CIFAR" in dataset:
#             MEAN, STD = CIFAR_MEAN, CIFAR_STD
#             size = 32
#         else:  # IMAGENET
#             MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
#             size = 224

#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     size,
#                     scale=(0.2, 1.0),
#                     interpolation=transforms.InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomApply(
#                     [
#                         transforms.ColorJitter(
#                             brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
#                         )
#                     ],
#                     p=0.0,
#                 ),
#                 transforms.RandomGrayscale(p=0.2),
#                 (
#                     GaussianBlur(1.0)
#                     if "CIFAR" not in dataset
#                     else transforms.Lambda(lambda x: x)  # if CIFAR do nothing
#                 ),
#                 Solarization(p=0.0),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=MEAN,
#                     std=STD,
#                 ),
#             ]
#         )
#         self.transform_prime = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     size, scale=(0.2, 1.0), interpolation=Image.BICUBIC
#                 ),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomApply(
#                     [
#                         transforms.ColorJitter(
#                             brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
#                         )
#                     ],
#                     p=0.0,
#                 ),
#                 transforms.RandomGrayscale(p=0.2),
#                 (
#                     GaussianBlur(0.1)
#                     if "CIFAR" not in dataset
#                     else transforms.Lambda(lambda x: x)  # if CIFAR do nothing
#                 ),
#                 Solarization(p=0.2),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=MEAN,
#                     std=STD,
#                 ),
#             ]
#         )

#     def __call__(self, x):
#         y1 = self.transform(x)
#         y2 = self.transform_prime(x)
#         return y1, y2


# class Sampler:
#     def __init__(self, transforms: list):
#         self.transforms = transforms

#     def __call__(self, x):
#         views = []
#         for t in self.transforms:
#             views.append(t(x))
#         if len(self.transforms) == 1:
#             return views[0]
#         return views


# class ValSampler:
#     def __init__(self, dataset="CIFAR10"):

#         if "CIFAR" in dataset:
#             MEAN, STD = CIFAR_MEAN, CIFAR_STD
#         else:  # IMAGENET
#             MEAN, STD = IMAGENET_MEAN, IMAGENET_STD

#         self.transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=MEAN, std=STD),
#             ]
#         )

#     def __call__(self, x):
#         return self.transform(x)
