from dataclasses import dataclass
from torchvision.transforms import v2
import torch
from PIL import Image
from io import BytesIO
import numpy as np
from scipy.ndimage import zoom as scizoom

# some are from
# https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py


@dataclass
class TransformsConfig:
    """Configuration for the data used for training the model.

    Parameters
    ----------
    transforms : list[tuple[str, dict]]
        The transformations to apply. For example:
        ```
        "RandomResizedCrop":{}
        ```
        for the default one, or
        ```
        "RandomResizedCrop":{"ratio":(0.3,2)}
        ```
        for custom settings.
    """

    name: str = None
    transforms: list[dict] = None

    def __post_init__(self):
        """Initialize the transformation configuration."""
        extra = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if self.transforms is None:
            self.transforms = [{}]
            self._transform = v2.Compose(extra)
        else:
            self._transform = v2.Compose(
                [TransformConfig(**t) for t in self.transforms] + extra
            )

    def __call__(self, x):
        return self._transform(x)


@dataclass
class TransformConfig:
    """Configuration for the data used for training the model.

    Parameters
    ----------
    transforms : list[tuple[str, dict]]
        The transformations to apply. For example:
        ```
        "RandomResizedCrop":{}
        ```
        for the default one, or
        ```
        "RandomResizedCrop":{"ratio":(0.3,2)}
        ```
        for custom settings.
    """

    name: str
    args: list
    kwargs: dict
    p: float = 1.0

    def __init__(self, name, args=None, kwargs=None, p=1):
        self.name = name
        self.args = args or []
        self.kwargs = kwargs or {}
        self.p = p
        if self.name is not None:
            if self.name in v2.__dict__:
                t = v2.__dict__[self.name](*self.args, **self.kwargs)
            else:
                t = globals()[self.name](*self.args, **self.kwargs)
            if self.p < 1:
                self._transform = v2.RandomApply(torch.nn.ModuleList([t]), p=self.p)
            elif self.p == 0:
                self._transform = v2.Identity()
            else:
                self._transform = t
        else:
            self._transform = v2.Identity()

    def __call__(self, x):
        return self._transform(x)


# def disk(radius, alias_blur=0.1, dtype=np.float32):
#     if radius <= 8:
#         L = np.arange(-8, 8 + 1)
#         ksize = (3, 3)
#     else:
#         L = np.arange(-radius, radius + 1)
#         ksize = (5, 5)
#     X, Y = np.meshgrid(L, L)
#     aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
#     aliased_disk /= np.sum(aliased_disk)

#     # supersample disk to antialias
#     return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py


def plasma_fractal(mapsize=256, wibbledecay=3):
    """Generate a heightmap using diamond-square algorithm.

    Returns a square 2D array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.

    Parameters
    ----------
    mapsize : int, optional
        Size of the map (must be a power of two). Default is 256.
    wibbledecay : int, optional
        The rate of decay for the randomness. Default is 3.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """Calculate middle value of squares as mean of points plus wibble."""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """Calculate middle value of diamonds as mean of points plus wibble."""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    """Apply clipped zoom to an image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.
    zoom_factor : float
        Factor by which to zoom the image.

    Returns
    -------
    numpy.ndarray
        The zoomed and clipped image.
    """
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=1
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top : trim_top + h, trim_top : trim_top + h]


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions ///////////////


class GaussianNoise(torch.nn.Module):
    """Apply Gaussian noise to an image.

    Parameters
    ----------
    severity : int, optional
        Severity level of the noise. Default is 1.
    """

    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity

    def forward(self, x):
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        c = [0.08, 0.12, 0.18, 0.26, 0.38][self.severity - 1]

        x = np.array(x) / 255.0
        x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        return x


class ShotNoise(torch.nn.Module):
    """Apply Shot noise to an image.

    Parameters
    ----------
    severity : int, optional
        Severity level of the noise. Default is 1.
    """

    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity

    def forward(self, x):
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        c = [60, 25, 12, 5, 3][self.severity - 1]

        x = np.array(x) / 255.0
        x = np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        return x


# class ImpulseNoise:
#     def __init__(self, severity=1):
#         self.severity = severity

#     def forward(self, x):
#         c = [0.03, 0.06, 0.09, 0.17, 0.27][self.severity - 1]

#         x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
#         return np.clip(x, 0, 1) * 255


class SpeckleNoise(torch.nn.Module):
    """Apply Speckle noise to an image.

    Parameters
    ----------
    severity : int, optional
        Severity level of the noise. Default is 1.
    """

    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity

    def forward(self, x):
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        c = [0.15, 0.2, 0.35, 0.45, 0.6][self.severity - 1]

        x = np.array(x) / 255.0
        x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        return x


# def fgsm(x, source_net, severity=1):
#     c = [8, 16, 32, 64, 128][severity - 1]

#     x = V(x, requires_grad=True)
#     logits = source_net(x)
#     source_net.zero_grad()
#     loss = F.cross_entropy(
#         logits, V(logits.data.max(1)[1].squeeze_()), size_average=False
#     )
#     loss.backward()

#     return standardize(
#         torch.clamp(
#      unstandardize(x.data) + c / 255.0 * unstandardize(torch.sign(x.grad.data)),
#             0,
#             1,
#         )
#     )


# def gaussian_blur(x, severity=1):
#     c = [1, 2, 3, 4, 6][severity - 1]

#     x = gaussian(np.array(x) / 255.0, sigma=c, multichannel=True)
#     return np.clip(x, 0, 1) * 255


# def glass_blur(x, severity=1):
#     # sigma, max_delta, iterations
#     c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

#     x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], multichannel=True) * 255)

#     # locally shuffle pixels
#     for i in range(c[2]):
#         for h in range(224 - c[1], c[1], -1):
#             for w in range(224 - c[1], c[1], -1):
#                 dx, dy = np.random.randint(-c[1], c[1], size=(2,))
#                 h_prime, w_prime = h + dy, w + dx
#                 # swap
#                 x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

#     return np.clip(gaussian(x / 255.0, sigma=c[0], multichannel=True), 0, 1) * 255


# def defocus_blur(x, severity=1):
#     c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

#     x = np.array(x) / 255.0
#     kernel = disk(radius=c[0], alias_blur=c[1])

#     channels = []
#     for d in range(3):
#         channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
#     channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

#     return np.clip(channels, 0, 1) * 255


# def motion_blur(x, severity=1):
#     c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

#     output = BytesIO()
#     x.save(output, format="PNG")
#     x = MotionImage(blob=output.getvalue())

#     x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

#     x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

#     if x.shape != (224, 224):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


class ZoomBlur(torch.nn.Module):
    """Apply zoom blur to an image.

    Parameters
    ----------
    severity : int, optional
        Severity level of the blur. Default is 1.
    """

    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity

    def forward(self, x):
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        c = [
            np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.02),
            np.arange(1, 1.26, 0.02),
            np.arange(1, 1.31, 0.03),
        ][self.severity - 1]

        x = (np.array(x) / 255.0).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        x = np.clip(x, 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        return x


class Fog(torch.nn.Module):
    """Apply fog effect to an image.

    Parameters
    ----------
    severity : int, optional
        Severity level of the fog. Default is 1.
    """

    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity

    def forward(self, x):
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][self.severity - 1]

        x = np.array(x) / 255.0
        max_val = x.max()
        x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
        x = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        return x

# def frost(x, severity=1):
#     c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
#     idx = np.random.randint(5)
#     filename = [
#         resource_filename(__name__, "frost/frost1.png"),
#         resource_filename(__name__, "frost/frost2.png"),
#         resource_filename(__name__, "frost/frost3.png"),
#         resource_filename(__name__, "frost/frost4.jpg"),
#         resource_filename(__name__, "frost/frost5.jpg"),
#         resource_filename(__name__, "frost/frost6.jpg"),
#     ][idx]
#     frost = cv2.imread(filename)
#     # randomly crop and convert to rgb
#     x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(
#         0, frost.shape[1] - 224
#     )
#     frost = frost[x_start : x_start + 224, y_start : y_start + 224][..., [2, 1, 0]]

#     return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


# def snow(x, severity=1):
#     c = [
#         (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
#         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
#         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
#         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
#         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
#     ][severity - 1]

#     x = np.array(x, dtype=np.float32) / 255.0
#     snow_layer = np.random.normal(
#         size=x.shape[:2], loc=c[0], scale=c[1]
#     )  # [:2] for monochrome

#     snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
#     snow_layer[snow_layer < c[3]] = 0

#     snow_layer = Image.fromarray(
#         (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L"
#     )
#     output = BytesIO()
#     snow_layer.save(output, format="PNG")
#     snow_layer = MotionImage(blob=output.getvalue())

#  snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

#     snow_layer = (
#         cv2.imdecode(
#             np.fromstring(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
#         )
#         / 255.0
#     )
#     snow_layer = snow_layer[..., np.newaxis]

#     x = c[6] * x + (1 - c[6]) * np.maximum(
#         x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5
#     )
#     return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


# def spatter(x, severity=1):
#     c = [
#         (0.65, 0.3, 4, 0.69, 0.6, 0),
#         (0.65, 0.3, 3, 0.68, 0.6, 0),
#         (0.65, 0.3, 2, 0.68, 0.5, 0),
#         (0.65, 0.3, 1, 0.65, 1.5, 1),
#         (0.67, 0.4, 1, 0.65, 1.5, 1),
#     ][severity - 1]
#     x = np.array(x, dtype=np.float32) / 255.0

#     liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

#     liquid_layer = gaussian(liquid_layer, sigma=c[2])
#     liquid_layer[liquid_layer < c[3]] = 0
#     if c[5] == 0:
#         liquid_layer = (liquid_layer * 255).astype(np.uint8)
#         dist = 255 - cv2.Canny(liquid_layer, 50, 150)
#         dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
#         _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
#         dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
#         dist = cv2.equalizeHist(dist)
#         ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
#         dist = cv2.filter2D(dist, cv2.CV_8U, ker)
#         dist = cv2.blur(dist, (3, 3)).astype(np.float32)

#         m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
#         m /= np.max(m, axis=(0, 1))
#         m *= c[4]

#         # water is pale turqouise
#         color = np.concatenate(
#             (
#                 175 / 255.0 * np.ones_like(m[..., :1]),
#                 238 / 255.0 * np.ones_like(m[..., :1]),
#                 238 / 255.0 * np.ones_like(m[..., :1]),
#             ),
#             axis=2,
#         )

#         color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
#         x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

#         return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
#     else:
#         m = np.where(liquid_layer > c[3], 1, 0)
#         m = gaussian(m.astype(np.float32), sigma=c[4])
#         m[m < 0.8] = 0

#         # mud brown
#         color = np.concatenate(
#             (
#                 63 / 255.0 * np.ones_like(x[..., :1]),
#                 42 / 255.0 * np.ones_like(x[..., :1]),
#                 20 / 255.0 * np.ones_like(x[..., :1]),
#             ),
#             axis=2,
#         )

#         color *= m[..., np.newaxis]
#         x *= 1 - m[..., np.newaxis]

#         return np.clip(x + color, 0, 1) * 255


class Contrast(torch.nn.Module):
    """Apply contrast adjustment to an image.

    Parameters
    ----------
    severity : int, optional
        Severity level of the contrast adjustment. Default is 1.
    """

    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity

    def forward(self, x):
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        c = [0.4, 0.3, 0.2, 0.1, 0.05][self.severity - 1]

        x = np.array(x) / 255.0
        means = np.mean(x, axis=(0, 1), keepdims=True)
        x = np.clip((x - means) * c + means, 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        return x


# def brightness(x, severity=1):
#     c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

#     x = np.array(x) / 255.0
#     x = sk.color.rgb2hsv(x)
#     x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
#     x = sk.color.hsv2rgb(x)

#     return np.clip(x, 0, 1) * 255


# def saturate(x, severity=1):
#     c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

#     x = np.array(x) / 255.0
#     x = sk.color.rgb2hsv(x)
#     x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
#     x = sk.color.hsv2rgb(x)

#     return np.clip(x, 0, 1) * 255


class JPEGCompression(torch.nn.Module):
    """Apply JPEG compression to an image.

    Parameters
    ----------
    severity : int, optional
        Severity level of the compression. Default is 1.
    """

    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity

    def forward(self, x):
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        c = [25, 18, 15, 10, 7][self.severity - 1]

        output = BytesIO()
        x.save(output, "JPEG", quality=c)
        x = Image.open(output)

        return x


class Pixelate(torch.nn.Module):
    """Apply pixelation effect to an image.

    Parameters
    ----------
    size: int
        The size of the final pixelated image. For CIFAR it's supposed to be 32x32 and for ImageNet it's 224x224.
    severity : int, optional
        Severity level of the pixelation. Default is 1.
    """

    def __init__(self, size, severity=1):
        super().__init__()
        self.severity = severity
        self.size = size

    def forward(self, x):
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        """
        x: PIL.Image
            Needs to be a PIL image in the range (0-255)
        """
        c = [0.6, 0.5, 0.4, 0.3, 0.25][self.severity - 1]

        x = x.resize((int(self.size * c), int(self.size * c)), Image.BOX)
        x = x.resize((self.size, self.size), Image.BOX)
        return x
