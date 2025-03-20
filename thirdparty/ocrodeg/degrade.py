import random as pyr
import warnings
from random import randint

import numpy as np
# import pylab
import scipy.ndimage as ndi
from numpy import *
import cv2 as cv


def autoinvert(image):
    assert amin(image) >= 0
    assert amax(image) <= 1, f"{amax(image)}"
    if sum(image>0.9) > sum(image<0.1):
        return 1-image
    else:
        return image

#
# random geometric transformations
#

def random_transform(translation=(-0.05, 0.05), rotation=(-2, 2), scale=(-0.1, 0.1), aniso=(-0.1, 0.1)):
    dx = pyr.uniform(*translation)
    dy = pyr.uniform(*translation)
    angle = pyr.uniform(*rotation)
    angle = angle * pi / 180.0
    scale = 10**pyr.uniform(*scale)
    aniso = 10**pyr.uniform(*aniso)
    return dict(angle=angle, scale=scale, aniso=aniso, translation=(dx, dy))

def transform_image(image, angle=0.0, scale=1.0, aniso=1.0, translation=(0, 0), order=1):
    dx, dy = translation
    scale = 1.0/scale
    c = cos(angle)
    s = sin(angle)
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], 'f')
    m = np.array([[c, -s], [s, c]], 'f')
    m = np.dot(sm, m)
    w, h = image.shape
    c = np.array([w, h]) / 2.0
    d = c - np.dot(m, c) + np.array([dx * w, dy * h])
    return ndi.affine_transform(image, m, offset=d, order=order, mode="nearest", output=dtype("f"))

# mainly based on https://github.com/Calamari-OCR/calamari/blob/master/calamari_ocr/thirdparty/ocrodeg/degrade.py
def random_pad(image, border=(0, 100, 0, 100)):
    if border[1] != 0:
        t, b = np.random.randint(border[0],border[1]), np.random.randint(border[0],border[1])
    else:
        t, b = 0, 0
    if border[3] != 0:
        l, r = np.random.randint(border[2],border[3]), np.random.randint(border[2],border[3])
    else:
        l, r = 0, 0
    # copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) -> dst
    return cv.copyMakeBorder(
        image,
        t,
        b,
        l,
        r,
        cv.BORDER_CONSTANT,
        value=[1] * (1 if image.ndim == 2 else image.shape[-1]),
    )


#
# random distortions
#

def bounded_gaussian_noise(shape, sigma, maxdelta):
    n, m = shape
    deltas = np.random.rand(2, n, m)
    deltas = ndi.gaussian_filter(deltas, (0, sigma, sigma))
    deltas -= np.amin(deltas)
    deltas /= np.amax(deltas)
    deltas = (2*deltas-1) * maxdelta
    return deltas

def distort_with_noise(image, deltas, order=1):
    assert deltas.shape[0] == 2
    assert image.shape == deltas.shape[1:], (image.shape, deltas.shape)
    n, m = image.shape
    xy = np.transpose(np.array(np.meshgrid(
        range(n), range(m))), axes=[0, 2, 1])
    deltas += xy
    return ndi.map_coordinates(image, deltas, order=order, mode="reflect")


def noise_distort1d(shape, sigma=100.0, magnitude=100.0):
    h, w = shape
    noise = ndi.gaussian_filter(np.random.randn(w), sigma)
    noise *= magnitude / amax(abs(noise))
    dys = array([noise]*h)
    deltas = array([dys, zeros((h, w))])
    return deltas

#
# mass preserving blur
#

def percent_black(image):
    n = prod(image.shape)
    k = sum(image < 0.5)
    return k * 100.0 / n

def binary_blur(image, sigma, noise=0.0):
    p = percent_black(image)
    blurred = ndi.gaussian_filter(image, sigma)
    if noise > 0:
        blurred += np.random.randn(*blurred.shape) * noise
    t = percentile(blurred, p)
    return array(blurred > t, 'f')

#
# multiscale noise
#

def make_noise_at_scale(shape, scale):
    h, w = shape
    h0, w0 = int(h/scale+1), int(w/scale+1)
    data = np.random.rand(h0, w0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ndi.zoom(data, scale)
    return result[:h, :w]

def make_multiscale_noise(shape, scales, weights=None, limits=(0.0, 1.0)):
    if weights is None: weights = [1.0] * len(scales)
    result = make_noise_at_scale(shape, scales[0]) * weights[0]
    for s, w in zip(scales, weights):
        result += make_noise_at_scale(shape, s) * w
    lo, hi = limits
    result -= amin(result)
    result /= amax(result)
    result *= (hi-lo)
    result += lo
    return result

def make_multiscale_noise_uniform(shape, srange=(1.0, 100.0), nscales=4, limits=(0.0, 1.0)):
    lo, hi = log10(srange[0]), log10(srange[1])
    scales = np.random.uniform(size=nscales)
    scales = add.accumulate(scales)
    scales -= amin(scales)
    scales /= amax(scales)
    scales *= hi-lo
    scales += lo
    scales = 10**scales
    weights = 2.0 * np.random.uniform(size=nscales)
    return make_multiscale_noise(shape, scales, weights=weights, limits=limits)

#
# random blobs
#

def random_blobs(shape, blobdensity, size, roughness=2.0):
    from random import randint
    from builtins import range  # python2 compatible
    h, w = shape
    numblobs = int(blobdensity * w * h)
    mask = np.zeros((h, w), 'i')
    for i in range(numblobs):
        mask[randint(0, h-1), randint(0, w-1)] = 1
    dt = ndi.distance_transform_edt(1-mask)
    mask =  np.array(dt < size, 'f')
    mask = ndi.gaussian_filter(mask, size/(2*roughness))
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    noise = np.random.rand(h, w)
    noise = ndi.gaussian_filter(noise, size/(2*roughness))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, 'f')

def random_blotches(image, fgblobs, bgblobs, fgscale=10, bgscale=10):
    fg = random_blobs(image.shape, fgblobs, fgscale)
    bg = random_blobs(image.shape, bgblobs, bgscale)
    return minimum(maximum(image, fg), 1-bg)

#
# random fibers
#

def make_fiber(l, a, stepsize=0.5):
    angles = np.random.standard_cauchy(l) * a
    angles[0] += 2 * pi * np.random.rand()
    angles = add.accumulate(angles)
    coss = add.accumulate(cos(angles)*stepsize)
    sins = add.accumulate(sin(angles)*stepsize)
    return array([coss, sins]).transpose(1, 0)

def make_fibrous_image(shape, nfibers=300, l=300, a=0.2, stepsize=0.5, limits=(0.1, 1.0), blur=1.0):
    h, w = shape
    lo, hi = limits
    result = zeros(shape)
    for i in range(nfibers):
        v = np.random.rand() * (hi-lo) + lo
        fiber = make_fiber(l, a, stepsize=stepsize)
        y, x = randint(0, h-1), randint(0, w-1)
        fiber[:, 0] += y
        fiber[:, 0] = clip(fiber[:, 0], 0, h-.1)
        fiber[:, 1] += x
        fiber[:, 1] = clip(fiber[:, 1], 0, w-.1)
        for y, x in fiber:
            result[int(y), int(x)] = v
    result = ndi.gaussian_filter(result, blur)
    result -= amin(result)
    result /= amax(result)
    result *= (hi-lo)
    result += lo
    return result


#
# print-like degradation with multiscale noise
#

def printlike_multiscale(image, blur=1.0, blotches=5e-5):
    # selector = autoinvert(image)
    selector = image
    selector = random_blotches(selector, 3*blotches, blotches)
    paper = make_multiscale_noise_uniform(image.shape, limits=(0.6, 1.0))
    ink = make_multiscale_noise_uniform(image.shape, limits=(0.0, 0.2))
    blurred = ndi.gaussian_filter(selector, blur)
    printed = blurred * ink + (1-blurred) * paper
    return printed


def printlike_fibrous(image, blur=1.0, blotches=5e-5):
    # selector = autoinvert(image)
    selector = image
    selector = random_blotches(selector, 3*blotches, blotches)
    paper = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], weights=[1.0, 0.3, 0.5, 0.3], limits=(0.8, 1.0))
    paper -= make_fibrous_image(image.shape, 300, 500, 0.01, limits=(0.0, 0.25), blur=0.5)
    ink = make_multiscale_noise(image.shape, [1.0, 5.0, 10.0, 50.0], limits=(0.0, 0.2))
    blurred = ndi.gaussian_filter(selector, blur)
    printed = blurred * ink + (1-blurred) * paper
    return printed

class FastPrintlike():
    def __init__(self, blur=1.0, blotches=5e-5):
        self.blur = blur
        self.blotches = blotches
        max_shape = (196,6000)
        # max_shape = (128,1500)
        paper_multiscale = np.stack([make_multiscale_noise_uniform(max_shape, limits=(0.6, 1.0)) for i in range(10)])
        paper_fibrous = np.stack([make_multiscale_noise(max_shape, [1.0, 5.0, 10.0, 50.0], weights=[1.0, 0.3, 0.5, 0.3],
                                      limits=(0.8, 1.0))-make_fibrous_image(max_shape, 300, 500, 0.01, limits=(0.0, 0.25), blur=0.5) for i in range(10)])
        self.papers = [paper_multiscale, paper_fibrous]
        self.ink = np.stack([make_multiscale_noise(max_shape, [1.0, 5.0, 10.0, 50.0], limits=(0.0, 0.2)) for i in range(10)])

    def __call__(self, x):
        blurred = ndi.gaussian_filter(x, self.blur)
        xs = x.shape
        ink_x, ink_y = np.random.randint(0,self.ink.shape[1]-xs[0]), np.random.randint(0,self.ink.shape[2]-xs[1])
        ink = self.ink[np.random.randint(0,10),ink_x:ink_x+xs[0],ink_y:ink_y+xs[1]]
        assert ink.shape==xs
        paper = self.papers[np.random.randint(0,len(self.papers))]
        paper_x, paper_y = (np.random.randint(0,paper.shape[1]-xs[0]),
                     np.random.randint(0,paper.shape[2]-xs[1]))
        paper = paper[np.random.randint(0,10),paper_x:paper_x+xs[0],paper_y:paper_y+xs[1]]
        assert paper.shape == xs
        printed = blurred * ink + (1 - blurred) * paper
        return printed