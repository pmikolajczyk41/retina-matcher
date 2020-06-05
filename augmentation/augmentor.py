from functools import reduce
from itertools import product
from random import random

import cv2 as cv
import numpy as np
from scipy import ndimage

from augmentation import uint
from data.loader import DataLoader


class Augmentor:
    def __init__(self,
                 rotation_rng=(-20, 20),
                 g_shift_x_rng=(-10, 10),
                 g_shift_y_rng=(-10, 10),
                 noise_prob=0.01,
                 l_shift_x_rng=(-10, 10),
                 l_shift_y_rng=(-10, 10),
                 perspective_corners_rng=((5, 5), (5, 5), (5, 5), (5, 5)),
                 light_rng=(.7, 1.1)):
        self._rotation_rng = rotation_rng
        self._g_shift_x_rng = g_shift_x_rng
        self._g_shift_y_rng = g_shift_y_rng
        self._noise_prob = noise_prob
        self._l_shift_x_rng = l_shift_x_rng
        self._l_shift_y_rng = l_shift_y_rng
        self._perspective_corners_rng = perspective_corners_rng
        self._light_rng = light_rng

    def augment(self, img, n):
        yield img
        for _ in range(n):
            yield self._generate_from(img)

    def _generate_from(self, img):
        pipeline = [self._rotate, self._shift, self._noise, self._move, self._change_perspective, self._light]
        return reduce(lambda op, fun: fun(op), pipeline, img)

    def _rotate(self, img):
        return ndimage.rotate(img, np.random.uniform(*self._rotation_rng), reshape=False)

    def _shift(self, img):
        dx = int(np.random.uniform(*self._g_shift_x_rng))
        dy = int(np.random.uniform(*self._g_shift_y_rng))
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        return cv.warpAffine(img, M, img.shape[::-1])

    def _noise(self, img):
        h, w = img.shape
        for x, y in product(range(h), range(w)):
            if random() <= self._noise_prob:
                img[x, y] = 255 * random()
        return img

    def _move(self, img):
        binarized = np.where(img > 0, 1, 0).astype(uint)
        nlabels, labels = cv.connectedComponents(binarized, 8, cv.CV_32S)
        shifts = [(int(np.random.uniform(*self._g_shift_x_rng)),
                   int(np.random.uniform(*self._g_shift_y_rng)))
                  for _ in range(nlabels)]
        h, w = img.shape
        clip_h = lambda v: 0 if v < 0 else min(v, h - 1)
        clip_w = lambda v: 0 if v < 0 else min(v, w - 1)

        moved = np.zeros_like(img, dtype=uint)
        for x, y in product(range(h), range(w)):
            label = labels[x, y]
            if label == 0: continue
            dx, dy = shifts[label]
            moved[clip_h(x + dx), clip_w(y + dy)] = img[x, y]
        return moved

    def _change_perspective(self, img):
        ds = [[int(np.random.uniform(r[0], r[0])), int(np.random.uniform(r[1], r[1]))]
              for r in self._perspective_corners_rng]
        h, w = img.shape
        corners = list(product([0, w], [0, h]))
        clip_h = lambda v, r: v + r if v + r < h else v - r
        clip_w = lambda v, r: v + r if v + r < w else v - r
        pts1 = np.float32([[clip_w(x, r[0]), clip_h(y, r[1])] for (x, y), r in zip(corners, ds)])
        pts2 = np.float32(corners)
        M = cv.getPerspectiveTransform(pts1, pts2)
        return cv.warpPerspective(img, M, (w, h))

    def _light(self, img):
        return (np.random.uniform(*self._light_rng) * img).astype(uint)


if __name__ == '__main__':
    for img in DataLoader().get_data():
        (h, w), ratio = img.shape, 4
        for i, a in enumerate(Augmentor().augment(img, 3)):
            cv.imshow(f'v{i}', cv.resize(a, (ratio * w, ratio * h), interpolation=cv.INTER_NEAREST))
        cv.waitKey()
        cv.destroyAllWindows()
