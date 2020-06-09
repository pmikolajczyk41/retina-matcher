import random
from collections import defaultdict

import numpy as np
from numpy.random import choice

from data.loader import DataLoader

SAME, DIFF = 1, 0


class Provider:
    def __init__(self):
        data = defaultdict(list)
        for key, img in DataLoader().get_data():
            data[key].append(img)
        y, X = zip(*data.items())
        self._y, self._X = np.array(y)[:, None], np.array(X)
        self._n = len(y)

    def get_same_sample(self):
        y = choice(self._n)
        xs = choice(len(self._X[y]), 2, replace=False)
        return self._X[y, xs]

    def get_diff_sample(self):
        ys = choice(self._n, 2, replace=False)
        x1, x2 = self._X[ys]
        return np.array([x1[choice(len(x1))],
                         x2[choice(len(x2))]])

    def get_batch(self, nsamples, ratio):
        same = [(self.get_same_sample(), SAME) for _ in range(int(ratio * nsamples))]
        diff = [(self.get_diff_sample(), DIFF) for _ in range(int((1. - ratio) * nsamples + 2))]
        data = (same + diff)[:nsamples]
        random.shuffle(data)
        batch, labels = zip(*data)
        return np.array(batch), np.array(labels)


if __name__ == '__main__':
    p = Provider()
    batch, labels = p.get_batch(32, 0.2)

    import cv2 as cv
    for (r1, r2), lbl in zip(batch, labels):
        (h, w), ratio = r1.shape, 6
        cv.imshow('retina', cv.resize(r1, (ratio * w, ratio * h), interpolation=cv.INTER_NEAREST))
        second = 'same' if lbl == SAME else 'diff'
        cv.imshow(second, cv.resize(r2, (ratio * w, ratio * h), interpolation=cv.INTER_NEAREST))

        cv.waitKey()
        cv.destroyAllWindows()

