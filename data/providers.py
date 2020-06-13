from collections import defaultdict
from random import sample, random

from data.loader import DataLoader

SAME, DIFF = 1, 0


class Provider:
    def __init__(self, data):
        classified = defaultdict(list)
        for key, img in data:
            classified[key].append(img)
        self._y, self._X = zip(*classified.items())
        self._n = len(self._y)

    def get_same_sample(self):
        x = sample(self._X, k=1)[0]
        return sample(x, k=2)

    def get_diff_sample(self):
        xs = sample(self._X, k=2)
        return [sample(x, k=1)[0] for x in xs]

    def provide(self, ratio=.5):
        while True:
            if random() <= ratio:
                yield self.get_diff_sample(), DIFF
            else:
                yield self.get_same_sample(), SAME


class ImageProvider(Provider):
    def __init__(self):
        super().__init__(DataLoader().get_img_data())


class GraphStatsProvider(Provider):
    def __init__(self):
        super().__init__(DataLoader().get_graph_data())


if __name__ == '__main__':
    p = ImageProvider()

    import cv2 as cv

    for _, ((r1, r2), lbl) in zip(range(6), p.provide(.5)):
        (h, w), ratio = r1.shape, 6
        cv.imshow('retina', cv.resize(r1, (ratio * w, ratio * h), interpolation=cv.INTER_NEAREST))
        second = 'same' if lbl == SAME else 'diff'
        cv.imshow(second, cv.resize(r2, (ratio * w, ratio * h), interpolation=cv.INTER_NEAREST))
        cv.waitKey()
        cv.destroyAllWindows()

    p = GraphStatsProvider()
    for _, ((r1, r2), lbl) in zip(range(6), p.provide(.5)):
        print(r1)
        print(r2)
        print(lbl)
        print()
