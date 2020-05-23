import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize

from data.loader import DataLoader
from graphify.marker import Marker, uint


class Graph:
    def __init__(self, img):
        img = self._simplify(img)
        marked = Marker().mark(img)
        cv.imshow('', cv.resize(120 * marked, (1300, 1000), interpolation=cv.INTER_NEAREST))
        cv.waitKey()
        cv.destroyAllWindows()

    @staticmethod
    def _simplify(img):
        _, img = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        skeleton = skeletonize(img > 0)
        return np.where(skeleton, 1, 0).astype(uint)


if __name__ == '__main__':
    for img in DataLoader().get_data():
        g = Graph(img)
