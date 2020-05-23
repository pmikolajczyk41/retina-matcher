import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize

from data.loader import DataLoader
from graphify import uint
from graphify.builder import Builder
from graphify.marker import Marker


class Graph:
    def __init__(self, img):
        simplified = self._simplify(img)
        marked = Marker().mark(simplified, True)
        self._V, self._E = Builder.build(marked, True)

    @staticmethod
    def _simplify(img):
        _, img = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        skeleton = skeletonize(img > 0)
        return np.where(skeleton, 1, 0).astype(uint)


if __name__ == '__main__':
    for img in DataLoader().get_data():
        g = Graph(img)
        cv.waitKey()
        cv.destroyAllWindows()
