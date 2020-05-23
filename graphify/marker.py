import cv2 as cv
import numpy as np

uint = np.uint8

BACKGROUND = uint(0)
EDGE = uint(1)
VERTEX = uint(2)


class Marker:
    _K1 = np.array([[1, 1, 1],
                    [1, 16, 1],
                    [1, 1, 1]], dtype=uint)
    _NULL_THRSH, _MAYB_VAL = uint(15), uint(18)

    _K2 = np.array([[1, 2, 4],
                    [128, 0, 8],
                    [64, 32, 16]], dtype=uint)
    _ADJACENT_SUMS = np.array([3, 6, 12, 24, 48, 96, 192, 129], dtype=uint)
    _NULL, _MAYB, _VERT = uint(0), uint(2), uint(4)

    _K3 = np.array([[0, 0, 0],
                    [1, 4, 1],
                    [0, 0, 0]], dtype=uint)
    _K4 = np.array([[0, 1, 0],
                    [0, 4, 0],
                    [0, 1, 0]], dtype=uint)
    _K5 = np.array([[1, 0, 0],
                    [0, 4, 0],
                    [0, 0, 1]], dtype=uint)
    _K6 = np.array([[0, 0, 1],
                    [0, 4, 0],
                    [1, 0, 0]], dtype=uint)
    _K7 = np.array([[0, 1, 0],
                    [0, 4, 1],
                    [0, 0, 0]], dtype=uint)
    _K8 = np.array([[0, 0, 0],
                    [0, 4, 1],
                    [0, 1, 0]], dtype=uint)
    _K9 = np.array([[0, 0, 0],
                    [1, 4, 0],
                    [0, 1, 0]], dtype=uint)
    _K10 = np.array([[0, 1, 0],
                     [1, 4, 0],
                     [0, 0, 0]], dtype=uint)
    _EXC_VERT_VAL = uint(6)

    def mark(self, img):
        c1 = cv.filter2D(img, -1, self._K1)
        c1 = np.where(c1 <= self._NULL_THRSH, self._NULL, c1)
        c1 = np.where(c1 == self._MAYB_VAL, self._MAYB, c1)
        c1 = np.where(np.logical_and(c1 > self._NULL_THRSH, c1 != self._MAYB_VAL), self._VERT, c1)

        c2 = cv.filter2D(c1, -1, self._K2)
        c2 = np.isin(c2, self._ADJACENT_SUMS)
        c2 = np.where(c2, uint(1), uint(0))

        mapped = self._map(c1 + c2).astype(uint)
        compressed = self._compress_vertices(mapped).astype(uint)
        return compressed

    @staticmethod
    def _map(img):
        def pointwise_map(x):
            if x in [0, 1]: return BACKGROUND
            if x in [2]: return EDGE
            if x in [3, 4, 5]: return VERTEX
            assert False

        return ((np.vectorize(pointwise_map))(img)).astype(uint)

    def _compress_vertices(self, img):
        all_verts = np.where(img == VERTEX, uint(1), uint(0))
        excess_verts = np.zeros_like(all_verts, dtype=uint)
        for k in [self._K3, self._K4, self._K5, self._K6, self._K7, self._K8, self._K9, self._K10]:
            verts = cv.filter2D(all_verts, -1, k)
            verts = np.where(verts == self._EXC_VERT_VAL, uint(1), uint(0))
            excess_verts = np.logical_or(excess_verts, verts)

        return (img + (int(EDGE) - int(VERTEX)) * excess_verts).astype(uint)
