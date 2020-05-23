from itertools import product

import cv2 as cv
import numpy as np

from graphify import uint, EDGE, VERTEX


class Edge:
    def __init__(self):
        self._weight, self._vertices = None, set()

    def get_weight(self): return self._weight

    def set_weight(self, weight):
        self._weight = weight
        return self

    def get_vertices(self): return self._vertices

    def add_vertex(self, vertex):
        self._vertices.add(vertex)
        return self


class Vertex:
    def __init__(self, i):
        self._id = i
        self._edges = set()

    def get_id(self): return self._id

    def add_edge(self, edge):
        self._edges.add(edge)
        return self


class Builder:
    @staticmethod
    def build(img, draw=False):
        edges = np.where(img == EDGE, 1, 0).astype(uint)
        nlabels, labels, _, _ = cv.connectedComponentsWithStats(edges, connectivity=8, ltype=cv.CV_32S)
        labels = np.where(edges == 0, 0, labels)
        e_ids, counts = np.unique(labels, return_counts=True)
        E = {i: Edge().set_weight(c) for i, c in zip(e_ids, counts)}

        vertices = np.where(img == VERTEX, 1, 0).astype(uint)
        vertices = list(zip(*np.nonzero(vertices)))
        V = [Vertex(i) for i in range(len(vertices))]
        for v, (vx, vy) in zip(V, vertices):
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1), (0, 1), (1, 0), (1, 1)]:
                eid = labels[vx + dx, vy + dy]
                if eid != 0:
                    E[eid].add_vertex(v)
                    v.add_edge(E[eid])

        if draw:
            Builder._draw(V, E, vertices, img.shape, 6)

        return V, E

    @staticmethod
    def _draw(V, E, vertices, shape, ratio):
        board = np.zeros((shape[0] * ratio, shape[1] * ratio))
        for v in V:
            vx, vy = vertices[v.get_id()][::-1]
            cv.circle(board, (ratio * vx, ratio * vy), 3, 255, 3)
        for e in E.values():
            for v1, v2 in product(e.get_vertices(), e.get_vertices()):
                a = vertices[v1.get_id()][::-1]
                b = vertices[v2.get_id()][::-1]
                cv.line(board, (ratio * a[0], ratio * a[1]), (ratio * b[0], ratio * b[1]), 128, 1)
        cv.imshow('Built', board)
