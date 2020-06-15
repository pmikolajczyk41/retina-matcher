import cv2 as cv
import networkx as nx
import numpy as np
from networkx.algorithms.approximation.treewidth import treewidth_decomp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import johnson, connected_components
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.morphology import skeletonize

from graphify import uint
from graphify.builder import Builder
from graphify.marker import Marker
from graphify.stats import GraphStats


class Graph:
    def __init__(self, img):
        simplified = self._simplify(img)
        marked = Marker().mark(simplified, False)
        self._V, self._E = Builder.build(marked, False)
        self._compute()

    def get_stats(self):
        return GraphStats(self._n, self._m, self._weight_sum, self._leaves, self._min_degree, self._max_degree,
                          self._avg_degree, self._med_degree, self._std_degree, self._diameter,
                          self._unweighted_diameter, self._cc, self._msf, self._treewidth)

    def _compute(self):
        self._compute_basics()
        self._compute_leaves()
        self._compute_degrees()

        m = self._get_adj_matrix()
        self._compute_diameters(m)
        self._compute_cc(m)
        self._compute_msf(m)

        g = self._get_nx_graph()
        self.compute_advanced(g)

    def _compute_basics(self):
        self._n = len(self._V)
        self._m = len(self._E)
        self._weight_sum = sum(map(lambda e: e.get_weight(), self._E)) // 2

    def _compute_leaves(self):
        self._leaves = len(list(filter(lambda v: len(v.get_edges()) == 1, self._V)))

    def _compute_degrees(self):
        degrees = [len(v.get_edges()) for v in self._V]
        self._min_degree = min(degrees)
        self._max_degree = max(degrees)
        self._avg_degree = np.mean(degrees)
        self._med_degree = np.median(degrees)
        self._std_degree = np.std(degrees)

    def _get_adj_matrix(self):
        m = np.zeros((self._n, self._n))
        for e in self._E:
            u, v = tuple(e.get_vertices())
            uid, vid = u.get_id(), v.get_id()
            m[uid, vid] = m[vid, uid] = e.get_weight()
        return m

    def _compute_diameters(self, adj_matrix):
        dists = johnson(csr_matrix(adj_matrix), directed=False, unweighted=False)
        dists = np.where(dists == np.inf, 0, dists)
        self._diameter = np.max(dists)
        dists = johnson(csr_matrix(adj_matrix), directed=False, unweighted=True)
        dists = np.where(dists == np.inf, 0, dists)
        self._unweighted_diameter = np.max(dists)

    def _compute_cc(self, adj_matrix):
        self._cc = connected_components(csr_matrix(adj_matrix), directed=False, return_labels=False)

    def _compute_msf(self, adj_matrix):
        msf = minimum_spanning_tree(csr_matrix(adj_matrix))
        self._msf = np.sum(msf)

    def _get_nx_graph(self):
        g = nx.Graph()
        for e in self._E:
            u, v = e.get_vertices()
            g.add_edge(u.get_id(), v.get_id(), weight=e.get_weight())
        return g

    def compute_advanced(self, g):
        self._treewidth, _ = treewidth_decomp(g)

    @staticmethod
    def _simplify(img):
        _, img = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
        skeleton = skeletonize(img > 0)
        thin = np.where(skeleton, 1, 0).astype(uint)
        return Graph.remove_noise(thin)

    @staticmethod
    def remove_noise(img, thresh=4):
        nlabels, labels, stats, _ = cv.connectedComponentsWithStats(img, connectivity=8)
        for id in range(nlabels):
            if stats[id][cv.CC_STAT_AREA] <= thresh:
                img = np.where(labels == id, 0, img).astype(uint)
        return img
