import json
import os
import shutil
import urllib.request
from os import path
from urllib.error import HTTPError

import cv2 as cv
import numpy as np
from PIL import Image

from data.augmentor import Augmentor
from graphify.graph import Graph
from graphify.stats import GraphStats

URL = 'http://cecas.clemson.edu/~ahoover/stare/icon-images/vessels-images'
RAW_DATA_DIR = '../resources/vessels/raw'
READY_DATA_DIR = '../resources/vessels/ready'
GRAPHIFIED_DATA_DIR = '../resources/vessels/graphified'
DEFAULT_RANGE = range(1, 403)
COPIES = 3


class DataLoader:
    def fetch_raw_data(self, rng=DEFAULT_RANGE):
        self._prepare_dir(RAW_DATA_DIR)
        temp = 'temp.gif'
        for idx, i in enumerate(rng):
            try:
                urllib.request.urlretrieve(f'{URL}/im{i:04d}.net.gif', temp)
                img = Image.open(temp)
                img.save(self._raw_ith(i), 'png', optimize=True)
            except HTTPError: pass
            self._log_progress('Fetching', (idx + 1) / len(rng))
        os.remove(temp)
        self._end_logging()

    def sieve_raw_data(self, excl_rng):
        for idx, i in enumerate(excl_rng):
            try: os.remove(self._raw_ith(i))
            except FileNotFoundError: pass
            self._log_progress('Sieving', (idx + 1) / len(excl_rng))
        self._end_logging()

    def augment(self, rng=DEFAULT_RANGE):
        self._prepare_dir(READY_DATA_DIR)
        aug = Augmentor()
        for idx, i in enumerate(rng):
            if not path.exists(self._raw_ith(i)): continue
            img = cv.imread(self._raw_ith(i), cv.IMREAD_GRAYSCALE)
            assert img is not None
            for id, copy in enumerate(aug.augment(img, COPIES)):
                cv.imwrite(f'{READY_DATA_DIR}/{i:04d}_{id}.png', copy)
            self._log_progress('Preprocessing', (idx + 1) / len(rng))
        self._end_logging()

    def convert_to_graph_stats(self):
        def cvt(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            else: raise TypeError

        self._prepare_dir(GRAPHIFIED_DATA_DIR)
        files = os.listdir(READY_DATA_DIR)
        for idx, file in enumerate(files):
            pth = os.path.join(READY_DATA_DIR, file)
            img = cv.imread(pth, cv.IMREAD_GRAYSCALE)
            gs = Graph(img).get_stats()
            with open(f'{GRAPHIFIED_DATA_DIR}/{file[:file.find(".png")]}.gstats', 'w') as output:
                json.dump(gs._asdict(), output, default=cvt)
            self._log_progress('Graphifing', (idx + 1) / len(files))
        self._end_logging()

    @staticmethod
    def get_img_data():
        for file in os.listdir(READY_DATA_DIR):
            pth = os.path.join(READY_DATA_DIR, file)
            img = cv.imread(pth, cv.IMREAD_GRAYSCALE)
            if img is not None: yield file[:file.find('_')], img

    @staticmethod
    def get_graph_data():
        for file in os.listdir(GRAPHIFIED_DATA_DIR):
            pth = os.path.join(GRAPHIFIED_DATA_DIR, file)
            with open(pth, 'r') as input:
                stats = json.load(input, object_hook=lambda d: GraphStats(**d))
            if stats is not None: yield file[:file.find('_')], stats

    @staticmethod
    def _prepare_dir(dirname):
        shutil.rmtree(dirname, ignore_errors=True)
        os.mkdir(dirname)

    @staticmethod
    def _raw_ith(i): return f'{RAW_DATA_DIR}/{i:04d}.png'

    @staticmethod
    def _log_progress(title, fraction): print(f'\r{title}... [{100 * fraction:.2f}%]', end='')

    @staticmethod
    def _end_logging(): print()


if __name__ == '__main__':
    DataLoader().fetch_raw_data()
    DataLoader().sieve_raw_data(
            [10, 35, 72, 79, 86, 96, 126, 127, 130, 131, 132, 147, 152, 171, 176, 203, 242, 261, 276, 305, 306, 310,
             311, 312, 313, 314, 315, 316, 345, 346, 350, 352, 356, 367, 387, ])
    DataLoader().augment()
    DataLoader().convert_to_graph_stats()
