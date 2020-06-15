import json
import os
import pathlib
import random
import shutil
import urllib.request
from os.path import join
from urllib.error import HTTPError

import cv2 as cv
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from data.augmentor import Augmentor
from graphify.graph import Graph
from graphify.stats import GraphStats

URL = 'http://cecas.clemson.edu/~ahoover/stare/icon-images/vessels-images'
RAW_DATA_DIR = '../resources/vessels/raw'
READY_DATA_DIR = '../resources/vessels/ready'
DEFAULT_RANGE = range(1, 403)
COPIES = 3
TEST_SIZE = 40
EXCL_RNG = [10, 35, 72, 79, 86, 96, 126, 127, 130, 131, 132, 147, 152, 171, 176, 203, 242, 261, 276, 305, 306, 310,
            311, 312, 313, 314, 315, 316, 345, 346, 350, 352, 356, 367, 387, ]


class DataLoader:
    def load_data(self):
        self.fetch_raw_data()
        self.sieve_raw_data(EXCL_RNG)
        self.split(TEST_SIZE)
        self.augment()
        self.postprocess()
        self.convert_to_graph_stats()

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

    def split(self, test_size):
        self._prepare_dir(f'{READY_DATA_DIR}/imgs/registered')
        self._prepare_dir(f'{READY_DATA_DIR}/imgs/unregistered')

        imgs = set(os.listdir(RAW_DATA_DIR))
        unreg_imgs = set(random.sample(imgs, test_size))
        reg_imgs = imgs - unreg_imgs

        for idx, unreg_img in enumerate(unreg_imgs):
            shutil.copyfile(join(RAW_DATA_DIR, unreg_img), join(f'{READY_DATA_DIR}/imgs/unregistered', unreg_img))
            self._log_progress('Splitting', (idx + 1) / len(imgs))

        for idx, unreg_img in enumerate(reg_imgs):
            shutil.copyfile(join(RAW_DATA_DIR, unreg_img), join(f'{READY_DATA_DIR}/imgs/registered', unreg_img))
            self._log_progress('Splitting', (idx + 1 + len(unreg_imgs)) / len(imgs))

        self._end_logging()

    def augment(self):
        aug = Augmentor()
        pth = f'{READY_DATA_DIR}/imgs/registered'
        reg_imgs = os.listdir(pth)
        for idx, img_file in enumerate(reg_imgs):
            img = cv.imread(join(pth, img_file), cv.IMREAD_GRAYSCALE)
            assert img is not None

            img_name = img_file[:img_file.find('.png')]
            for id, copy in enumerate(aug.augment(img, COPIES)):
                cv.imwrite(join(pth, f'{img_name}_{id}.png'), copy)
            os.remove(join(pth, img_file))

            self._log_progress('Augmenting', (idx + 1) / len(reg_imgs))
        self._end_logging()

    def postprocess(self):
        self._postprocess_single_set(f'{READY_DATA_DIR}/imgs/registered', 'registered')
        self._postprocess_single_set(f'{READY_DATA_DIR}/imgs/unregistered', 'unregistered')

    def _postprocess_single_set(self, pth, dataset):
        imgs = os.listdir(pth)
        for idx, img_file in enumerate(imgs):
            img_pth = join(pth, img_file)
            img = cv.imread(img_pth, cv.IMREAD_GRAYSCALE)
            _, img = cv.threshold(img, 5, 255, cv.THRESH_BINARY)
            skeleton = skeletonize(img > 0)
            cv.imwrite(img_pth, np.where(skeleton, 255, 0).astype(np.uint8))
            self._log_progress(f'Postprocessing {dataset} set', (idx + 1) / len(imgs))
        self._end_logging()

    def convert_to_graph_stats(self):
        pth = f'{READY_DATA_DIR}/gstats'
        self._prepare_dir(f'{pth}/registered')
        self._prepare_dir(f'{pth}/unregistered')

        self._convert_to_graph_stats_single_set(pth, 'registered')
        self._convert_to_graph_stats_single_set(pth, 'unregistered')

    def _convert_to_graph_stats_single_set(self, gstats_dir, suffix):
        def int_cvt(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            else: raise TypeError

        img_dir = f'{READY_DATA_DIR}/imgs'
        imgs = os.listdir(join(img_dir, suffix))
        for idx, img_file in enumerate(imgs):
            img = cv.imread(join(img_dir, suffix, img_file), cv.IMREAD_GRAYSCALE)
            gs = Graph(img).get_stats()
            img_name = img_file[:img_file.find(".png")]
            with open(join(gstats_dir, suffix, f'{img_name}.json'), 'w') as output:
                json.dump(gs._asdict(), output, default=int_cvt)
            self._log_progress(f'Graphifing {suffix} set', (idx + 1) / len(imgs))
        self._end_logging()

    @staticmethod
    def get_img_data(dataset='registered'):
        base_dir = f'{READY_DATA_DIR}/imgs/{dataset}'
        for file in os.listdir(base_dir):
            pth = os.path.join(base_dir, file)
            img = cv.imread(pth, cv.IMREAD_GRAYSCALE)
            if img is not None: yield file[:file.find('_')], img

    @staticmethod
    def get_graph_data(dataset='registered'):
        base_dir = f'{READY_DATA_DIR}/gstats/{dataset}'
        for file in os.listdir(base_dir):
            pth = os.path.join(base_dir, file)
            with open(pth, 'r') as input:
                stats = json.load(input, object_hook=lambda d: GraphStats(**d))
            if stats is not None: yield file[:file.find('_')], stats

    @staticmethod
    def _prepare_dir(dirname):
        shutil.rmtree(dirname, ignore_errors=True)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _raw_ith(i): return f'{RAW_DATA_DIR}/{i:04d}.png'

    @staticmethod
    def _log_progress(title, fraction): print(f'\r{title}... [{100 * fraction:.2f}%]', end='')

    @staticmethod
    def _end_logging(): print()


if __name__ == '__main__':
    dl = DataLoader()
    dl.load_data()
