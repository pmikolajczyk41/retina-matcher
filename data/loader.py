import os
import shutil
from os import path

import cv2 as cv
import urllib.request
from urllib.error import HTTPError

from PIL import Image

URL = 'http://cecas.clemson.edu/~ahoover/stare/icon-images/vessels-images'
RAW_DATA_DIR = '../resources/vessels/raw'
READY_DATA_DIR = '../resources/vessels/ready'
DEFAULT_RANGE = range(1, 403)


class DataLoader:
    def fetch_raw_data(self, rng=DEFAULT_RANGE):
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

    def preprocess(self, rng=DEFAULT_RANGE):
        shutil.rmtree(READY_DATA_DIR)
        os.mkdir(READY_DATA_DIR)
        for idx, i in enumerate(rng):
            if not path.exists(self._raw_ith(i)): continue
            img = cv.imread(self._raw_ith(i), cv.IMREAD_GRAYSCALE)
            assert img is not None
            img = cv.resize(img, (100, 100))
            cv.imwrite(self._ready_ith(i), img)
            self._log_progress('Preprocessing', (idx + 1) / len(rng))
        self._end_logging()

    def get_data(self, rng=DEFAULT_RANGE):
        for i in rng:
            yield cv.imread(self._ready_ith(i), cv.IMREAD_GRAYSCALE)

    @staticmethod
    def _raw_ith(i): return f'{RAW_DATA_DIR}/{i:04d}.png'

    @staticmethod
    def _ready_ith(i): return f'{READY_DATA_DIR}/{i:04d}.png'

    @staticmethod
    def _log_progress(title, fraction): print(f'\r{title}... [{100 * fraction:.2f}%]', end='')

    @staticmethod
    def _end_logging(): print()


if __name__ == '__main__':
    DataLoader().fetch_raw_data()
    DataLoader().sieve_raw_data(
            [10, 35, 72, 79, 86, 96, 126, 127, 130, 131, 132, 147, 152, 171, 176, 203, 242, 261, 276, 305, 306, 310,
             311, 312, 313, 314, 315, 316, 345, 346, 350, 352, 356, 367, 387, ])
    DataLoader().preprocess()
