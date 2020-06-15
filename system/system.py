import numpy as np

from data.loader import DataLoader
from graphify.graph import Graph
from models.image_net import ImageNet, IMG_SHAPE


class System:
    def __init__(self):
        self._imagenet, self._graphnet = ImageNet(), None
        self._imagenet.prepare()
        self._create_database()

    def _create_database(self):
        self._images = dict()
        for key, img in DataLoader.get_img_data():
            if key not in self._images.keys():
                self._images[key] = img
        self._graphs = {k: Graph(i).get_stats() for k, i in self._images.items()}

    def verify(self, target, id):
        ids, imgs = zip(*list(self._images.items()))
        imgs = np.array(imgs).reshape((len(ids), *IMG_SHAPE))

        r = np.repeat(target, len(ids)).reshape((len(ids), *IMG_SHAPE))
        probs = self._imagenet.net.predict([r, imgs]).flatten()
        min_prob = np.min(probs)
        print(min_prob)
        if str(id) in ids: print(probs[ids.index(str(id))])
        if min_prob < .8:
            return False, None
        return True, ids[np.argmin(probs)]


if __name__ == '__main__':
    s = System()
    for (id, img), _ in zip(DataLoader.get_img_data(), range(6)):
        print(id)
        print(s.verify(img, id))
