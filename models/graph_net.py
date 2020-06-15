from itertools import islice

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from data.providers import GraphStatsProvider
from models.net import Net

IMG_SHAPE = (14, 1)


class GraphNet(Net):
    def __init__(self):
        super().__init__()
        left_input, right_input = Input(IMG_SHAPE), Input(IMG_SHAPE)

        model = Sequential()
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='sigmoid'))

        encoded_l, encoded_r = model(left_input), model(right_input)

        L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)

        self.net = Model(inputs=[left_input, right_input], outputs=prediction)

        optimizer = Adam(lr=1e-4)
        self.net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[BinaryAccuracy()])
        self.net.summary()

    def train(self, save=True):
        x, y = zip(*islice(GraphStatsProvider().provide(), 20000))
        x, y = np.array(x), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=.66)

        X_train = X_train.transpose(1, 0, 2)
        X_test = X_test.transpose(1, 0, 2)

        self.net.fit([X_train[0], X_train[1]], y_train,
                     batch_size=8, validation_split=0.1, epochs=60)

        if save:
            self.net.save('graphnet')

        m = BinaryAccuracy()
        y_pred = self.net.predict([X_test[0][..., np.newaxis], X_test[1][..., np.newaxis]])
        m.update_state(y_test, y_pred)
        print(m.result())

    def load(self):
        self.net.load_weights('graphnet')

    def prepare(self):
        try: self.load()
        except: self.train()


if __name__ == '__main__':
    im = GraphNet()
    im.train()
