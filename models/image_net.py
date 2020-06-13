from itertools import islice

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from data.providers import ImageProvider

IMG_SHAPE = (130, 150, 1)


class ImageNet:
    def __init__(self):
        left_input, right_input = Input(IMG_SHAPE), Input(IMG_SHAPE)

        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=IMG_SHAPE))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256, activation='sigmoid'))

        encoded_l, encoded_r = model(left_input), model(right_input)

        L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)

        self.net = Model(inputs=[left_input, right_input], outputs=prediction)

        optimizer = Adam(lr=1e-6)
        self.net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[BinaryAccuracy()])
        self.net.summary()

    def train(self, save=True):
        x, y = zip(*islice(ImageProvider().provide(), 20000))
        x, y = np.array(x), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=.66)

        X_train = X_train.transpose(1, 0, 2, 3)
        X_test = X_test.transpose(1, 0, 2, 3)

        self.net.fit([X_train[0][..., np.newaxis], X_train[1][..., np.newaxis]], y_train,
                     batch_size=16, validation_split=0.1, epochs=30)

        if save:
            self.net.save('imagenet')

        m = BinaryAccuracy()
        y_pred = self.net.predict([X_test[0][..., np.newaxis], X_test[1][..., np.newaxis]])
        m.update_state(y_test, y_pred)
        print(m.result())


if __name__ == '__main__':
    im = ImageNet()
    im.train()
