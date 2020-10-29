import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from random import randint
import sys
from PIL import Image
import os
from uitwerkingen import confEls, confData

def image_to_matrix(image):
    return keras.preprocessing.image.img_to_array(image).reshape((75, 75))


if __name__ == '__main__':
    X = np.zeros(shape=(1000, 75, 75))
    y = np.zeros(shape=(1000,))
    labels = []

    root_folder = "./Fundus-data/"

    category_index = 0
    index = 0

    print("loading images...")
    for category in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, category)):
            category_name = os.path.basename(category)
            category_location = root_folder + category_name + "/"
            labels.append(category_name)
            for image in os.listdir(category_location):
                X[index] = image_to_matrix(Image.open(category_location + image))
                y[index] = category_index
                index += 1
            category_index += 1
    print("Done!")

    # The image is inverted, but this shouldn't matter
    X /= np.amax(X)
    plt.imshow(X[0], cmap='Greys')
    plt.show()

    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, input_shape=(5625,), activation='relu'),
        keras.layers.Dense(units=np.amax(y) + 1, activation='softmax')
    ])

    #TODO no copied code from exercise 3
    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=1000)

    predictions = np.argmax(model.predict(X), axis=1)
    conf = tf.math.confusion_matrix(y, predictions)

    sess = tf.compat.v1.Session()
    with sess.as_default():
        data = conf.numpy()

    plt.figure()
    plt.matshow(data)
    plt.show()

    metrics = confEls(conf, labels)
    print(metrics)

    scores = confData(metrics)
    print(scores)
