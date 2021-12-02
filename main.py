import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

if __name__ == "__main__":
    epochs = 8
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print("Fully connected")
    fully_connected = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28)),
         tf.keras.layers.Dense(180, activation='relu'),
         tf.keras.layers.Dense(128, activation='relu'),
         tf.keras.layers.Dense(10)])
    fully_connected.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
    fully_connected.fit(X_train, Y_train, epochs=epochs, verbose=2)
    fully_connected.evaluate(X_test, Y_test, verbose=2)

    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    print("Convolutional")
    convolutional = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(10, activation="softmax")])
    convolutional.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
    convolutional.fit(X_train, Y_train, epochs=epochs, verbose=2)
    convolutional.evaluate(X_test, Y_test, verbose=2)

    print("Convolutional + pooling")
    convolutional = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
         tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax")])
    convolutional.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
    convolutional.fit(X_train, Y_train, epochs=epochs, verbose=2)
    convolutional.evaluate(X_test, Y_test, verbose=2)
