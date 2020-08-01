import csv

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

class_names = ['personal', 'important', 'transaction', 'advertisement', 'spam']


def merge(labels):
    for i, label in enumerate(labels):
        if label == 4:
            labels[i] = 3
    return labels


if __name__ == '__main__':
    # load data
    print("loading data")
    data = np.array(list(csv.reader(open('data/train_db/dataset.csv', encoding='utf8'))))

    # split into features and labels
    print("parsing data")
    features = data[:, :-1].astype(float)
    labels = data[:, -1].astype(float)

    # size variables
    m, n = features.shape

    # split data into train, test and cv data
    test_features = features[:int(m*0.1)]
    test_labels = labels[:int(m*0.1)]
    print(test_features[0])

    cv_features = features[-int(m*0.15):]
    cv_labels = labels[-int(m*0.15):]

    train_features = features[int(m*0.1):-int(m*0.15)]
    train_labels = labels[int(m*0.1):-int(m*0.15)]

    print("prepping model")
    # prep NN model
    model = keras.Sequential([
        keras.Input(shape=n),  # input layer
        keras.layers.Dense(32, activation='relu'),  # hidden layer (1)
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(5e-3)),  # output layer (2)
    ])

    # compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # train the model
    print('training the model')
    model.fit(train_features, train_labels, epochs=9)

    # test set accuracy
    test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=1)
    print('test accuracy:', test_acc)

    # cv set accuracy
    cv_loss, cv_acc = model.evaluate(cv_features, cv_labels, verbose=1)
    print('cv accuracy:', cv_acc)

    model.save('models/model.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("models/model.tflite", "wb").write(tflite_model)
