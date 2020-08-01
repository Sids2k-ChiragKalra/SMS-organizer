from tensorflow import keras
import numpy as np
import pandas as pd

import data.normalise as nm
from data.features import compute_features
from prune import nw_features_disc

class_names = ['personal', 'important', 'transaction', 'advertisement', 'spam']


def predict(**kwargs):
    data = {arg: np.array(val, dtype=str) for arg, val in kwargs.items()}

    nw_features = pd.DataFrame(index=range(len(data['message'])), columns=[feat for feat in nw_features_disc])
    for feature, disc in nw_features_disc.items():
        nw_features[feature] = np.vectorize(disc['func'])(np.array(data[disc['input']]))
    nw_features = nw_features.to_numpy()

    _, w_features = compute_features(data['message'])
    number_words = nm.number_words(w_features)
    nw_features = np.append(nw_features, number_words, axis=1)
    features = np.append(nw_features, w_features, axis=1)

    # get predictions from model
    model = keras.models.load_model('models/model.h5')
    predictions = [class_names[x] for x in np.argmax(model.predict(features), axis=1)]
    print(predictions)


predict(message=['corona is a bad thing, haha'], time=['12 June 2020 15:21'])
