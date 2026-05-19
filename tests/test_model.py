import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from starter.starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference
)


def test_train_model():

    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])

    y = np.array([0, 1, 0, 1])

    model = train_model(X, y)

    assert isinstance(
        model,
        RandomForestClassifier
    )


def test_compute_model_metrics():

    y = np.array([1, 0, 1, 1])

    preds = np.array([1, 0, 1, 0])

    precision, recall, fbeta = compute_model_metrics(
        y,
        preds
    )

    assert isinstance(precision, (float, np.floating))
    assert isinstance(recall, (float, np.floating))
    assert isinstance(fbeta, (float, np.floating))


def test_inference():

    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])

    y = np.array([0, 1, 0, 1])

    model = train_model(X, y)

    preds = inference(model, X)

    assert len(preds) == 4