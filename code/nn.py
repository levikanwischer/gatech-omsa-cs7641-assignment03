# -*- coding: utf -*-

"""This module contains general code for Neural Networks."""

import importlib
import logging
import os
from itertools import product

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier as Classifier

helpers = importlib.import_module("helpers")

np.random.seed(42)


@helpers.log_func_edges
def search_best_params(X, y, params=None):
    """GridSearch for best params, returning results and best model.

    Parameters
    ----------
    X : <array-like>
        Features for fitting classifier with.
    y : <array-like>
        Target for scoring classifier against.
    params : dict, optional (default=None)
        classifier hyper parameters to fit against.
        If None, will use broad GridSearch

    Returns
    -------
    anonymous : tuple
        Tuple of results array and best key model.

    """
    lMin = len(y.unique())
    lMax = (X.shape[1] + lMin) * 1.2
    lJump = min(lMax - lMin, 8)

    lSteps = np.unique(np.ceil(np.linspace(lMin, lMax, lJump))).astype("int")

    # NOTE: Quick workaround around for np.int64 JSON encoding issue
    lSteps = [int(i) for i in lSteps]

    layers = product(lSteps, lSteps, lSteps, lSteps)
    layers = [l for l in layers if l[1] * lMin <= l[0] >= l[2] > l[3]]

    logging.info(f"Preparing to run {len(layers)} different hidden layer configurations.")
    logging.info(f"Hidden Layers: {layers}")

    params = params or {"hidden_layer_sizes": layers, "max_iter": [5000]}

    clf = Classifier(random_state=42)
    return helpers.search_model_param_space(clf, params, X, y)


@helpers.log_func_edges
def run_gridsearch(datasets, targets):
    """Run GridSearch on hyper-params for Neural Network."""
    return helpers._run_fn_gridsearch("NN", datasets, targets, search_best_params)


@helpers.log_func_edges
def plot_confusion_matrix(datasets, targets):
    """Plot Confusion matrix for datasets using 'best' Neural Network params."""
    return helpers._plot_confusion_matrix("NN", Classifier, datasets, targets)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(module)s - %(levelname)s - %(message)s")

    datasets = ("cardio", "pulsar")
    targets = ["class"] * len(datasets)

    run_gridsearch(datasets, targets)
    plot_confusion_matrix(datasets, targets)
