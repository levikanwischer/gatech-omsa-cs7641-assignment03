# -*- coding: utf -*-

"""This module contains general code for NMF."""

import importlib
import logging
import os

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import NMF as DimRedux

from sklearn.ensemble import RandomForestClassifier as SimpleClf1
from sklearn.naive_bayes import GaussianNB as SimpleClf2

helpers = importlib.import_module("helpers")

np.random.seed(42)


@helpers.log_func_edges
def search_best_k(datasets, targets):
    """Search for best K by Classifier Score."""

    plt.figure(figsize=(8, 4))

    subindex = 0
    for dataset, target in zip(datasets, targets):
        subindex += 1

        logging.info(f"Initializing NMF search for {dataset}...")
        data = helpers.load_dataset_df(dataset)

        train, test = helpers.split_test_train(data, target)
        train, test = helpers.scale_test_train(train, test)

        slf = dict()
        for k in range(1, train.X.shape[1]):
            dim = DimRedux(k, max_iter=5000, random_state=42)
            dim.fit(train.X)

            split_ = train.X.shape[0] // 3 * 2

            clf1, clf2 = SimpleClf1(), SimpleClf2()
            clf1.fit(dim.transform(train.X[:split_,]), train.y[:split_,])
            clf2.fit(dim.transform(train.X[:split_,]), train.y[:split_,])

            sco1 = clf1.score(dim.transform(train.X[split_:,]), train.y[split_:,])
            sco2 = clf2.score(dim.transform(train.X[split_:,]), train.y[split_:,])

            slf[k] = (sco1 + sco2) / 2

        plt.subplot(1, len(datasets), subindex)
        plt.plot(list(slf.keys()), list(slf.values()))
        plt.xlabel("Components")
        plt.ylabel("Classifier Train Score")
        plt.xticks(np.arange(1, train.X.shape[1] + 1, step=1))
        plt.title(f"{dataset}", fontsize=10)

        plt.tight_layout()

        outpath = os.path.join(helpers.BASEDIR, "img", f"dim-nmf-both.png")
        plt.savefig(outpath)

    return None


@helpers.log_func_edges
def append_cluster_labels(train, test, bestK):
    """Append best K label to train & test."""
    clu = DimRedux(bestK, max_iter=5000, random_state=42)
    clu.fit(train.X)

    train = helpers.Data(clu.transform(train.X), train.y)
    test = helpers.Data(clu.transform(test.X), test.y)

    return helpers.scale_test_train(train, test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(module)s - %(levelname)s - %(message)s")

    datasets = ("cardio", "pulsar")
    targets = ["class"] * len(datasets)

    search_best_k(datasets, targets)
