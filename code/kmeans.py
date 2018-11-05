# -*- coding: utf -*-

"""This module contains general code for KMeans."""

import importlib
import logging
import os

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans as Clusterizer

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score

helpers = importlib.import_module("helpers")

np.random.seed(42)


@helpers.log_func_edges
def search_best_k(datasets, targets, kMax=20, kMin=2):
    """Search for best K by SSE & Silhouette."""

    for dataset, target in zip(datasets, targets):

        logging.info(f"Initializing KMeans search for {dataset}...")
        data = helpers.load_dataset_df(dataset)

        train, test = helpers.split_test_train(data, target)
        train, test = helpers.scale_test_train(train, test)

    return _sub_search_best_k(dataset, train, test, kMax, kMin)


@helpers.log_func_edges
def _sub_search_best_k(dataset, train, test, kMax, kMin, extra=""):
    """Search for best K by SSE & Silhouette."""
    plt.figure(figsize=(8, 4))

    sse, sil, nmi = dict(), dict(), dict()
    for k in range(kMin, kMax + 1):
        clu = Clusterizer(k, max_iter=1000, random_state=42)
        clu.fit(train.X)

        sse[k] = clu.inertia_
        logging.info(f"Sum Squared Errors for {k} clusters is {sse[k]:.4f}")

        sil[k] = silhouette_score(train.X, clu.predict(train.X), random_state=42)
        logging.info(f"Silhouette Coefficient for {k} clusters is {sil[k]:.4f}")

        nmi[k] = normalized_mutual_info_score(train.y, clu.predict(train.X))
        logging.info(f"Norm Mutual Info for {k} clusters is {nmi[k]:.4f}")

    plt.subplot(1, 3, 1)
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum Squared Errors")
    plt.xticks(np.arange(kMin, kMax + 1, step=1))
    plt.title(f"{dataset} - Elbow", fontsize=10)

    plt.subplot(1, 3, 2)
    plt.plot(list(sil.keys()), list(sil.values()))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.xticks(np.arange(kMin, kMax + 1, step=1))
    plt.title(f"{dataset} - Silhouette", fontsize=10)

    plt.subplot(1, 3, 3)
    plt.plot(list(nmi.keys()), list(nmi.values()))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Normalized Mutual Info")
    plt.xticks(np.arange(kMin, kMax + 1, step=1))
    plt.title(f"{dataset} - NMI", fontsize=10)

    plt.tight_layout()

    outpath = os.path.join(helpers.BASEDIR, "img", f"kmeans-bestk-{extra}{dataset}.png")
    plt.savefig(outpath)
    return None


@helpers.log_func_edges
def append_cluster_labels(train, test, bestK, replace=False, dummies=False):
    """Append best K label to train & test."""
    clu = Clusterizer(bestK, max_iter=1000, random_state=42)
    clu.fit(train.X)

    trainPredict = clu.predict(train.X).reshape(-1, 1)
    testPredict = clu.predict(test.X).reshape(-1, 1)

    if dummies:
        trainPredict = pd.get_dummies(pd.DataFrame(trainPredict)[0], drop_first=True).values
        testPredict = pd.get_dummies(pd.DataFrame(testPredict)[0], drop_first=True).values

    if not replace:
        train = helpers.Data(np.append(train.X, trainPredict, axis=1), train.y)
        test = helpers.Data(np.append(test.X, testPredict, axis=1), test.y)

    else:
        train = helpers.Data(trainPredict, train.y)
        test = helpers.Data(testPredict, test.y)

    if not dummies:
        train, test = helpers.scale_test_train(train, test)

    return train, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(module)s - %(levelname)s - %(message)s")

    datasets = ("cardio", "pulsar")
    targets = ["class"] * len(datasets)

    search_best_k(datasets, targets)
