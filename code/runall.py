# -*- coding: utf -*-

"""This module contains general code for processing all algorithms."""

import importlib
import logging
import os

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

helpers = importlib.import_module("helpers")

nn = importlib.import_module("nn")

kmeans = importlib.import_module("kmeans")
em = importlib.import_module("em")

pca = importlib.import_module("pca")
ica = importlib.import_module("ica")
rp = importlib.import_module("rp")
lda = importlib.import_module("lda")

matplotlib.rc("font", weight="normal", size=8)
sns.set_style("darkgrid")

np.random.seed(42)


_DIMS = {
    "pca": {"cardio": 7, "pulsar": 4},
    "ica": {"cardio": 7, "pulsar": 4},
    "rp": {"cardio": 12, "pulsar": 4},
    "lda": {"cardio": 5, "pulsar": 5},

    "kmeans": {"cardio": 4, "pulsar": 4},  # Said 4, not 13
    "em": {"cardio": 8, "pulsar": 7},

    "pca-kmeans": {"cardio": 8, "pulsar": 9},
    "ica-kmeans": {"cardio": 5, "pulsar": 11},
    "rp-kmeans": {"cardio": 6, "pulsar": 4},
    "lda-kmeans": {"cardio": 4, "pulsar": 8},

    "pca-em": {"cardio": 16, "pulsar": 9},
    "ica-em": {"cardio": 4, "pulsar": 10},
    "rp-em": {"cardio": 7, "pulsar": 6},
    "lda-em": {"cardio": 4, "pulsar": 10},
}


@helpers.log_func_edges
def search_best_k(funcname, func, datasets, targets, kMax=20, kMin=2):
    """Search for best K for dim redux."""

    for dataset, target in zip(datasets, targets):

        logging.info(f"Initializing KMeans search for {dataset}...")
        data = helpers.load_dataset_df(dataset)

        train, test = helpers.split_test_train(data, target)
        train, test = helpers.scale_test_train(train, test)
        train, test = func(train, test, _DIMS[funcname][dataset])

        kmeans._sub_search_best_k(dataset, train, test, kMax, kMin, f"{funcname}-")
        em._sub_search_best_k(dataset, train, test, kMax, kMin, f"{funcname}-")

    return None


@helpers.log_func_edges
def dim_reconstruction_loss(datasets, targets):
    """Reconstruction loss for various dim reduction algos."""
    for dataset, target in zip(datasets, targets):

        logging.info(f"Initializing reconstruction loss search for {dataset}...")
        data = helpers.load_dataset_df(dataset)

        train, test = helpers.split_test_train(data, target)
        train, test = helpers.scale_test_train(train, test)

        mse = dict()

        pca = PCA(_DIMS["pca"][dataset], whiten=True, random_state=42)
        ica = FastICA(_DIMS["ica"][dataset], max_iter=5000, whiten=True, random_state=42)

        pcaT = pca.fit_transform(train.X)
        icaT = ica.fit_transform(train.X)

        mse["pca"] = ((train.X - pca.inverse_transform(pcaT)) ** 2).mean()
        mse["ica"] = ((train.X - ica.inverse_transform(icaT)) ** 2).mean()

        print(dataset, mse)

    return None


@helpers.log_func_edges
def class_cluster_seperation(datasets, targets):
    """Visualizing class separation as clusters."""
    for dataset, target in zip(datasets, targets):

        _sets = [n for n in range(3)]

        plt.figure(figsize=(8, 4))
        axs = (f"ax{i}" for i in _sets)
        colors = ("blue", "green", "red", "cyan", "magenta", "yellow", "black")
        fig, axs = plt.subplots(ncols=len(_sets), figsize=(8, 4))

        logging.info(f"Initializing class separation viz for {dataset}...")
        data = helpers.load_dataset_df(dataset)

        train, test = helpers.split_test_train(data, target, split=0.10)
        train, test = helpers.scale_test_train(train, test)

        pca = PCA(2, whiten=True, random_state=42)

        pBase = pca.fit_transform(train.X)

        trainKM, testKM = kmeans.append_cluster_labels(train, test, _DIMS["kmeans"][dataset], False, True)
        pKMean = pca.fit_transform(trainKM.X)

        trainEM, testEM = em.append_cluster_labels(train, test, _DIMS["em"][dataset], False, True)
        pExpMax = pca.fit_transform(trainEM.X)

        axs[0].scatter(
            pBase[train.y == 1, 0], pBase[train.y == 1, 1], color="blue", label="1", alpha=0.5, marker="s"
        )
        axs[0].scatter(
            pBase[train.y == 0, 0], pBase[train.y == 0, 1], color="green", label="0", alpha=0.5, marker="^"
        )

        axs[1].scatter(
            pKMean[trainKM.y == 1, 0], pKMean[trainKM.y == 1, 1], color="blue", label="1", alpha=0.5, marker="s"
        )
        axs[1].scatter(
            pKMean[trainKM.y == 0, 0], pKMean[trainKM.y == 0, 1], color="green", label="0", alpha=0.5, marker="^"
        )

        axs[2].scatter(
            pExpMax[trainEM.y == 1, 0], pExpMax[trainEM.y == 1, 1], color="blue", label="1", alpha=0.5, marker="s"
        )
        axs[2].scatter(
            pExpMax[trainEM.y == 0, 0], pExpMax[trainEM.y == 0, 1], color="green", label="0", alpha=0.5, marker="^"
        )

        axs[0].set_title(f"{dataset}: Base w/ pca", fontsize=10)
        axs[1].set_title(f"{dataset}: KMeans w/ pca", fontsize=10)
        axs[2].set_title(f"{dataset}: ExpMax w/ pca", fontsize=10)

        for index in _sets:
            axs[index].set_xlabel("1st principal component")
            axs[index].set_ylabel("2nd principal component")
            axs[index].legend(loc="upper right")
            axs[index].grid()

        plt.tight_layout()

        outpath = os.path.join(helpers.IMGDIR, f"feature-separation-{dataset}.png")
        plt.savefig(outpath)

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(module)s - %(levelname)s - %(message)s")

    datasets = ("cardio", "pulsar")
    targets = ["class"] * len(datasets)

    # dim_reconstruction_loss(datasets, targets)
    # class_cluster_seperation(datasets, targets)

    # search_best_k("pca", pca.append_cluster_labels, datasets, targets)
    # search_best_k("ica", ica.append_cluster_labels, datasets, targets)
    # search_best_k("rp", rp.append_cluster_labels, datasets, targets)
    # search_best_k("lda", lda.append_cluster_labels, datasets, targets)

    datasets = ("cardio",)
    targets = ["class"] * len(datasets)

    models = helpers.load_best_model_json()

    for dataset, target in zip(datasets, targets):

        data = helpers.load_dataset_df(dataset)
        train, test = helpers.split_test_train(data, target)
        trainO, testO = helpers.scale_test_train(train, test)

        scores = dict()
        clf = nn.Classifier(**models[dataset]["NN"]["params"])

        ## Original Neural Network
        clf.fit(trainO.X, trainO.y)
        predicted = clf.predict(testO.X)
        scores["Base"] = helpers.recall_score(testO.y, predicted)

        ## PCA Neural Network
        train, test = pca.append_cluster_labels(trainO, testO, _DIMS["pca"][dataset])
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["PCA"] = helpers.recall_score(test.y, predicted)

        ## ICA Neural Network
        train, test = ica.append_cluster_labels(trainO, testO, _DIMS["ica"][dataset])
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ICA"] = helpers.recall_score(test.y, predicted)

        ## RP Neural Network
        train, test = rp.append_cluster_labels(trainO, testO, _DIMS["rp"][dataset])
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["RP"] = helpers.recall_score(test.y, predicted)

        ## LDA Neural Network
        train, test = lda.append_cluster_labels(trainO, testO, _DIMS["lda"][dataset])
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["LDA"] = helpers.recall_score(test.y, predicted)

        ## KMeans(PCA) Neural Network
        train, test = pca.append_cluster_labels(trainO, testO, _DIMS["pca"][dataset])
        train, test = kmeans.append_cluster_labels(train, test, _DIMS["pca-kmeans"][dataset], False, False)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["Kmeans & PCA"] = helpers.recall_score(test.y, predicted)

        ## Kmeans(ICA) Neural Network
        train, test = ica.append_cluster_labels(trainO, testO, _DIMS["ica"][dataset])
        train, test = kmeans.append_cluster_labels(train, test, _DIMS["ica-kmeans"][dataset], False, False)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["Kmeans & ICA"] = helpers.recall_score(test.y, predicted)

        ## Kmeans(RP) Neural Network
        train, test = rp.append_cluster_labels(trainO, testO, _DIMS["rp"][dataset])
        train, test = kmeans.append_cluster_labels(train, test, _DIMS["rp-kmeans"][dataset], False, False)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["Kmeans & RP"] = helpers.recall_score(test.y, predicted)

        ## Kmeans(LDA) Neural Network
        train, test = lda.append_cluster_labels(trainO, testO, _DIMS["lda"][dataset])
        train, test = kmeans.append_cluster_labels(train, test, _DIMS["lda-kmeans"][dataset], False, False)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["Kmeans & LDA"] = helpers.recall_score(test.y, predicted)

        ## ExpMax(PCA) Neural Network
        train, test = pca.append_cluster_labels(trainO, testO, _DIMS["pca"][dataset])
        train, test = em.append_cluster_labels(train, test, _DIMS["pca-em"][dataset], False, False)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ExpMax & PCA"] = helpers.recall_score(test.y, predicted)

        ## ExpMax(ICA) Neural Network
        train, test = ica.append_cluster_labels(trainO, testO, _DIMS["ica"][dataset])
        train, test = em.append_cluster_labels(train, test, _DIMS["ica-em"][dataset], False, False)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ExpMax & ICA"] = helpers.recall_score(test.y, predicted)

        ## ExpMax(RP) Neural Network
        train, test = rp.append_cluster_labels(trainO, testO, _DIMS["rp"][dataset])
        train, test = em.append_cluster_labels(train, test, _DIMS["rp-em"][dataset], False, False)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ExpMax & RP"] = helpers.recall_score(test.y, predicted)

        ## ExpMax(LDA) Neural Network
        train, test = lda.append_cluster_labels(trainO, testO, _DIMS["lda"][dataset])
        train, test = em.append_cluster_labels(train, test, _DIMS["lda-em"][dataset], False, False)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ExpMax & LDA"] = helpers.recall_score(test.y, predicted)


        ## -- START TESTING SECTIONS --#
        logging.info("**************** START TESTING SECTION ****************")

        ## KMeans(PCA) Neural Network
        logging.info("Starting on Kmeans post PCA NN scoring.")
        train, test = pca.append_cluster_labels(trainO, testO, _DIMS["pca"][dataset])
        train, test = kmeans.append_cluster_labels(train, test, _DIMS["pca-kmeans"][dataset], False, True)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["Kmeans pPCA"] = helpers.recall_score(test.y, predicted)

        ## Kmeans(ICA) Neural Network
        logging.info("Starting on Kmeans post ICA NN scoring.")
        train, test = ica.append_cluster_labels(trainO, testO, _DIMS["ica"][dataset])
        train, test = kmeans.append_cluster_labels(train, test, _DIMS["ica-kmeans"][dataset], False, True)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["Kmeans pICA"] = helpers.recall_score(test.y, predicted)

        ## Kmeans(RP) Neural Network
        logging.info("Starting on Kmeans post RP NN scoring.")
        train, test = rp.append_cluster_labels(trainO, testO, _DIMS["rp"][dataset])
        train, test = kmeans.append_cluster_labels(train, test, _DIMS["rp-kmeans"][dataset], False, True)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["Kmeans pRP"] = helpers.recall_score(test.y, predicted)

        ## Kmeans(LDA) Neural Network
        logging.info("Starting on Kmeans post LDA NN scoring.")
        train, test = lda.append_cluster_labels(trainO, testO, _DIMS["lda"][dataset])
        train, test = kmeans.append_cluster_labels(train, test, _DIMS["lda-kmeans"][dataset], False, True)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["Kmeans pLDA"] = helpers.recall_score(test.y, predicted)

        ## ExpMax(PCA) Neural Network
        logging.info("Starting on ExpMax post PCA NN scoring.")
        train, test = pca.append_cluster_labels(trainO, testO, _DIMS["pca"][dataset])
        train, test = em.append_cluster_labels(train, test, _DIMS["pca-em"][dataset], False, True)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ExpMax pPCA"] = helpers.recall_score(test.y, predicted)

        ## ExpMax(ICA) Neural Network
        logging.info("Starting on ExpMax post ICA NN scoring.")
        train, test = ica.append_cluster_labels(trainO, testO, _DIMS["ica"][dataset])
        train, test = em.append_cluster_labels(train, test, _DIMS["ica-em"][dataset], False, True)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ExpMax pICA"] = helpers.recall_score(test.y, predicted)

        ## ExpMax(RP) Neural Network
        logging.info("Starting on ExpMax post RP NN scoring.")
        train, test = rp.append_cluster_labels(trainO, testO, _DIMS["rp"][dataset])
        train, test = em.append_cluster_labels(train, test, _DIMS["rp-em"][dataset], False, True)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ExpMax pRP"] = helpers.recall_score(test.y, predicted)

        # ExpMax(LDA) Neural Network
        logging.info("Starting on ExpMax post LDA NN scoring.")
        train, test = lda.append_cluster_labels(trainO, testO, _DIMS["lda"][dataset])
        train, test = em.append_cluster_labels(train, test, _DIMS["lda-em"][dataset], False, True)
        clf.fit(train.X, train.y)
        predicted = clf.predict(test.X)
        scores["ExpMax pLDA"] = helpers.recall_score(test.y, predicted)


        print(scores)
        logging.info("***************** END TESTING SECTION *****************")
        ## -- END TESTING SECTIONS --#

        plt.figure(figsize=(12, 4))
        labels = np.arange(len(scores))
        plt.bar(labels, list(scores.values()), align="center", alpha=0.75)
        plt.xticks(labels, list(scores.keys()), fontsize="x-small")
        plt.xlabel("Models")
        plt.ylabel("Test Recall Scores")
        plt.title(f"Neural Network - {dataset}", fontsize=10)

        for x, y in zip(labels, list(scores.values())):
            plt.text(x, y, f"{round(y * 100, 1)}%", fontweight="bold", ha="center", va="bottom")

        plt.tight_layout()

        outpath = os.path.join(helpers.BASEDIR, "img", f"{dataset}-combined-test-recall.png")
        plt.savefig(outpath)

    # data = helpers.load_dataset_df("cardio")
    # train, test = helpers.split_test_train(data, "class")
    # trainO, testO = helpers.scale_test_train(train, test)

    # scores = dict()
    # clf = nn.Classifier(**models["cardio"]["NN"]["params"])

    # icaScore, kmScore, emScore = dict(), dict(), dict()
    # for i in range(5, 19):

    #     ## ICA Neural Network
    #     train, test = ica.append_cluster_labels(trainO, testO, i)
    #     clf.fit(train.X, train.y)
    #     predicted = clf.predict(test.X)
    #     icaScore[i] = helpers.recall_score(test.y, predicted)

    #     for k in range(4, 18+1):

    #         _key = f"{i:02d}{k:02d}"

    #         ## Kmeans(ICA) Neural Network
    #         train, test = ica.append_cluster_labels(trainO, testO, i)
    #         train, test = kmeans.append_cluster_labels(train, test, k, False, False)
    #         clf.fit(train.X, train.y)
    #         predicted = clf.predict(test.X)
    #         kmScore[_key] = helpers.recall_score(test.y, predicted)

    #         ## ExpMax(ICA) Neural Network
    #         train, test = ica.append_cluster_labels(trainO, testO, i)
    #         train, test = em.append_cluster_labels(train, test, k, False, False)
    #         clf.fit(train.X, train.y)
    #         predicted = clf.predict(test.X)
    #         emScore[_key] = helpers.recall_score(test.y, predicted)

    # plt.figure(figsize=(8, 4))
    # # plt.plot(list(icaScore.keys()), list(icaScore.values()))
    # plt.plot(list(kmScore.keys()), list(kmScore.values()))
    # plt.plot(list(emScore.keys()), list(emScore.values()))
    # plt.show()
