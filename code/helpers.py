# -*- coding: utf -*-

"""This module contains general "helper" methods."""

import csv
import functools
import itertools
import json
import logging
import os
from collections import namedtuple
from itertools import product
from operator import itemgetter

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

matplotlib.rc("font", weight="normal", size=8)
sns.set_style("darkgrid")

np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.dirname(HERE)
DATASETS = os.path.join(BASEDIR, "data")
IMGDIR = os.path.join(BASEDIR, "img")

Data = namedtuple("Data", ["X", "y"])


def log_func_edges(func):
    """Decorator to log enter and exit of function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Generic wrapper function."""
        logging.info(f"Entering `{func.__name__}` for processing...")
        results = func(*args, **kwargs)
        logging.info(f"Exiting processing for `{func.__name__}`")
        return results

    return wrapper


@log_func_edges
def qualify_full_filepath(filename, path=None):
    """Forms filepath and validates it exists.

    Parameters
    ----------
    filename : str
        Name of file (without extension) to load.
    path : str, optional (default=None)
        Path to csv to load filename from.
        If None, uses current path.

    Returns
    -------
    filepath : str
        Validated path to given file.

    Raises
    ------
    OSError
        If filepath is not a valid/available file.

    """
    filepath = os.path.join(path or "", filename)
    if not os.path.isfile(filepath):
        raise OSError(f"No available file found at: {filename}.")
    return filepath


@log_func_edges
def load_dataset_csv(filename, path=DATASETS):
    """Load CSV dataset into memory from path.

    Parameters
    ----------
    filename : str
        Name of file (without extension) to load.
    path : str, optional (default=DATASETS)
        Path to csv to load filename from.

    Returns
    -------
    data : list of dict
        Loaded dataset as list of column/row pairs.

    """
    fpath = qualify_full_filepath(f"{filename}.csv", path)
    with open(fpath, "r", newline="\n") as infile:
        data = [row for row in csv.DictReader(infile)]
    return data


@log_func_edges
def load_dataset_df(filename, path=DATASETS):
    """Load CSV dataset into memory from path as DF.

    Parameters
    ----------
    filename : str
        Name of file (without extension) to load.
    path : str, optional (default=DATASETS)
        Path to csv to load filename from.

    Returns
    -------
    data : <object> pandas.DataFrame
        Loaded dataset as a pandas DataFrame.

    """
    data = pd.DataFrame(load_dataset_csv(filename, path))
    data = data.apply(lambda c: pd.to_numeric(c, errors="ignore"))
    return data


@log_func_edges
def load_best_model_json():
    """Load JSON of best models by dataset.

    Returns
    -------
    models : dict
        Dict of current best models.

    """
    with open(qualify_full_filepath(f"models.json", HERE), "r") as infile:
        models = json.load(infile)
    return models


@log_func_edges
def update_best_model_json(updated):
    """Update JSON of best models by dataset.

    Parameters
    ----------
    updated : dict
        Dict of updated model/datasets params.

    """
    with open(qualify_full_filepath(f"models.json", HERE), "w") as outfile:
        json.dump(updated, outfile, sort_keys=True, indent=2)
    return None


@log_func_edges
def split_test_train(data, target="class", split=0.20):
    """Split data into test and train sets."""
    np.random.seed(42)

    X = data[[c for c in list(data.columns) if c != target]]
    # y = data[target].astype("int")
    y = data[target].astype("category")

    train, test = Data(X, y), None
    if split is not None or split > 0:
        splits = train_test_split(X, y, test_size=split, stratify=y, random_state=42)
        train, test = Data(splits[0], splits[2]), Data(splits[1], splits[3])

    return train, test


@log_func_edges
def scale_test_train(train, test=None, scale=QuantileTransformer):
    """Scale train & test data sets."""

    scaler = scale()
    train = Data(scaler.fit_transform(train.X), train.y)
    test = None if test is None else Data(scaler.transform(test.X), test.y)

    return train, test


@log_func_edges
def search_model_param_space(model, params, X, y):
    """Run Cross Validated GridSearch for model on data & params."""
    np.random.seed(42)

    classes = len(np.unique(y))

    score, _score = "recall", make_scorer(recall_score)
    logging.info(f"Found {classes} unique classes, using {score} for scoring")

    grid = GridSearchCV(model, params, cv=5, scoring=_score)
    grid.fit(X, y)

    processed, estimated = (len(grid.cv_results_["params"]), np.prod([len(v) for v in params.values()]))

    if estimated != processed:
        logging.warning(f"Mismatch of processed ({processed}) and expected ({estimated}) model counts")

    if len(grid.cv_results_["mean_test_score"]) != len(grid.cv_results_["params"]):
        logging.warning(f"Mismatch output from VC results.")
        for k, v in grid.cv_results_.items():
            print(k, v)

    results = list(zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"]))
    if processed != len(results):
        logging.warning(f"Mismatch of results ({len(results)}) and processed ({processed}) model counts")

    return sorted(results, key=itemgetter(1), reverse=True), grid.best_estimator_, score


@log_func_edges
def plot_single_confusion_matrix(CM, labels, title, fname, cmap=plt.cm.Blues):
    """Visualize confusion matrix as text and graph outputs.

    Parameters
    ----------
    CM : <object> sklearn.confusion_matrix
        Confusion Matrix to visualize from sklearn.
    labels : list
        Labels of Confusion Matrix to include in visuals.
    title : str
        Title to include in grid plot.
    fname : str
        Name of dataset/model pair for file name.
    cmap : object, optional (default=plt.cm.Blues)
        Color map to use for grid plot.

    """
    print("Confusion Matrix")
    print(labels, "\n", CM)

    plt.figure()
    plt.imshow(CM, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=10)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)

    for i, j in product(range(CM.shape[0]), range(CM.shape[1])):
        color = "white" if CM[i, j] > CM.max() * 0.5 else "black"
        plt.text(j, i, format(CM[i, j], "d"), horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")

    outpath = os.path.join(BASEDIR, "img", f"confusion-matrix-{fname}.png")
    plt.savefig(outpath)

    return None


@log_func_edges
def _run_fn_gridsearch(clfname, datasets, targets, fnsearch):
    """Run gridsearch over dataset for fn."""
    for dataset, target in zip(datasets, targets):

        logging.info(f"Initializing {clfname} search for {dataset}...")
        data = load_dataset_df(dataset)

        train, test = split_test_train(data, target)
        train, test = scale_test_train(train, test)
        results, best, scorer = fnsearch(*train)

        models = load_best_model_json()
        if dataset not in models:
            models[dataset] = dict()

        predicted = best.predict(test.X)
        tested = recall_score(test.y, predicted)

        models[dataset][clfname] = {
            "params": best.get_params(),
            "train": results[0][1],
            "test": tested,
            "scorer": scorer,
        }

        update_best_model_json(models)

        print(f"{'*'*20} Running {clfname} GridSearch for {dataset} {'*'*20}")
        print(round(data[target].value_counts() / len(data[target]) * 100, 1), "\n")

        print(f"{'*'*10} Best fit {clfname} model {'*'*10}")
        print(best, "\n")

        print(f"Score of best fit on test is: {round(tested, 5)}\n")

        print(f"{'*'*10} Top N model params {'*'*10}")
        for index in range(min(15, len(results))):
            print(f"{index+1:02}. {results[index][0]} == {round(results[index][1], 5)}")

    return None


@log_func_edges
def _plot_confusion_matrix(clfname, clfobj, datasets, targets, cmap=plt.cm.Blues):
    """Plot CM for datasets using 'best' model params."""

    # TODO: Make this plot look better.. Kind of sucks

    best = load_best_model_json()
    plt.figure(figsize=(8, 4))

    subindex = 0
    for index, dataset in enumerate(datasets):
        subindex += 1

        data = load_dataset_df(dataset)
        labels = list(data[targets[index]].unique())

        train, test = split_test_train(data, targets[index], split=0.20)
        train, test = scale_test_train(train, test)

        logging.info(f"Plotting matrix for {dataset} with {clfname}...")

        if clfname not in best[dataset]:
            logging.warning(f"No model ({clfname}) found for {dataset}, ignoring...")
            continue

        clf = clfobj(**best[dataset][clfname]["params"])
        clf.fit(train.X, train.y)
        CM = confusion_matrix(test.y, clf.predict(test.X), labels)

        plt.subplot(np.ceil(len(datasets) / 2), 2, subindex)
        plt.imshow(CM, interpolation="nearest", cmap=cmap)
        plt.title(dataset, fontsize=10)
        plt.colorbar()
        plt.grid()

        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels)
        plt.yticks(ticks, labels)

        for i, j in product(range(CM.shape[0]), range(CM.shape[1])):
            color = "white" if CM[i, j] > CM.max() * 0.5 else "black"
            plt.text(j, i, format(CM[i, j], "d"), horizontalalignment="center", color=color)

        plt.ylabel("True Labels")
        plt.xlabel("Predicted Labels")

    plt.tight_layout()

    outpath = os.path.join(BASEDIR, "img", f"confusion-matrix-{clfname}.png")
    plt.savefig(outpath)

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(module)s - %(levelname)s - %(message)s")
