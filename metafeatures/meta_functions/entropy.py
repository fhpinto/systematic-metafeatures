from __future__ import division
import numpy as np


def entropy(labels):
    labels_counted = labels.value_counts()._values
    n_labels = len(labels_counted)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels_counted)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)
