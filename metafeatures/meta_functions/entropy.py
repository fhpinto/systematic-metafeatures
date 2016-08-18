from __future__ import division
import numpy as np
import scipy.stats

from metafeatures.meta_functions.base import MetaFunction


class Entropy(MetaFunction):

    def get_numerical_arity(self):
        return 0

    def get_categorical_arity(self):
        return 1

    def get_output_type(self):
        return 'numerical'

    def get_matrix_applicable(self):
        return False

    def _calculate(self, input):
        # TODO use scipy.stats.entropy
        return scipy.stats.entropy(input, base=2)

        labels_counted = input.value_counts()._values
        n_labels = len(labels_counted)

        if n_labels <= 1:
            return 0

        counts = np.bincount(labels_counted)
        probs = counts[np.nonzero(counts)] / n_labels
        n_classes = len(probs)

        if n_classes <= 1:
            return 0
        return - np.sum(probs * np.log(probs)) / np.log(n_classes)
