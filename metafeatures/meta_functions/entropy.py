from __future__ import division
import numpy as np
#import scipy.stats

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
        #return scipy.stats.entropy(input, base=2)

        probs = [np.mean(input == c) for c in set(input)]
        return np.sum(-p * np.log2(p) for p in probs)
