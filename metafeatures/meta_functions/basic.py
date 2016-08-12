import numpy as np
import scipy.stats

from metafeatures.meta_functions.base import MetaFunction



class Mean(MetaFunction):
    def __init__(self):
        """Computes the mean of a set of values."""
        pass

    def get_input_types(self):
        return True

    def get_output_types(self):
        return True

    def _calculate(self, input):
        return np.mean(input)


class StandardDeviation(MetaFunction):
    def __init__(self):
        """Computes the standard deviation of a set of values."""
        pass

    def get_input_types(self):
        return True

    def get_output_types(self):
        return True

    def _calculate(self, input):
        return np.std(input)


class Kurtosis(MetaFunction):
    def __init__(self):
        """Computes the kurtosis of a set of values."""
        pass

    def get_input_types(self):
        return True

    def get_output_types(self):
        return True

    def _calculate(self, input):
        return scipy.stats.kurtosis(input)


class Skew(MetaFunction):
    def __init__(self):
        """Computes the skew of a set of values."""
        pass

    def get_input_types(self):
        return True

    def get_output_types(self):
        return True

    def _calculate(self, input):
        return scipy.stats.skew(input)