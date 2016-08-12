import abc
import time
import scipy.sparse


class PostProcessing(object):
    """Abstract post-processing function."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_input_types(self):
        pass

    @abc.abstractmethod
    def get_output_types(self):
        pass

    @abc.abstractmethod
    def _calculate(self, input):
        pass

    def __call__(self, input):
        starttime = time.time()

        try:
            value = self._calculate(input)
            comment = ""
        except MemoryError as e:
            value = None
            comment = "Memory Error"

        endtime = time.time()
        calculation_time = endtime - starttime
        return value, calculation_time, comment