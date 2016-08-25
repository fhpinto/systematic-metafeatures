import pandas as pd
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions.basic import Kurtosis
from metafeatures.post_processing_functions.basic import Mean
from metafeatures.post_processing_functions.basic import StandardDeviation
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator


#Load a dataset in Pandas DataFrame format
data = pd.read_csv('../datasets/weather_year.csv')

#Instantiate metafunctions and post-processing functions
entropy = Entropy()
kurtosis = Kurtosis()
_mean = Mean()
_sd = StandardDeviation()
_nagg = NonAggregated()

#Run experiments
result = metafeature_generator(data, ['Events'], [kurtosis, entropy], [_mean, _sd, _nagg])
print(result)


