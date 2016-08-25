import pandas as pd
import numpy as np
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions.basic import Kurtosis
from metafeatures.meta_functions.pearson_correlation import PearsonCorrelation
from metafeatures.post_processing_functions.basic import Mean
from metafeatures.post_processing_functions.basic import StandardDeviation
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator
from scipy.stats import pearsonr
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.core.object_to_mf_mapper import map_object_to_mf


#Load a dataset in Pandas DataFrame format
data = pd.read_csv('../datasets/weather_year.csv')

#Instantiate metafunctions and post-processing functions
entropy = Entropy()
kurtosis = Kurtosis()
correlation = PearsonCorrelation()
_mean = Mean()
_sd = StandardDeviation()
_nagg = NonAggregated()

#Run experiments
result = metafeature_generator(data, ['Events'], [correlation, entropy], [_mean, _sd, _nagg])
print(result)
#data_numpy, attributes = analyze_pd_dataframe(data, ['Events'])
#ticketsFeatures, ticketsLabels, ticketsFeaturesLabels = map_object_to_mf(attributes, correlation)
#print(entropy._calculate(data_numpy[:,0]))
#print(ticketsFeatures[0])
#print(data_numpy[:,ticketsFeatures[0]][:,0])
#cols = [1,2]
#print(data_numpy[:,cols][:,0])