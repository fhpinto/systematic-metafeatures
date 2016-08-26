import pandas as pd
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions.basic import Kurtosis
from metafeatures.meta_functions.pearson_correlation import PearsonCorrelation
from metafeatures.meta_functions.mutual_information import MutualInformation
from metafeatures.meta_functions.spearman_correlation import SpearmanCorrelation
from metafeatures.post_processing_functions.basic import Mean
from metafeatures.post_processing_functions.basic import StandardDeviation
from metafeatures.post_processing_functions.basic import NonAggregated
from metafeatures.core.engine import metafeature_generator
#from sklearn.metrics.cluster import normalized_mutual_info_score
#from metafeatures.core.object_analyzer import analyze_pd_dataframe
#from metafeatures.core.object_to_mf_mapper import map_object_to_mf

#Load a dataset in Pandas DataFrame format
data = pd.read_csv('../datasets/weather_year.csv')

#Instantiate metafunctions and post-processing functions
entropy = Entropy()
kurtosis = Kurtosis()
correlation = PearsonCorrelation()
mutual_information =  MutualInformation()
scorrelation = SpearmanCorrelation()
_mean = Mean()
_sd = StandardDeviation()
_nagg = NonAggregated()

#Run experiments
metafeatures_values, metafeatures_names = metafeature_generator(
    data, # Pandas Dataframe
    ['Events'], # Name of the target variable
    [mutual_information, entropy, correlation, scorrelation, kurtosis], # Metafunctions
    [_mean, _sd, _nagg] # Post-processing functions
)
print(metafeatures_names)
print(metafeatures_values)
#data_numpy, attributes = analyze_pd_dataframe(data, ['Events'])
#ticketsFeatures, ticketsLabels, ticketsFeaturesLabels = map_object_to_mf(attributes, entropy)
