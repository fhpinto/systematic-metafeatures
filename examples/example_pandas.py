import pandas as pd
from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.core.object_to_mf_mapper import map_object_to_mf

#Load a dataset
data = pd.read_csv('../datasets/weather_year.csv')

#Functions that characterize a pandas dataframe - might be useful for object characterization
data.get_dtype_counts()
data.select_dtypes(['object'])

#Entropy example
data, attributes = analyze_pd_dataframe(data, ['Events'])
entropy = Entropy()
entropy.get_categorical_arity()

numAttr = {k for k, v in attributes.items() if v['type'] == 'numerical'}
catAttr  = {k for k, v in attributes.items() if v['type'] == 'categorical'}
regLabel = {k for k, v in attributes.items() if (v['type'] == 'numerical') and (v['is_target'] == True)}
classLabel = {k for k, v in attributes.items() if (v['type'] == 'categorical') and (v['is_target'] == True)}

a, b, c = map_object_to_mf(attributes, entropy)
print(a)
print(b)
print(c)