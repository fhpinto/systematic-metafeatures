import pandas as pd

from metafeatures.meta_functions.entropy import entropy

#Load a dataset
data = pd.read_csv('datasets/weather_year.csv')

#Functions that characterize a pandas dataframe - might be useful for object characterization
data.get_dtype_counts()
data.select_dtypes(['object'])

#Entropy example
print entropy(data['PrecipitationIn'])
