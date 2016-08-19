import pandas as pd
from metafeatures.core.object_analyzer import analyze_pd_dataframe
#Load a dataset
data = pd.read_csv('datasets/weather_year.csv')

#Functions that characterize a pandas dataframe - might be useful for object characterization
data.get_dtype_counts()
data.select_dtypes(['object'])

#Entropy example
data, attributes = analyze_pd_dataframe(data, ['Events'])
