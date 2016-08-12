import numpy as np
import pandas as pd

from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.post_processing_functions.basic import Mean, \
    StandardDeviation, Skew, Kurtosis


# Load a dataset
data = pd.read_csv('../datasets/weather_year.csv')
columns = data.columns
columns = [column_name.replace(' ', '') for column_name in columns]
data.columns = columns

categorical = ['PrecipitationIn', 'Events']
categorical = [True if column_name in categorical else False
               for column_name in data.columns]
for i, cat in enumerate(categorical):
    if cat:
        column = data.iloc[:, i]
        unique_values = column.unique()
        mapping = {uv: j for j, uv in enumerate(unique_values)}
        column = column.replace(mapping).astype(float)
        data.iloc[:, i] = column

X = data.iloc[:, 1:-1].astype(float).values
categorical = categorical[1:-1]
y = data.iloc[:, -1].astype(float).values

post_processing_steps = {'mean': Mean(),
                         'std': StandardDeviation(),
                         'skew': Skew(),
                         'kurtosis': Kurtosis()}
meta_functions_categorical = {'entropy': Entropy()}
meta_functions_numerical = {'mean': basic_meta_functions.Mean(),
                            'std': basic_meta_functions.StandardDeviation(),
                            'skew': basic_meta_functions.Skew(),
                            'kurtosis': basic_meta_functions.Kurtosis()}

meta_features = {}


for name, mfc in meta_functions_categorical.items():
    values = []
    for i, cat in enumerate(categorical):
        if cat:
            raw_value = mfc(X[:, i])[0]
            values.append(raw_value)

    for pps_name, pps in post_processing_steps.items():
        metafeature_name = '%s:%s' % (name, pps_name)
        value = pps(values)[0]
        meta_features[metafeature_name] = value

for name, mfc in meta_functions_numerical.items():
    values = []
    for i, cat in enumerate(categorical):
        if cat:
            raw_value = mfc(X[:, i])[0]
            values.append(raw_value)

    for pps_name, pps in post_processing_steps.items():
        metafeature_name = '%s:%s' % (name, pps_name)
        value = pps(values)[0]
        meta_features[metafeature_name] = value

print(len(meta_features))
print(meta_features)
