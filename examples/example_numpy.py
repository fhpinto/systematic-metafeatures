import pandas as pd

from metafeatures.core.object_analyzer import analyze_pd_dataframe
from metafeatures.meta_functions.entropy import Entropy
from metafeatures.meta_functions import basic as basic_meta_functions
from metafeatures.post_processing_functions.basic import Mean, \
    StandardDeviation, Skew, Kurtosis


# Load a dataset
data = pd.read_csv('../datasets/weather_year.csv')
data, attributes = analyze_pd_dataframe(data, ['Events'])
# Remove information

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
    for index, meta_information in attributes.items():
        column_type = meta_information['type']
        if column_type == 'categorical':
            raw_value = mfc(data[:, index])[0]
            values.append(raw_value)

    for pps_name, pps in post_processing_steps.items():
        metafeature_name = '%s:%s' % (name, pps_name)
        value = pps(values)[0]
        meta_features[metafeature_name] = value

for name, mfc in meta_functions_numerical.items():
    values = []
    for index, meta_information in attributes.items():
        column_type = meta_information['type']
        if column_type == 'numerical':
            raw_value = mfc(data[:, index])[0]
            values.append(raw_value)

    for pps_name, pps in post_processing_steps.items():
        metafeature_name = '%s:%s' % (name, pps_name)
        value = pps(values)[0]
        meta_features[metafeature_name] = value

print(len(meta_features))
print(meta_features)
