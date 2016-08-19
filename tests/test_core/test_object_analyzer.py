import os
import unittest

import numpy as np
import pandas as pd

import metafeatures.core.object_analyzer


class TestObjectAnalyzer(unittest.TestCase):

    def setUp(self):
        current_directory = os.path.dirname(__file__)
        data_path = os.path.join(current_directory,
                                 '../../datasets/weather_year.csv')
        self.data = pd.read_csv(data_path)

    def test_object_analyzer(self):
        data, metadata = metafeatures.core.object_analyzer.analyze_pd_dataframe(
            self.data, None)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.dtype, np.float)
        self.assertIsInstance(metadata, dict)

    def test__normalize_pd_column_names(self):
        data = metafeatures.core.object_analyzer._normalize_pd_column_names(
            self.data)
        self.assertEqual(list(data.columns),
                         ['EDT', 'MaxTemperatureF', 'MeanTemperatureF',
                          'MinTemperatureF', 'MaxDewPointF', 'MeanDewPointF',
                          'MinDewpointF', 'MaxHumidity', 'MeanHumidity',
                          'MinHumidity', 'MaxSeaLevelPressureIn',
                          'MeanSeaLevelPressureIn', 'MinSeaLevelPressureIn',
                          'MaxVisibilityMiles', 'MeanVisibilityMiles',
                          'MinVisibilityMiles', 'MaxWindSpeedMPH',
                          'MeanWindSpeedMPH', 'MaxGustSpeedMPH',
                          'PrecipitationIn', 'CloudCover', 'Events',
                          'WindDirDegrees'])

    def test__get_pd_attribute_types(self):
        # Test for int
        data = pd.DataFrame([{'column1': 1, 'column2': 2},
                             {'column1': 3, 'column2': 4}])
        attributes = metafeatures.core.object_analyzer\
            ._get_pd_attribute_types(data, None)
        self.assertEqual(attributes, {0: {'type': 'numerical',
                                          'name': 'column1',
                                          'is_target': False},
                                      1: {'type': 'numerical',
                                          'name': 'column2',
                                          'is_target': False}})

        # Test for float
        data = pd.DataFrame([{'column1': 1.3, 'column2': 2.7},
                             {'column1': 3.2, 'column2': 3.9}])
        attributes = metafeatures.core.object_analyzer \
            ._get_pd_attribute_types(data, None)
        self.assertEqual(attributes, {0: {'type': 'numerical',
                                          'name': 'column1',
                                          'is_target': False},
                                      1: {'type': 'numerical',
                                          'name': 'column2',
                                          'is_target': False}})

        # Test for object
        data = pd.DataFrame([{'column1': 1.3, 'column2': "Bla"},
                             {'column1': 3.2, 'column2': "Bla"}])
        attributes = metafeatures.core.object_analyzer \
            ._get_pd_attribute_types(data, None)
        self.assertEqual(attributes, {0: {'type': 'numerical',
                                          'name': 'column1',
                                          'is_target': False},
                                      1: {'type': 'categorical',
                                          'name': 'column2',
                                          'is_target': False}})

    def test__get_pd_attribute_types_target_types(self):
        data = pd.DataFrame([{'column1': 1, 'column2': 2},
                             {'column1': 3, 'column2': 4}])
        attributes = metafeatures.core.object_analyzer \
            ._get_pd_attribute_types(data, [0])
        print(attributes)
        self.assertTrue(attributes[0]['is_target'])
        self.assertFalse(attributes[1]['is_target'])

        data = pd.DataFrame([{'column1': 1, 'column2': 2},
                             {'column1': 3, 'column2': 4}])
        attributes = metafeatures.core.object_analyzer \
            ._get_pd_attribute_types(data, 1)
        self.assertFalse(attributes[0]['is_target'])
        self.assertTrue(attributes[1]['is_target'])

        data = pd.DataFrame([{'column1': 1, 'column2': 2},
                             {'column1': 3, 'column2': 4}])
        attributes = metafeatures.core.object_analyzer \
            ._get_pd_attribute_types(data, [])
        self.assertFalse(attributes[0]['is_target'])
        self.assertFalse(attributes[1]['is_target'])

        data = pd.DataFrame([{'column1': 1, 'column2': 2},
                             {'column1': 3, 'column2': 4}])
        attributes = metafeatures.core.object_analyzer \
            ._get_pd_attribute_types(data, [0, 1])
        self.assertTrue(attributes[0]['is_target'])
        self.assertTrue(attributes[1]['is_target'])

        data = pd.DataFrame([{'column1': 1, 'column2': 2},
                             {'column1': 3, 'column2': 4}])
        attributes = metafeatures.core.object_analyzer \
            ._get_pd_attribute_types(data, ['column1'])
        self.assertTrue(attributes[0]['is_target'])
        self.assertFalse(attributes[1]['is_target'])

        data = pd.DataFrame([{'column1': 1, 'column2': 2},
                             {'column1': 3, 'column2': 4}])
        attributes = metafeatures.core.object_analyzer \
            ._get_pd_attribute_types(data, 'column2')
        self.assertFalse(attributes[0]['is_target'])
        self.assertTrue(attributes[1]['is_target'])

    def test__replace_objects_by_integers(self):
        data = pd.DataFrame([{'column1': 1.3, 'column2': "Bla"},
                             {'column1': 3.2, 'column2': "Bla"},
                             {'column1': 2.7, 'column2': "Aha"}])
        data = metafeatures.core.object_analyzer \
            ._replace_objects_by_integers(data,
                                          {0: {'type': 'numerical',
                                               'name': 'column1',
                                               'is_target': False},
                                           1: {'type': 'categorical',
                                               'name': 'column2',
                                               'is_target': False}})
        print(data)
        self.assertEqual(data.dtypes[0], np.float)
        self.assertEqual(data.dtypes[1], np.float)
        np.testing.assert_allclose(data.iloc[:, 1], [0, 0, 1])