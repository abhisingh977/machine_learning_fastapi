from app.function import preprocess_data, transform_data_to_dict
import pytest
from tests.data.sample_input import mock_input_data, list_of_mock_input_data
import pandas as pd
from app.constants import subset_train_data_for_prediction


def test_preprocess_dict_data():
    data = transform_data_to_dict(mock_input_data)

    # read list of dictionaries into a pandas dataframe
    data = pd.json_normalize(data)

    # merge with subset_train_data_for_prediction and data df
    data = pd.concat(
        [subset_train_data_for_prediction, data], axis=0, ignore_index=True
    )

    test_imputed_std = preprocess_data(data)

    assert test_imputed_std.shape == (839, 96)
    assert test_imputed_std["x12"].dtype == float
    assert test_imputed_std["x63"].dtype == float


def test_preprocess_list_dict_data():
    data = transform_data_to_dict(list_of_mock_input_data)

    # read list of dictionaries into a pandas dataframe
    data = pd.json_normalize(data)

    # merge with subset_train_data_for_prediction and data df
    data = pd.concat(
        [subset_train_data_for_prediction, data], axis=0, ignore_index=True
    )

    test_imputed_std = preprocess_data(data)

    assert test_imputed_std.shape == (843, 96)
    assert test_imputed_std["x12"].dtype == float
    assert test_imputed_std["x63"].dtype == float
