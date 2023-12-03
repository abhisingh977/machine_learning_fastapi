from function import create_dummies, preprocess_data, transform_data_to_dict
import pytest
from tests.data.sample_input import list_of_mock_input_data, mock_input_data
import pandas as pd
from constants import train_unique

# write test create_dummies for dict input data
def test_create_dummies_mock_input_data():
    data = transform_data_to_dict(mock_input_data)

    # read list of dictionaries into a pandas dataframe
    data = pd.json_normalize(data)

    # merge with train_unique and data df
    data = pd.concat([train_unique, data], axis=0, ignore_index=True)

    test_imputed_std = preprocess_data(data)

    result = create_dummies(data, test_imputed_std)

    assert result.shape == (839, 121)

    assert isinstance(result, pd.DataFrame)


# write test create_dummies for list of dict input data
def test_create_dummies_list_of_mock_input_data():

    data = transform_data_to_dict(list_of_mock_input_data)
    # read list of dictionaries into a pandas dataframe
    data = pd.json_normalize(data)

    # merge with train_unique and data df
    data = pd.concat([train_unique, data], axis=0, ignore_index=True)

    test_imputed_std = preprocess_data(data)

    result = create_dummies(data, test_imputed_std)

    assert result.shape == (843, 121)
    
    assert isinstance(result, pd.DataFrame)
