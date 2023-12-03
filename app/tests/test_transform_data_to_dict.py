import pytest
from tests.data.sample_input import mock_input_data, list_of_mock_input_data
from function import transform_data_to_dict

# test for transform_data_to_dict fuction
def test_transform_data_to_dict_mock_input_data():
    result = transform_data_to_dict(mock_input_data)
    assert isinstance(result, dict)
    assert len(result) == 100

# test for transform_data_to_dict fuction for list mock_input_data
def test_transform_data_to_dict_list_of_mock_input_data():
    result = transform_data_to_dict(list_of_mock_input_data)
    assert isinstance(result, list)
    assert len(result) == 5
