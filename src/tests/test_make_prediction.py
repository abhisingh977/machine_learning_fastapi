from app.function import (
    create_dummies,
    preprocess_data,
    transform_data_to_dict,
    make_prediction,
)
import pytest
from tests.data.sample_input import list_of_mock_input_data, mock_input_data
import pandas as pd
from app.constants import subset_train_data_for_prediction
from app.constants import load_model, variables


# write test create_dummies for dict input data
def test_create_dummies_mock_input_data():
    data = transform_data_to_dict(mock_input_data)

    # read list of dictionaries into a pandas dataframe
    data = pd.json_normalize(data)

    # merge with subset_train_data_for_prediction and data df
    data = pd.concat(
        [subset_train_data_for_prediction, data], axis=0, ignore_index=True
    )

    test_imputed_std = preprocess_data(data)

    result = create_dummies(data, test_imputed_std)
    print(result.shape)
    user_input = result[838:]

    final_input = user_input[variables]

    model = load_model()

    list_response = make_prediction(model, final_input)

    assert isinstance(list_response, list)

    assert len(list_response) == 1

    # Example assertions for specific values
    assert list_response[0]["business_outcome"] == 0
    assert list_response[0]["phat"] == 0.36432704992115167
    assert list_response[0]["x12"] == 0.9570362947692982
    assert list_response[0]["x31_asia"] == 0.0
    assert list_response[0]["x31_germany"] == 1.0
    assert list_response[0]["x31_japan"] == 0.0


# write test create_dummies for list of dict input data
def test_create_dummies_list_of_mock_input_data():
    data = transform_data_to_dict(list_of_mock_input_data)
    # read list of dictionaries into a pandas dataframe
    data = pd.json_normalize(data)

    # merge with subset_train_data_for_prediction and data df
    data = pd.concat(
        [subset_train_data_for_prediction, data], axis=0, ignore_index=True
    )

    test_imputed_std = preprocess_data(data)

    result = create_dummies(data, test_imputed_std)

    user_input = result[838:]

    final_input = user_input[variables]

    model = load_model()

    list_response = make_prediction(model, final_input)
    assert isinstance(list_response, list)

    assert len(list_response) == 5

    # Example assertions for specific values
    assert list_response[0]["business_outcome"] == 0
    assert list_response[0]["phat"] == 0.36432704992115167
    assert list_response[0]["x12"] == 0.9570362947692982
    assert list_response[0]["x31_asia"] == 0.0
    assert list_response[0]["x31_germany"] == 1.0
    assert list_response[0]["x31_japan"] == 0.0

    # Example assertions for specific values
    assert list_response[1]["business_outcome"] == 0
    assert list_response[1]["phat"] == 0.36432704992115167
    assert list_response[1]["x12"] == 0.9570362947692982
    assert list_response[1]["x31_asia"] == 0.0
    assert list_response[1]["x31_germany"] == 1.0
    assert list_response[1]["x31_japan"] == 0.0
