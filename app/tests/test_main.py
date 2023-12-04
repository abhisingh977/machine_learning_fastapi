from fastapi.testclient import TestClient
from tests.data.sample_input import dict_data, list_dict_data
from main import app

import pytest

@pytest.fixture
def client():
    return TestClient(app)


# write a test function for the endpoint read_root
def test_read_root(client):
    # call the function
    response = client.get("/")
    # assert the status code
    assert response.status_code == 200
    # Use the follow method to handle redirects
    
    # assert the response text
    assert response.json() == {"message": "api is up and running"}



# write a test function for the endpoint get_prediction_from_data dict_data
def test_get_prediction_dict_data(client):
    # call the function
    response = client.post("/predict", json=dict_data)
    # assert the status code
    assert response.status_code == 200
    # assert the response text
    # Assert the response content
    response_data = response.json()
    
    assert len(response_data) == 1

    # Example assertions for specific values
    assert response_data[0]["business_outcome"] == 0
    assert response_data[0]["phat"] == 0.36432704992115167
    assert response_data[0]["x12"] == 0.9570362947692982
    assert response_data[0]["x31_asia"] == 0.0
    assert response_data[0]["x31_germany"] == 1.0
    assert response_data[0]["x31_japan"] == 0.0



# write a test function for the endpoint get_prediction_from_data list_dict_data
def test_get_prediction_list_dict_data(client):
    # call the function
    response = client.post("/predict", json=list_dict_data)
    # assert the status code
    assert response.status_code == 200
    # assert the response text
    # Assert the response content
    response_data = response.json()

    assert len(response_data) == 6
    
    # Example assertions for specific values
    assert response_data[0]["business_outcome"] == 0
    assert response_data[0]["phat"] == 0.36432704992115167
    assert response_data[0]["x12"] == 0.9570362947692982
    assert response_data[0]["x31_asia"] == 0.0
    assert response_data[0]["x31_germany"] == 1.0
    assert response_data[0]["x31_japan"] == 0.0

    # Example assertions for specific values
    assert response_data[1]["business_outcome"] == 1
    assert response_data[1]["phat"] == 0.8227526744727083
    assert response_data[1]["x12"] == -0.9529553121992512
    assert response_data[1]["x31_asia"] == 0.0
    assert response_data[1]["x31_germany"] == 0.0
    assert response_data[1]["x31_japan"] == 0.0