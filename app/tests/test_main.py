from fastapi.testclient import TestClient
from tests.data.sample_input import dict_data, list_dict_data
from main import app

import pytest

client = TestClient(app)


# write a test function for the endpoint read_root
def test_read_root():
    # call the function
    response = client.get("/")
    # assert the status code
    assert response.status_code == 200
    # assert the response text
    assert response.json() == {"message": "api is up and running"}


# write a test function for the endpoint get_prediction_from_data
def test_get_prediction_dict_data():
    # call the function
    response = client.post("/predict", json=dict_data)
    # assert the status code
    assert response.status_code == 200
    # assert the response text

    assert response.json() == pytest.approx([0.36432704992115167])


# write a test function for the endpoint get_prediction_from_data
def test_get_prediction_list_dict_data():
    # call the function
    response = client.post("/predict", json=list_dict_data)
    # assert the status code
    assert response.status_code == 200

    # assert the response json
    assert response.json() == pytest.approx([
            0.36432704992115167,
            0.8227526744727083,
            0.1341062506470991,
            0.4696811033305632,
            0.3220713947355858,
            0.5512301005945599,
        ])
    
