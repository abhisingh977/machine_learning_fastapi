from fastapi import FastAPI
from typing import List, Union, Dict
from app.constants import load_model, variables
from app.function import create_dummies, preprocess_data, transform_data_to_dict, transform_data_to_dataframe, make_prediction
from app.schemas import InputData

# Create an instance of the FastAPI class
app = FastAPI()

model = load_model()
# Define a simple endpoint
@app.get("/")
async def read_root():
    return {"message": "api is up and running"}


@app.post("/predict")
async def get_prediction_from_data(
    data: Union[InputData, List[InputData]],
)  -> List[Dict]:
    
    data = transform_data_to_dict(data)

    data = transform_data_to_dataframe(data)

    test_imputed_std = preprocess_data(data)

    test_imputed_std = create_dummies(data, test_imputed_std)

    user_input = test_imputed_std[838:]

    final_input = user_input[variables]

    list_response = make_prediction(model, final_input)

    return list_response
