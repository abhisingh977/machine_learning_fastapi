from fastapi import FastAPI
import uvicorn
from typing import List, Union, Dict
import pandas as pd
from constants import train_unique, load_model, variables
from function import create_dummies, preprocess_data, transform_data_to_dict, transform_data_to_dataframe
from schemas import InputData
from pydantic import ValidationError


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

    list_response = []
    response = {}

    #we will say the cutoff is at the 75th percentile.  F
    # or the API, please return the predicted outcome (variable name is business_outcome),
    #  predicted probability (variable name is phat),
    #  and all model inputs;
    #  the variables should be returned in alphabetical order in the API return as json.
    value = model.predict(final_input).tolist()

    for i in range(len(value)):
        if value[i] > 0.75:
            response["business_outcome"] = 1

        else:
            response["business_outcome"] = 0

        response["phat"] = value[i]
        final_input_dict = final_input.iloc[i].to_dict()

        # add the final_input_dict to the response dictionary
        response.update(final_input_dict)
        
        # sort the dictionary by key alphabetically
        sorted_response = {k: response[k] for k in sorted(response)}

        list_response.append(sorted_response)

    return list_response

