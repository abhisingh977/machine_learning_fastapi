from fastapi import FastAPI, HTTPException
from typing import List, Union, Dict
from app.constants import load_model, variables
from app.function import (
    create_dummies,
    preprocess_data,
    transform_data_to_dict,
    transform_data_to_dataframe,
    make_prediction,
)
from app.schemas import InputData
import logging

# Configure logging settings (optional, but recommended)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create an instance of the FastAPI class
app = FastAPI()

model = load_model()


# Define a simple endpoint
@app.get("/")
async def read_root():
    return {"message": "api is up and running"}


@app.post("/predict")
async def predict(
    data: Union[InputData, List[InputData]],
) -> List[Dict]:
    try:
        data = transform_data_to_dict(data)
        logging.info("transformed_data_to_dict")

        data = transform_data_to_dataframe(data)
        logging.info("transformed_data_to_dataframe")

        test_imputed_std = preprocess_data(data)
        logging.info("preprocessed_data")

        test_imputed_std = create_dummies(data, test_imputed_std)
        logging.info("created_dummies_to_df")

        user_input = test_imputed_std[838:]
        final_input = user_input[variables]
        list_response = make_prediction(model, final_input)
        logging.info("made_prediction!!!!")

        return list_response
    
    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))
