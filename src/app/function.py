import pandas as pd
from app.constants import imputer, std_scaler, subset_train_data_for_prediction
from typing import List, Dict, Union
from app.schemas import InputData
import logging

# Get the logger for this module
logger = logging.getLogger(__name__)

def make_prediction(model, final_input: pd.DataFrame) -> List[Dict]:
    # cutoff is at the 75th percentile.  
    #  predicted outcome (variable name is business_outcome),
    #  predicted probability (variable name is phat),
    #  and all model inputs
    #  the variables should be returned in alphabetical order in the API return as json.

    list_response = []
    response = {}

    value = model.predict(final_input).tolist()

    for i in range(len(value)):
        if value[i] >= 0.75:
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


def transform_data_to_dataframe(data: Union[List[Dict], Dict]) -> pd.DataFrame:
    # read list of dictionaries into a pandas dataframe
    data = pd.json_normalize(data)
    logging.info(f"size of input: {len(data)}")
    # merge with subset_train_data_for_prediction and data df
    data = pd.concat([subset_train_data_for_prediction, data], axis=0, ignore_index=True)
    return data


def transform_data_to_dict(
    data: Union[InputData, List[InputData]]
) -> Union[List[Dict], Dict]:
    if isinstance(data, list):
        # Handle the case where data is a list
        result = []
        for item in data:
            result.append(item.dict())
        return result
    else:
        # Handle the case where data is an InputData object
        return data.dict()


def create_dummies(data: pd.DataFrame, test_imputed_std: pd.DataFrame) -> pd.DataFrame:
    dumb5 = pd.get_dummies(
        data["x5"], drop_first=True, prefix="x5", prefix_sep="_", dummy_na=True
    )
    test_imputed_std = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)
    dumb31 = pd.get_dummies(
        data["x31"], drop_first=True, prefix="x31", prefix_sep="_", dummy_na=True
    )
    test_imputed_std = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(
        data["x81"], drop_first=True, prefix="x81", prefix_sep="_", dummy_na=True
    )
    test_imputed_std = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(
        data["x82"], drop_first=True, prefix="x82", prefix_sep="_", dummy_na=True
    )
    test_imputed_std = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)

    del dumb5, dumb31, dumb81, dumb82
    
    logging.info(f"size of input after dummies: {test_imputed_std.shape}")
    return test_imputed_std


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # apply same changes to my json of x0, x1, ..., x99 values that are applied to the training data train_val

    data["x12"] = data["x12"].str.replace("$", "", regex=True)
    data["x12"] = data["x12"].str.replace(":", "", regex=True)
    data["x12"] = data["x12"].str.replace(",", "", regex=True)
    data["x12"] = data["x12"].str.replace(")", "", regex=True)
    data["x12"] = data["x12"].str.replace("(", "-", regex=True)
    data["x12"] = data["x12"].astype(float)
    data["x63"] = data["x63"].str.replace("%", "", regex=True)
    data["x63"] = data["x63"].astype(float)

    new_data_dropped = data.drop(columns=["x5", "x31", "x81", "x82"])
    test_imputed = pd.DataFrame(
        imputer.transform(new_data_dropped), columns=new_data_dropped.columns
    )
    test_imputed_std = pd.DataFrame(
        std_scaler.transform(test_imputed), columns=test_imputed.columns
    )

    return test_imputed_std
