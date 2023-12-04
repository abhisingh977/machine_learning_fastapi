import pandas as pd
from app.constants import imputer, std_scaler, train_unique
from typing import List, Dict, Union
from app.schemas import InputData


def transform_data_to_dataframe(data: Union[List[Dict], Dict]) -> pd.DataFrame:
    # read list of dictionaries into a pandas dataframe
    data = pd.json_normalize(data)

    # merge with train_unique and data df
    data = pd.concat([train_unique, data], axis=0, ignore_index=True)
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
