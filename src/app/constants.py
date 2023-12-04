import joblib
import pickle
import pandas as pd

# Load the train data
subset_train_data_for_prediction = pd.read_csv("required_data/subset_train_data_for_prediction.csv")

# Load the imputer and std_scaler objects
imputer = joblib.load("model/train_data_imputer.joblib")
std_scaler = joblib.load("model/train_data_std_scaler.joblib")

# Load and return the pre-trained model
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    return model


# Final features needed for prediction
variables = [
    "x5_saturday",
    "x81_July",
    "x81_December",
    "x31_japan",
    "x81_October",
    "x5_sunday",
    "x31_asia",
    "x81_February",
    "x91",
    "x81_May",
    "x5_monday",
    "x81_September",
    "x81_March",
    "x53",
    "x81_November",
    "x44",
    "x81_June",
    "x12",
    "x5_tuesday",
    "x81_August",
    "x81_January",
    "x62",
    "x31_germany",
    "x58",
    "x56",
]
