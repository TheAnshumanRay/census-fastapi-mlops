from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference


app = FastAPI()


# Load model artifacts
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "..", "model", "model.pkl")
)

encoder = joblib.load(
    os.path.join(BASE_DIR, "..", "model", "encoder.pkl")
)

lb = joblib.load(
    os.path.join(BASE_DIR, "..", "model", "lb.pkl")
)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Pydantic schema
class CensusData(BaseModel):

    age: int = Field(example=37)

    workclass: str = Field(
        example="Private"
    )

    fnlgt: int = Field(example=284582)

    education: str = Field(
        example="Bachelors"
    )

    education_num: int = Field(
        alias="education-num",
        example=13
    )

    marital_status: str = Field(
        alias="marital-status",
        example="Married-civ-spouse"
    )

    occupation: str = Field(
        example="Exec-managerial"
    )

    relationship: str = Field(
        example="Husband"
    )

    race: str = Field(
        example="White"
    )

    sex: str = Field(
        example="Male"
    )

    capital_gain: int = Field(
        alias="capital-gain",
        example=0
    )

    capital_loss: int = Field(
        alias="capital-loss",
        example=0
    )

    hours_per_week: int = Field(
        alias="hours-per-week",
        example=40
    )

    native_country: str = Field(
        alias="native-country",
        example="United-States"
    )

    class Config:
        allow_population_by_field_name = True


@app.get("/")
def welcome():
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
def predict(data: CensusData):

    data_dict = data.dict(by_alias=True)

    df = pd.DataFrame([data_dict])

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = inference(model, X)[0]

    if prediction == 1:
        result = ">50K"
    else:
        result = "<=50K"

    return {
        "prediction": result
    }