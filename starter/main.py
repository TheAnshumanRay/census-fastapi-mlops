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
    " workclass",
    " education",
    " marital-status",
    " occupation",
    " relationship",
    " race",
    " sex",
    " native-country",
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

    education_num: int = Field(alias="education-num")

    marital_status: str = Field(alias="marital-status")

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

    capital_gain: int = Field(alias="capital-gain")

    capital_loss: int = Field(alias="capital-loss")

    hours_per_week: int = Field(alias="hours-per-week")

    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True


@app.get("/")
def welcome():
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
def predict(data: CensusData):

    data_df = pd.DataFrame([data.dict(by_alias=True)])

    X, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = inference(model, X)

    result = lb.inverse_transform(prediction)[0]

    return {"prediction": result}