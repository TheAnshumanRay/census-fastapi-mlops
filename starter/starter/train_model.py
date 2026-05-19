# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")
data.columns = data.columns.str.strip()
data = data.applymap(
    lambda x: x.strip() if isinstance(x, str) else x
)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(
    y_test,
    preds
)

print("Precision:", precision)
print("Recall:", recall)
print("Fbeta:", fbeta)

joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")

with open("slice_output.txt", "w") as file:

    for feature in cat_features:

        for value in test[feature].unique():

            slice_df = test[test[feature] == value]

            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds_slice = inference(model, X_slice)

            precision, recall, fbeta = compute_model_metrics(
                y_slice,
                preds_slice
            )

            result = (
                f"Feature: {feature}={value}\n"
                f"Precision: {precision:.3f}\n"
                f"Recall: {recall:.3f}\n"
                f"Fbeta: {fbeta:.3f}\n\n"
            )

            file.write(result)