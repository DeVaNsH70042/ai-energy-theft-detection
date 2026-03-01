import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

MODEL = None


def train_model():
    global MODEL

    file_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "baseline_normal_data.xlsx"
    )

    df = pd.read_excel(file_path)

    X = df["mismatch_ratio"].values.reshape(-1, 1)

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    model.fit(X)

    MODEL = model


def score_reading(mismatch_value):
    global MODEL

    if MODEL is None:
        return 0

    prediction = MODEL.predict(np.array([[mismatch_value]]))

    if prediction[0] == -1:
        return 1
    else:
        return 0