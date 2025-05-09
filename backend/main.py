# ------------------ FastAPI ------------------
from src.scripts.predict import (
    predict_next_months,
    predict_with_dataset,
    predict_last_week,
)
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class PredictionRequest(BaseModel):
    past_values: list[float]


app = FastAPI()
# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""
# ------------------ Predict Endpoint ------------------
@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    return predict(request.past_values)


# ------------------ Predict CSV Endpoint ------------------
@app.get("/predict-csv")
def predict_csv_endpoint():
    return predict_from_csv()

"""


# ------------------ Predict Next Month Endpoint ------------------
@app.get("/predict-next-month")
def predict_next_month_endpoint():
    return predict_next_months()


@app.get("/predict_with_dataset")
def predict_with_dataset_endpoint():
    return predict_with_dataset()


@app.get("/api/predict-last-week")
def predict_last_week_endpoint():
    results = predict_last_week()
    return results
