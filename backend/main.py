from src.scripts.predict import predict, predict_from_csv, predict_next_month
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class PredictionRequest(BaseModel):
    past_values: list[float]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    return predict(request.past_values)


@app.get("/predict-csv")
def predict_csv_endpoint():
    return predict_from_csv()


@app.get("/predict-next-month")
def predict_next_month_endpoint():
    return predict_next_month()
