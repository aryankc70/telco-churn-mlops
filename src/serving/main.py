"""FastAPI application serving the Telco churn model."""
from fastapi import FastAPI, HTTPException
from src.serving.schemas import CustomerData, PredictionResponse
from src.serving.inference import predict_churn, load_artifacts

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/")
def health_check():
    return {"status": "ok", "service": "telco-churn-api"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        result = predict_churn(customer.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))