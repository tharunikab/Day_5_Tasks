from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from schemas import PredictRequest, PredictResponse


# A simple placeholder ML model (Replace this with your real model later!)
class DummyModel:
    def predict(self, text: str) -> dict:
        if "charged" in text.lower() or "bill" in text.lower():
            return {"category": "Billing", "priority": "High", "confidence": 0.93}
        return {"category": "General", "priority": "Low", "confidence": 0.85}


# 1. Startup & Shutdown Logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything BEFORE 'yield' runs ONCE when the server starts
    print("Loading ML model into memory...")
    app.state.model = DummyModel()
    yield
    # Everything AFTER 'yield' runs when the server stops
    print("Shutting down and cleaning up...")


# 2. Create the FastAPI app and attach the lifespan
app = FastAPI(lifespan=lifespan)


# 3. Health Check Endpoint (GET = asking for information)
@app.get("/health")
def health_check():
    return {"status": "ok"}


# 4. Prediction Endpoint (POST = sending data to be processed)
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    # Grab the pre-loaded model from app.state
    model = request.app.state.model
    result = model.predict(payload.text)
    return result
