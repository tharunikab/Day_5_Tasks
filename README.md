# Support Ticket Classifier API

A FastAPI service that classifies customer support tickets into a **category**, **priority**, and **confidence** score.

## Features
- **Single Startup Model Load:** Loads the ML model once at server startup using FastAPI `lifespan`.
- **Pydantic Input Validation:** Rejects empty or whitespace-only text with a `422 Unprocessable Entity` error.
- **Health Monitoring & Tests:** Includes `GET /health` and automated `pytest` tests using `TestClient`.

## Setup & Run Steps

1. **Clone the repository and enter the project folder:**
   ```bash
   git clone <your-repo-url>
   cd ticket-api
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```
   The API will run at `http://127.0.0.1:8000` (Interactive Swagger UI at `http://127.0.0`).

## Running Automated Tests

Run the `TestClient` suite with `pytest`:
```bash
pytest -v
```

## Example `curl` Commands

### 1. Health Check (`GET /health`)
```bash
curl -X GET http://127.0.0
```
**Expected Response:**
```json
{"status":"ok"}
```

### 2. Predict Ticket Category & Priority (`POST /predict`)
```bash
curl -X POST http://127.0.0 \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I was charged twice\"}"
```
**Expected Response:**
```json
{"category":"Billing","priority":"High","confidence":0.93}
```

### 3. Validation Error Check (Empty Input -> `422`)
```bash
curl -i -X POST http://127.0.0 \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"\"}"
```
