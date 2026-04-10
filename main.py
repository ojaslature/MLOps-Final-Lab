import logging
import sys
import pickle
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# 1. LOGGING SETUP (Expt 3)
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_logs.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mlops-api")

# ---------------------------------------------------------
# 2. APP INITIALIZATION & CORS (Expt 9 Fix)
# ---------------------------------------------------------
app = FastAPI(title="Iris MLOps API - Versioned")

# This allows your Streamlit dashboard (localhost) to talk to AWS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 3. AUTHENTICATION CONFIG (Expt 4)
# ---------------------------------------------------------
API_KEY = "my_super_secret_mlops_key"
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_value: str = Security(api_key_header)):
    if header_value == API_KEY:
        return header_value
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

# ---------------------------------------------------------
# 4. MULTI-MODEL LOADING (Expt 7)
# ---------------------------------------------------------
try:
    with open("models/model.pkl", "rb") as f:
        model_v1 = pickle.load(f)
    with open("models/model_v2.pkl", "rb") as f:
        model_v2 = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Models V1, V2 and Scaler loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Critical Error: Model files missing. {e}")
    sys.exit(1)

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ---------------------------------------------------------
# 5. REQUEST LOGGING MIDDLEWARE (Expt 3)
# ---------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Finished Request: Status {response.status_code}")
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"message": "Invalid input type"})

# ---------------------------------------------------------
# 6. VERSIONED PREDICT ENDPOINT (Expt 7)
# ---------------------------------------------------------
@app.post("/predict/{version}")
async def predict(version: str, data: IrisRequest, api_key: str = Depends(get_api_key)):
    # Version routing logic
    if version == "v1":
        active_model = model_v1
    elif version == "v2":
        active_model = model_v2
    else:
        logger.warning(f"Invalid version requested: {version}")
        raise HTTPException(status_code=400, detail="Invalid version. Use v1 or v2.")

    # Data Processing
    input_array = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    scaled_data = scaler.transform(input_array)
    
    # Prediction
    prediction = int(active_model.predict(scaled_data)[0])
    target_names = ['setosa', 'versicolor', 'virginica']
    
    logger.info(f"Auth successful. Version: {version}. Prediction: {target_names[prediction]}")
    
    return {
        "model_version": version,
        "prediction": prediction, 
        "species": target_names[prediction]
    }