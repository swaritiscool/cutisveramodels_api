"""
RUN WITH THIS:

uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from fastai.learner import load_learner
from fastai.vision.core import PILImage
import base64
import torch
import torch.nn.functional as F

# Load Fastai Image Model
MODEL_PATH = "fastai_resnet_101_model_DERMNET_gradual_10.pkl"  # Update with your model's filename

# Load Torch Text Model
TXT_MODEL_PATH = "finetuned_text3.pkl"  # Update with your model's filename

try:
    learn = load_learner(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load image model: {e}")

# Load Torch text model (assumes simple architecture)
try:
    learn_txt = load_learner(TXT_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load text model: {e}")

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Skin Disease Detection API is running!"}

# Image Prediction Endpoint (File Upload)
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image = PILImage.create(BytesIO(await file.read()))
        pred, pred_idx, probs = learn.predict(image)

        return {
            "prediction": pred,
            "confidence": float(probs[pred_idx]),
            "all_labels": probs.tolist(),
            "list_order": ["Acne & Rosacea", "Atopic Dermatitis", "Bullous Disease", "Eczema", "Normal"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image Prediction Endpoint (Base64)
@app.post("/predict-base64/")
async def predict_base64(data: dict):
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Missing 'image' key")

        img_data = base64.b64decode(data["image"])
        image = PILImage.create(BytesIO(img_data))

        pred, pred_idx, probs = learn.predict(image)

        return {
            "prediction": pred,
            "confidence": float(probs[pred_idx])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic model for input
class TextInput(BaseModel):
    input_data: str

@app.post("/predict-text/")
async def predict_text(input: TextInput):
    try:
        # Use the string directly, not the whole object
        pred, pred_idx, probs = learn_txt.predict(input.input_data)

        return {
            "prediction": pred,
            "confidence": float(probs[pred_idx]),
            "all_labels": probs.tolist(),
            "list_order": [
                "Acne & Rosacea",
                "Atopic Dermatitis",
                "Bullous Disease",
                "Eczema",
                "Normal"
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text inference failed: {e}")
