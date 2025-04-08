"""
RUN WITH THIS:

uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from fastai.learner import load_learner
from fastai.vision.core import PILImage
import base64
import os
from dotenv import load_dotenv
import pathlib
import platform
from download_google_drive import download_google_drive_models
import uvicorn
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

load_dotenv()
API_KEY = os.getenv("API_KEY")

download_google_drive_models()

print("Available files:", os.listdir())

print("Downloaded txt size:", os.path.getsize("txt_model.pkl"))
print("Downloaded img size:", os.path.getsize("img_model.pkl"))

path = os.getcwd()

# Load Fastai Image Model
MODEL_PATH = path+"/img_model.pkl"  # Update with your model's filename

# Load Torch Text Model
TXT_MODEL_PATH = path+"/txt_model.pkl"  # Update with your model's filename

try:
    learn = load_learner(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load image model: {e}")

# Load Torch text model (assumes simple architecture)
try:
    learn_txt = load_learner(TXT_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load text model: {e}")

# API key check dependency
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized access")

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
async def predict_image(file: UploadFile = File(...), auth=Depends(verify_api_key)):
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
async def predict_base64(data: dict, auth=Depends(verify_api_key)):
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
async def predict_text(input: TextInput, auth=Depends(verify_api_key)):
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
if __name__ == "__main__":
    port = os.getenv("PORT") or 8000
    uvicorn.run(app, host="0.0.0.0", port=int(port))