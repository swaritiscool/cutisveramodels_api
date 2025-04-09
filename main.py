"""
RUN WITH THIS:

uvicorn main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
import torch
from mangum import Mangum
import os
from dotenv import load_dotenv
import pathlib
import gdown
import platform
import uvicorn
from PIL import Image
import torchvision.transforms as transforms
import torch
import spacy
import gc
# os.system("python -m spacy download en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

load_dotenv()
API_KEY = os.getenv("API_KEY")

def load_model_custom(fname):
    "Load a `Learner` object in `fname`, by default putting it on the `cpu`"
    try:
        res = torch.load(fname, weights_only=False)
    except AttributeError as e:
        e.args = [f"Custom classes or functions exported with your `Learner` not available in namespace. Re-declare/import before loading:\n\t{e.args[0]}"]
        raise
    return res

def download_google_drive_models():
    path = os.getcwd()
    if os.path.exists(path+"/img_model.pth") == False:
        img_model_id = "13cv8AMSd7Ff1xsU98Pjec5Ds1aZ7ucvW"
        gdown.download(id = img_model_id, output=path+"/img_model.pth", fuzzy=True)
        print("Downloaded img model")

    if os.path.exists(path+"/txt_model.pth") == False:
        txt_model_id = "1v55qhP3FjtHPyABcYuurkm2K120z08hE"
        gdown.download(id = txt_model_id, output=path+"/txt_model.pth", fuzzy=True)
        print("Downloaded txt model")
    
    if os.path.exists(path+"/vocab.pth") == False:
        vocab_model_id = "1Wb-1Y3Vf2yZ2Q4rBbaK9oe-wpy4BhIgp"
        gdown.download(id = vocab_model_id, output=path+"/vocab.pth", fuzzy=True)
        print("Downloaded txt model")

download_google_drive_models()

print("Available files:", os.listdir())

print("Downloaded txt size:", os.path.getsize("txt_model.pth"))
print("Downloaded img size:", os.path.getsize("img_model.pth"))
print("Downloaded vocab size:", os.path.getsize("vocab.pth"))

path = os.getcwd()

# Load Fastai Image Model
MODEL_PATH = path+"/img_model.pth"  # Update with your model's filename

# Load Torch Text Model
TXT_MODEL_PATH = path+"/txt_model.pth"  # Update with your model's filename
VOCAB_PATH = path+"/vocab.pth"  # Update with your model's filename

try:
    learn = load_model_custom(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load image model: {e}")

# Load Torch text model (assumes simple architecture)
try:
    learn_txt = load_model_custom(TXT_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load text model: {e}")

# Load Torch text model (assumes simple architecture)
try:
    vocab = torch.load(VOCAB_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load vocab model: {e}")

def spacy_tokenizer(text):
    return [token.text.lower() for token in nlp(text) if not token.is_space]

# API key check dependency
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized access")

# Initialize FastAPI
app = FastAPI()
handler = Mangum(app)

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
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Transform to tensor
        transform = transforms.Compose([
            transforms.Resize((150, 150)),  # Change this to your model’s expected input size
            transforms.ToTensor()])

        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Inference
        learn.eval()
        with torch.inference_mode():
            logits = learn(input_tensor).to(dtype=torch.float32)
            probs = torch.nn.functional.softmax(logits, dim=1)  # Now real probabilities
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        # Define the label list you used during training    
        label_list = ["Acne & Rosacea", "Atopic Dermatitis", "Bullous Disease", "Eczema", "Normal"]
        prediction = label_list[pred_idx]
        return {
            "all_labels": [round(p, 6) for p in probs[0].tolist()],
            "list_order": ["Acne & Rosacea", "Atopic Dermatitis", "Bullous Disease", "Eczema", "Normal"],
        }
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 🔥 Cleanup
        for var in ['image', 'input_tensor', 'logits', 'probs']:
            if var in locals():
                del locals()[var]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Pydantic model for input
class TextInput(BaseModel):
    input_data: str

@app.post("/predict-text/")
async def predict_text(input: TextInput, auth=Depends(verify_api_key)):
    try:
        global vocab
        # Convert vocab list to a dictionary if not already
        if isinstance(vocab, list):
            vocab = {token: idx for idx, token in enumerate(vocab)}
        tokens = spacy_tokenizer(input.input_data)
        token_ids = [vocab.get(token, vocab.get('<unk>', 0)) for token in tokens]
        input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0) 
        # Inference
        learn_txt.eval()
        with torch.inference_mode():
            logits = learn_txt(input_tensor)
    
            # If logits is a tuple, take the first element (logits)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        return {
            "all_labels": probs.tolist()[0],
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
    
    finally:
        # 🔥 Memory cleanup
        for var in ['tokens', 'token_ids', 'input_tensor', 'logits', 'probs']:
            if var in locals():
                del locals()[var]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    port = os.getenv("PORT") or 8000
    uvicorn.run(app, host="0.0.0.0", port=int(port))