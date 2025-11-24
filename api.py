# api.py
import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from deepface import DeepFace
from fastapi.staticfiles import StaticFiles
import uvicorn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPFACE_BACKEND"] = "torch"
os.environ["OMP_NUM_THREADS"] = "1"

app = FastAPI(title="Real-Time Pet Emotion")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images/JS/CSS
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html
@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

# Model to receive base64 image
class ImageData(BaseModel):
    image_base64: str

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

# Analyze emotion from base64 image
@app.post("/analyze-emotion/")
async def analyze_emotion(data: ImageData):
    try:
        header, encoded = data.image_base64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"emotion": "neutral", "error": "invalid_image"}

        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )

        if isinstance(result, list):
            dominant = result[0].get("dominant_emotion", "neutral")
        else:
            dominant = result.get("dominant_emotion", "neutral")

        return {"emotion": dominant.lower()}

    except Exception as e:
        return {"emotion": "neutral", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)
