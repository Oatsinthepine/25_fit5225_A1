"""
main.py
只负责：
	•	HTTP 接收
	•	数据解码
	•	调用 pose_service
	•	返回 JSON

"""

from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2


app = FastAPI()

class PoseRequest(BaseModel):
    id: str
    image: str

@app.post("/api/pose_estimation")
async def pose_estimation(request: PoseRequest):

    image_bytes = base64.b64decode(request.image)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Image decoding failed"}

    return {
        "id": request.id,
        "shape": img.shape
    }

