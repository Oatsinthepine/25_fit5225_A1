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

from .pose_detection import detect_base64_img, annotate_img



app = FastAPI()

class PoseRequest(BaseModel):
    id: str
    image: str

@app.post("/api/pose_estimation")
async def pose_estimation(request: PoseRequest):
    """
    调用 detect_base64_img 进行pose检测
    HTTP 接收
    数据解码
    调用 pose_service
    返回 JSON
    """

    result = detect_base64_img(request.image)

    return {
        "id": request.id,
        **result
    }


@app.post("/api/pose_estimation_annotation")
async def pose_estimation_annotation(request: PoseRequest):
    """
    调用 annotate_img 进行pose检测并标注

    """

    result = annotate_img(request.image)

    if "annotated_img" not in result:
        return {
            "id":request.id,
            "error":result.get("err", "Unknown error")
        }

    return {
        "id":request.id,
        "image": result["annotated_img"]
    }