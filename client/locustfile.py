from locust import HttpUser, task, between
import glob
import uuid
import base64
import random
import os

BASE_DIR = os.path.dirname(__file__)
IMAGE_FOLDER = os.path.join(BASE_DIR, "inputfolder")


def load_images():
    images = []
    path = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    for img in path:
        with open(img, "rb") as f:
            images.append(base64.b64encode(f.read()).decode("utf-8"))

    print(f"Loaded {len(images)} images")    
    return images    

IMAGES = load_images()

class CloudPoseUser(HttpUser):

    host = "http://161.33.91.68:30007" # 这个 IP 就是我 OCI k8s master 的 public IP 地址
    wait_time = between(0.1, 0.5)  # 模拟真实用户间隔

    @task
    def pose_estimation(self):

        image = random.choice(IMAGES)

        payload = {
            "id": str(uuid.uuid4()),
            "image": image
        }

        with self.client.post(
            "/api/pose_estimation",
            json=payload,
            name="/api/pose_estimation",
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status: {response.status_code}, Body: {response.text[:200]}")