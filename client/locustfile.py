# from locust import HttpUser, task
# import glob
# import uuid
# import base64
# import json
# import threading
# import time


# def get_images_to_be_processed(input_folder):
#     return glob.glob(input_folder + "*.jpg")

# class InternetUser(HttpUser):
#     @task
#     def index(self):
#         images = get_images_to_be_processed("inputfolder/")
#         for i, image in enumerate(images):
#             try:
#                 data = {}
#                 with open(image, 'rb') as image_file:
#                     data['src_img'] = base64.b64encode(image_file.read()).decode('utf-8')
#                 data['id'] = str(uuid.uuid4())
                
#                 response = self.client.post("/api/pose_estimation", json= data)

#                 if response.ok:
#                     output = f"Thread: {threading.current_thread().name}, Image: {image}, Output: {json.dumps(response.json(), indent=2)}"
#                     print(output)
                
#                 else:
#                     print("Error:", response.status_code)
            
#             except Exception as e:
#                 print("Exception in webservice call:", e)

from locust import HttpUser, task, between
import glob
import uuid
import base64
import random


host = "http://161.33.91.68:30007" # 这个 IP 就是我 OCI k8s master 的 public IP 地址



def get_images(input_folder):
    return glob.glob(input_folder + "*.jpg")


class CloudPoseUser(HttpUser):

    wait_time = between(0.1, 0.5)  # 模拟真实用户间隔

    @task
    def pose_estimation(self):

        IMAGES = []

        for img in get_images("client/inputfolder/"):
            with open(img, "rb") as f:
                IMAGES.append(base64.b64encode(f.read()).decode('utf-8'))

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
                response.failure("Request failed")