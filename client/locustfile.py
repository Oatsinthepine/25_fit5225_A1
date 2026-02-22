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

def get_images(input_folder):
    return glob.glob(input_folder + "*.jpg")

IMAGES = get_images("client/inputfolder/")

class CloudPoseUser(HttpUser):

    wait_time = between(0.1, 0.5)  # 模拟真实用户间隔

    @task
    def pose_estimation(self):

        image = random.choice(IMAGES)

        with open(image, 'rb') as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        payload = {
            "id": str(uuid.uuid4()),
            "image": img_base64
        }

        self.client.post(
            "/api/pose_estimation",
            json=payload,
            name="/api/pose_estimation"
        )