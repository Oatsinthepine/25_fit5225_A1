from locust import HttpUser, task
import glob
import uuid
import base64
import json
import threading
import time

class InternetUser(HttpUser):
    @task
    def index(self):
        images = get_images_to_be_processed("inputfolder/")
        for i, image in enumerate(images):
            try:
                data = {}
                with open(image, 'rb') as image_file:
                    data['src_img'] = base64.b64encode(image_file.read()).decode('utf-8')
                data['id'] = str(uuid.uuid4())
                headers = {'Content-Type': 'application/json'}

                start_time = time.time()
                response = self.client.post("/api/pose_estimation", data=json.dumps(data), headers=headers)

                if response.ok:
                    output = f"Thread: {threading.current_thread().name}, Image: {image}, Output: {json.dumps(response.json(), indent=2)}"
                    print(output)
                else:
                    print("Error:", response.status_code)
            except Exception as e:
                print("Exception in webservice call:", e)

def get_images_to_be_processed(input_folder):
    return glob.glob(input_folder + "*.jpg")