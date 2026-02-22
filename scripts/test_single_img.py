

import requests
import base64
import uuid

IMAGE_PATH = "client/inputfolder/000000007454.jpg"
URL = "http://localhost:60001/api/pose_estimation_annotation"

with open(IMAGE_PATH, "rb") as image_file:
    img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

payload = {
    "id": str(uuid.uuid4()),
    "image": img_base64
}    

response = requests.post(URL, json=payload)

print("Status:", response.status_code)

if response.ok:
    result = response.json()

    if "image" in result:
        with open("annotated_output.jpg", "wb") as f:
            f.write(base64.b64decode(result["image"]))

        print("Annotated image saved as annotated_output.jpg")
    else:
        print("Error:", result)
else:
    print(response.text)