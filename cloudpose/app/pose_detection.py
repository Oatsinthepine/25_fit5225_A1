from ultralytics import YOLO
import os
import cv2
import logging
import base64
import numpy as np
import threading
from PIL import Image
from io import BytesIO

log = logging.getLogger(__name__)

def predict(model, src_img, dst_img):
    log.info(f"Predicting with source image: {src_img}, output to {dst_img}")

    img = cv2.imread(src_img)
    if img is None:
        log.error(f"Error: Could not read image at {src_img}")
        exit(1)

    results = model(src_img)
    print(results)
    # Process the results
    for result in results:
        keypoints = result.keypoints  # Keypoints object
        #print(keypoints)

        if keypoints is not None and len(keypoints.xy) > 0: #Check if keypoints are detected
            # Get keypoints coordinates (x, y) - Shape: (num_people, num_keypoints, 2)
            keypoints_xy = keypoints.xy[0]  # Assuming only one person detected for simplicity. Adapt if multiple people are expected.
            #print(keypoints_xy)
            # Get confidence scores for each keypoint - Shape: (num_people, num_keypoints)
            keypoints_conf = keypoints.conf[0]
            #print(keypoints_conf)

            # Plot keypoints on the image
            for k, (x, y) in enumerate(keypoints_xy):
                if keypoints_conf[k] > 0.5: # Only plot if confidence is above a threshold (e.g., 0.5)
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circles
                    cv2.putText(img, str(k), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red labels

            # Draw lines connecting the keypoints (optional - customize as needed)
            # Example: Connect shoulders and elbows
            # You'll need to define the connections based on the model's keypoint indices
            # Example connections for COCO format (adjust if using a different dataset):
            connections = [[5, 6], [5, 11], [6, 12], [11, 12]] # Example connections (left shoulder - right shoulder, left shoulder - left hip, right shoulder - right hip, left hip - right hip)

            for connection in connections:
                p1 = (int(keypoints_xy[connection[0]][0]), int(keypoints_xy[connection[0]][1]))
                p2 = (int(keypoints_xy[connection[1]][0]), int(keypoints_xy[connection[1]][1]))

                if keypoints_conf[connection[0]] > 0.5 and keypoints_conf[connection[1]] > 0.5: #only draw the line if the keypoints are above the confidence level
                    cv2.line(img, p1, p2, (0, 0, 255), 2)  # Red lines

            # Save the image
            cv2.imwrite(dst_img, img)
        else:
            print("No keypoints were detected in the image.")
    return results

log.info("Loading YOLO pose detection model...")


'''
这里进行修改，使用os模块进行动态模型路径加载防止路径错误.
这样：
	•	本地 OK
	•	Docker OK
	•	K8s OK
'''

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'yolo11x-pose.pt')
model = YOLO(MODEL_PATH)

lock = threading.Lock()

def detect_base64_img(base64_img_str, img_format='.jpg'):
    base64_img_str = base64_img_str.split(",")[-1]  #Remove the base64 header such as "data:image/jpeg;base64"

    #decode the base64 string
    img_bytes = base64.b64decode(base64_img_str)
    # Convert bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = None
    
    with lock:
        results = model(decoded_img)

    res = {
        "count":0,
        "boxes": [],
        "keypoints": [],
        "speed_preprocess": 0,
        "speed_inference": 0,
        "speed_postprocess": 0,
    }
    
    if results is None:
        log.error("Error: Could not detect pose in the image")
        res["err"] = "Error: Could not detect pose in the image and results is None"
        return res
    
    # assuming only one result for person as we can tell form: https://docs.ultralytics.com/tasks/pose/
    if len(results) > 1:
        log.error("Error: More than one result detected")
        res["err"] = "Error: More than one result detected"
        return res
    
    # Process the results
    for result in results:
        keypoints = result.keypoints   # keypoints object
        
        res["speed_preprocess"] = result.speed['preprocess']
        res["speed_inference"] = result.speed['inference']
        res["speed_postprocess"] = result.speed['postprocess']  # Do I need to get average speed?
        # print("debug result:")
        # print(result.speed)
        # print(result.boxes)
        # print(result.names)
        # print("name:")
        # print(result.names)
        # Print keypoints
        if keypoints is not None and len(keypoints.xy) > 0: #Check if keypoints are detected
            # for keypoints coordinates (x, y) - Shape: (num_people, num_keypoints, 2)
            # print("keypoints:")
            # print(keypoints)
            res["count"] = len(keypoints.xy) # number of keypoints detected

            for i, keypoints_xy in enumerate(keypoints.xy):
                
                # add keypoints to the result
                keypoints_probability = []
                for k, (x, y) in enumerate(keypoints_xy):
                    keypoints_probability.append([x.item(), y.item(), keypoints.conf[i][k].item()])
                res["keypoints"].append(keypoints_probability)
            """
                # Annotate the image with keypoints
                # Get confidence scores for each keypoint - Shape: (num_people, num_keypoints)
                keypoints_conf = keypoints.conf[i]

                # Plot keypoints on the image
                for k, (x, y) in enumerate(keypoints_xy):
                    if keypoints_conf[k] > 0.5: # Only plot if confidence is above a threshold (e.g., 0.5)
                        cv2.circle(decoded_img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circles
                        cv2.putText(decoded_img, str(k), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red labels

                # Draw lines connecting the keypoints (optional - customize as needed)
                # Example: Connect shoulders and elbows
                # You'll need to define the connections based on the model's keypoint indices
                # Example connections for COCO format (adjust if using a different dataset):
                connections = [[5, 6], [5, 11], [6, 12], [11, 12]] # Example connections (left shoulder - right shoulder, left shoulder - left hip, right shoulder - right hip, left hip - right hip)

                for connection in connections:
                    p1 = (int(keypoints_xy[connection[0]][0]), int(keypoints_xy[connection[0]][1]))
                    p2 = (int(keypoints_xy[connection[1]][0]), int(keypoints_xy[connection[1]][1]))

                    if keypoints_conf[connection[0]] > 0.5 and keypoints_conf[connection[1]] > 0.5: #only draw the line if the keypoints are above the confidence level
                        cv2.line(decoded_img, p1, p2, (0, 0, 255), 2)  # Red lines
            """
        else:
            print("No keypoints were detected in the image.")
            res["err"] = "Error: No keypoints were detected in the first result"
            return res
        
        # Add boxes to the result
        boxes = result.boxes
        for i, (x,y,w,h) in enumerate(boxes.xyxy):
            box = {
                "x": x.item(),
                "y": y.item(),
                "width": w.item(),
                "height": h.item(),
                "probability": boxes.conf[i].item(),
            }
            res["boxes"].append(box)

    return res



def annotate_img(base64_img_str, img_format='.jpg'):
  

    base64_img_str = base64_img_str.split(",")[-1]  #Remove the base64 header such as "data:image/jpeg;base64"

    #decode the base64 string
    img_bytes = base64.b64decode(base64_img_str)
    # Convert bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = None
    
    with lock:
        results = model(decoded_img)
    
    res = dict()
    
    if results is None:
        log.error("Error: Could not detect pose in the image")
        res["err"] = "Error: Could not detect pose in the image and results is None"
        return res
    
    # assuming only one result for person as we can tell form: https://docs.ultralytics.com/tasks/pose/
    if len(results) > 1:
        log.error("Error: More than one result detected")
        res["err"] = "Error: More than one result detected"
        return res   
    
    # Process the results
    for result in results:
        keypoints = result.keypoints   # keypoints object

        # Print keypoints
        if keypoints is not None and len(keypoints.xy) > 0: #Check if keypoints are detected
            # for keypoints coordinates (x, y) - Shape: (num_people, num_keypoints, 2)

            for i, keypoints_xy in enumerate(keypoints.xy):
                # Annotate the image with keypoints
                # Get confidence scores for each keypoint - Shape: (num_people, num_keypoints)
                keypoints_conf = keypoints.conf[i]

                # Plot keypoints on the image
                for k, (x, y) in enumerate(keypoints_xy):
                    if keypoints_conf[k] > 0.5: # Only plot if confidence is above a threshold (e.g., 0.5)
                        cv2.circle(decoded_img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circles
                        cv2.putText(decoded_img, str(k), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red labels

                # Draw lines connecting the keypoints (optional - customize as needed)
                # Example: Connect shoulders and elbows
                # You'll need to define the connections based on the model's keypoint indices
                # Example connections for COCO format (adjust if using a different dataset):
                connections = [[5, 6], [5, 11], [6, 12], [11, 12]] # Example connections (left shoulder - right shoulder, left shoulder - left hip, right shoulder - right hip, left hip - right hip)

                for connection in connections:
                    p1 = (int(keypoints_xy[connection[0]][0]), int(keypoints_xy[connection[0]][1]))
                    p2 = (int(keypoints_xy[connection[1]][0]), int(keypoints_xy[connection[1]][1]))

                    if keypoints_conf[connection[0]] > 0.5 and keypoints_conf[connection[1]] > 0.5: #only draw the line if the keypoints are above the confidence level
                        cv2.line(decoded_img, p1, p2, (0, 0, 255), 2)  # Red lines

        else:
            print("No keypoints were detected in the image.")
            res["err"] = "Error: No keypoints were detected in the first result"
            return res

        # Save the image
        img_rgb = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)  # OpenCV default code function is BGR, PIL needs RGB
        # Convert PIL image
        pil_img = Image.fromarray(img_rgb)
        # write to disk cache
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

        res["annotated_img"] = base64_img
    
    return res



def main():
    """
    CLI 入口：用本地图片做一次 pose 预测并保存标注图。
    路径基于当前模块所在目录，与 MODEL_PATH 一致，避免受 CWD 影响。
    """
    try:
        # 使用模块已加载的全局 model，不再重复加载
        image_path = os.path.join(BASE_DIR, 'test.jpg')
        output_image = os.path.join(BASE_DIR, 'test_with_keypoints.jpg')

        if not os.path.isfile(image_path):
            log.warning(f"Test image not found: {image_path}. Put test.jpg in cloudpose/app/ or pass path.")
            return

        log.info("Running pose prediction (using already-loaded model).")
        result = predict(model, image_path, output_image)
        log.info(f"Result: {result}; output saved to {output_image}")
    except Exception as e:
        log.exception("Exception in main")
        raise



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()