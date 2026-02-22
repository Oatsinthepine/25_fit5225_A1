# Generate the parallel requests based on the ThreadPool Executor
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import sys
import time
import glob
import requests
import threading
import uuid
import base64
import json
import os
import argparse


def parse_argmuments() -> argparse.Namespace:
    """
    当 --debug 开启时：
	•	打印每个请求详细信息
	•	打印失败原因
	•	打印响应时间

    不开 debug 时：
	•	只打印总时间

    python cloudpose_client.py \
    --input-folder client/inputfolder \
    --url http://localhost:60001/api/pose_estimation \
    --workers 4 \
    --debug  
    """
    parser = argparse.ArgumentParser(
        description="CloudPose Client - Send parallel requests to pose estimation service"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to input folder containing images"
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="CloudPose API endpoint URL"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default = 1,
        help="Number of parallel worker threads"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logs"
    )

    parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to CSV file to store experiment results"
    )

    return parser.parse_args()


#send http request
def call_cloudpose_service(image:str, url: str, debug: bool = False) -> None:

    """

    """
    
    try:
        data = dict()
        #generate uuid for image
        img_id = uuid.uuid5(uuid.NAMESPACE_OID, image)
        # Encode image into base64 string
        with open (image, 'rb') as image_file:
            data['image'] =  base64.b64encode(image_file.read()).decode('utf-8')

        data ['id'] = str(img_id)
        
        # 这里是不对的，json = xxx 直接会添加 json.dumps, 不需要额外做一遍，然后也会添加header
        # headers = {'Content-Type': 'application/json'}
        # response = requests.post(url, json= json.dumps(data), headers = headers)

        start_time = time.time()
        response = requests.post(url, json = data)
        elapsed = time.time() - start_time

        if response.ok:
            output = "Thread : {},  input image: {},  output:{}".format(threading.current_thread().getName(),
                                                                        image,  response.text)
            print(output)
        
        else:
            print ("Error, response status:{}".format(response))

        if debug:
            print(f"[DEBUG] Thread: {threading.current_thread().name}")
            print(f"[DEBUG] Image: {image}")
            print(f"[DEBUG] Response time: {elapsed:.4f} sec")
            print(f"[DEBUG] Status code: {response.status_code}")

    except Exception as e:
        print("Exception in webservice call: {}".format(e))



# gets list of all images path from the input folder
def get_images_to_be_processed(input_folder: str) -> list[str]:
    images = []
    for image_file in glob.iglob(input_folder + "*.jpg"):
        images.append(image_file)
    return images



def main() -> None:
    """
    Main function to parse arguments and call the cloudpose service
    """
    
    args = parse_argmuments()

    input_folder = os.path.join(args.input_folder, "")
    images = get_images_to_be_processed(input_folder)

    url = args.url
    
    num_images = len(images)
    num_workers = args.workers
    start_time = time.time()
    
    #craete a worker  thread  to  invoke the requests in parallel
    with PoolExecutor(max_workers=num_workers) as executor:
        # for _ in executor.map(call_cloudpose_service,  images):
        for _ in executor.map(lambda img: call_cloudpose_service(img, url, args.debug), images):
            pass

    elapsed_time =  time.time() - start_time
    print("Total time spent: {} average response time: {}".format(elapsed_time, elapsed_time/num_images))

    if args.output:
        import csv
        file_exists = os.path.isfile(args.output)

        with open(args.output, mode = "a", newline = "") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["workers", "total_time", "avg_response_time"])
            
            writer.writerow([num_workers, round(elapsed_time,4), round(elapsed_time/num_images,6)])



if __name__ == "__main__":
    main()