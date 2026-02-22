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
        required=True,
        help="Number of parallel worker threads"
    )

    return parser.parse_args()


#send http request
def call_cloudpose_service(image:str, url: str) -> None:

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

        response = requests.post(url, json = data)

        if response.ok:
            output = "Thread : {},  input image: {},  output:{}".format(threading.current_thread().getName(),
                                                                        image,  response.text)
            print(output)
        else:
            print ("Error, response status:{}".format(response))

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
        for _ in executor.map(lambda img: call_cloudpose_service(img, url), images):
            pass

    elapsed_time =  time.time() - start_time
    print("Total time spent: {} average response time: {}".format(elapsed_time, elapsed_time/num_images))





if __name__ == "__main__":
    main()