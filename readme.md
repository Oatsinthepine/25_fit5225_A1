# cloudpose_client.py

cloudpose_client is a Python script to invoke your web service endpoint according to the Assignment 1 specification

## Installation

All the required packages are part of the basic python installation. Please use python 3.10 or higher.

## Usage format

python cloudpose_client.py  <input folder name> <URL> <num_threads>

## 目前开发的时候用这个命令测试 （我下面打算重构 cloudpose_client.py， 使用argparse，这样更工程化）：
`python3 ./client/cloudpose_client.py ./client/inputfolder http://localhost:60001/api/pose_estimation 1`


## Sample run command

python cloudpose_client.py  inputfolder/  http://localhost:8000/api/pose_detection 4

## 目前开发的时候仍然用这个命令启动fastapi server：
`uvicorn cloudpose.app.main:app --host 0.0.0.0 --port 60001`


---

## 为什么做 Service Layer Separation

因为未来：
	•	你可能做第二个 endpoint（annotated image）
	•	你可能做 GPU 加速
	•	你可能改模型

但 API 层不用动。