# cloudpose_client.py

cloudpose_client is a Python script to invoke your web service endpoint according to the Assignment 1 specification

## Installation

All the required packages are part of the basic python installation. Please use python 3.10 or higher.

## Usage format

python cloudpose_client.py  <input folder name> <URL> <num_threads>

## Sample run command

python cloudpose_client.py  inputfolder/  http://localhost:8000/api/pose_detection 4

## 为什么做 Service Layer Separation

因为未来：
	•	你可能做第二个 endpoint（annotated image）
	•	你可能做 GPU 加速
	•	你可能改模型

但 API 层不用动。