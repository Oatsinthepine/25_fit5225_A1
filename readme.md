# CloudPose Kubernetes Benchmarking

## Overview

CloudPose Benchmark is a cloud-native performance evaluation project designed to analyze the scalability and performance characteristics of an AI inference service deployed on Kubernetes.

The project benchmarks a human pose estimation service built using FastAPI and YOLO-based inference, deployed in a containerized environment on a Kubernetes cluster hosted on Oracle Cloud Infrastructure (OCI).

To simulate realistic workloads, Locust is used as a distributed load testing framework running on a separate Azure virtual machine. The benchmarking workflow is fully automated using scripting pipelines that dynamically scale Kubernetes deployments and execute headless load tests.

The project demonstrates how AI inference services can be evaluated and scaled using cloud-native infrastructure.


### **For the detailed report, please view this experiment report pdf: [FIT5225 A1 Experiment Report.pdf](FIT5225%20A1%20Experiment%20Report.pdf)**

---

## Project Architecture

The benchmarking architecture follows a distributed setup to simulate realistic production workloads.

```plaintext
Load Generator (Azure VM)
        │
        │ HTTP Requests (Locust)
        ▼
Kubernetes NodePort Service
        │
OCI Kubernetes Cluster
        │
CloudPose Pods
(FastAPI + YOLO Inference)
        │
Pose Estimation Results
(JSON Response)
```

Key components:

| Component              | Description                                  |
| ---------------------- | -------------------------------------------- |
| Azure VM               | Load generator running Locust                |
| OCI Kubernetes Cluster | Hosts containerized inference service (1 Controller / master; 2 workers)       |
| CloudPose Service      | FastAPI-based API performing pose estimation |
| NodePort Service       | Exposes inference API externally             |
| Automation Script      | Runs benchmarking experiments                |


## Features

- Cloud-native AI inference deployment

- Automated load testing using Locust with orchestration

- Distributed benchmarking architecture

- Performance evaluation and visualization


## Project Structure

```plaintext
CloudPose-Benchmark
│
├── client
│   ├── locustfile.py
│   └── inputfolder
│       └── sample images (128 images, omitted to pushed to remote repo)
│
├── experiment
│   ├── run_experiment.ps1
│   └── results
│       └── locust csv outputs
│
├── plot_results.py
│
├── deployment
│   ├── cloudpose-deployment.yaml
│   └── cloudpose-service.yaml
│
└── report
```

---

## Inference Service

The CloudPose service exposes a REST API for pose estimation.

### Endpoint

```plaintext
POST /api/pose_estimation
```

### Request format

```json
{
  "id": "uuid",
  "image": "base64 encoded image"
}
```

### Response format

```json
{
  "id": "...",
  "count": 1,
  "boxes": [...],
  "keypoints": [...],
  "speed_inference": ...
}
```

The service uses a YOLO-based deep learning model to detect human pose keypoints from input images.


## Benchmarking Workflow

The benchmarking pipeline automates the entire experiment process.

```plaintext
run_experiment.ps1
        │
        ├─ scale Kubernetes deployment
        │
        ├─ wait for pods ready
        │
        ├─ run Locust headless load test
        │
        └─ export performance metrics
```        

Each experiment automatically generates Locust CSV files containing performance statistics.

---

## Technology Used

| Technology | Purpose                            |
| ---------- | ---------------------------------- |
| Kubernetes | Container orchestration            |
| Docker     | Containerized inference service    |
| FastAPI    | Inference API                      |
| YOLO       | Pose estimation model              |
| Locust     | Load testing framework             |
| Python     | Experiment automation and analysis |
| PowerShell | Benchmark orchestration            |


## Key Learning Outcomes

This project demonstrates several important cloud-native concepts:

        - deploying AI inference services on Kubernetes

        - evaluating performance of containerized ML workloads

        - designing distributed benchmarking experiments

        - implementing automated load testing pipelines

        - analyzing scalability of microservices architectures

---

## Experiment Reproducibility

Experiments are reproducible because:

- **Fixed dataset:** 128 images in `client/inputfolder`
- **Fixed runtime:** 2 minutes per Locust run
- **Fixed spawn rate:** same rate across runs
- **Fixed experiment matrix:** same (users, replicas) combinations each time

---

## Troubleshooting

- **OOMKilled**  
  Solution: increase container memory limit (e.g., 1024Mi) in the deployment YAML.

- **NodePort not reachable**  
  Solution: verify public IP and firewall rules allow traffic to the NodePort (e.g., 30007).

- **ImagePullBackOff**  
  Solution: verify container image exists and is accessible (image name and registry).

- **Windows path issues**  
  Solution: run scripts from repository root so relative paths resolve correctly.

---

## License

This project is intended for educational purposes as part of Monash Faculty of IT 2025 Fit5225 assignment. For educational purpose only.