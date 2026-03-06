# CloudPose Benchmark Experiment Runner
#
# Runs automated benchmarking experiments.
#
# Steps:
# 1) Scale Kubernetes deployment on OCI cluster
# 2) Wait for rollout
# 3) Run Locust load test from Azure VM
# 4) Save CSV results to experiment/results
#
# Requirements:
# - Windows PowerShell
# - Locust installed
# - SSH key configured for OCI master node
#
# Usage:
# Run from repository root:
# .\experiment\run_experiment.ps1

cd C:\locust_test\25_fit5225_A1

$HOST_URL = "http://161.33.91.68:30007"
$LOCUSTFILE = "client\locustfile.py"
$RESULT_DIR = "experiment\results"

Write-Host "==========================================="
Write-Host " - CloudPose Benchmark Experiment Start - "
Write-Host "==========================================="

$experiments = @(
    @{users=5; pods=1},
    @{users=10; pods=1},
    @{users=20; pods=1},

    @{users=20; pods=4},
    @{users=40; pods=4},

    @{users=40; pods=8}
)

foreach ($exp in $experiments) {

    $users = $exp.users
    $pods = $exp.pods

    Write-Host ""
    Write-Host "--------------------------------------"
    Write-Host "Running experiment:"
    Write-Host "Users = $users"
    Write-Host "Pods  = $pods"
    Write-Host "--------------------------------------"


    # Step 1 change replicas
    Write-Host "Scaling Kubernetes deployment..."
    ssh -i $HOME/.ssh/k8s-master-node.key ubuntu@161.33.91.68 "kubectl scale deployment cloudpose-deployment --replicas=$pods -n jacky"

    # Step 2 wait for rollout of the pods and update then ready
    Write-Host "Waiting for rollout to complete..."
    ssh -i $HOME/.ssh/k8s-master-node.key ubuntu@161.33.91.68 "kubectl rollout status deployment cloudpose-deployment -n jacky"


    Write-Host "Pods are ready."

    # Run locust test
    Write-Host "Starting Locust load test..."

    # Step 3 run locust
    locust `
        -f $LOCUSTFILE `
        --host $HOST_URL `
        --headless `
        -u $users `
        -r 2 `
        --run-time 2m `
        --csv "$RESULT_DIR\exp_${users}u_${pods}p"

    Write-Host "Experiment completed."    
}

Write-Host ""
Write-Host "======================================"
Write-Host " All experiments finished successfully"
Write-Host " Results saved in experiment/results/"
Write-Host "======================================"