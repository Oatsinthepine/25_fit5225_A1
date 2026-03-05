import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

RESULT_DIR = "experiment/results"

records = []

# 读取所有 locust csv
for file in os.listdir(RESULT_DIR):
    if file.endswith("_stats.csv"):

        path = os.path.join(RESULT_DIR, file)

        # 从文件名解析 users / pods
        match = re.search(r'exp_(\d+)u_(\d+)p', file)
        users = int(match.group(1))
        pods = int(match.group(2))

        df = pd.read_csv(path)

        # 找到 pose_estimation API 的行
        row = df[df["Name"] == "/api/pose_estimation"]

        if row.empty:
            continue

        latency = row["Average Response Time"].values[0]
        rps = row["Requests/s"].values[0]

        records.append({
            "users": users,
            "pods": pods,
            "latency": latency,
            "rps": rps
        })

data = pd.DataFrame(records)

print(data)

sns.set(style="whitegrid")

# -----------------------------
# 图1 Users vs Latency
# -----------------------------
latency_data = data[data["pods"] == 1]

plt.figure(figsize=(7,5))
sns.lineplot(data=latency_data, x="users", y="latency", marker="o")

plt.title("Users vs Average Latency")
plt.xlabel("Concurrent Users")
plt.ylabel("Average Response Time (ms)")
plt.tight_layout()

plt.savefig("users_vs_latency.png", dpi=300)
plt.show()


# -----------------------------
# 图2 Users vs Throughput
# -----------------------------
throughput_data = data[data["pods"] == 1]

plt.figure(figsize=(7,5))
sns.lineplot(data=throughput_data, x="users", y="rps", marker="o")

plt.title("Users vs Throughput (RPS)")
plt.xlabel("Concurrent Users")
plt.ylabel("Requests per Second")
plt.tight_layout()

plt.savefig("users_vs_rps.png", dpi=300)
plt.show()


# -----------------------------
# 图3 Pods vs Throughput
# -----------------------------
scaling_data = data[data["users"] == 40]

plt.figure(figsize=(7,5))
sns.lineplot(data=scaling_data, x="pods", y="rps", marker="o")

plt.title("Pods vs Throughput")
plt.xlabel("Number of Pods")
plt.ylabel("Requests per Second")
plt.tight_layout()

plt.savefig("pods_vs_rps.png", dpi=300)
plt.show()