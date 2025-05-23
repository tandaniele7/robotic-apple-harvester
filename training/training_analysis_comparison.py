""" 
This script analyzes and compares training results from multiple model runs:
- Reads training results and configurations from CSV and YAML files
- Extracts metrics like mAP50-95(B) scores for different optimizers (AdamW vs SGD)
- Plots comparison graphs showing performance across different training runs
- Processes results from train2 to train214 directories
- Creates subplots comparing optimizer performance with customized formatting
"""

import matplotlib.pyplot as plt
import csv
import yaml
import os

dictionary = {}
res = {}
best_model_epoch = {}
model_info = {}
train_info = {}

curDir = os.getcwd()

for train in range(2, 214):
    dir_path = f"{curDir}/segment/train{train}"
    
    files = os.listdir(path=dir_path)
    
    if "results.csv" not in files or len(files) != 28:
        continue

    file_path = f"{dir_path}/results.csv"

    yaml_file_path = (
        f"{dir_path}/args.yaml"
    )

    with open(yaml_file_path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    model_info[f"train_{train}"] = data

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        data = list(reader)
    dictionary[f"train_{train}"] = {}

    res[f"train_{train}"] = {}
    for i, line in enumerate(data):
        if i == 0:
            for x in line:
                res[f"train_{train}"][str(x).strip()] = []

    for i, content in enumerate(res[f"train_{train}"]):
        dictionary[f"train_{train}"][i] = str(content)

    for i, line in enumerate(data):
        if i != 0:
            for idx, val in enumerate(line):
                res[f"train_{train}"][dictionary[f"train_{train}"][idx]].append(
                    float(val)
                )

    best_model_epoch[f"train_{train}"] = int(
        res[f"train_{train}"]["epoch"][
            res[f"train_{train}"]["metrics/mAP50-95(B)"].index(
                max(res[f"train_{train}"]["metrics/mAP50-95(B)"])
            )
        ]
    )
    best_model_idx = res[f"train_{train}"]["metrics/mAP50-95(B)"].index(
        max(res[f"train_{train}"]["metrics/mAP50-95(B)"])
    )
    train_info[f"train_{train}"] = {
        "id": train,
        "mAP50-95(B)": res[f"train_{train}"]["metrics/mAP50-95(B)"][best_model_idx],
        "optimizer": model_info[f"train_{train}"]["optimizer"]
    }

Adam_count = 0
AdamW_mAP = []
AdamW_train = []

SGD_mAP = []
SGD_train = []

for idx, train in enumerate(train_info):
    if train_info[train]["optimizer"] == "AdamW":
        AdamW_mAP.append(train_info[train]["mAP50-95(B)"])
        AdamW_train.append(train_info[train]["id"])
        
    elif train_info[train]["optimizer"] == "SGD":
        SGD_mAP.append(train_info[train]["mAP50-95(B)"])
        SGD_train.append(train_info[train]["mAP50-95(B)"])
        

ln_width = 4
font = {"family": "sans-serif", "color": "black", "size": 20}
font_title = {"family": "sans-serif", "color": "black", "size": 30}

plt.subplot(2, 1, 1)
plt.plot(
    AdamW_mAP,
    AdamW_train,
    linewidth=ln_width,
    label="AdamW",
)
plt.subplot(2, 1, 2)
plt.plot(
    SGD_mAP,
    SGD_train,
    linewidth=ln_width,
    label="SGD",
)
plt.title("Trainings", fontdict=font_title)
plt.xlabel("train", fontdict=font), plt.ylabel("mAP50-95(B)", fontdict=font)
plt.grid()
plt.xticks(fontsize = 30) , plt.yticks(fontsize = 30)
plt.legend(fontsize=20)

# for idx, title in enumerate(dictionary):
#     title = dictionary[idx + 1][6:]
#     plt.subplot(2, 2, idx + 1)
#     plt.plot(
#         np.array(res["epoch"]),
#         np.array(res[f"train/{title}"]),
#         linewidth=ln_width,
#         label="train_err",
#     )
#     plt.plot(
#         np.array(res["epoch"]),
#         np.array(res[f"val/{title}"]),
#         linewidth=ln_width,
#         label="val_err",
#     )
#     plt.title(f"{title}", fontdict=font_title)
#     plt.xlabel("epochs", fontdict=font), plt.ylabel("loss", fontdict=font)
#     plt.grid()
#     plt.xticks(fontsize = 30) , plt.yticks(fontsize = 30)
#     plt.legend(fontsize=20)
#     if idx == 3:
#         break
