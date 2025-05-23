import array
from opcode import opname
from matplotlib.font_manager import json_load
from sympy import false
from ultralytics import YOLO
import os
import yaml
import csv
import json
import matplotlib.pyplot as plt

curDir = os.getcwd()
os.system("clear")

dictionary = {}
res = {}
best_model_epoch = {}
model_info = {}
train_info = {}

choice = int(input("Press: \n1. to create json file \n2. to analyse json file\n"))
while choice != 1 and choice != 2:
    os.system("clear")
    print(choice)
    choice = int(
        input("Error\n\n Press: \n1. to create json file \n2. to analyse json file\n")
    )


if choice == 1:

    for train in range(2, 268):
        dir_path = (
            f"{curDir}/training/runs/segment/train{train}"
        )

        files = os.listdir(path=dir_path)

        if "results.csv" not in files or len(files) != 28:
            continue
        model = YOLO(
            f"{curDir}/training/runs/segment/train{train}/weights/best.pt"
        )
        results = model.val(
            save=False, data=f"{curDir}/training/data.yaml"
        )

        file_path = f"{dir_path}/results.csv"

        yaml_file_path = f"{dir_path}/args.yaml"

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
            "optimizer": model_info[f"train_{train}"]["optimizer"],
            "maps": [x for x in results.box.maps.tolist()],
            "maps_masks": [x for x in results.seg.maps.tolist()],
        }

    with open("train_info.json", "w") as outfile:
        json.dump(train_info, outfile)
else:
    train_info = {}
    with open("train_info.json") as f:
        train_info = json.load(f)

    AdamW_mAP_avg = []
    AdamW_map_apple = []
    AdamW_map_branches = []
    AdamW_map_metalwire = []
    AdamW_map_backg = []
    AdamW_train = []

    SGD_mAP_avg = []
    SGD_map_apple = []
    SGD_map_branches = []
    SGD_map_metalwire = []
    SGD_map_backg = []
    SGD_train = []

    for idx, train in enumerate(train_info):
        if train_info[train]["optimizer"] == "AdamW":
            AdamW_mAP_avg.append(train_info[train]["mAP50-95(B)"])
            AdamW_train.append(train_info[train]["id"])
            AdamW_map_apple.append(train_info[train]["maps"][0])
            AdamW_map_branches.append(train_info[train]["maps"][1])
            AdamW_map_metalwire.append(train_info[train]["maps"][2])
            AdamW_map_backg.append(train_info[train]["maps"][3])

        elif train_info[train]["optimizer"] == "SGD":
            SGD_mAP_avg.append(train_info[train]["mAP50-95(B)"])
            SGD_train.append(train_info[train]["mAP50-95(B)"])
            SGD_map_apple.append(train_info[train]["maps"][0])
            SGD_map_branches.append(train_info[train]["maps"][1])
            SGD_map_metalwire.append(train_info[train]["maps"][2])
            SGD_map_backg.append(train_info[train]["maps"][3])

    ln_width = 2
    ms_size = 8
    font = {"family": "sans-serif", "color": "black", "size": 20}
    font_title = {"family": "sans-serif", "color": "black", "size": 30}

    plt.subplot(2, 2, 1)
    plt.plot(
        AdamW_train,
        AdamW_mAP_avg,
        "o-b",
        ms=ms_size,
        linewidth=ln_width,
        label="AdamW",
    )
    plt.subplot(2, 2, 2)
    plt.plot(
        AdamW_train,
        AdamW_map_apple,
        "o-r",
        ms=ms_size,
        linewidth=ln_width,
        label="map apple",
    )
    plt.plot(
        AdamW_train,
        AdamW_map_branches,
        "o-y",
        ms=ms_size,
        linewidth=ln_width,
        label="map branches",
    )
    plt.plot(
        AdamW_train,
        AdamW_map_metalwire,
        "o-m",
        ms=ms_size,
        linewidth=ln_width,
        label="map metal wire",
    )
    plt.plot(
        AdamW_train,
        AdamW_map_backg,
        "o-b",
        ms=ms_size,
        linewidth=ln_width,
        label="map background",
    )
    plt.subplot(2, 2, 3)
    plt.plot(
        SGD_train,
        SGD_mAP_avg,
        "o-b",
        ms=ms_size,
        linewidth=ln_width,
        label="SGD",
    )
    plt.subplot(2, 2, 4)
    plt.plot(
        SGD_train,
        SGD_map_apple,
        "o-r",
        ms=ms_size,
        linewidth=ln_width,
        label="map apple",
    )
    plt.plot(
        SGD_train,
        SGD_map_branches,
        "o-y",
        ms=ms_size,
        linewidth=ln_width,
        label="map branches",
    )
    plt.plot(
        SGD_train,
        SGD_map_metalwire,
        "o-m",
        ms=ms_size,
        linewidth=ln_width,
        label="map metal wire",
    )
    plt.plot(
        SGD_train,
        SGD_map_backg,
        "o-b",
        ms=ms_size,
        linewidth=ln_width,
        label="map background",
    )

    plt.title("Trainings", fontdict=font_title)
    plt.xlabel("train", fontdict=font), plt.ylabel("mAP50-95(B)", fontdict=font)
    plt.grid()
    plt.xticks(fontsize=30), plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    plt.show()
