import numpy as np
import matplotlib.pyplot as plt
import csv
import os

curDir = os.getcwd()
trainingNumber = 100

file_path = f"{curDir}/training/runs/segment/train{trainingNumber}/results.csv"

with open(file_path, newline="") as f:
    reader = csv.reader(f)
    data = list(reader)
dictionary = {}


res = {}
for i, line in enumerate(data):
    if i == 0:
        for x in line:
            res[str(x).strip()] = []

for i, content in enumerate(res):
    dictionary[i] = str(content)

for i, line in enumerate(data):
    if i != 0:
        for idx, val in enumerate(line):
            res[dictionary[idx]].append(float(val))


ln_width = 4
font = {"family": "sans-serif", "color": "black", "size": 20}
font_title = {"family": "sans-serif", "color": "black", "size": 30}

for idx, title in enumerate(dictionary):
    title = dictionary[idx + 1][6:]
    plt.subplot(2, 2, idx + 1)
    plt.plot(
        np.array(res["epoch"]),
        np.array(res[f"train/{title}"]),
        linewidth=ln_width,
        label="train_err",
    )
    plt.plot(
        np.array(res["epoch"]),
        np.array(res[f"val/{title}"]),
        linewidth=ln_width,
        label="val_err",
    )
    plt.title(f"{title}", fontdict=font_title)
    plt.xlabel("epochs", fontdict=font), plt.ylabel("loss", fontdict=font)
    plt.grid()
    plt.xticks(fontsize=30), plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    if idx == 3:
        break

plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")
