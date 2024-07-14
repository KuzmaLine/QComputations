#!/opt/intel/oneapi/intelpython/latest/bin/python

import os
import json
import pandas as pd
import seaborn as sns

from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from joblib import parallel_backend
import imageio

def read_files(dir):
    time_vec = pd.read_csv("./" + dir + "/time.csv", header=None).to_numpy().squeeze().tolist()
    basis = pd.read_csv("./" + dir + "/basis.csv", header=None).to_numpy().squeeze().tolist()
    probs = pd.read_csv("./" + dir + "/probs.csv", header=None)

    probs.index=time_vec
    probs.columns = basis

    return probs

json_file = os.environ.get('SEABORN_CONFIG')

with open(json_file) as json_data:
    config = json.load(json_data)

format = config.get("format")
if (format == "gif"):
    dirs = config.get("dirs")
    plotname = config.get("filename")

    fig = plt.figure(figsize=(int(config.get("width")), int(config.get("height"))))

    dir_list = []
    for p in Path('.').glob(dirs):
        if (os.path.isdir(str(p))):
            dir_list.append(str(p))

    dir_list.sort()
    probs = read_files(dir_list[0])

    def save_frame(filename, frame_data):
        fig = plt.figure(figsize=(int(config.get("width")), int(config.get("height"))))
        sns.lineplot(data=frame_data)
        plt.title(filename)
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig(filename)
        plt.close(fig)

    def process_dir(dir_path):
        probs = read_files(dir_path)
        save_frame(f"{dir_path}.png", probs)

    Parallel(n_jobs=-1)(delayed(process_dir)(dir_path) for dir_path in dir_list)

    images = []
    for i, dir_path in enumerate(dir_list):
        images.append(imageio.imread(f"{dir_path}.png"))

    imageio.mimsave(config.get("filename"), images)

    for dir_path in dir_list:
        os.remove(f"{dir_path}.png")
    #plt.show()
elif (format == "ready_gif"):
    dirs = config.get("dirs")
    dir_list = []
    for p in Path('.').glob(dirs):
        dir_list.append(str(p))

    images = Parallel(n_jobs=-1)(delayed(imageio.imread)(f"{dir_path}") for dir_path in dir_list)

    print("HERE")
    imageio.mimsave(config.get("filename"), images)

    for dir_path in dir_list:
        os.remove(f"{dir_path}")
else:
    dirs = config.get("dirs")

    fig = plt.figure(figsize=(int(config.get("width")), int(config.get("height"))))

    dir_list = []
    for p in Path('.').glob(dirs):
        if (os.path.isdir(str(p))):
            dir_list.append(str(p))

    dir_list.sort()

    def process_directory(dir):
        probs = read_files(dir)

        fig = plt.figure(figsize=(int(config.get("width")), int(config.get("height"))))
        sns.lineplot(data=probs)
        plt.title(dir)
        plt.xlabel("Time (6.626 * 10^(-34) seconds)")
        plt.ylabel("Probabilty")
        plt.grid()
        plt.savefig(dir + ".png", format=config.get("format"))
        plt.close(fig)

    # Распараллеливаем цикл по всем директориям
    Parallel(n_jobs=-1)(delayed(process_directory)(dir) for dir in dir_list)
