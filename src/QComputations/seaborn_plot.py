#!/opt/intel/oneapi/intelpython/latest/bin/python

import os
import json
import pandas as pd
import seaborn as sns

from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def read_files(dir):
    time_vec = pd.read_csv("./" + dir + "/time.csv", header=None).to_numpy().squeeze().tolist()
    basis = pd.read_csv("./" + dir + "/basis.csv", header=None).to_numpy().squeeze().tolist()
    probs = pd.read_csv("./" + dir + "/probs.csv", header=None)

    print("TIME - ", len(time_vec))
    print("Probs - ", probs.shape[0], probs.shape[1])

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

    fig = plt.figure(figsize = (int(config.get("width")), int(config.get("height"))))

    dir_list = []
    for p in Path('.').glob(dirs):
        dir_list.append(str(p))

    dir_list.sort()
    probs = read_files(dir_list[0])
    def init():
        global probs
        sns.lineplot(data=probs)
        plt.title(dir_list[0])
        plt.legend(loc='upper right')
        plt.grid()

    index = 1
    def update(frame):
        global dir_list
        global index
        global config

        probs = read_files(dir_list[index % int(config.get('frames'))])
        plt.clf()
        sns.lineplot(data=probs)
        plt.title(dir_list[index % int(config.get('frames'))])
        plt.legend(loc='upper right')
        plt.grid()
        index += 1

    ani = FuncAnimation(fig, update, frames=int(config.get("frames")),
                        init_func=init, interval=int(config.get("interval")))
    ani.save(config.get("filename"))
    plt.show()
else:
    dir = config.get("dir")
    plotname = config.get("filename")

    probs = read_files(dir)

    plt.figure(figsize = (int(config.get("width")), int(config.get("height"))))
    sns.lineplot(data=probs)
    plt.grid()
    plt.savefig(plotname, format=config.get("format"))
    plt.show()