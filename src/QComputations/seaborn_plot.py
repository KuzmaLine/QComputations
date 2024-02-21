#!/opt/intel/oneapi/intelpython/latest/bin/python

import json
import pandas as pd
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import sys

json_file = "$SEABORN_CONFIG"

with open(json_file) as json_data:
    config = json.load(json_data)

dir = config.get("dir")
plotname = config.get("filename")

# hamiltonian = pandas.read_csv("./" + dir + "/hamiltonian.csv", header=None)
time_vec = pd.read_csv("./" + dir + "/time.csv", header=None).to_numpy().squeeze().tolist()
basis = pd.read_csv("./" + dir + "/basis.csv", header=None).to_numpy().squeeze().tolist()
probs = pd.read_csv("./" + dir + "/probs.csv", header=None)

print("TIME - ", len(time_vec))
print("Probs - ", probs.shape[0], probs.shape[1])

probs.index=time_vec
probs.columns = basis

print(probs)

# print(hamiltonian)
# print(hamiltonian.head(1))
# print(time_vec)
# print(basis)

plt.figure(figsize = (int(config.get("width")), int(config.get("height"))))
#plt.figure()
sns.lineplot(data=probs)
plt.grid()
plt.savefig(plotname, format=config.get("format"))
plt.show()