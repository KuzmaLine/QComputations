#!/opt/intel/oneapi/intelpython/latest/bin/python

import pandas as pd
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import sys

dir = sys.argv[1]
plotname = sys.argv[2]

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

plt.figure(figsize = (int(sys.argv[4]), int(sys.argv[5])))
#plt.figure()
sns.lineplot(data=probs)
plt.grid()
plt.savefig(plotname, format=sys.argv[3])
#plt.show()


