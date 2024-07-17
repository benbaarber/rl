import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

path = "examples/policy_iteration_car_rental/out/"

data = np.loadtxt(path + "data.csv", delimiter=',', dtype=int).reshape(-1, 20, 20)
data = np.concat([np.zeros((1, 20, 20)), data], axis=0)

_, axes = plt.subplots(2, 3, figsize=(40, 20))
axes = axes.flatten()
plt.subplots_adjust(wspace=0.5, hspace=0.5)

for i, policy in enumerate(data):
  fig = sns.heatmap(policy, ax=axes[i], cmap="viridis")
  fig.invert_yaxis()
  fig.set_ylabel("# cars at first location", fontsize=18)
  fig.set_xlabel("# cars at second location", fontsize=18)
  fig.set_title(f"$\pi_{i}$", fontsize=24)
  
plt.savefig(path + "fig.png")
plt.show()
