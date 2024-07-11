import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('examples/policy_iteration_car_rental/out/data.csv', delimiter=',', dtype=int).reshape(-1, 20, 20)
data = np.concat([np.zeros((1, 20, 20)), data], axis=0)

sns.set_theme()

_, axes = plt.subplots(2, 3, figsize=(40, 20))
axes = axes.flatten()
plt.subplots_adjust(wspace=0.2, hspace=0.2)

for i, policy in enumerate(data):
  fig = sns.heatmap(np.flipud(policy), ax=axes[i], cmap="viridis")
  fig.set_ylabel('# cars at first location', fontsize=18)
  fig.set_yticks(list(reversed(range(21))))
  fig.set_xlabel('# cars at second location', fontsize=18)
  fig.set_title('policy {}'.format(i), fontsize=24)
  
plt.savefig('examples/policy_iteration_car_rental/out/policy.png')
plt.show()
