import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

path = "examples/sarsa_windy_gridworld/out/"

df = pd.read_csv(path + "data.csv")
sns.relplot(data=df, x="steps", y="episodes", kind="line")

plt.savefig(path + "fig.png")
plt.show()