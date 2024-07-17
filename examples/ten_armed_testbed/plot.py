import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = "examples/ten_armed_testbed/out/"

df = pd.read_csv(path + "data.csv")
sns.relplot(data=df, x="param", y="reward", hue="algo", kind="line")
plt.xscale("log")

ticks = np.logspace(-7, 2, base=2, num=10)
plt.xticks(ticks, [f"$2^{{ {int(np.log2(x))} }}$" for x in ticks])

plt.savefig(path + "fig.png")
plt.show()