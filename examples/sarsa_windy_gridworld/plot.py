import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("examples/sarsa_windy_gridworld/out/data.csv")
sns.relplot(data=df, x="steps", y="episodes", kind="line")

plt.show()