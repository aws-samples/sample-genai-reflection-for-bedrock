import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(current_dir + "/grid_search.csv")

sns.set_theme(font_scale=1.3, style="whitegrid")

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    df["Cost"],
    df["Latency"],
    c=df["Score"],
    cmap="coolwarm",
    s=200,
    alpha=0.7,
)
for i, label in enumerate(df["Config"]):
    plt.annotate(
        label,
        (df["Cost"].iloc[i], df["Latency"].iloc[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )
plt.colorbar(scatter, label="Accuracy", format=lambda x, pos: "{:.0f}%".format(x))
plt.xlabel("Cost ($) / sample")
plt.ylabel("Latency (seconds) / sample")
plt.title("Performance of <Model (# Reflections)>")

plt.tight_layout()
plt.savefig(current_dir + "/scatter_grid_search.png", dpi=500)
