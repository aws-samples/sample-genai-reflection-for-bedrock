import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


def plot_bar(name: str, df: pd.DataFrame):
    df["model_combination"] = ["\n".join(x) for x in df["models"]]
    n_reflections_unique = df["n_reflections"].unique()

    colors = plt.cm.plasma(np.linspace(0, 1, len(n_reflections_unique)))
    color_map = {n: colors[i] for i, n in enumerate(n_reflections_unique)}
    df["color"] = df["n_reflections"].map(color_map)

    model_combinations = df["model_combination"].unique()
    n_reflections = df["n_reflections"].unique()

    bar_width = 0.25  # Wider bars for better visibility
    index = np.arange(len(model_combinations))

    plt.figure(figsize=(14, 10))  # Larger figure size for clarity
    for i, n in enumerate(n_reflections):
        subset = df[df["n_reflections"] == n]
        plt.barh(
            index + i * bar_width,
            subset["mean"],
            height=bar_width,
            color=color_map[n],
            label=f"{n} reflections",
            xerr=subset["std_err"],
            capsize=5,
            edgecolor="black",
        )

    plt.yticks(index + bar_width * (len(n_reflections) - 1) / 2, model_combinations, fontsize=12)
    plt.xlabel("Duration (seconds) - $\mu \pm \sigma$", fontsize=14)
    plt.title(f"Time Taken: {name}", fontsize=16, fontweight="bold")
    plt.legend(title="# Reflections", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(f"{dir_path}/grouped_time_trial.png")


if __name__ == "__main__":
    df = pd.read_json(f"{dir_path}/timing_results.json", orient="records", lines=True)
    if "aggregator" in df.columns:
        df = df[df["aggregator"].isnull()]
    plot_bar("(without aggregation)", df)
