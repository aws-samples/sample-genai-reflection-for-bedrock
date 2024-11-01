import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the current directory and data path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = current_dir + "/timing_results.csv"
timing_data = pd.read_csv(data_path)

# Assuming acc_data is defined as per your original code
acc_data = pd.read_json("./examples/benchmarks/arithmetic/result.json")
acc_data.index = [int(x.replace("reflection", "").strip()) for x in acc_data.index]
acc_data["n_reflections"] = acc_data.index

data = pd.merge(timing_data, acc_data, on="n_reflections", how="outer")

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.set_theme(font_scale=1.3)

# Define the difficulties
difficulties = ["Easy", "Medium", "Hard"]
for difficulty in difficulties:
    data[difficulty] *= 100

# Plot 1: Performance vs Reflections
palette = sns.color_palette("viridis", n_colors=len(difficulties))
for i, difficulty in enumerate(difficulties):
    sns.lineplot(data=data, x="n_reflections", y=difficulty, color=palette[-(i + 1)], ax=axes[0])
    sns.scatterplot(
        data=data,
        x="n_reflections",
        y=difficulty,
        color=palette[-(i + 1)],
        s=200,
        ax=axes[0],
        label=difficulty,
    )

axes[0].set_ylabel("% Correct", fontsize=23, weight="bold")
axes[0].set_xlabel("# Reflections", fontsize=23, weight="bold")
axes[0].set_title("Performance vs # Reflections", fontsize=24, weight="bold")
axes[0].legend(title="Difficulty", fontsize=22, title_fontsize="23")
axes[0].tick_params(axis="both", labelsize=22)
axes[0].grid(axis="both", linestyle="--", alpha=0.7)
axes[0].set_ylim(0, 105)

# Plot 2: Cost vs Latency
palette = sns.color_palette("cividis", n_colors=data["n_reflections"].nunique())
sns.scatterplot(
    data=data, y="cost", x="latency_mean", hue="n_reflections", s=250, ax=axes[1], palette=palette
)

axes[1].set_xlabel("Mean Latency per sample (s)", fontsize=23, weight="bold")
axes[1].set_ylabel("Cost per sample ($)", fontsize=23, weight="bold")
axes[1].set_title("Cost vs Latency", fontsize=24, weight="bold")
axes[1].legend(title="# Reflections", fontsize=22, title_fontsize="23")
axes[1].tick_params(axis="both", labelsize=22)
axes[1].grid(axis="both", linestyle="--", alpha=0.7)
axes[1].set_xlim(0, 65)

plt.tight_layout()
plt.savefig(current_dir + "/latency_vs_cost_performance.png", dpi=500)
