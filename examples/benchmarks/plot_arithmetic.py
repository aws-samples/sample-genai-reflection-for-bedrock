import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
folder = "./figures"

# Data
benchmark, title, metric = "arithmetic.json", "Arithmetic", "% Correct"
benchmark_name = benchmark.split(".")[0]
with open(f"{current_dir}/{benchmark_name}/cleaned_result.json", "r") as f:
    data = json.load(f)

# Prepare data for plotting
categories = list(data.keys())
methods = list(data[categories[0]].keys())
values = {cat: [data[cat][method] for method in methods] for cat in categories}

# Convert data to long format for seaborn
plot_data = []
for cat in categories:
    for i, method in enumerate(methods):
        plot_data.append({"Category": cat, "Method": method, "Value": values[cat][i] * 100})

plot_df = pd.DataFrame(plot_data)

plt.figure(figsize=(18, 7))  # Larger figure size for better visibility
sns.set_theme(font_scale=1.3)  # Increase font scale for better readability
colours = ["#546A76", "#FFB238"]
bar_plot = sns.barplot(data=plot_df, x="Category", y="Value", hue="Method", palette=colours)

for bar in bar_plot.patches:
    print(bar.get_height())

for i in range(len(categories)):
    second_bar_index = i + len(categories)
    second_row_index = (i * len(methods)) + 1

    first_value = plot_df.iloc[second_row_index - 1]["Value"]
    second_value = plot_df.iloc[second_row_index]["Value"]
    percentage_diff = ((second_value - first_value) / first_value) * 100

    x_value = (
        bar_plot.patches[second_bar_index].get_x()
        + bar_plot.patches[second_bar_index].get_width() / 2
    )
    y_value = bar_plot.patches[second_bar_index].get_height()

    percent_label = "+" if percentage_diff > 0 else ""
    bar_plot.text(
        x_value,
        y_value,
        f"{percent_label}{percentage_diff:.1f}%",
        ha="center",
        va="bottom",
        fontsize=20,
        weight="bold",
        color="black",
    )

# Improve labels and titles
plt.ylabel(metric, fontsize=23, weight="bold")
plt.ylim(0, 106)
plt.title(f"{title} Performance", fontsize=24, weight="bold")
plt.legend(facecolor="white", fontsize=22)

# Customize ticks
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xlabel("")

# Show plot
plt.tight_layout()
plt.savefig(f"{current_dir}/{benchmark_name}/result.png", dpi=500)
