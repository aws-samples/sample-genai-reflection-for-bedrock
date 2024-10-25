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
        plot_data.append(
            {"Category": cat.capitalize(), "Method": method, "Value": values[cat][i] * 100}
        )

plot_df = pd.DataFrame(plot_data)

plt.figure(figsize=(18, 7))  # Larger figure size for better visibility
sns.set_theme(font_scale=1.3)  # Increase font scale for better readability
colours = ["#546A76", "#FFB238"]
bar_plot = sns.barplot(data=plot_df, x="Category", y="Value", hue="Method", palette=colours)

for i in range(len(categories)):
    second_bar_index = (i * len(methods)) + 1
    first_value = plot_df.iloc[second_bar_index - 1]["Value"]
    second_value = plot_df.iloc[second_bar_index]["Value"]
    percentage_diff = ((second_value - first_value) / first_value) * 100
    x_value = (
        bar_plot.patches[second_bar_index].get_x()
        + bar_plot.patches[second_bar_index].get_width() / 2
    )
    y_value = bar_plot.patches[second_bar_index].get_height()

    # HACK, for some reason the bar index of 0th and 1st are swapped
    if i == 0:
        x_value += bar_plot.patches[second_bar_index].get_width()
        y_value = bar_plot.patches[4].get_height()
        percentage_diff = 19.2307
    elif i == 1:
        percentage_diff = 5.2631

    bar_plot.text(
        x_value,
        y_value,
        f"+{percentage_diff:.1f}%",
        ha="center",
        va="bottom",
        fontsize=20,
        weight="bold",
        color="black",
    )

# Improve labels and titles
plt.ylabel(metric, fontsize=23, weight="bold")
plt.ylim(0, 106)
plt.title(f"Performance on {title}", fontsize=24, weight="bold")
plt.legend(facecolor="white", fontsize=22)

# Customize ticks
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xlabel("")

# Show plot
plt.tight_layout()

# Save the plot
## this further reduced in size before display
plt.savefig(f"{current_dir}/{benchmark_name}/result.png", dpi=500)
