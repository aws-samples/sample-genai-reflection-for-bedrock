import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
folder = "./figures"

# Data
task_to_name = {"arithmetic": "Arithmetic", "text2sql": "Text2SQL", "math500": "Math500"}
data: list[pd.DataFrame] = []
for benchmark in task_to_name:
    with open(f"{current_dir}/{benchmark}/cleaned_result.json", "r") as f:
        json_results: dict = json.load(f)
    categories = list(json_results.keys())
    methods = list(json_results[categories[0]].keys())
    values = {cat: [json_results[cat][method] for method in methods] for cat in categories}

    # Convert data to long format for seaborn
    plot_data = []
    for cat in categories:
        for i, method in enumerate(methods):
            plot_data.append({"Category": cat, "Method": method, "Value": values[cat][i] * 100})

    plot_df = pd.DataFrame(plot_data)
    plot_df["Task"] = task_to_name.get(benchmark)
    data.append(plot_df)

plot_df = pd.concat(data)

fig, ax = plt.subplots(layout="constrained", figsize=(18, 7))
sns.set_theme(font_scale=1.3, style="whitegrid")  # Increase font scale for better readability
colours = ["#546A76", "#FFB238"]

methods = sorted(plot_df["Method"].unique().tolist(), reverse=True)
plot_df["CategoryNum"] = plot_df["Category"].apply(
    lambda x: {
        "Easy": 0,
        "Medium": 1,
        "Hard": 2,
        "museum_visit": 3,
        "voter_1": 4,
        "orchestra": 5,
        "employee_hire": 6,
        "battle_death": 7,
        "50 qs": 8,
    }[x]
)
plot_df.sort_values(by=["Task", "CategoryNum"], inplace=True, ascending=[True, True])
plot_df.reset_index(drop=True, inplace=True)
for method, color in zip(methods, colours):
    method_data = plot_df[plot_df["Method"] == method]
    method_data["Label"] = method_data["Task"] + "\n" + method_data["Category"]
    bars = ax.bar(method_data["Label"], method_data["Value"], color=color, label=method)

for bar_index, category in enumerate(method_data["Category"].unique()):
    cat_df = plot_df[plot_df["Category"] == category]["Value"].values
    percent_diff = ((cat_df[1] - cat_df[0]) / cat_df[0]) * 100
    ax.text(
        bar_index,
        cat_df[1],
        f"{round(percent_diff)}%",
        ha="center",
        va="bottom",
        fontsize=20,
        weight="bold",
        color="black",
    )

plt.ylabel("% Correct", fontsize=23, weight="bold")
plt.ylim(0, 106)
plt.legend(facecolor="gainsboro", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{current_dir}/result.png", dpi=500)
