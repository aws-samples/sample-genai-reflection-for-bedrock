import os
import pandas as pd
import numpy as np
from bhive import TrialConfig, Hive
from bhive.evaluators import answer_in_tags

current_dir = os.path.dirname(os.path.abspath(__file__))

test_dataset = []
for _ in range(20):
    a, b, c, d, e, f = np.random.randint(250, 750, size=6)
    question = "What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer in <answer> </answer> tags without any commas.".format(
        a, b, c, d, e, f
    )
    answer = a + b * c + d - e * f
    test_dataset.append((question, str(answer)))

trial_config = TrialConfig(
    bedrock_model_combinations=[
        ["anthropic.claude-3-sonnet-20240229-v1:0"],
        ["anthropic.claude-3-haiku-20240307-v1:0"],
        ["mistral.mistral-small-2402-v1:0"],
        ["mistral.mistral-large-2402-v1:0"],
    ],
    reflection_range=[0, 1, 3],
)
hive_client = Hive()
results = hive_client.optimise(
    test_dataset,
    trial_config,
    evaluator=answer_in_tags,
)
scores = [r.score * 100 for r in results.individual_results]
cost = [r.avg_cost_dollars for r in results.individual_results]
latency = [r.avg_latency_seconds for r in results.individual_results]
labels = [
    f"{r.config.bedrock_model_ids[0]} - {r.config.num_reflections}"
    for r in results.individual_results
]
df = pd.DataFrame(
    {
        "Config": labels,
        "Cost": cost,
        "Latency": latency,
        "Score": scores,
    }
)
df.to_csv(current_dir + "/grid_search.csv", index=False)
