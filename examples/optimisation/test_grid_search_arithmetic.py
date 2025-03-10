"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import os
import pandas as pd
import numpy as np
from bhive import TrialConfig, Hive
from bhive.evaluators import answer_in_tags

current_dir = os.path.dirname(os.path.abspath(__file__))

test_dataset = []
for _ in range(20):
    a, b, c, d, e, f = np.random.randint(250, 750, size=6)  # "medium" difficulty
    question = f"What is the result of {a}+{b}*{c}+{d}-{e}*{f}? Make sure to state your answer in <answer> </answer> tags without any commas."
    answer = str(a + b * c + d - e * f)
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

result_data: dict[str, list] = {
    "Config": [],
    "Cost": [],
    "Latency": [],
    "Score": [],
}
for result in results.individual_results:
    result_data["Config"].append(
        f"{result.config.bedrock_model_ids[0]} - {result.config.num_reflections}"
    )  # NOTE assumes only one model was used
    result_data["Cost"].append(result.avg_cost_dollars)
    result_data["Latency"].append(result.avg_latency_seconds)
    result_data["Score"].append(result.score * 100)

df = pd.DataFrame(result_data)
df.to_csv(current_dir + "/new_grid_search_arithmetic.csv", index=False)
