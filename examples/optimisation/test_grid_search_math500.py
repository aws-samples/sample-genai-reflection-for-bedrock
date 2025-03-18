"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

This expects the math-500 jsonlines dataset to be stored locally.
"""

import os
import pandas as pd
import random
from bhive import TrialConfig, Hive
from .math import answer_math_equal

current_dir = os.path.dirname(os.path.abspath(__file__))
n_samples = 20
math500_path = f"{current_dir}/math500.jsonl"

with open(math500_path, "r") as f:
    raw_data: list[str] = f.readlines()
    parsed_raw_data: list[dict[str, str]] = [eval(line) for line in raw_data]
random.shuffle(parsed_raw_data)

if not os.path.exists(current_dir + "/math500_subset.jsonl"):
    print("Creating subset of math500 dataset...")
    dataset_subset = []
    for i in range(n_samples):
        sample = parsed_raw_data[i]
        problem, answer = sample["problem"], sample["answer"]
        question = f"""
        What is the answer to the following math problem:
        {problem}

        Make sure to always state your final answer in <answer> </answer> tags.
        """
        dataset_subset.append({"q": question, "a": answer})

    # Save subset
    with open(current_dir + "/math500_subset.jsonl", "w") as f:
        for sample in dataset_subset:
            f.write(str(sample) + "\n")

# Load subset into list[tuple]
with open(current_dir + "/math500_subset.jsonl", "r") as f:
    test_dataset = [eval(line) for line in f.readlines()]
    test_dataset = [(s["q"], s["a"]) for s in test_dataset]


def math_in_tags(expected_answer: str, response: str):
    start_tag = "<answer>"
    end_tag = "</answer>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)
    if start_index != -1 and end_index != -1:
        start_index += len(start_tag)
        response = response[start_index:end_index].strip()
        return answer_math_equal(expected_answer, response)
    else:
        return False


trial_config = TrialConfig(
    bedrock_model_combinations=[
        ["us.anthropic.claude-3-7-sonnet-20250219-v1:0"],
        ["us.anthropic.claude-3-5-sonnet-20241022-v2:0"],
        # ["amazon.nova-pro-v1:0"],
        # ["amazon.nova-lite-v1:0"],
        # ["amazon.nova-micro-v1:0"],
        # ["anthropic.claude-3-5-haiku-20241022-v1:0"],
        # ["anthropic.claude-3-5-sonnet-20240620-v1:0"],
        # ["meta.llama3-3-70b-instruct-v1:0"],
    ],
    reflection_range=[0, 1, 3],
)
hive_client = Hive()
results = hive_client.optimise(
    test_dataset,
    trial_config,
    evaluator=math_in_tags,
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
df.to_csv(current_dir + "/new_grid_search_math500.csv", index=False)
