import os
import dotenv
import time
import math
import statistics
import pandas as pd
from botocore.config import Config
from bhive import Hive, HiveConfig, set_logger_level

set_logger_level("DEBUG")

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
dotenv.load_dotenv(dotenv_path)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Available models and configuration
AVAILABLE_MODELS = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    # "mistral.mistral-large-2402-v1:0",
    # "amazon.titan-text-premier-v1:0",
]
BEDROCK_CONFIG = Config(
    region_name="us-east-1",
    connect_timeout=120,
    read_timeout=120,
    retries={"max_attempts": 5},
)
CLIENT = Hive(client_config=BEDROCK_CONFIG)
N_REPLICATES = 1
Q = "What is the result of 674+492*613+485-623*429? Make sure to always state your final answer in <answer> </answer> tags."


def profile_hive(models, n_reflections, replicates=N_REPLICATES):
    durations = []

    for _ in range(replicates):
        _config = HiveConfig(bedrock_model_ids=models, num_reflections=n_reflections)
        start_time = time.time()
        messages = [{"role": "user", "content": [{"text": Q}]}]
        _ = CLIENT.converse(messages, _config)  # same simple question each time
        durations.append(time.time() - start_time)

    std_err = statistics.stddev(durations) / math.sqrt(replicates) if replicates > 1 else None
    return {
        "cost": None,
        "models": models,
        "n_reflections": n_reflections,
        "latency_mean": statistics.mean(durations),
        "latency_stderr": std_err,
    }


if __name__ == "__main__":
    all_results = []

    # Iterate over combinations of models and reflection rounds
    for num_models in range(1, len(AVAILABLE_MODELS) + 1):
        models = AVAILABLE_MODELS[:num_models]
        for n_reflections in [0, 1, 2, 3, 5]:  # number of reflections
            print(f"PROFILING {models=} AND {n_reflections=}\n")
            results = profile_hive(models, n_reflections)
            all_results.append(results)

    # Convert results to DataFrame and save as JSON
    df = pd.DataFrame(all_results)
    output_file = f"{dir_path}/timing_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Profiling results saved to {output_file}")
