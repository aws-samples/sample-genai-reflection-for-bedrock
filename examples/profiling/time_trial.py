import os
import dotenv
import time
import math
import statistics
import pandas as pd
from botocore.config import Config
from bhive import BedrockHive, HiveConfig

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
dotenv.load_dotenv(dotenv_path)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Available models and configuration
AVAILABLE_MODELS = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "amazon.titan-text-premier-v1:0",
]
BEDROCK_CONFIG = Config(
    region_name="us-east-1",
    connect_timeout=120,
    read_timeout=120,
    retries={"max_attempts": 5},
)
CLIENT = BedrockHive(client_config=BEDROCK_CONFIG)


def profile_bedrock_hive(models, n_reflections, aggregator_id, replicates):
    durations = []

    for _ in range(replicates):
        _config = HiveConfig(
            bedrock_model_ids=models,
            num_reflections=n_reflections,
            aggregator_model_id=aggregator_id,
        )

        start_time = time.time()
        _ = CLIENT.converse("What is 2 + 2?", _config)  # same simple question each time
        durations.append(time.time() - start_time)

    return {
        "models": models,
        "n_reflections": n_reflections,
        "mean": statistics.mean(durations),
        "std_err": statistics.stdev(durations) / math.sqrt(replicates),
    }


if __name__ == "__main__":
    all_results = []

    # Iterate over combinations of models and reflection rounds
    for num_models in range(1, len(AVAILABLE_MODELS) + 1):
        models = AVAILABLE_MODELS[:num_models]  # could test out variants
        for n_reflections in range(0, 3):  # number of reflections
            for use_aggregator in [False]:  # remove aggregator as constant cost
                aggregator_id = AVAILABLE_MODELS[0] if use_aggregator else None
                print(f"Profiling models: {models} with {n_reflections} reflections")
                results = profile_bedrock_hive(models, n_reflections, aggregator_id, replicates=5)
                time.sleep(5)
                all_results.append(results)

    # Convert results to DataFrame and save as JSON
    df = pd.DataFrame(all_results)
    output_file = f"{dir_path}/timing_results.json"
    df.to_json(output_file, orient="records", lines=True)
    print(f"Profiling results saved to {output_file}")
