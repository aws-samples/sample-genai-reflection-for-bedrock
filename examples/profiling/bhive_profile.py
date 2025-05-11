import os
import dotenv
import argparse
from botocore.config import Config
from bhive import Hive, HiveConfig, set_logger_level
import cProfile

set_logger_level("WARNING")

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
dotenv.load_dotenv(dotenv_path)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Available models and configuration
AVAILABLE_MODELS = ["us.anthropic.claude-3-7-sonnet-20250219-v1:0", "amazon.nova-pro-v1:0"]
AGGREGATOR = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
BEDROCK_CONFIG = Config(region_name="us-west-2")

client = Hive(client_config=BEDROCK_CONFIG)

with open(f"{dir_path}/test_prompt.txt", "r") as f:
    sample_question = f.read()


def profile_hive(
    models, n_reflections, output_file, aggregator=None, use_prompt_caching=True
) -> None:
    pr = cProfile.Profile()
    pr.enable()

    _config = HiveConfig(
        bedrock_model_ids=models,
        num_reflections=n_reflections,
        aggregator_model_id=aggregator,
        use_prompt_caching=use_prompt_caching,
    )
    messages = [{"role": "user", "content": [{"text": sample_question}]}]
    _out = client.converse(messages, _config)  # same simple question each time

    print("OUTPUT:")
    for m, u in _out.usage.items():
        print(f"{m}: {u} {_out.metrics[m]}")
    print(_out.cost)
    print()

    pr.disable()
    pr.dump_stats(output_file)

    print(f"Profiling results saved to {output_file}. Use SnakeViz to visualize.")
    print()


if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument("--choice", default=None)
    args = config.parse_args()

    output_file = f"{dir_path}/profiling_results.prof"  # Specify the output file for SnakeViz
    choices = list(range(0, 8)) if args.choice is None else [int(args.choice)]

    for choice in choices:
        if choice == 0:
            # Single model call with 0 reflections
            print("Profiling single model with 0 reflections")
            profile_hive([AVAILABLE_MODELS[0]], 0, output_file)
        elif choice == 1:
            # Single model call with 2 reflections
            print("Profiling single model with 2 reflections")
            profile_hive([AVAILABLE_MODELS[0]], 2, output_file)
        elif choice == 2:
            # Multi-model call with 0 reflections
            print("Profiling multiple models with 0 reflections")
            profile_hive(AVAILABLE_MODELS, 0, output_file)
        elif choice == 3:
            # Multi-model call with 2 reflections
            print("Profiling multiple models with 2 reflections")
            profile_hive(AVAILABLE_MODELS, 2, output_file)
        elif choice == 4:
            print("Profiling same model twice with 0 reflections")
            profile_hive([AVAILABLE_MODELS[0]] * 2, 0, output_file)
        elif choice == 5:
            print("Profiling same model twice with 2 reflections")
            profile_hive([AVAILABLE_MODELS[0]] * 2, 2, output_file)
        elif choice == 6:
            print("Profiling multi-model twice with 2 reflections and aggregator")
            profile_hive(AVAILABLE_MODELS, 0, output_file, aggregator=AGGREGATOR)
            profile_hive(AVAILABLE_MODELS, 0, output_file, aggregator=AVAILABLE_MODELS[0])
        elif choice == 7:
            print("Profiling single model with 2 reflections and prompt caching")
            profile_hive([AVAILABLE_MODELS[0]], 2, output_file, use_prompt_caching=False)
            profile_hive([AVAILABLE_MODELS[0]], 2, output_file, use_prompt_caching=True)
        else:
            print("Invalid choice. Please provide an integer between 0 and 5 inclusive.")
