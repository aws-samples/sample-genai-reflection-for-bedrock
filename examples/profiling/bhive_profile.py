import os
import dotenv
import argparse
from botocore.config import Config
from bhive import Hive, HiveConfig, set_logger_level
import cProfile

set_logger_level("DEBUG")

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
dotenv.load_dotenv(dotenv_path)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Available models and configuration
AVAILABLE_MODELS = ["anthropic.claude-3-sonnet-20240229-v1:0", "mistral.mistral-large-2402-v1:0"]
BEDROCK_CONFIG = Config(region_name="us-east-1")

client = Hive(client_config=BEDROCK_CONFIG)


def profile_hive(models, n_reflections, output_file) -> None:
    pr = cProfile.Profile()
    pr.enable()

    _config = HiveConfig(bedrock_model_ids=models, num_reflections=n_reflections)
    messages = [{"role": "user", "content": [{"text": "What is 2 + 2?"}]}]
    _out = client.converse(messages, _config)  # same simple question each time
    print(_out)

    pr.disable()
    pr.dump_stats(output_file)

    print(f"Profiling results saved to {output_file}. Use SnakeViz to visualize.")


if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument("--choice", type=int, default=0)
    args = config.parse_args()

    output_file = f"{dir_path}/profiling_results.prof"  # Specify the output file for SnakeViz

    if args.choice == 0:
        # Single model call with 0 reflections
        print("Profiling single model with 0 reflections")
        profile_hive([AVAILABLE_MODELS[0]], 0, output_file)
    elif args.choice == 1:
        # Single model call with 2 reflections
        print("Profiling single model with 2 reflections")
        profile_hive([AVAILABLE_MODELS[0]], 2, output_file)
    elif args.choice == 2:
        # Multi-model call with 0 reflections
        print("Profiling multiple models with 0 reflections")
        profile_hive(AVAILABLE_MODELS, 0, output_file)
    elif args.choice == 3:
        # Multi-model call with 2 reflections
        print("Profiling multiple models with 2 reflections")
        profile_hive(AVAILABLE_MODELS, 2, output_file)
    elif args.choice == 4:
        print("Profiling same model twice with 0 reflections")
        profile_hive([AVAILABLE_MODELS[0]] * 2, 0, output_file)
    elif args.choice == 5:
        print("Profiling same model twice with 2 reflections")
        profile_hive([AVAILABLE_MODELS[0]] * 2, 2, output_file)
    else:
        print("Invalid choice. Please provide an integer between 0 and 5 inclusive.")
