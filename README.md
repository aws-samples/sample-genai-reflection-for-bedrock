# 🐝 BedrockHive

A configurable extension to Bedrock text generation, focused on enhancing reasoning performance.

## 📦 Installation

Follow the [package registry guidance](https://quip-amazon.com/DHVAAHndixT7/GitLab-Package-Registry) to setup a local `pip` configuration for installing GitLab packages.

Then install the library:
```bash
pip install bedrock_hive
```

## 💬 Usage

```python
from botocore.config import Config
from bhive import BedrockHive, HiveConfig

client_config = Config(region_name="us-east-1", connect_timeout=120, read_timeout=120, retries={"max_attempts": 5})

models = ["anthropic.claude-3-sonnet-20240229-v1:0", "mistral.mistral-large-2402-v1:0"]
n_reflections = 2
aggregator_model = models[0]

bhive_client = BedrockHive(client_config=client_config)
bhive_config = HiveConfig(
    bedrock_model_ids=models,
    num_reflections=n_reflections,
    aggregator_model_id=aggregator_model,
)

response = bhive_client.converse("What is 2 + 2?", bhive_config)
print(response)
```

## 🤝 Contributors

Chat to [`@jackbtlr`](https://phonetool.amazon.com/users/jackbtlr) if you have feature suggestions or bug report.

### UV

We use `uv`, a fast rust-based python tool for managing dependencies. Although you don't have to use `uv` for working on this package, I recommend you try it out and read more on [their website](https://docs.astral.sh/uv/).

Some convenient example commands are;

```bash
uv python install / list / uninstall # for handling python versions

uv add / remove / sync / lock # for handling python dependencies

uv run example.py # for running scripts inside an environment

```

### Logging

Logging is handled via [`loguru`](https://github.com/Delgan/loguru) as it's very simple to use and sufficient for most use cases. It is by default set to the `INFO` level but developers can change it to `DEBUG` to see more detailed output or `WARNING` to see less by running the following snippet locally:

```python
import sys
from loguru import logger

logger.remove() # removes existing logger
logger.add(sys.stderr, level="<level>") # adds a logger with DEBUG or WARNING or another level
```

### Pre-Commit

[`pre-commit`](https://pre-commit.com/) is used for handling linting, type checking and other code hygiene related factors.

```bash
uv run pre-commit run --all-files
```

### Pytest

We use `pytest` as our testing framework of choice, read more about their documentation [here](https://docs.pytest.org/en/stable/). In particular, we use a convention for starting all test functions with `should_` as it encourages a more declarative mindset around test writing. If you don't use this convention, the tests will not be picked up in `pytest`.

```bash
uv run pytest -v
```

### Sphinx

Package documentation is handled via [`sphinx`](https://www.sphinx-doc.org/en/) and can be built by running:
```bash
uv run sphinx-build -b html docs _build/html # builds documentation
open _build/html/index.html # views it locally
```

## FAQs

1. How does this scale with the number of models or rounds of reflection?
> You should expect the pipeline to scale roughly linearly with the # reflections as these calls must happen in serial but you should not see any dramatic increases as you scale the # models.

2. Can I use this with any model on Bedrock?
> The model must support conversation history to be used, this rules out certain models such as `Jurassic-2 Ultra` which do not have this capability.

3. Does it support multimodal queries?
> At the moment we only support text generation but this may be added in the future.

4. Can I authenticate with my own `boto3` client?
> Yes, you can pass an initialised client instance to `BedrockHive`, otherwise we will try to create a client from the `AWS_PROFILE` environment variable.
