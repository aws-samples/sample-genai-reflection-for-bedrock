# 🐝 BedrockHive

A configurable extension to Bedrock text generation, focused on enhancing performance ...

## 📦 Installation

Follow the [package registry guidance](https://quip-amazon.com/DHVAAHndixT7/GitLab-Package-Registry) to setup a local `pip` configuration for installing GitLab packages.

Then install the library:
```bash
pip install X # TBD
```

## 💬 Usage

The model must support conversation history to be used, `Jurassic-2 Ultra` does not ...

... small example

## Contributors

Chat to [`@jackbtlr`](https://phonetool.amazon.com/users/jackbtlr) if you have feature suggestions or bug report ...

### UV

We use `uv`, a fast rust-based python tool for managing dependencies. Although you don't have to use `uv` for working on this package, I recommend you try it out and read more on [their website](https://docs.astral.sh/uv/).

Some convenient example commands are;

```bash
uv python install / list / uninstall # for handling python versions

uv add / remove / sync / lock # for handling python dependencies

uv run example.py # for running scripts inside an environment

uv run pre-commit run --all-files # for running pre-commit

uv run pytest -v # running tests (with verbose flag)
```

### Logging

Logging is handled via [`loguru`](https://github.com/Delgan/loguru) as it's very simple to use and sufficient for most use cases. It is by default set to the `INFO` level but developers can change it to `DEBUG` by running the following snippet locally:
```python
from loguru import logger

logger.remove() # removes existing logger
logger.add(sys.stderr, level="DEBUG") # adds a logger with DEBUG level
```

### Pre-Commit

[`pre-commit`](https://pre-commit.com/) is used for handling linting, type checking and other code hygiene related factors.

### Pytest

We use `pytest` as our testing framework of choice, read more about their documentation [here](https://docs.pytest.org/en/stable/). In particular, we use a convention for starting all test functions with `should_` as it encourages a more declarative mindset around test writing. If you don't use this convention, the tests will not be picked up in `pytest`.
