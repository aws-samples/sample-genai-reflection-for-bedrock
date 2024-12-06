# NOTE hardcoded costs is far from ideal
## but it's surprisingly hard to programmatically retrieve bedrock costs / token
## see > https://amzn-aws.slack.com/archives/C053FAVNMT3/p1718310667442179
## somebody please help if a better way is discovered :pray

import pydantic

from bhive.chat import ConverseMetrics, ConverseUsage


class TokenPrices(pydantic.BaseModel):
    input_per_1000: float = pydantic.Field(ge=0.0)
    output_per_1000: float = pydantic.Field(ge=0.0)
    currency: str = "USD"


MODELID_COSTS_PER_TOKEN: dict[str, TokenPrices] = {
    "anthropic.claude-3-5-haiku-20241022-v1:0": TokenPrices(
        input_per_1000=0.008, output_per_1000=0.004
    ),
    "anthropic.claude-3-haiku-20240307-v1:0": TokenPrices(
        input_per_1000=0.00025, output_per_1000=0.00125
    ),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": TokenPrices(
        input_per_1000=0.003, output_per_1000=0.015
    ),
    "anthropic.claude-3-5-sonnet-20240620-v1:0": TokenPrices(
        input_per_1000=0.003, output_per_1000=0.015
    ),
    "anthropic.claude-3-sonnet-20240229-v1:0": TokenPrices(
        input_per_1000=0.003, output_per_1000=0.015
    ),
    "anthropic.claude-3-opus-20240229-v1:0": TokenPrices(
        input_per_1000=0.015, output_per_1000=0.075
    ),
    "amazon.nova-pro-v1:0": TokenPrices(input_per_1000=0.0008, output_per_1000=0.0032),
    "amazon.nova-lite-v1:0": TokenPrices(input_per_1000=0.00006, output_per_1000=0.00024),
    "amazon.nova-micro-v1:0": TokenPrices(input_per_1000=0.000035, output_per_1000=0.00014),
    "meta.llama3-2-90b-instruct-v1:0": TokenPrices(input_per_1000=0.00072, output_per_1000=0.00072),
    "meta.llama3-2-11b-instruct-v1:0": TokenPrices(input_per_1000=0.00016, output_per_1000=0.00016),
    "meta.llama3-2-3b-instruct-v1:0": TokenPrices(input_per_1000=0.00015, output_per_1000=0.00015),
    "meta.llama3-2-1b-instruct-v1:0": TokenPrices(input_per_1000=0.0001, output_per_1000=0.0001),
    "meta.llama3-1-70b-instruct-v1:0": TokenPrices(input_per_1000=0.00099, output_per_1000=0.00099),
    "mistral.mistral-small-2402-v1:0": TokenPrices(input_per_1000=0.001, output_per_1000=0.003),
    "mistral.mistral-large-2402-v1:0": TokenPrices(input_per_1000=0.004, output_per_1000=0.012),
    "mistral.mistral-large-2402-v1:0": TokenPrices(input_per_1000=0.004, output_per_1000=0.012),
    "mistral.mistral-7b-instruct-v0:2": TokenPrices(input_per_1000=0.00015, output_per_1000=0.0002),
    "mistral.mixtral-8x7b-instruct-v0:1" TokenPrices(input_per_1000=0.00045, output_per_1000=0.0007),
}


def calculate_model_cost(modelid: str, input_tokens: int, output_tokens: int) -> float:
    model_cost = MODELID_COSTS_PER_TOKEN.get(modelid)
    if not model_cost:
        raise ValueError(
            f"Model {modelid} not found in cost table, override `BudgetConfig.cost_dictionary.`"
        )
    input_price = model_cost.input_per_1000 * (input_tokens / 1000)
    output_price = model_cost.output_per_1000 * (output_tokens / 1000)
    return input_price + output_price


def calculate_cost(usage: dict[str, ConverseUsage]) -> float:
    total_cost = 0.0
    for modelid, tokens in usage.items():
        total_cost += calculate_model_cost(modelid, tokens.inputTokens, tokens.outputTokens)
    return total_cost


def average_latency(metrics: dict[str, ConverseMetrics]) -> float:
    latencies = [metric.latencySecs for metric in metrics.values()]
    return sum(latencies) / len(latencies)
