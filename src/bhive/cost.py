"""
Copyright © Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

# NOTE hardcoded costs is far from ideal
## but it's surprisingly hard to programmatically retrieve bedrock costs / token
## see > https://amzn-aws.slack.com/archives/C053FAVNMT3/p1718310667442179
## somebody please help if a better way is discovered :pray

import pydantic
from loguru import logger


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
    "mistral.mistral-7b-instruct-v0:2": TokenPrices(input_per_1000=0.00015, output_per_1000=0.0002),
    "mistral.mixtral-8x7b-instruct-v0:1": TokenPrices(
        input_per_1000=0.00045, output_per_1000=0.0007
    ),
}


class ConverseUsage(pydantic.BaseModel):
    inputTokens: int = 0
    outputTokens: int = 0

    @property
    def totalTokens(self) -> int:
        return self.inputTokens + self.outputTokens


class ConverseMetrics(pydantic.BaseModel):
    latencyMs: int = 0

    @property
    def latencySecs(self) -> float:
        return self.latencyMs / 1000.0


class TotalCost(pydantic.BaseModel):
    cost: float = pydantic.Field(ge=0.0)
    currency: str = "USD"


def calculate_model_cost(cost_per_token: TokenPrices, usage: ConverseUsage) -> float:
    input_price = cost_per_token.input_per_1000 * (usage.inputTokens / 1000)
    output_price = cost_per_token.output_per_1000 * (usage.outputTokens / 1000)
    return input_price + output_price


def calculate_cost(
    usage: dict[str, ConverseUsage],
    cost_dictionary: dict[str, TokenPrices] = MODELID_COSTS_PER_TOKEN,
    strict: bool = False,  # enforces that all models exist
) -> float:
    total_cost = 0.0
    for modelid, tokens in usage.items():
        if modelid.startswith(("us.", "eu.")):
            logger.info("Removing cross-region prefix from the model ID")
            modelid = modelid.split(".", 1)[1]
        cost_per_token = cost_dictionary.get(modelid)
        if cost_per_token:
            total_cost += calculate_model_cost(cost_per_token, tokens)
        else:
            msg = f"{modelid} not found in cost_dictionary and will not be included in total."
            logger.warning(msg)
            if strict:
                raise ValueError(msg)

    return total_cost


def average_latency(metrics: dict[str, ConverseMetrics]) -> float:
    latencies = [metric.latencySecs for metric in metrics.values()]
    return sum(latencies) / len(latencies)
