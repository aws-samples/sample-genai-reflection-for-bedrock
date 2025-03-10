"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import pydantic
from loguru import logger

from bhive import config, cost


class BudgetConfig(pydantic.BaseModel):
    max_dollar_per_sample: float = pydantic.Field(ge=0.0)
    max_seconds_per_sample: float = pydantic.Field(ge=0.0)
    cost_dictionary: dict[str, cost.TokenPrices] = cost.MODELID_COSTS_PER_TOKEN

    @pydantic.field_validator("cost_dictionary")
    def validate_cost_dictionary(cls, v) -> dict[str, cost.TokenPrices]:
        default_costs = cost.MODELID_COSTS_PER_TOKEN
        if v is None:
            logger.warning(
                "Using default cost dictionary, this is not actively maintained and may be out of date."
            )
        else:
            logger.info("Updating default cost dictionary with custom values.")
            default_costs.update(v)
        return default_costs

    def check_budget(self, result: "TrialResult") -> bool:
        return (
            result.avg_cost_dollars < self.max_dollar_per_sample
            and result.avg_latency_seconds < self.max_seconds_per_sample
        )


class TrialResult(pydantic.BaseModel):
    config: config.HiveConfig
    score: float
    avg_cost_dollars: float = pydantic.Field(ge=0.0)
    avg_latency_seconds: float = pydantic.Field(ge=0.0)


class GridResults(pydantic.BaseModel):
    best: TrialResult = pydantic.Field(default=None)
    individual_results: list[TrialResult] = pydantic.Field(default=[])

    def best_score(self, candidate: TrialResult):
        return self.best.score < candidate.score

    def better_resource_usage(self, candidate: TrialResult):
        return (
            candidate.score == self.best.score
            and candidate.avg_cost_dollars < self.best.avg_cost_dollars
            and candidate.avg_latency_seconds < self.best.avg_latency_seconds
        )
