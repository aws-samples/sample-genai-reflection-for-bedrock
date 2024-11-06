import pydantic
from bhive import cost, config


class BudgetConfig(pydantic.BaseModel):
    max_dollar_per_sample: float = pydantic.Field(ge=0.0)
    max_seconds_per_sample: float = pydantic.Field(ge=0.0)
    cost_dictionary: dict[str, cost.TokenPrices] = cost.MODELID_COSTS_PER_TOKEN

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


def answers_equal(expected_response: str, generated_response: str) -> bool:
    """Compares the string responses directly"""
    return expected_response.strip().lower() == generated_response.strip().lower()


def answer_in_text(expected_response: str, generated_response: str) -> bool:
    """Checks if the expected response is present in the generated response."""
    return expected_response.strip().lower() in generated_response.strip().lower()


def answer_in_tags(expected_response: str, generated_response: str, tag: str = "<answer>") -> bool:
    """Parses using the tag and checks against the expected response"""
    generated_response = generated_response.split(tag)[1]
    generated_response = generated_response.split("</answer>")[0]
    return answers_equal(expected_response, generated_response)
