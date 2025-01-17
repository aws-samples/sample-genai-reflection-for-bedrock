"""
Copyright © Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

from typing import Callable

import pydantic

from bhive import logger


class HiveConfig(pydantic.BaseModel):
    """
    Configuration class for Hive, managing model settings and validation.

    Attributes:
        bedrock_model_ids (list[str]): A list of Bedrock model identifiers.
        num_reflections (int): The number of reflections to perform, must be zero or positive.
        aggregator_model_id (str | None): An optional aggregator model, to combine multiple model's responses.
        verifier (Callable[[str], str] | None): An optional callable for verifying thinking steps.
    """

    bedrock_model_ids: list[str]
    num_reflections: int = pydantic.Field(default=0, ge=0)
    aggregator_model_id: str | None = None
    verifier: Callable[[str], str] | None = None

    @pydantic.field_validator("bedrock_model_ids")
    @classmethod
    def ensure_at_least_one_model(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("bedrock_model_ids must have at least one model.")
        return v

    @pydantic.model_validator(mode="after")
    def validate_configuration(self: "HiveConfig") -> "HiveConfig":
        if self.n_models > 1 and not self.aggregator_model_id:
            logger.warning("We recommend a final aggregator_model when using multiple models.")
        if self.aggregator_model_id and self.n_models == 1:
            logger.warning("No need for an aggregator_model when using a single model.")
        if self.single_model_single_call and self.verifier:
            raise ValueError("verifier cannot be provided when using a single model call.")
        return self

    @property
    def n_models(self) -> int:
        return len(self.bedrock_model_ids)

    @property
    def no_reflections(self) -> bool:
        return self.num_reflections == 0

    @property
    def single_model_single_call(self) -> bool:
        return self.n_models == 1 and self.no_reflections

    @property
    def multi_model_single_call(self) -> bool:
        return self.n_models > 1 and self.no_reflections

    @property
    def single_model_multi_call(self) -> bool:
        return self.n_models == 1 and not self.no_reflections


class TrialConfig(pydantic.BaseModel):
    """Configuration class for Hive trials, managing trial settings and validation."""

    bedrock_model_combinations: list[list[str]]
    reflection_range: list[int] = pydantic.Field(default=[0])
    aggregator_model_ids: list[str | None] | None = pydantic.Field(default=[None])
    verifier_functions: list[Callable[[str], str] | None] | None = pydantic.Field(default=[None])

    def _all_configuration_options(self) -> list[HiveConfig]:
        """Captures all valid combinations for the grid search"""
        config_params = []
        verifiers = self.verifier_functions if self.verifier_functions else [None]
        aggregators = self.aggregator_model_ids if self.aggregator_model_ids else [None]
        for model_ids in self.bedrock_model_combinations:
            for reflection_val in self.reflection_range:
                for verifier in verifiers:
                    for aggregator in aggregators:
                        try:
                            _config = HiveConfig(
                                bedrock_model_ids=model_ids,
                                num_reflections=reflection_val,
                                aggregator_model_id=aggregator,
                                verifier=verifier,
                            )
                            config_params.append(_config)
                        except Exception as e:
                            logger.warning(f"Skipping invalid configuration, error:{e}")
        return config_params
