"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

from typing import Callable, Literal

import pydantic
from bhive import logger

AugmentationMethod = Literal["semantic", "lexical", "visual"]


class HiveConfig(pydantic.BaseModel):
    """
    Configuration class for Hive, managing model settings and validation.

    Attributes:
        bedrock_model_ids (list[str]): A list of Bedrock model identifiers, duplicate ids lead to parallel samples.
        num_reflections (int): The number of reflections to perform, must be zero or positive.
        aggregator_model_id (str | None): An optional aggregator model, to combine multiple model's responses.
        verifier (Callable[[str], str] | None): An optional callable for verifying thinking steps.
        use_prompt_caching (bool): An optional flag to enable Bedrock prompt caching during reflection.
        output_model (type[pydantic.BaseModel]): An optional Pydantic BaseModel for structured outputs.
        max_reasoning_seconds (type[int]): An optional maximum reasoning time in seconds before returning a response.
        augmentation_method (str | None): Input augmentation strategy: 'semantic', 'lexical', or 'visual'.
        augmentation_model_id (str | None): Model used for 'semantic' augmentation. Defaults to first model if not provided.
    """

    bedrock_model_ids: list[str]
    num_reflections: int = pydantic.Field(default=0, ge=0)
    aggregator_model_id: str | None = None
    verifier: Callable[[str], str] | None = None
    use_prompt_caching: bool = False
    output_model: type[pydantic.BaseModel] | None = None
    max_reasoning_seconds: float | None = pydantic.Field(default=None, gt=0)
    augmentation_method: AugmentationMethod | None = None
    augmentation_model_id: str | None = None

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
        if self.n_models == 1 and self.no_reflections and self.verifier:
            raise ValueError("verifier cannot be provided when using a single model call.")
        if self.use_prompt_caching:
            logger.warning("Cache read / write pricing is approximate but may not be exact.")
        if self.augmentation_method and self.n_models < 2:
            logger.warning(
                "Augmentation requires multiple model slots (duplicate model IDs) to be effective."
            )
        if self.augmentation_model_id and not self.augmentation_method:
            logger.warning("augmentation_model_id has no effect without augmentation_method.")
        if self.augmentation_method == "semantic" and not self.augmentation_model_id:
            self.augmentation_model_id = self.bedrock_model_ids[0]
        return self

    @property
    def n_models(self) -> int:
        return len(self.bedrock_model_ids)

    @property
    def no_reflections(self) -> bool:
        return self.num_reflections == 0


class TrialConfig(pydantic.BaseModel):
    """Configuration class for Hive trials, managing trial settings and validation."""

    bedrock_model_combinations: list[list[str]]
    reflection_range: list[int] = pydantic.Field(default=[0])
    aggregator_model_ids: list[str | None] | None = pydantic.Field(default=[None])
    verifier_functions: list[Callable[[str], str] | None] | None = pydantic.Field(default=[None])
    use_prompt_caching: bool = False
    augmentation_methods: list[AugmentationMethod | None] = pydantic.Field(default=[None])

    def _all_configuration_options(self) -> list[HiveConfig]:
        """Captures all valid combinations for the grid search"""
        config_params = []
        verifiers = self.verifier_functions if self.verifier_functions else [None]
        aggregators = self.aggregator_model_ids if self.aggregator_model_ids else [None]
        for model_ids in self.bedrock_model_combinations:
            for reflection_val in self.reflection_range:
                for verifier in verifiers:
                    for aggregator in aggregators:
                        for aug_method in self.augmentation_methods:
                            try:
                                _config = HiveConfig(
                                    bedrock_model_ids=model_ids,
                                    num_reflections=reflection_val,
                                    aggregator_model_id=aggregator,
                                    verifier=verifier,
                                    use_prompt_caching=self.use_prompt_caching,
                                    augmentation_method=aug_method,
                                )
                                config_params.append(_config)
                            except Exception as e:
                                logger.warning(f"Skipping invalid configuration, error:{e}")
        return config_params
