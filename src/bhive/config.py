import pydantic
from bhive import logger


class HiveConfig(pydantic.BaseModel):
    bedrock_model_ids: list[str]
    num_reflections: int
    aggregator_model_id: str | None = None

    @pydantic.field_validator("num_reflections")
    @classmethod
    def ensure_positive_or_zero(cls, v: int) -> int:
        if v < 0:
            raise ValueError("num_reflections must be a positive number or equal to 0.")
        return v

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
