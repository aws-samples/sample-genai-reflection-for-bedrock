import pytest
from pydantic_core import ValidationError
from bhive.config import HiveConfig


@pytest.mark.parametrize(
    "model_ids",
    [(["FAKE_MODEL"]), (["claude_model_123"]), (["claude_model_123", "anthropic_new_model"])],
)
def should_allow_arbitrary_modelids(model_ids: list[str]):
    hive_config = HiveConfig(
        bedrock_model_ids=model_ids,
        num_reflections=10,
        aggregator_model_id=model_ids[0],
    )
    assert hive_config.bedrock_model_ids == model_ids
    assert hive_config.aggregator_model_id == model_ids[0]


@pytest.mark.parametrize("num_reflections", [(-5), (-1)])
def should_raise_error_for_invalid_num_reflections(num_reflections: int):
    with pytest.raises(ValidationError):
        HiveConfig(
            bedrock_model_ids=["FAKE_MODEL"],
            num_reflections=num_reflections,
            aggregator_model_id="FAKE_MODEL",
        )


# --- Augmentation config validation ---


def should_accept_valid_augmentation_methods():
    for method in ("semantic", "lexical", "visual"):
        cfg = HiveConfig(
            bedrock_model_ids=["m1", "m1"],
            augmentation_method=method,
            aggregator_model_id="m1",
        )
        assert cfg.augmentation_method == method


def should_reject_invalid_augmentation_method():
    with pytest.raises(ValidationError):
        HiveConfig(bedrock_model_ids=["m1"], augmentation_method="invalid")


def should_default_augmentation_model_id_for_llm():
    cfg = HiveConfig(
        bedrock_model_ids=["model-a", "model-a"],
        augmentation_method="semantic",
        aggregator_model_id="model-a",
    )
    assert cfg.augmentation_model_id == "model-a"


def should_use_explicit_augmentation_model_id():
    cfg = HiveConfig(
        bedrock_model_ids=["model-a", "model-a"],
        augmentation_method="semantic",
        augmentation_model_id="model-b",
        aggregator_model_id="model-a",
    )
    assert cfg.augmentation_model_id == "model-b"


def should_allow_none_augmentation_method():
    cfg = HiveConfig(bedrock_model_ids=["m1"])
    assert cfg.augmentation_method is None
    assert cfg.augmentation_model_id is None
