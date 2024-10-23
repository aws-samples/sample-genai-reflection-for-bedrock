import pytest
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
    with pytest.raises(ValueError):
        HiveConfig(
            bedrock_model_ids=["FAKE_MODEL"],
            num_reflections=num_reflections,
            aggregator_model_id="FAKE_MODEL",
        )


def should_raise_error_for_duplicate_model_ids():
    with pytest.raises(ValueError):
        HiveConfig(
            bedrock_model_ids=["FAKE_MODEL", "FAKE_MODEL"],
            num_reflections=10,
            aggregator_model_id="FAKE_MODEL",
        )
