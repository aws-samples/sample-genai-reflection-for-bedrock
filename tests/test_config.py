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
