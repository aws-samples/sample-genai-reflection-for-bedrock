import pytest
import pydantic
from bhive import client, config


@pytest.fixture
def response_factory():
    def _create_response(
        input_message: str,
        input_tokens: int = 5,
        output_tokens: int = 10,
        latency_ms: int = 120,
    ) -> dict:
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "output": {"message": {"role": "assistant", "content": [{"text": input_message}]}},
            "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
            "metrics": {"latencyMs": latency_ms},
            "trace": {},
            "stopReason": "end_turn",
        }

    return _create_response


@pytest.fixture
def mock_runtime_client(mocker):
    return mocker.MagicMock()


def _make_hive(mock_client, response_or_side_effect):
    if callable(response_or_side_effect) or isinstance(response_or_side_effect, list):
        mock_client.converse.side_effect = response_or_side_effect
    else:
        mock_client.converse.return_value = response_or_side_effect
    return client.Hive(client=mock_client)


def _messages(text="Hello"):
    return [{"role": "user", "content": [{"text": text}]}]


def _get_call_messages(mock_client, call_index):
    return mock_client.converse.call_args_list[call_index][1]["messages"]


def _last_user_msg_text(messages):
    user_msgs = [m for m in messages if m["role"] == "user"]
    return user_msgs[-1]["content"][0]["text"]


# --- Single model, no reflection ---


def should_call_model_once_with_no_reflection(mock_runtime_client, response_factory):
    hive = _make_hive(mock_runtime_client, response_factory("answer"))
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"])
    result = hive.converse(_messages(), cfg)
    assert result.response == "answer"
    assert mock_runtime_client.converse.call_count == 1


# --- Single model, with reflection ---


def should_reflect_correct_number_of_rounds(mock_runtime_client, response_factory):
    responses = [response_factory(f"round-{i}") for i in range(4)]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"], num_reflections=3)
    result = hive.converse(_messages(), cfg)
    assert result.response == "round-3"
    assert mock_runtime_client.converse.call_count == 4  # initial + 3 reflections


def should_include_verifier_feedback_during_reflection(mock_runtime_client, response_factory):
    responses = [response_factory("wrong"), response_factory("4")]
    hive = _make_hive(mock_runtime_client, responses)

    def verifier(ctx: str) -> str:
        return "correct" if "4" in ctx else "wrong"

    cfg = config.HiveConfig(bedrock_model_ids=["model-a"], num_reflections=1, verifier=verifier)
    result = hive.converse(_messages("What is 2+2?"), cfg)
    assert result.response == "4"
    # The second call's messages should contain the reflection prompt with verifier feedback
    second_call_msgs = _get_call_messages(mock_runtime_client, 1)
    reflection_text = _last_user_msg_text(second_call_msgs)
    assert "external verifier" in reflection_text
    assert "wrong" in reflection_text


def should_include_original_question_reminder_during_reflection(
    mock_runtime_client, response_factory
):
    responses = [response_factory("first"), response_factory("second")]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"], num_reflections=1)
    hive.converse(_messages("What is 2+2?"), cfg)
    second_call_msgs = _get_call_messages(mock_runtime_client, 1)
    reflection_text = _last_user_msg_text(second_call_msgs)
    assert "original question" in reflection_text
    assert "What is 2+2?" in reflection_text


# --- Multi model, no reflection ---


def should_return_list_for_multi_model_no_reflection(mock_runtime_client, response_factory):
    hive = _make_hive(mock_runtime_client, response_factory("same"))
    cfg = config.HiveConfig(bedrock_model_ids=["model-a", "model-b"])
    result = hive.converse(_messages(), cfg)
    assert isinstance(result.response, list)
    assert len(result.response) == 2


def should_aggregate_multi_model_responses(mock_runtime_client, response_factory):
    def side_effect(**kwargs):
        model = kwargs["modelId"]
        if model == "aggregator":
            return response_factory("aggregated")
        return response_factory(f"answer-from-{model}")

    mock_runtime_client.converse.side_effect = side_effect
    hive = client.Hive(client=mock_runtime_client)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a", "model-b"],
        aggregator_model_id="aggregator",
    )
    result = hive.converse(_messages(), cfg)
    # aggregator is added to _all_models so it participates in parallel call too,
    # but the final aggregation call overwrites its chat history entry
    # get_last_answer returns all 3 model entries' last messages
    assert "aggregated" in result.response


# --- Multi model, with reflection (debate) ---


def should_debate_across_models_with_reflections(mock_runtime_client, response_factory):
    hive = _make_hive(mock_runtime_client, response_factory("debated"))
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a", "model-b"],
        num_reflections=1,
        aggregator_model_id="aggregator",
    )
    result = hive.converse(_messages(), cfg)
    # 3 models (a, b, aggregator) * 2 rounds + 1 aggregation = 7
    assert mock_runtime_client.converse.call_count == 7
    assert "debated" in result.response


def should_inject_other_agent_responses_during_debate(mock_runtime_client, response_factory):
    call_log = []

    def side_effect(**kwargs):
        call_log.append(kwargs)
        return response_factory(f"answer-{len(call_log)}")

    mock_runtime_client.converse.side_effect = side_effect
    hive = client.Hive(client=mock_runtime_client)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a", "model-b"],
        num_reflections=1,
        aggregator_model_id="agg",
    )
    hive.converse(_messages(), cfg)
    # Round 2 calls (after initial 3) should contain "One agent response"
    for call in call_log[3:6]:
        last_user = _last_user_msg_text(call["messages"])
        assert "One agent response" in last_user


def should_apply_verifier_during_debate(mock_runtime_client, response_factory):
    call_log = []

    def side_effect(**kwargs):
        call_log.append(kwargs)
        return response_factory(f"ans-{len(call_log)}")

    mock_runtime_client.converse.side_effect = side_effect
    hive = client.Hive(client=mock_runtime_client)

    def verifier(ctx: str) -> str:
        return "verified"

    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a", "model-b"],
        num_reflections=1,
        aggregator_model_id="agg",
        verifier=verifier,
    )
    hive.converse(_messages(), cfg)
    for call in call_log[3:6]:
        last_user = _last_user_msg_text(call["messages"])
        assert "external verifier" in last_user


# --- Structured output ---


class MathAnswer(pydantic.BaseModel):
    answer: int


def should_parse_structured_output_single_model(mock_runtime_client, response_factory):
    json_response = '<thinking>simple</thinking>\n<json>{"answer": 42}</json>'
    hive = _make_hive(mock_runtime_client, response_factory(json_response))
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"], output_model=MathAnswer)
    result = hive.converse(_messages(), cfg)
    assert result.parsed_response is not None
    assert result.parsed_response.answer == 42


# --- Prompt caching ---


def should_add_cache_points_when_enabled(mock_runtime_client, response_factory):
    responses = [response_factory("r1"), response_factory("r2")]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"], num_reflections=1, use_prompt_caching=True
    )
    hive.converse(_messages(), cfg)
    first_call_messages = _get_call_messages(mock_runtime_client, 0)
    assert any(
        "cachePoint" in item for item in first_call_messages[0]["content"] if isinstance(item, dict)
    )


# --- Token and metric accumulation across reflections ---


def should_accumulate_tokens_across_reflections(mock_runtime_client, response_factory):
    responses = [
        response_factory("r1", input_tokens=10, output_tokens=20),
        response_factory("r2", input_tokens=30, output_tokens=40),
    ]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"], num_reflections=1)
    result = hive.converse(_messages(), cfg)
    assert result.usage["model-a"].inputTokens == 40
    assert result.usage["model-a"].outputTokens == 60


def should_accumulate_latency_across_reflections(mock_runtime_client, response_factory):
    responses = [
        response_factory("r1", latency_ms=100),
        response_factory("r2", latency_ms=200),
    ]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"], num_reflections=1)
    result = hive.converse(_messages(), cfg)
    assert result.metrics["model-a"].latencyMs == 300


# --- Chat history tracking ---


def should_build_conversation_history_across_reflections(mock_runtime_client, response_factory):
    responses = [response_factory("first"), response_factory("second")]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"], num_reflections=1)
    result = hive.converse(_messages("question"), cfg)
    history = result.chat_history[0].chat_history
    # user, assistant, user (reflection), assistant
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "user"
    assert history[3]["role"] == "assistant"


# --- Validation ---


def should_reject_verifier_with_single_call_no_reflection():
    with pytest.raises(ValueError, match="verifier cannot be provided"):
        config.HiveConfig(
            bedrock_model_ids=["model-a"],
            num_reflections=0,
            verifier=lambda x: x,
        )


# --- max_reasoning_seconds ---


def should_exit_early_when_timeout_exceeded(mock_runtime_client, response_factory, mocker):
    # Simulate: first call takes 0s, then time jumps past the limit before round 2
    mock_time = mocker.patch("bhive.inference.time")
    mock_time.monotonic.side_effect = [0.0, 5.0]  # start=0, check before round 1=5s

    hive = _make_hive(mock_runtime_client, response_factory("initial"))
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"], num_reflections=3, max_reasoning_seconds=2.0
    )
    result = hive.converse(_messages(), cfg)
    assert result.response == "initial"
    assert mock_runtime_client.converse.call_count == 1  # only the initial call


def should_complete_all_rounds_when_timeout_not_exceeded(
    mock_runtime_client, response_factory, mocker
):
    mock_time = mocker.patch("bhive.inference.time")
    # start=0, then each check returns well under the limit
    mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 0.3]

    responses = [response_factory(f"round-{i}") for i in range(4)]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"], num_reflections=3, max_reasoning_seconds=10.0
    )
    result = hive.converse(_messages(), cfg)
    assert result.response == "round-3"
    assert mock_runtime_client.converse.call_count == 4


def should_allow_partial_reflections_before_timeout(mock_runtime_client, response_factory, mocker):
    mock_time = mocker.patch("bhive.inference.time")
    # start=0, round 1 check=1s (ok), round 2 check=6s (over limit)
    mock_time.monotonic.side_effect = [0.0, 1.0, 6.0]

    responses = [response_factory("r0"), response_factory("r1"), response_factory("r2")]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"], num_reflections=3, max_reasoning_seconds=5.0
    )
    result = hive.converse(_messages(), cfg)
    assert result.response == "r1"  # completed round 0 and 1, exited before round 2
    assert mock_runtime_client.converse.call_count == 2


def should_not_check_timeout_when_not_configured(mock_runtime_client, response_factory):
    responses = [response_factory(f"r{i}") for i in range(3)]
    hive = _make_hive(mock_runtime_client, responses)
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"], num_reflections=2)
    assert cfg.max_reasoning_seconds is None
    result = hive.converse(_messages(), cfg)
    assert result.response == "r2"
    assert mock_runtime_client.converse.call_count == 3


def should_reject_negative_max_reasoning_seconds():
    with pytest.raises(Exception):
        config.HiveConfig(bedrock_model_ids=["model-a"], max_reasoning_seconds=-1.0)


def should_reject_zero_max_reasoning_seconds():
    with pytest.raises(Exception):
        config.HiveConfig(bedrock_model_ids=["model-a"], max_reasoning_seconds=0.0)
