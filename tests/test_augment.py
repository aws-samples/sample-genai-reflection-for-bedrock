import io
from unittest.mock import MagicMock

import pytest

from bhive.augment import (
    augment_image,
    augment_images_in_content,
    augment_llm,
    detect_image_format,
)
from bhive.chat import ChatLog, ConverseResponse


# --- detect_image_format ---


@pytest.mark.parametrize(
    "header,expected",
    [
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 20, "jpeg"),
        (b"GIF89a" + b"\x00" * 20, "gif"),
        (b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20, "webp"),
        (b"\x00\x00\x00\x00" + b"\x00" * 20, "png"),  # fallback
    ],
)
def should_detect_image_format(header, expected):
    assert detect_image_format(header) == expected


# --- augment_llm ---


def should_parse_llm_rephrasings():
    mock_converse = MagicMock()
    mock_converse.return_value = ConverseResponse(
        answer="<q1>Rephrased one</q1>\n<q2>Rephrased two</q2>"
    )
    result = augment_llm("original question", 2, mock_converse, "model-a")
    assert result == ["Rephrased one", "Rephrased two"]
    mock_converse.assert_called_once()


def should_pad_with_original_when_llm_returns_fewer():
    mock_converse = MagicMock()
    mock_converse.return_value = ConverseResponse(answer="<q1>Only one</q1>")
    result = augment_llm("original", 3, mock_converse, "model-a")
    assert len(result) == 3
    assert result[0] == "Only one"
    assert result[1] == "original"
    assert result[2] == "original"


def should_forward_images_to_llm_rephrase():
    mock_converse = MagicMock()
    mock_converse.return_value = ConverseResponse(answer="<q1>rephrased</q1>")
    image_block = {"image": {"format": "png", "source": {"bytes": b"fake"}}}
    augment_llm("question", 1, mock_converse, "model-a", content_blocks=[image_block])
    call_messages = mock_converse.call_args[1]["messages"]
    content = call_messages[0]["content"]
    assert any("image" in b for b in content)


# --- augment_image ---


def _make_png_bytes():
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def should_augment_image_returns_valid_png():
    original = _make_png_bytes()
    result = augment_image(original)
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert result[:4] == b"\x89PNG"


def should_augment_image_produces_different_bytes():
    """At least one of several augmentations should differ from original."""
    original = _make_png_bytes()
    results = [augment_image(original) for _ in range(10)]
    assert any(r != original for r in results)


# --- augment_images_in_content ---


def should_augment_only_image_blocks_in_content():
    original_png = _make_png_bytes()
    content = [
        {"image": {"format": "png", "source": {"bytes": original_png}}},
        {"text": "some question"},
    ]
    result = augment_images_in_content(content)
    assert result[1]["text"] == "some question"
    assert isinstance(result[0]["image"]["source"]["bytes"], bytes)
    assert len(result[0]["image"]["source"]["bytes"]) > 0


# --- Hive._apply_augmentation ---


def _make_chatlog(n_models, text="What is 2+2?", content=None):
    """Helper to create a ChatLog with n_models slots."""
    if content is None:
        content = [{"text": text}]
    messages = [{"role": "user", "content": content}]
    model_ids = ["model-a"] * n_models
    return ChatLog(model_ids, messages)


def _make_hive(mock_client):
    from bhive.client import Hive

    return Hive(client=mock_client)


def should_apply_llm_augmentation_to_chatlog(mocker):
    from bhive import config

    mock_client = mocker.MagicMock()
    # The rephrase call goes through _converse -> runtime_client.converse
    mock_client.converse.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "<q1>Variant A</q1>\n<q2>Variant B</q2>"}],
            }
        },
        "usage": {"inputTokens": 5, "outputTokens": 5},
        "metrics": {"latencyMs": 10},
        "trace": {},
        "stopReason": "end_turn",
    }
    hive = _make_hive(mock_client)
    chatlog = _make_chatlog(3)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"] * 3,
        augmentation_method="semantic",
        aggregator_model_id="model-a",
    )
    hive._apply_augmentation(cfg, chatlog, hive._converse)
    assert chatlog.history[0].chat_history[0]["content"][0]["text"] == "What is 2+2?"
    assert chatlog.history[1].chat_history[0]["content"][0]["text"] == "Variant A"
    assert chatlog.history[2].chat_history[0]["content"][0]["text"] == "Variant B"


def should_skip_augmentation_with_no_method(mocker):
    from bhive import config

    mock_client = mocker.MagicMock()
    mock_client.converse.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {"message": {"role": "assistant", "content": [{"text": "42"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "metrics": {"latencyMs": 50},
        "trace": {},
        "stopReason": "end_turn",
    }
    hive = _make_hive(mock_client)
    cfg = config.HiveConfig(bedrock_model_ids=["model-a"])
    messages = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]
    spy = mocker.spy(hive, "_apply_augmentation")
    hive.converse(messages, cfg)
    spy.assert_not_called()


def should_skip_augmentation_with_single_model(mocker):
    from bhive import config

    mock_client = mocker.MagicMock()
    hive = _make_hive(mock_client)
    chatlog = _make_chatlog(1)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"],
        augmentation_method="semantic",
    )
    hive._apply_augmentation(cfg, chatlog, hive._converse)
    mock_client.converse.assert_not_called()


def should_apply_image_augmentation_to_chatlog(mocker):
    from bhive import config

    mock_client = mocker.MagicMock()
    hive = _make_hive(mock_client)
    original_png = _make_png_bytes()
    content = [
        {"image": {"format": "png", "source": {"bytes": original_png}}},
        {"text": "Describe this image"},
    ]
    chatlog = _make_chatlog(2, content=content)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"] * 2,
        augmentation_method="visual",
        aggregator_model_id="model-a",
    )
    hive._apply_augmentation(cfg, chatlog, hive._converse)
    assert (
        chatlog.history[0].chat_history[0]["content"][0]["image"]["source"]["bytes"] == original_png
    )
    aug_bytes = chatlog.history[1].chat_history[0]["content"][0]["image"]["source"]["bytes"]
    assert isinstance(aug_bytes, bytes)
    assert len(aug_bytes) > 0
    assert chatlog.history[0].chat_history[0]["content"][1]["text"] == "Describe this image"
    assert chatlog.history[1].chat_history[0]["content"][1]["text"] == "Describe this image"


def should_skip_image_augmentation_without_images(mocker):
    from bhive import config

    mock_client = mocker.MagicMock()
    hive = _make_hive(mock_client)
    chatlog = _make_chatlog(2)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"] * 2,
        augmentation_method="visual",
        aggregator_model_id="model-a",
    )
    hive._apply_augmentation(cfg, chatlog, hive._converse)
    assert chatlog.history[0].chat_history[0]["content"][0]["text"] == "What is 2+2?"
    assert chatlog.history[1].chat_history[0]["content"][0]["text"] == "What is 2+2?"


def should_raise_for_unknown_augmentation_method(mocker):
    from bhive import config

    mock_client = mocker.MagicMock()
    hive = _make_hive(mock_client)
    chatlog = _make_chatlog(2)
    # bypass config validation by setting method after construction
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a"] * 2,
        augmentation_method="lexical",
        aggregator_model_id="model-a",
    )
    cfg.augmentation_method = "unknown"
    with pytest.raises(ValueError, match="Unknown augmentation method"):
        hive._apply_augmentation(cfg, chatlog, hive._converse)


# --- Integration: augmentation + inference ---


def should_augment_then_infer_with_mock_client(mocker):
    """End-to-end: HiveConfig with augmentation_method='semantic' augments slot inputs."""
    from bhive import client, config

    mock_client = mocker.MagicMock()

    def side_effect(**kwargs):
        messages = kwargs["messages"]
        last_text = messages[-1]["content"][-1]["text"]
        if "Rephrase" in last_text:
            return {
                "ResponseMetadata": {"HTTPStatusCode": 200},
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "<q1>Rephrased version</q1>"}],
                    }
                },
                "usage": {"inputTokens": 5, "outputTokens": 5},
                "metrics": {"latencyMs": 10},
                "trace": {},
                "stopReason": "end_turn",
            }
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "output": {"message": {"role": "assistant", "content": [{"text": "42"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "metrics": {"latencyMs": 50},
            "trace": {},
            "stopReason": "end_turn",
        }

    mock_client.converse.side_effect = side_effect
    hive = client.Hive(client=mock_client)
    cfg = config.HiveConfig(
        bedrock_model_ids=["model-a", "model-a"],
        augmentation_method="semantic",
        aggregator_model_id="model-a",
    )
    messages = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]
    result = hive.converse(messages, cfg)
    assert mock_client.converse.call_count >= 3
    assert "42" in result.response
