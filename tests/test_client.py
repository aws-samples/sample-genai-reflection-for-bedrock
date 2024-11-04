import pytest
from botocore.config import Config
from bhive import client, config


@pytest.fixture
def response_factory():
    def _create_response(
        input_message: str,
        role: str = "assistant",
        input_tokens: int = 5,
        output_tokens: int = 10,
        latency_ms: int = 120,
    ) -> dict:
        message = {"role": role, "content": [{"text": input_message}]}
        usage = {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        }
        response_status = {"HTTPStatusCode": 200}
        return {
            "ResponseMetadata": response_status,
            "output": {"message": message},
            "usage": usage,
            "metrics": {"latencyMs": latency_ms},
        }

    return _create_response


@pytest.fixture
def example_boto_config():
    return Config(
        region_name="us-east-1",
        signature_version="v4",
        retries={"max_attempts": 10, "mode": "standard"},
    )


@pytest.fixture
def mock_boto_client(mocker):
    return mocker.patch("boto3.client")


@pytest.fixture
def mock_runtime_client(mocker):
    return mocker.MagicMock()


def should_instantiate_using_client_config(
    mock_boto_client, mock_runtime_client, example_boto_config
):
    mock_boto_client.return_value = mock_runtime_client
    bedrock_hive = client.Hive(client_config=example_boto_config)
    mock_boto_client.assert_called_once_with(
        service_name=client._RUNTIME_CLIENT_NAME, config=example_boto_config
    )
    assert bedrock_hive.runtime_client == mock_runtime_client


def should_instantiate_using_client(mock_runtime_client):
    bedrock_hive = client.Hive(client=mock_runtime_client)
    assert bedrock_hive.runtime_client == mock_runtime_client


def should_fail_when_using_neither_client_or_config():
    with pytest.raises(ValueError):
        client.Hive()


def should_fail_when_using_client_and_config():
    with pytest.raises(ValueError):
        client.Hive(client_config="example", client="example")


@pytest.mark.parametrize("message", ["test", "test2"])
def should_return_correct_response(message, mock_runtime_client, response_factory):
    mock_runtime_client.converse.return_value = response_factory(message)
    bedrock_hive = client.Hive(client=mock_runtime_client)
    _config = config.HiveConfig(bedrock_model_ids=["test"])
    response = bedrock_hive.converse("Hello", _config)
    assert response.response == message


@pytest.mark.parametrize("input_tokens, output_tokens", [(5, 10), (200, 100)])
def should_correctly_count_tokens(
    input_tokens, output_tokens, mock_runtime_client, response_factory
):
    mock_runtime_client.converse.return_value = response_factory(
        "test", input_tokens=input_tokens, output_tokens=output_tokens
    )
    bedrock_hive = client.Hive(client=mock_runtime_client)
    _config = config.HiveConfig(bedrock_model_ids=["test"])
    response = bedrock_hive.converse("Hello", _config)
    assert response.usage.inputTokens == input_tokens
    assert response.usage.outputTokens == output_tokens
    assert response.usage.totalTokens == input_tokens + output_tokens


@pytest.mark.parametrize("latency_ms", [120, 300])
def should_correctly_count_latency_ms(latency_ms, mock_runtime_client, response_factory):
    mock_runtime_client.converse.return_value = response_factory("test", latency_ms=latency_ms)
    bedrock_hive = client.Hive(client=mock_runtime_client)
    _config = config.HiveConfig(bedrock_model_ids=["test"])
    response = bedrock_hive.converse("Hello", _config)
    assert response.metrics.latencyMs == latency_ms
