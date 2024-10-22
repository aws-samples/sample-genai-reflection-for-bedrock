import pytest
from botocore.config import Config
from bhive import client


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
    bedrock_hive = client.BedrockHive(client_config=example_boto_config)
    mock_boto_client.assert_called_once_with(
        service_name=client._RUNTIME_CLIENT_NAME, config=example_boto_config
    )
    assert bedrock_hive.runtime_client == mock_runtime_client


def should_instantiate_using_client(mock_runtime_client):
    bedrock_hive = client.BedrockHive(client=mock_runtime_client)
    assert bedrock_hive.runtime_client == mock_runtime_client


def should_fail_when_using_neither_client_or_config():
    with pytest.raises(ValueError):
        client.BedrockHive()


def should_fail_when_using_client_and_config():
    with pytest.raises(ValueError):
        client.BedrockHive(client_config="example", client="example")
