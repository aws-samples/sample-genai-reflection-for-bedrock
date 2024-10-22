import functools
import boto3
from bhive import inference, logger
from botocore.config import Config
from bhive import config, chat
from bhive.utils import parse_bedrock_output

_RUNTIME_CLIENT_NAME = "bedrock-runtime"


class BedrockHive:
    """
    A class for enhancing reasoning capabilities by leveraging multiple models
    and maintaining conversation history.

    Attributes:
        runtime_client (boto3.Client): The Boto3 client used to communicate
            with the Bedrock runtime service.

    Parameters:
        client_config (botocore.config.Config | None):
            Configuration for the Boto3 client. If provided, a new client
            will be created using this configuration.
        client (boto3.Client | None):
            An existing Boto3 client. If provided, it will be used directly
            instead of creating a new client. Only one of `client_config`
            or `client` should be provided.

    Raises:
        ValueError: If both `client_config` and `client` are provided, or if
            neither is provided.
    """

    def __init__(self, client_config: Config | None = None, client=None) -> None:
        """Initializes a BedrockHive instance connected to a Boto3 client.

        This constructor either creates a new Boto3 client using the provided
        `client_config` or uses the existing `client`.

        Parameters:
            client_config (botocore.config.Config | None): Configuration for the Boto3 client.
            client (boto3.Client | None): Existing Boto3 client.

        Raises:
            ValueError: If both `client_config` and `client` are provided,
                or if neither is provided.
        """
        if client and client_config:
            raise ValueError("Only one of client or client_config should be provided.")
        if client_config:
            self.runtime_client = boto3.client(
                service_name=_RUNTIME_CLIENT_NAME, config=client_config
            )
        elif client:
            self.runtime_client = client
        else:
            raise ValueError("Either client_config or client must be provided.")

    def converse(
        self, message: str, config: config.HiveConfig, **converse_kwargs
    ) -> chat.HiveOutput:
        """Invokes conversation with BedrockHive using inference parameters.

        This method sends a message to the configured models and processes
        their responses based on the provided configuration.

        Parameters:
            message (str): A message or task to be solved by the models.
            config (config.HiveConfig): An inference configuration that outlines
                the models to be used, the number of rounds of reflection/debate,
                and the choice of aggregation model.

        Returns:
            chat.HiveOutput: A response object containing the answer and
                chat history.
        """
        chatlog = chat.ChatLog(config.bedrock_model_ids)
        for m in config.bedrock_model_ids:
            chatlog.add_user_msg(message, m)
        logger.info(f"Starting reasoning chain with {message=}, {config=} and {converse_kwargs=}")

        _converse_func = functools.partial(self._converse, **converse_kwargs)
        response: str | list[str] | None = None
        if config.single_model_single_call:
            # single model call
            response = inference.single_model_single_call(config, chatlog, _converse_func)

        elif config.multi_model_single_call:
            # multi model / single round debate
            response = inference.multi_model_single_call(config, chatlog, _converse_func, message)

        elif config.single_model_multi_call:
            # single model but reflection
            response = inference.single_model_multi_call(config, chatlog, _converse_func, message)

        else:
            # multi model + multi round debate
            response = inference.multi_model_multi_call(config, chatlog, _converse_func, message)

        logger.info(f"Retrieved final answer of {response}")
        return chat.HiveOutput(response=response, chat_history=chatlog.history)

    def _converse(self, model_id: str, messages: list[dict], **runtime_kwargs) -> str:
        response = self.runtime_client.converse(
            messages=messages,
            modelId=model_id,
            **runtime_kwargs,
        )
        status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        if status_code != 200:
            logger.error(f"Converse call failed for {model_id=} with {status_code=}")
            return "Failed to provide a response."
        answer = parse_bedrock_output(response)
        logger.debug(f"Received answer from {model_id}:\n{answer}")
        return answer
