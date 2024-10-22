import functools
import boto3
from bhive import logger
from botocore.config import Config
from bhive import prompt, config, chat
from bhive.utils import parse_bedrock_output, parallel_bedrock_exec

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
            self.runtime_client = boto3.client(service_name=_RUNTIME_CLIENT_NAME, config=client_config)
        elif client:
            self.runtime_client = client
        else:
            raise ValueError("Either client_config or client must be provided.")

    def converse(self, message: str, config: config.HiveConfig, **converse_kwargs) -> chat.HiveOutput:
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
            modelid = config.bedrock_model_ids[0]
            logger.info(f"Calling {modelid} with no self-reflection")
            answer = _converse_func(model_id=modelid, messages=chatlog.history[modelid])
            chatlog.add_assistant_msg(answer, modelid)
            response = answer

        elif config.multi_model_single_call:
            # multi model / single round debate
            logger.info(f"Calling {config.bedrock_model_ids} with no self-reflection")
            answers = parallel_bedrock_exec(
                _converse_func,
                model_ids=config.bedrock_model_ids,
                chathistory=chatlog.history,
            )
            for modelid, answer in answers.items():
                chatlog.add_assistant_msg(answer, modelid)

            if config.aggregator_model_id:
                # aggregate an answer
                agg_msg = prompt.aggregate
                for recent_ans in answers:
                    agg_msg += f"\n\n One agent response: ```{recent_ans}```"
                agg_msg += f"\n\n As a reminder, the original question is {message}"
                fmt_msg = chatlog.wrap_user_msg(agg_msg)
                logger.info(f"Aggregating a final response using {config.aggregator_model_id=}")
                response = self._converse(config.aggregator_model_id, [fmt_msg])
            else:
                # give back all answers
                response = [m[-1]["content"][0]["text"] for m in chatlog.history.values()]

        elif config.single_model_multi_call:
            # single model but reflection
            modelid = config.bedrock_model_ids[0]
            logger.info(f"Calling {modelid} with {config.num_reflections} rounds of self-reflection")
            for n_reflect in range(config.num_reflections + 1):
                if 0 < n_reflect:
                    reflect_msg = prompt.reflect
                    reflect_msg += f"\n\n As a reminder, the original question is {message}"
                    chatlog.add_user_msg(reflect_msg, modelid)
                answer = _converse_func(model_id=modelid, messages=chatlog.history[modelid])
                chatlog.add_assistant_msg(answer, modelid)
            response = chatlog.history[modelid][-1]["content"][0]["text"]

        else:
            # multi model + multi round debate
            logger.info(f"Calling {config.bedrock_model_ids} with {config.num_reflections} rounds of debate")
            for n_reflect in range(config.num_reflections + 1):
                if 0 < n_reflect:
                    # consider others & debate
                    for modelid in config.bedrock_model_ids:
                        recent_other_answers = chatlog.get_recent_other_answers(modelid)
                        debate_msg = prompt.debate
                        for recent_ans in recent_other_answers:
                            # NOTE could alternatively summarise messages
                            debate_msg += f"\n\n One agent response: ```{recent_ans}```"
                        debate_msg += f"\n\n {prompt.careful}"
                        debate_msg += f"\n\n As a reminder, the original question is {message}"
                        logger.debug(f"Sending request to {modelid}:\n{debate_msg}")
                        chatlog.add_user_msg(debate_msg, modelid)

                logger.info(f"Fetching debate #{n_reflect+1} answers from all {config.bedrock_model_ids=}")
                answers = parallel_bedrock_exec(
                    _converse_func,
                    model_ids=config.bedrock_model_ids,
                    chathistory=chatlog.history,
                )
                for modelid, answer in answers.items():
                    chatlog.add_assistant_msg(answer, modelid)

            if config.aggregator_model_id:
                # aggregate an answer
                agg_msg = prompt.aggregate
                for recent_ans in answers:
                    agg_msg += f"\n\n One agent response: ```{recent_ans}```"
                agg_msg += f"\n\n As a reminder, the original question is {message}"
                fmt_msg = chatlog.wrap_user_msg(agg_msg)
                logger.info(f"Aggregating a final response using {config.aggregator_model_id=}")
                response = self._converse(config.aggregator_model_id, [fmt_msg])
            else:
                # give back all answers
                response = [m[-1]["content"][0]["text"] for m in chatlog.history.values()]

        logger.info(f"Retrieved final answer of {response}")
        return chat.HiveOutput(responses=response, chat_history=chatlog.history)

    def _converse(self, model_id: str, messages: list[dict], **runtime_kwargs) -> str:
        response = self.runtime_client.converse(messages=messages, modelId=model_id, **runtime_kwargs)
        status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        if status_code != 200:
            logger.error(f"Converse call failed for {model_id=} with {status_code=}")
            return "Failed to provide a response."
        answer = parse_bedrock_output(response)
        logger.debug(f"Received answer from {model_id}:\n{answer}")
        return answer
