from typing import Callable
from loguru import logger
from bhive import chat, prompt
from bhive.config import HiveConfig
from bhive.utils import parallel_bedrock_exec


def aggregate_last_responses(
    config: HiveConfig, chatlog: chat.ChatLog, _converse_func: Callable, message: str
) -> chat.ConverseResponse:
    last_answers = [m[-1]["content"][0]["text"] for m in chatlog.history.values()]
    agg_msg = prompt.aggregate
    for ans in last_answers:
        agg_msg += f"\n\nOne agent response: ```{ans}```\n"
        if config.verifier:
            agg_msg += apply_verification(ans, config.verifier)
    agg_msg += f"\nAs a reminder, the original question is {message}"
    fmt_msg = chatlog.wrap_user_msg(agg_msg)
    logger.info(f"Aggregating a final response using {config.aggregator_model_id=}")
    return _converse_func(config.aggregator_model_id, [fmt_msg])


def single_model_single_call(
    config: HiveConfig, chatlog: chat.ChatLog, _converse_func: Callable
) -> tuple[str, chat.ChatLog]:
    modelid = config.bedrock_model_ids[0]
    logger.info(f"Calling {modelid} with no self-reflection")
    response: chat.ConverseResponse = _converse_func(
        model_id=modelid, messages=chatlog.history[modelid]
    )
    chatlog.add_assistant_msg(response.answer, modelid)
    chatlog.update_stats(modelid, response)
    return response.answer, chatlog


def multi_model_single_call(
    config: HiveConfig, chatlog: chat.ChatLog, _converse_func: Callable, message: str
) -> tuple[str | list[str], chat.ChatLog]:
    logger.info(f"Calling {config.bedrock_model_ids} with no self-reflection")
    responses: dict[str, chat.ConverseResponse] = parallel_bedrock_exec(
        _converse_func,
        model_ids=config.bedrock_model_ids,
        chathistory=chatlog.history,
    )
    for modelid, response in responses.items():
        chatlog.add_assistant_msg(response.answer, modelid)
        chatlog.update_stats(modelid, response)

    if config.aggregator_model_id:
        # aggregate an answer
        return aggregate_last_responses(config, chatlog, _converse_func, message), chatlog
    else:
        return [m[-1]["content"][0]["text"] for m in chatlog.history.values()], chatlog


def single_model_multi_call(
    config: HiveConfig, chatlog: chat.ChatLog, _converse_func: Callable, message: str
) -> tuple[str, chat.ChatLog]:
    modelid = config.bedrock_model_ids[0]
    logger.info(f"Calling {modelid} with {config.num_reflections} rounds of self-reflection")
    for n_reflect in range(config.num_reflections + 1):
        if 0 < n_reflect:
            reflect_msg = prompt.reflect + "\n"
            if config.verifier:
                past_answer = chatlog.get_last_answer(modelid)
                reflect_msg += apply_verification(past_answer, config.verifier)
            reflect_msg += f"\nAs a reminder, the original question is {message}"
            chatlog.add_user_msg(reflect_msg, modelid)
        response: chat.ConverseResponse = _converse_func(
            model_id=modelid, messages=chatlog.history[modelid]
        )
        chatlog.add_assistant_msg(response.answer, modelid)
        chatlog.update_stats(modelid, response)
    return response.answer, chatlog


def multi_model_multi_call(
    config: HiveConfig, chatlog: chat.ChatLog, _converse_func: Callable, message: str
) -> tuple[str | list[str], chat.ChatLog]:
    logger.info(f"Calling {config.bedrock_model_ids} with {config.num_reflections} rounds")
    for n_reflect in range(config.num_reflections + 1):
        if 0 < n_reflect:
            # consider others & debate
            for modelid in config.bedrock_model_ids:
                recent_other_answers = chatlog.get_recent_other_answers(modelid)
                debate_msg = prompt.debate
                for recent_ans in recent_other_answers:
                    # NOTE could alternatively summarise messages
                    answer_text = recent_ans["content"][0]["text"]
                    debate_msg += f"\n\nOne agent response: ```{answer_text}```"
                    if config.verifier:
                        debate_msg += apply_verification(answer_text, config.verifier)
                debate_msg += (
                    f"\n\n {prompt.careful}\nAs a reminder, the original question is {message}"
                )
                logger.debug(f"Sending request to {modelid}:\n{debate_msg}")
                chatlog.add_user_msg(debate_msg, modelid)

        logger.info(f"Fetching debate #{n_reflect+1} answers from all {config.bedrock_model_ids=}")
        responses: dict[str, chat.ConverseResponse] = parallel_bedrock_exec(
            _converse_func,
            model_ids=config.bedrock_model_ids,
            chathistory=chatlog.history,
        )
        for modelid, response in responses.items():
            chatlog.add_assistant_msg(response.answer, modelid)
            chatlog.update_stats(modelid, response)

    if config.aggregator_model_id:
        # aggregate an answer
        return aggregate_last_responses(config, chatlog, _converse_func, message), chatlog
    else:
        return [m[-1]["content"][0]["text"] for m in chatlog.history.values()], chatlog


def apply_verification(past_answer: str, verifier: Callable[[str], str]) -> str:
    # Applies a verification function to add more context
    verifier_context = verifier(past_answer)
    logger.debug(f"External verification function returned: {verifier_context}")
    return (
        f"\nAn external verification function has added context to this answer: {verifier_context}"
    )
