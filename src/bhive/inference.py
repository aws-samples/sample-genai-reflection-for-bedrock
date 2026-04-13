"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import time
from typing import Callable

from loguru import logger

from bhive import chat, prompt
from bhive.config import HiveConfig
from bhive.utils import parallel_bedrock_exec


def run_inference(
    config: HiveConfig, chatlog: chat.ChatLog, _converse_func: Callable, message: str | None = None
) -> tuple[str | list[str], chat.ChatLog]:
    is_single = config.n_models == 1
    start_time = time.monotonic()

    for n_reflect in range(config.num_reflections + 1):
        if n_reflect > 0:
            if config.max_reasoning_seconds is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= config.max_reasoning_seconds:
                    logger.info(
                        f"Exiting early at round {n_reflect}/{config.num_reflections} "
                        f"after {elapsed:.1f}s (limit: {config.max_reasoning_seconds}s)"
                    )
                    break
        if n_reflect > 0:
            if is_single:
                reflect_msg = prompt.reflect + "\n"
                if config.verifier:
                    past_answer = chatlog.get_last_answer()
                    reflect_msg += apply_verification(past_answer, config.verifier)  # type: ignore[arg-type]
                if message:
                    reflect_msg += f"\nAs a reminder, the original question is {message}"
                chatlog.add_user_msg(reflect_msg, invoke_index=0)
            else:
                for index in range(config.n_models):
                    recent_other_answers = chatlog.get_recent_other_answers(index)
                    debate_msg = prompt.debate
                    for recent_ans in recent_other_answers:
                        answer_text = recent_ans["content"][0]["text"]
                        debate_msg += f"\n\nOne agent response: ```{answer_text}```"
                        if config.verifier:
                            debate_msg += apply_verification(answer_text, config.verifier)
                    debate_msg += f"\n\n {prompt.careful}\n"
                    if message:
                        debate_msg += f"\nAs a reminder, the original question is {message}"
                    chatlog.add_user_msg(debate_msg, index)

        if is_single:
            modelid = config.bedrock_model_ids[0]
            response = _converse_func(model_id=modelid, messages=chatlog.history[0].chat_history)
            _record_response(chatlog, 0, modelid, response)
        else:
            responses = parallel_bedrock_exec(_converse_func, chathistory=chatlog.history)
            for (index, modelid), response in responses.items():
                _record_response(chatlog, index, modelid, response)

    if config.aggregator_model_id:
        chatlog = aggregate_last_responses(config, chatlog, _converse_func, message)

    return chatlog.get_last_answer(), chatlog


def _record_response(
    chatlog: chat.ChatLog, index: int, modelid: str, response: chat.ConverseResponse
):
    chatlog.add_assistant_msg(response.answer, index)
    if response.thinking:
        chatlog.add_thinking_trace(response.thinking, index)
    chatlog.update_stats(modelid, response)
    chatlog.add_stop_reason(response.stopReason)
    chatlog.add_trace(response.trace)


def aggregate_last_responses(
    config: HiveConfig, chatlog: chat.ChatLog, _converse_func: Callable, message: str | None = None
) -> chat.ChatLog:
    assert isinstance(config.aggregator_model_id, str), (
        f"Must have a valid model id to aggregate responses, found {config.aggregator_model_id=} "
    )

    agg_msg = prompt.aggregate
    for ans in chatlog.get_last_answer():
        agg_msg += f"\n\nOne agent response: ```{ans}```\n"
        if config.verifier:
            agg_msg += apply_verification(ans, config.verifier)
    if message:
        agg_msg += f"\nAs a reminder, the original question is {message}"
    fmt_msg = chatlog.wrap_user_msg(agg_msg)
    logger.info(f"Aggregating a final response using {config.aggregator_model_id=}")
    response: chat.ConverseResponse = _converse_func(config.aggregator_model_id, [fmt_msg])

    _record_response(chatlog, 0, config.aggregator_model_id, response)

    return chatlog


def apply_verification(past_answer: str, verifier: Callable[[str], str]) -> str:
    verifier_context = verifier(past_answer)
    logger.debug(f"External verification function returned: {verifier_context}")
    return f"An external verifier has added the following to this answer: {verifier_context}"
