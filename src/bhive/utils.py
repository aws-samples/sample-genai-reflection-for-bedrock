"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import concurrent.futures
from typing import Callable

import boto3

from bhive import logger
from bhive.chat import ModelChatLog

_RUNTIME_CLIENT_NAME = "bedrock-runtime"


def parse_bedrock_output(response: dict) -> tuple[str, str]:
    replies = response["output"]["message"]["content"]
    logger.debug(f"Request output:\n{response['output']=}")
    logger.debug(f"Request statistics:\n{response['usage']}\n{response['metrics']}")

    text_replies = [reply for reply in replies if "text" in reply]  # handle claude 3.7 reasoning
    text_answer = text_replies[0]["text"]

    thinking_replies = [reply for reply in replies if "reasoningContent" in reply]
    thinking = ""
    if thinking_replies:
        thinking = thinking_replies[0]["reasoningContent"]["reasoningText"]["text"]

    return text_answer, thinking


def parallel_bedrock_exec(func: Callable, chathistory: list[ModelChatLog]) -> dict:
    outputs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_input = {
            executor.submit(func, log.modelid, log.chat_history): (i, log.modelid)
            for i, log in enumerate(chathistory)
        }
        for future in concurrent.futures.as_completed(future_to_input):
            index, modelid = future_to_input[future]
            try:
                data = future.result()
                outputs[(index, modelid)] = data
            except Exception as exc:
                raise exc
    return outputs


def create_bedrock_client(client_config: dict | None = None):
    logger.info(f"Creating Bedrock client from environment with {client_config=}.")
    return boto3.client(
        service_name=_RUNTIME_CLIENT_NAME,
        config=client_config,
    )
