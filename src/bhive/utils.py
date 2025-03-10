"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import concurrent.futures
from typing import Callable

from bhive import logger
from bhive.chat import ModelChatLog
import boto3

_RUNTIME_CLIENT_NAME = "bedrock-runtime"


def parse_bedrock_output(response: dict) -> str:
    replies = response["output"]["message"]["content"]
    logger.debug(f"Request output:\n{response['output']=}")
    logger.debug(f"Request statistics:\n{response['usage']}\n{response['metrics']}")
    if not len(replies) == 1:
        raise ValueError("Model has returned multiple or no content blocks in this response.")
    return replies[0]["text"]


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
