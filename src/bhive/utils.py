"""
Copyright © Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import concurrent.futures
from typing import Callable

from bhive import logger


def parse_bedrock_output(response: dict) -> str:
    replies = response["output"]["message"]["content"]
    logger.debug(f"Request output:\n{response['output']=}")
    logger.debug(f"Request statistics:\n{response['usage']}\n{response['metrics']}")
    if not len(replies) == 1:
        raise ValueError("Model has returned multiple or no content blocks in this response.")
    return replies[0]["text"]


def parallel_bedrock_exec(
    func: Callable, model_ids: list[str], chathistory: dict[str, list[dict]]
) -> dict:
    outputs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_input = {
            executor.submit(func, mo, me): mo for mo, me in zip(model_ids, chathistory.values())
        }
        for future in concurrent.futures.as_completed(future_to_input):
            mo = future_to_input[future]
            try:
                data = future.result()
                outputs[mo] = data
            except Exception as exc:
                raise exc
    return outputs
