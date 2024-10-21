import concurrent.futures
from bhive import logger
from typing import Callable


def parse_bedrock_output(response: dict) -> str:
    replies = response["output"]["message"]["content"]
    logger.debug(f"Raw {response=}")
    assert len(replies) < 2, "Model has returned multiple content blocks for one task."
    assert 0 < len(replies), "Model has returned no content blocks for the task."
    return replies[0]["text"]


def parallel_bedrock_exec(func: Callable, model_ids: list[str], chathistory: dict[str, list[dict]]) -> dict:
    outputs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_input = {executor.submit(func, mo, me): mo for mo, me in zip(model_ids, chathistory.values())}
        for future in concurrent.futures.as_completed(future_to_input):
            mo = future_to_input[future]
            try:
                data = future.result()
                outputs[mo] = data
            except Exception as exc:
                raise exc
    return outputs
