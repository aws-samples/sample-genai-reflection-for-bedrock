"""
Copyright © Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import copy

import pydantic

from bhive import logger
from bhive.cost import ConverseMetrics, ConverseUsage, TotalCost


class ConverseResponse(pydantic.BaseModel):
    answer: str
    usage: ConverseUsage
    metrics: ConverseMetrics


class HiveOutput(pydantic.BaseModel):
    response: str | list[str]
    chat_history: dict[str, list[dict]]
    usage: dict[str, ConverseUsage]
    metrics: dict[str, ConverseMetrics]
    cost: TotalCost


class ChatLog:
    _USER = "user"
    _ASSISTANT = "assistant"

    def __init__(self, model_ids: list[str], messages: list[dict]) -> None:
        self.models = model_ids
        self.history: dict[str, list[dict]] = {m: copy.deepcopy(messages) for m in model_ids}
        self.metrics = {m: ConverseMetrics() for m in model_ids}
        self.usage = {m: ConverseUsage(modelid=m) for m in model_ids}

    def update_stats(self, modelid: str, stats: ConverseResponse):
        # update usage
        self.usage[modelid].inputTokens += stats.usage.inputTokens
        self.usage[modelid].outputTokens += stats.usage.outputTokens
        self.metrics[modelid].latencyMs += stats.metrics.latencyMs

    def add_assistant_msg(self, message: str, modelid: str):
        self._add_msg(message, self._ASSISTANT, modelid)

    def add_user_msg(self, message: str, modelid: str):
        self._add_msg(message, self._USER, modelid)

    def _add_msg(self, message: str, role: str, modelid: str):
        fmt_message = self._wrap_converse_msg(message, role)
        if modelid in self.history:
            self.history[modelid].append(fmt_message)
        else:
            logger.error(f"No {modelid} in chat history, only {self.models}.")

    def wrap_assistant_msg(self, message: str):
        return self._wrap_converse_msg(message, self._ASSISTANT)

    def wrap_user_msg(self, message: str):
        return self._wrap_converse_msg(message, self._USER)

    @staticmethod
    def _wrap_converse_msg(message: str, role: str) -> dict[str, str | list[dict]]:
        return {"role": role, "content": [{"text": message}]}

    def get_recent_other_answers(self, your_model_id: str) -> list[dict]:
        other_model_answers = []
        for m, log in self.history.items():
            if m == your_model_id:
                continue
            assistant_msgs = [msg for msg in log if msg.get("role") == self._ASSISTANT]
            other_model_answers.append(assistant_msgs[-1])
        return other_model_answers

    def get_last_answer(self, model_id: str) -> str:
        return self.history[model_id][-1]["content"][0]["text"]
