"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import copy

import pydantic

from bhive.cost import ConverseMetrics, ConverseUsage, TotalCost


class ConverseResponse(pydantic.BaseModel):
    answer: str
    usage: ConverseUsage
    metrics: ConverseMetrics


class HiveOutput(pydantic.BaseModel):
    response: str | list[str]
    chat_history: list["ModelChatLog"]
    usage: dict[str, ConverseUsage]
    metrics: dict[str, ConverseMetrics]
    cost: TotalCost


class ModelChatLog(pydantic.BaseModel):
    modelid: str
    chat_history: list[dict]


class ChatLog:
    _USER = "user"
    _ASSISTANT = "assistant"

    def __init__(self, model_ids: list[str], messages: list[dict]) -> None:
        self.models = model_ids
        self.history: list[ModelChatLog] = [
            ModelChatLog(modelid=m, chat_history=copy.deepcopy(messages)) for m in self.models
        ]
        self.metrics = {m: ConverseMetrics() for m in model_ids}
        self.usage = {m: ConverseUsage(modelid=m) for m in model_ids}

    def update_stats(self, modelid: str, stats: ConverseResponse):
        # update usage
        self.usage[modelid].inputTokens += stats.usage.inputTokens
        self.usage[modelid].outputTokens += stats.usage.outputTokens
        self.metrics[modelid].latencyMs += stats.metrics.latencyMs

    def add_assistant_msg(self, message: str, invoke_index: int):
        self._add_msg(message, self._ASSISTANT, invoke_index)

    def add_user_msg(self, message: str, invoke_index: int):
        self._add_msg(message, self._USER, invoke_index)

    def _add_msg(self, message: str, role: str, invoke_index: int):
        fmt_message = self._wrap_converse_msg(message, role)
        self.history[invoke_index].chat_history.append(fmt_message)

    def wrap_assistant_msg(self, message: str):
        return self._wrap_converse_msg(message, self._ASSISTANT)

    def wrap_user_msg(self, message: str):
        return self._wrap_converse_msg(message, self._USER)

    @staticmethod
    def _wrap_converse_msg(message: str, role: str) -> dict[str, str | list[dict]]:
        return {"role": role, "content": [{"text": message}]}

    def get_recent_other_answers(self, invoke_index: int) -> list[dict]:
        other_model_answers = []
        for index, model_log in enumerate(self.history):
            if index == invoke_index:
                continue
            assistant_msgs = [
                msg for msg in model_log.chat_history if msg.get("role") == self._ASSISTANT
            ]
            other_model_answers.append(assistant_msgs[-1])
        return other_model_answers

    def get_last_answer(self, invoke_index: int) -> str:
        return self.history[invoke_index].chat_history[-1]["content"][0]["text"]

    def get_all_last_text_answers(self) -> list[str]:
        return [m.chat_history[-1]["content"][0]["text"] for m in self.history]
