"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Guaranteed XML-tag structured output.

Many downstream evaluation harnesses need a *single, reliable* answer string from
an otherwise verbose model response. This module provides:

  * `instruction(tag)`  - a prompt suffix instructing the model to wrap its final
                          answer in <tag></tag>.
  * `extract(text, tag)`- robust extraction of the content inside the LAST matching
                          tag pair, tolerant of casing, whitespace, and the model
                          repeating the tag during reasoning.

It is intentionally dependency-free and backward compatible: it does not alter the
existing `struct_output` (Pydantic-JSON) path and can be used independently.
"""

from __future__ import annotations

import re
from typing import Optional

_INSTRUCTION = (
    "Provide your final answer wrapped in <{tag}></{tag}> XML tags. "
    "Only the final answer belongs inside the tags; put any reasoning outside them. "
    "Do not nest or repeat the <{tag}> tags."
)


def instruction(tag: str = "answer") -> str:
    """Prompt suffix that asks the model to emit its answer inside <tag></tag>."""
    return _INSTRUCTION.format(tag=tag)


def extract(text: str, tag: str = "answer", default: Optional[str] = None) -> Optional[str]:
    """Extract the content of the last <tag>...</tag> pair in `text`.

    Returns `default` (None by default) if the tag is not found. We take the LAST
    occurrence because models often restate the tag while reasoning and settle on
    the final answer last. Matching is case-insensitive and dot-matches-newline.
    """
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    # Fallback: an opening tag with no close (truncated output) - take everything after it.
    open_only = re.search(rf"<{re.escape(tag)}>(.*)", text, re.DOTALL | re.IGNORECASE)
    if open_only:
        return open_only.group(1).strip()
    return default


def extract_all(text: str, tag: str) -> list[str]:
    """Return the content of every <tag>...</tag> pair, in order."""
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in pattern.findall(text)]
