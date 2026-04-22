"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import copy
import io
import random
import re
from typing import Callable

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from PIL import Image, ImageEnhance

from bhive import logger, prompt


def detect_image_format(image_bytes: bytes) -> str:
    """Detect image format from magic bytes."""
    if image_bytes[:4] == b"\x89PNG":
        return "png"
    if image_bytes[:2] == b"\xff\xd8":
        return "jpeg"
    if image_bytes[:4] == b"GIF8":
        return "gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    return "png"


def augment_llm(
    text: str,
    n: int,
    converse_func: Callable,
    model_id: str,
    content_blocks: list[dict] | None = None,
) -> list[str]:
    """Use an LLM to rephrase the user prompt into n variants.

    If content_blocks is provided (e.g. containing images), they are included
    before the rephrase prompt so the model has full context.
    """
    rephrase_prompt = prompt.rephrase.format(n=n, prompt=text)
    content: list[dict] = []
    if content_blocks:
        content.extend(b for b in content_blocks if "image" in b)
    content.append({"text": rephrase_prompt})
    messages = [{"role": "user", "content": content}]
    response = converse_func(model_id=model_id, messages=messages)
    rephrased = []
    for i in range(1, n + 1):
        match = re.search(rf"<q{i}>(.*?)</q{i}>", response.answer, re.DOTALL)
        if match:
            rephrased.append(match.group(1).strip())
    while len(rephrased) < n:
        logger.warning(f"LLM produced {len(rephrased)}/{n} rephrasings, padding with original.")
        rephrased.append(text)
    return rephrased


def augment_character(text: str, n: int) -> list[str]:
    """Apply character-level perturbations using nlpaug."""
    augmenters = [
        nac.KeyboardAug(aug_char_p=0.1, aug_word_p=0.1),
        nac.OcrAug(aug_char_p=0.1, aug_word_p=0.1),
        nac.RandomCharAug(aug_char_p=0.1, aug_word_p=0.1),
        naw.SpellingAug(aug_p=0.1),
    ]
    results = []
    for _ in range(n):
        aug = random.choice(augmenters)
        try:
            out = aug.augment(text)
            if isinstance(out, list):
                out = out[0] if out else text
            results.append(out)
        except Exception:
            logger.warning("Character augmenter failed, using original text.")
            results.append(text)
    return results


def augment_image(image_bytes: bytes) -> bytes:
    """Apply a random visual transformation (rotate, brightness, contrast) to an image."""
    fmt = detect_image_format(image_bytes)
    img = Image.open(io.BytesIO(image_bytes))
    transform = random.choice(["rotate", "brightness", "contrast"])
    if transform == "rotate":
        img = img.rotate(random.uniform(-15, 15), expand=True)
    elif transform == "brightness":
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    else:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    buf = io.BytesIO()
    save_fmt = {"png": "PNG", "jpeg": "JPEG", "gif": "GIF", "webp": "WEBP"}[fmt]
    img.save(buf, format=save_fmt)
    return buf.getvalue()


def augment_images_in_content(content: list[dict]) -> list[dict]:
    """Augment all image blocks in a Converse API content list, preserving text."""
    new_content = copy.deepcopy(content)
    for block in new_content:
        if "image" in block:
            block["image"]["source"]["bytes"] = augment_image(block["image"]["source"]["bytes"])
    return new_content
