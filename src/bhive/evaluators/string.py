"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""


def answers_equal(expected_response: str, generated_response: str) -> bool:
    """Compares the string responses directly"""
    return expected_response.strip().lower() == generated_response.strip().lower()


def answer_in_text(expected_response: str, generated_response: str) -> bool:
    """Checks if the expected response is present in the generated response."""
    return expected_response.strip().lower() in generated_response.strip().lower()


def answer_in_tags(expected_response: str, generated_response: str, tag: str = "<answer>") -> bool:
    """Parses using the tag and checks against the expected response"""
    generated_response = generated_response.split(tag)[1]
    back_tag = tag.replace("<", "</")
    generated_response = generated_response.split(back_tag)[0]
    return answers_equal(expected_response, generated_response)
