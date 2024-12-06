"""
Copyright © Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import sys

from loguru import logger

from bhive.client import Hive as Hive
from bhive.config import HiveConfig as HiveConfig
from bhive.config import TrialConfig as TrialConfig
from bhive.cost import TokenPrices as TokenPrices
from bhive.evaluators import BudgetConfig as BudgetConfig

LOGGER_LEVELS = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


def set_logger_level(level: str) -> None:
    # users can change this to DEBUG
    if level not in LOGGER_LEVELS:
        raise ValueError(f"Invalid logger level: {level}, must be one of {LOGGER_LEVELS}")
    logger.remove()
    logger.add(sys.stderr, level=level)


set_logger_level("INFO")
