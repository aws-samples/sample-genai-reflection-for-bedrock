import sys
from loguru import logger
from bhive.client import BedrockHive as BedrockHive
from bhive.config import HiveConfig as HiveConfig

logger.remove()
logger.add(sys.stderr, level="INFO")  # users can change this to DEBUG
