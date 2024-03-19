from logging import Logger
from typing import Any

from omegaconf import DictConfig


class AbstractModel:
    def train(self, logger: Logger, config: DictConfig) -> tuple[Any, Any, Any]:
        raise NotImplementedError
