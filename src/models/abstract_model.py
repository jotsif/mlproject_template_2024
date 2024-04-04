from logging import Logger
from typing import Any

from aim import Run
from omegaconf import DictConfig


class AbstractModel:
    def train(
        self, logger: Logger, config: DictConfig, aim_run: Run
    ) -> tuple[Any, Any, Any]:
        raise NotImplementedError
