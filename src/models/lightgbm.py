from logging import Logger
from typing import Any

import lightgbm as lgb
import pandas as pd
from omegaconf import DictConfig

from models.abstract_model import AbstractModel
from utils.metrics import calc_metrics


class LightGBM(AbstractModel):
    def train(self, logger: Logger, config: DictConfig) -> tuple[Any, Any, Any]:
        preprocess_config = config.preprocessing
        features = preprocess_config.features
        target = preprocess_config.target
        model_config = config.model
        logger.info("Loading data from %s", config.training_dataset)
        training_data = pd.read_parquet(config.training_dataset)
        train_data = training_data[training_data["train"]]
        val_data = training_data[~training_data["train"]]
        train_lgb = lgb.Dataset(train_data[features], train_data[target])
        val_lgb = lgb.Dataset(val_data[features], val_data[target])
        model = lgb.train(dict(model_config.train), train_lgb, valid_sets=[val_lgb])
        logger.info("Training LightGBM model")

        val_predictions = model.predict(val_data[features])
        val_actuals = val_data[target]

        train_predictions = model.predict(train_data[features])
        train_actuals = train_data[target]

        return (
            model,
            calc_metrics(val_actuals, val_predictions),
            calc_metrics(train_actuals, train_predictions),
        )
