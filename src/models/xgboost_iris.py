from logging import Logger
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from xgboost import XGBRegressor

from models.abstract_model import AbstractModel
from utils.metrics import calc_metrics


class XGBoostIris(AbstractModel):
    def train(self, logger: Logger, config: DictConfig) -> tuple[Any, Any, Any]:
        preprocess_config = config.preprocessing
        features = preprocess_config.features
        target = preprocess_config.target
        model_config = config.model
        logger.info("Loading data from %s", config.training_dataset)
        training_data = pd.read_parquet(config.training_dataset)
        train_data = training_data[training_data["train"]]

        val_data = training_data[~training_data["train"]]
        # Train the model
        model = XGBRegressor(
            **model_config.train,
        )

        x_train = train_data[features]
        y_train = train_data[target]
        logger.info("Training an XGBoost model")

        model.fit(x_train, y_train)
        val_predictions = model.predict(val_data[features])
        val_actuals = val_data[target]

        train_predictions = model.predict(train_data[features])
        train_actuals = train_data[target]

        return (
            model,
            calc_metrics(val_actuals, val_predictions),
            calc_metrics(train_actuals, train_predictions),
        )
