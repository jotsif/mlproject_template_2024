from logging import Logger
from typing import Any

import pandas as pd
from xgboost import XGBRegressor

from utils.metrics import calc_metrics


class XGBoostIris:
    def train(self, logger: Logger, config: Any) -> tuple[Any, Any]:
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
        test_predictions = model.predict(val_data[features])
        test_actuals = val_data[target]

        return model, calc_metrics(test_actuals, test_predictions)
