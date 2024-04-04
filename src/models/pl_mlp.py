from logging import Logger
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from aim import Run
from omegaconf import DictConfig
from torch import nn

from models.abstract_model import AbstractModel
from utils.metrics import calc_metrics


class IrisData(pl.LightningDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: str) -> None:
        preprocess_config = self.config.preprocessing
        training_data = pd.read_parquet(self.config.training_dataset)
        train_data = training_data[training_data["train"]]
        val_data = training_data[~training_data["train"]]
        features = preprocess_config.features
        target = preprocess_config.target
        self.x_train = train_data[features]
        self.y_train = train_data[target]
        self.x_val = val_data[features]
        self.y_val = val_data[target]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(self.x_train.values, dtype=torch.float32),
                torch.tensor(self.y_train.values, dtype=torch.float32),
            ),
            batch_size=self.config.model.train.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(self.x_val.values, dtype=torch.float32),
                torch.tensor(self.y_val.values, dtype=torch.float32),
            ),
            batch_size=self.config.model.train.batch_size,
            shuffle=False,
        )


class MLP(pl.LightningModule):
    def __init__(self, config: DictConfig, aim_run: Run) -> None:
        super(MLP, self).__init__()
        self.config = config
        self.aim_run = aim_run
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.aim_run.track(
            value=loss,
            name="loss",
            epoch=self.current_epoch,
            context={"dataset": "train"},
        )
        return loss

    def valdation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.aim_run.track(
            value=loss,
            name="loss",
            epoch=self.current_epoch,
            context={"dataset": "val"},
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def val_metrics(self, data: IrisData) -> Any:
        val_predictions = []
        with torch.no_grad():
            val_actuals = data.y_val
            for xs, ys in data.val_dataloader():
                val_prediction = self.model(xs)
                val_predictions.append(val_prediction)
            metrics = calc_metrics(
                val_actuals, [v for vv in val_predictions for v in vv]
            )
        return metrics

    def train_metrics(self, data: IrisData) -> Any:
        train_predictions = []
        with torch.no_grad():
            train_actuals = data.y_train
            for xs, ys in data.train_dataloader():
                train_prediction = self.model(xs)
                train_predictions.append(train_prediction)
            metrics = calc_metrics(
                train_actuals, [v for vv in train_predictions for v in vv]
            )
        return metrics


class MLPtrainer(AbstractModel):
    def train(
        self, logger: Logger, config: DictConfig, aim_run: Run
    ) -> tuple[Any, Any, Any]:
        mlp = MLP(config.model.train, aim_run=aim_run)
        dataloader = IrisData(config)
        trainer = pl.Trainer(max_epochs=config.model.train.epochs)
        trainer.fit(mlp, dataloader)
        val_metrics = mlp.val_metrics(data=dataloader)
        train_metrics = mlp.train_metrics(data=dataloader)

        return (
            mlp,
            val_metrics,
            train_metrics,
        )
