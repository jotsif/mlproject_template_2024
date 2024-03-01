from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def prepare_training_data(config: DictConfig) -> None:
    root = Path.cwd()
    all_data = pd.read_parquet(root / config.dataset)

    # Read configs
    pre_conf = config.preprocessing
    features = pre_conf.features
    target = pre_conf.target

    # Filter the data
    all_data = all_data.dropna(subset=features + [target])

    all_data["train"] = np.random.random(len(all_data)) < pre_conf.train_fraction

    all_data.to_parquet(root / config.training_dataset)


if __name__ == "__main__":
    prepare_training_data()
