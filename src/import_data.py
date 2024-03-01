from pathlib import Path

import hydra
from omegaconf import DictConfig
from sklearn import datasets
from sklearn.utils import Bunch


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def import_data(config: DictConfig) -> None:
    Path.mkdir(Path(config.dataset).parent, exist_ok=True)
    iris: Bunch = datasets.load_iris(as_frame=True)
    iris.frame.to_parquet(config.dataset)


if __name__ == "__main__":
    import_data()
