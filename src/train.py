import json
import logging
import os
import pickle
from pathlib import Path

import hydra
from aim import Run
from aim.sdk.objects.plugins.dvc_metadata import DvcData
from git import Repo
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(config: DictConfig) -> float | None:
    hydra_conf = HydraConfig.get()
    hydra_run_mode = hydra_conf.mode
    model_config = config.model

    logger = logging.getLogger("train")

    # Use DVC experiment name if available, otherwise use the branch name
    repo = Repo(".")
    branch = repo.active_branch.name

    experiment_name = os.environ.get("DVC_EXP_NAME", branch)

    aim_run = Run(
        experiment=experiment_name,
        log_system_params=True,
        capture_terminal_logs=True,
        system_tracking_interval=1.0,
    )

    aim_run["hparams"] = {
        "model_params": model_config.train,
        "run_mode": str(hydra_run_mode),
        "model": model_config.model_class,
        "branch": branch,
    }

    # track DVC metadata
    path_to_dvc_repo = config.dvc_dir
    dvc_data = DvcData(path_to_dvc_repo)
    aim_run["dvc_info"] = dvc_data

    model = instantiate(model_config.model_class)

    model_object, val_metrics, train_metrics = model.train(logger=logger, config=config)

    # Track model parameters

    with open(config.metrics_file, "w") as f:
        f.write(json.dumps({"val": val_metrics, "train": train_metrics}))
    aim_run.track(val_metrics, context={"dataset": "val"})
    aim_run.track(train_metrics, context={"dataset": "train"})

    if hydra_run_mode == RunMode.RUN:
        # Save the model
        Path.mkdir(Path(config.model_path).parent, exist_ok=True)
        pickle.dump(model_object, open(config.model_path, "wb"))
        return None
    elif hydra_run_mode == RunMode.MULTIRUN:
        return float(val_metrics["r2"])

    raise ValueError(f"Unknown mode: {hydra_run_mode}")


if __name__ == "__main__":
    main()
