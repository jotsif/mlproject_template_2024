import json
import logging
import os
import pickle
from pathlib import Path

import hydra
from aim import Run
from aim.sdk.objects.plugins.dvc_metadata import DvcData
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

    # Use DVC experiment name if available, otherwise use the experiment name
    # from the config and override experiment name from Hydra
    experiment_name = os.environ.get("DVC_EXP_NAME", config.experiment_name)

    aim_run = Run(
        experiment=experiment_name,
        log_system_params=True,
        capture_terminal_logs=True,
        system_tracking_interval=1.0,
    )

    aim_run["hparams"] = {
        "model_params": model_config.train,
        "run_mode": str(hydra_run_mode),
    }

    # track DVC metadata
    path_to_dvc_repo = config.dvc_dir
    aim_run["dvc_info"] = DvcData(path_to_dvc_repo)

    model = instantiate(model_config.model_class)

    model_object, metrics = model.train(logger=logger, config=config)

    # Track model parameters

    with open(config.metrics_file, "w") as f:
        f.write(json.dumps(metrics))
    aim_run.track(metrics)

    if hydra_run_mode == RunMode.RUN:
        # Save the model
        Path.mkdir(Path(config.model_path).parent, exist_ok=True)
        pickle.dump(model_object, open(config.model_path, "wb"))
        return None
    elif hydra_run_mode == RunMode.MULTIRUN:
        return float(metrics["r2"])

    raise ValueError(f"Unknown mode: {hydra_run_mode}")


if __name__ == "__main__":
    main()
