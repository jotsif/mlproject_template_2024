# Background
The idea with this repo is to have example code
for creating a minimal setup for DVC, Hydra, Optuna and Aim.

Main ideas:
* All configurations are handled in Yaml files with minimum repetition.
* DVC pipeline is templated from the configuration files
* DVC handles the training pipeline and optuna runs
* Aim handles experiment tracking
* Most common commands are in Makefiles


# Prerequisites


## MacOS X
The code uses Yq to handle configuration files

```brew install yq```


# Run pipeline

Standard repro

```make repro```

To do a hyperparameter run and update the optimization_results.yaml file run
```make repro_with_hopt```
(the hyperoptimisation stage is frozen by default)

To edit parameters to sweep over go to model.yaml file and edit `trials` with standard Optuna
syntax.


# Run DVC Experiment

```EXPERIMENT=train.n_estimators=10 make run_exp```

these experiments will also be tracked in Aim UI.


# Start AIM

```make aim_ui```
