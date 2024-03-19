MODEL=$(shell yq '.defaults[1].model' config/config.yaml)
BUCKET=$(shell poetry run dvc remote list | sed 's/.*s3:\/\/\(.*\)\//\1/')


install:
	poetry install
	poetry run aim init
	poetry run pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit


generate_config:
	# Write hydra config to params.yaml to be loaded by dvc when running pipeline
	poetry run  python3 src/train.py -c all > params.yaml

repro: generate_config
	SHELL=/bin/bash poetry run dvc repro src/dvc.yaml

repro_with_hopt: generate_config
	yq -i '.stages.hyperparameter_tuning.frozen=False' src/dvc.yaml
	SHELL=/bin/bash poetry run dvc repro src/dvc.yaml
	yq -i '.stages.hyperparameter_tuning.frozen=True' src/dvc.yaml

run_exp: generate_config
	# TODO: How to run with more than one parameter changed
	SHELL=/bin/bash poetry run dvc exp run -S config/model/$(MODEL).yaml:$(EXPERIMENT) src/dvc.yaml

aim_ui:
	poetry run aim up

dvc_pull:
	poetry run dvc pull

dvc_push:
	poetry run dvc push


upload_aim_db:
	$(eval NAME=$(shell poetry run aim runs upload $(BUCKET) | sed 's/.*\(aim.*.zip\)./\1/'))
	echo $(NAME) > .aim_backup_name
	git add .aim_backup_name


merge_aim_db_with_remote:
	aws s3 cp s3://$(BUCKET)/$(shell cat .aim_backup_name) aim_db.zip
	mkdir -p aim_backup
	unzip aim_db.zip -d aim_backup/.aim
	rm aim_db.zip
	poetry run python3 copy_aim_runs.py
	rm -rf aim_backup


clean_backups:
	# Remove all backups
	echo "NOT IMPLEMENTED"
