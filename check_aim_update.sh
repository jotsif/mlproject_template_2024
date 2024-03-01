#!/bin/bash
let master_timestamp=$(git show --no-patch --format=%ct HEAD)
let aim_db_timestamp=$(stat -f%c .aim/run_metadata.sqlite)

# Check if AIM db has changed since last commit on master
if [ "$aim_db_timestamp" -gt "$master_timestamp" ]; then
	echo "Aim db has changed since last commit on master"
	# Check if the .aim_backup_name file has been updated
	# and if it is commited to git
	db_name_changed_master=$((git diff --no-patch --exit-code HEAD .aim_backup_name); echo $?)
	if [ $db_name_changed_master -eq 1 ]; then
		echo "- A backup of the aim db has been made"
	else
		echo "ERROR: But a new backup has not been made. Run make upload_aim_db and git add .aim_backup_name"
		exit 1
	fi
	db_name_changed_branch=$((git diff --no-patch --exit-code .aim_backup_name); echo $?)
	if [ $db_name_changed_branch -eq 0 ]; then
		echo "- The new backup name has been added to git"
		exit 0
	else
		echo "ERROR: But the new name has not been added to git. Run git add .aim_backup_name"
		exit 1
	fi
fi
