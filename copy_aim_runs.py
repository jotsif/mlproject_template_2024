import aim

# Import DvcData is necessary for the repos to be initialised correctly
from aim.sdk.objects.plugins.dvc_metadata import DvcData  # noqa: F401


def main() -> None:
    # Copy from backup repo that does not exist in the local repo
    old_repo = aim.sdk.repo.Repo("aim_backup")
    new_repo = aim.sdk.repo.Repo(".")
    runs = old_repo.list_all_runs()
    new_runs = new_repo.list_all_runs()
    old_repo.copy_runs([run for run in runs if run not in new_runs], new_repo)


if __name__ == "__main__":
    main()
