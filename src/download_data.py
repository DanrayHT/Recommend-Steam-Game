from huggingface_hub import hf_hub_download
from .preprocessing import load_config, resolve_path

HF_REPO_ID = "DanrayHT/steam_data_with_playtime"
USER_PARTS = ["game_info.csv", "user_final.csv"]


def download_user_files():
    data_dir = resolve_path(load_config().get("paths", "data_dir", fallback="data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename in USER_PARTS:
        local_path = data_dir / filename

        if local_path.exists():
            print(f"Skip: {local_path} already exists")
            continue

        print(f"Downloading: {filename}")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(data_dir),
        )
        print(f"Downloaded: {local_path}")


if __name__ == "__main__":
    download_user_files()