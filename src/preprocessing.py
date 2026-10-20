from pathlib import Path
from configparser import ConfigParser
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CONFIG_PATH = BASE_DIR / "config.ini"


def load_config():
    config = ConfigParser()

    if not config.read(CONFIG_PATH, encoding="utf-8"):
        raise FileNotFoundError(f"config.ini not found: {CONFIG_PATH}")

    if not config.has_section("paths"):
        raise ValueError("config.ini needs a [paths] section")

    return config


def resolve_path(value):
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


class DataPreprocessor:
    REVIEW_MAPPING = {
        "overwhelmingly negative": 0,
        "very negative": 1,
        "mostly negative": 2,
        "negative": 3,
        "mixed": 4,
        "positive": 5,
        "mostly positive": 6,
        "very positive": 7,
        "overwhelmingly positive": 8,
    }

    def __init__(self, config):
        self.data_dir = resolve_path(
            config.get("paths", "data_dir", fallback="data")
        )
        self.save_dir = resolve_path(
            config.get("paths", "save_dir", fallback="data/preprocessing")
        )

    def _read_csv(self, filename, usecols=None):
        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return pd.read_csv(path, low_memory=False, usecols=usecols)

    def load_game_data(self):
        return self._read_csv("game_info.csv")

    def load_user_data(self):
        return self._read_csv(
            "user_final.csv",
            usecols=lambda c: c.strip().lower()
            in {"user_id", "game_id", "playtime_forever"},
        )

    @staticmethod
    def check_columns(df, required_columns, dataset_name):
        df.columns = df.columns.str.strip().str.lower()

        missing = set(required_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"{dataset_name} missing columns: {sorted(missing)}"
            )

        return df

    def preprocess_game_data(self, df):
        df = self.check_columns(
            df.copy(),
            {"game_id", "last_30d_avg", "name", "review", "tags"},
            "game_info.csv",
        )

        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
        df["last_30d_avg"] = pd.to_numeric(df["last_30d_avg"], errors="coerce")
        df.loc[df["last_30d_avg"] < 0, "last_30d_avg"] = np.nan

        before = len(df)
        df = df.dropna(subset=["game_id"])
        df = (
            df.sort_values("last_30d_avg", ascending=False, na_position="last")
            .drop_duplicates(subset="game_id", keep="first")
            .sort_values("game_id")
            .reset_index(drop=True)
        )
        df["game_id"] = df["game_id"].astype("int64")
        print(f"Games: dropped {before - len(df):,} rows (missing/duplicate id)")

        df["review"] = (
            df["review"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace("", pd.NA)
        )
        df["review_score"] = df["review"].map(self.REVIEW_MAPPING).astype("float64")
        df["review_is_valid"] = df["review_score"].notna()

        df["tags_list"] = df["tags"].apply(self._parse_tags)
        df["tag_count"] = df["tags_list"].apply(len)
        df["tags"] = df["tags_list"].apply("|".join)
        df["tags_is_valid"] = df["tag_count"] > 0

        return df

    @staticmethod
    def _parse_tags(value):
        if pd.isna(value):
            return []

        result = []
        seen = set()

        for tag in str(value).split("|"):
            tag = tag.strip()

            if not tag:
                continue

            key = tag.casefold()

            if key not in seen:
                result.append(tag)
                seen.add(key)

        return result

    def preprocess_user_data(self, df, valid_game_ids=None):
        df = self.check_columns(
            df.copy(),
            {"user_id", "game_id", "playtime_forever"},
            "user_final.csv",
        )[["user_id", "game_id", "playtime_forever"]]

        for column in ["user_id", "game_id", "playtime_forever"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df.loc[df["playtime_forever"] < 0, "playtime_forever"] = np.nan

        before = len(df)
        df = df.dropna(subset=["user_id", "game_id"])

        df = (
            df.sort_values("playtime_forever", ascending=False, na_position="last")
            .drop_duplicates(subset=["user_id", "game_id"], keep="first")
        )
        print(f"Users: dropped {before - len(df):,} rows (missing/duplicate)")

        if valid_game_ids is not None:
            before = len(df)
            df = df[df["game_id"].isin(valid_game_ids)]
            print(f"Users: dropped {before - len(df):,} rows (unknown game_id)")

        df[["user_id", "game_id"]] = df[["user_id", "game_id"]].astype("int64")

        return df.sort_values(["user_id", "game_id"]).reset_index(drop=True)

    def validate_game_data(self, df):
        print("=" * 42)
        print("Game data quality report")
        print(f"Total games: {len(df):,}")

        print("\nMissing values:")
        print(df[["game_id", "name", "last_30d_avg", "review", "tags"]].isna().sum())

        print("\nReview validity:")
        print(df["review_is_valid"].value_counts(dropna=False))

        print("\nTag count distribution:")
        print(df["tag_count"].value_counts().sort_index())

        print("\nPlayer-count summary:")
        print(df["last_30d_avg"].describe())

    def validate_user_data(self, df):
        print("=" * 42)
        print("User data quality report")
        print(f"Total interactions: {len(df):,}")
        print(f"Missing playtime_forever: {df['playtime_forever'].isna().sum():,}")
        print(f"Zero playtime: {(df['playtime_forever'] == 0).sum():,}")

        print("\nPlaytime summary:")
        print(df["playtime_forever"].describe())

    def save_clean_data(self, game_df, user_df):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        game_df.drop(columns=["tags_list"]).to_csv(
            self.save_dir / "game_info_clean.csv", index=False
        )
        user_df.to_csv(self.save_dir / "user_final_clean.csv", index=False)


class FeatureEngineer:
    def __init__(self, min_tag_freq=1):
        self.min_tag_freq = min_tag_freq

    def create_game_features(self, game_df):
        df = game_df.drop(columns=["tags_list"]).copy()
        df["popularity_log"] = np.log1p(df["last_30d_avg"].clip(lower=0))
        return df

    def create_tag_multihot(self, game_df):
        tags = game_df["tags_list"].apply(
            lambda items: "|".join(t.casefold() for t in items)
        )

        onehot = tags.str.get_dummies(sep="|").astype("uint8")

        keep = onehot.columns[onehot.sum() >= self.min_tag_freq]
        onehot = onehot[sorted(keep)].add_prefix("tag_")
        onehot.insert(0, "game_id", game_df["game_id"].to_numpy())

        return onehot

def main():
    config = load_config()

    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(
        min_tag_freq=config.getint("features", "min_tag_freq", fallback=1)
    )
    save_dir = preprocessor.save_dir

    game_clean = preprocessor.preprocess_game_data(preprocessor.load_game_data())
    user_clean = preprocessor.preprocess_user_data(
        preprocessor.load_user_data(),
        valid_game_ids=set(game_clean["game_id"]),
    )

    preprocessor.validate_game_data(game_clean)
    preprocessor.validate_user_data(user_clean)
    preprocessor.save_clean_data(game_clean, user_clean)

    outputs = {
        "game_features.csv": feature_engineer.create_game_features(game_clean),
        "game_tags_multihot.csv": feature_engineer.create_tag_multihot(game_clean),
    }

    for filename, frame in outputs.items():
        frame.to_csv(save_dir / filename, index=False)

    n_tags = outputs["game_tags_multihot.csv"].shape[1] - 1
    print(
        f"\nDone | games: {len(game_clean):,} | "
        f"interactions: {len(user_clean):,} | tags: {n_tags:,} | output: {save_dir}"
    )

if __name__ == "__main__":
    main()