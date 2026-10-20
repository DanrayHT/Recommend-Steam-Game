import getpass

from src.preprocessing import CONFIG_PATH, load_config, resolve_path
from src.recommend import (
    N_PICKS,
    Catalog,
    CompatibilityRecommender,
    MLRecommender,
    PopularityRecommender,
    fetch_steam_library,
    profile_from_library,
    profile_from_picks,
)

SETTINGS = {
    "paths": {
        "data_dir": (str, "data", None, None),
        "save_dir": (str, "data/preprocessing", None, None),
    },
    "features": {
        "min_tag_freq": (int, 1, 1, None),
    },
    "train": {
        "min_playtime": (float, 120, 0, None),
        "min_liked": (int, 4, 1, None),
        "min_game_users": (int, 20, 1, None),
        "n_candidates": (int, 150, 1, None),
        "pick5_frac": (float, 0.5, 0, 1),
        "max_train_users": (int, 0, 0, None),
        "rerun": (("ask", "yes", "no"), "ask", None, None),
    },
}

RAW_FILES = ["game_info.csv", "user_final.csv"]
PROCESSED_FILES = [
    "game_info_clean.csv",
    "user_final_clean.csv",
    "game_features.csv",
    "game_tags_multihot.csv",
]

RECOMMENDERS = {
    "1": ("Popularity", PopularityRecommender),
    "2": ("Compatibility", CompatibilityRecommender),
    "3": ("ML pick", MLRecommender),
}


def ask_yes_no(question, default=False):
    hint = "Y/n" if default else "y/N"

    while True:
        answer = input(f"{question} [{hint}]: ").strip().lower()

        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def parse_value(raw, typ, lo, hi):
    raw = raw.strip()

    if isinstance(typ, tuple):
        value = raw.lower()
        if value not in typ:
            raise ValueError(f"Choose one of: {', '.join(typ)}.")
        return value

    if typ is str:
        if not raw:
            raise ValueError("Cannot be empty.")
        return raw

    try:
        value = typ(raw)
    except ValueError:
        raise ValueError(f"Enter a number ({typ.__name__}).") from None

    if (lo is not None and value < lo) or (hi is not None and value > hi):
        low = "-inf" if lo is None else lo
        high = "+inf" if hi is None else hi
        raise ValueError(f"Value must be in [{low}, {high}].")

    return value


def same_value(old, new, typ):
    if typ in (int, float):
        try:
            return typ(old) == new
        except ValueError:
            return False
    return str(old) == str(new)


def edit_config(config):
    items = [
        (section, key, *spec)
        for section, keys in SETTINGS.items()
        for key, spec in keys.items()
    ]
    for section in SETTINGS:
        if not config.has_section(section):
            config.add_section(section)

    changed = set()

    while True:
        print("\nCurrent config:")
        for i, (section, key, _, default, _, _) in enumerate(items, 1):
            value = config.get(section, key, fallback=str(default))
            print(f"  {i:>2}) [{section}] {key} = {value}")

        choice = input("Item number to edit (Enter to finish): ").strip()

        if not choice:
            break
        if not choice.isdigit() or not 1 <= int(choice) <= len(items):
            print("Invalid number.")
            continue

        section, key, typ, default, lo, hi = items[int(choice) - 1]
        try:
            value = parse_value(input(f"New value for {key}: "), typ, lo, hi)
        except ValueError as error:
            print(f"Error: {error}")
            continue

        if not same_value(config.get(section, key, fallback=str(default)), value, typ):
            changed.add(key)
        config.set(section, key, str(value))

    if changed:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            config.write(f)
        print(f"Saved: {CONFIG_PATH}")
        if "min_tag_freq" in changed:
            print("min_tag_freq changed: data will be reprocessed.")

    return changed


def ensure_processed(config, force=False):
    data_dir = resolve_path(config.get("paths", "data_dir", fallback="data"))
    save_dir = resolve_path(config.get("paths", "save_dir", fallback="data/preprocessing"))

    missing = [f for f in PROCESSED_FILES if not (save_dir / f).exists()]
    if not missing and not force:
        return False

    if missing:
        print(f"\nMissing: {', '.join(missing)}. Preprocessing...")
    else:
        print("\nPreprocessing config changed. Reprocessing...")

    if any(not (data_dir / f).exists() for f in RAW_FILES):
        print(f"Raw data missing in {data_dir}. Downloading...")
        try:
            import src.download_data
        except ImportError:
            raise RuntimeError(
                "Missing huggingface_hub. Run: pip install huggingface_hub"
            ) from None
        src.download_data.download_user_files()

    import src.preprocessing

    src.preprocessing.main()
    return True


def maybe_train(save_dir, rerun, force=False):
    model_dir = save_dir / "model"
    ready = (model_dir / "artifacts.pkl").exists() and (model_dir / "ranker.json").exists()

    if not ready:
        print("\nNo model found. Training...")
        retrain = True
    elif force:
        print("\nData or train config changed. Retraining...")
        retrain = True
    elif rerun == "yes":
        print("\nrerun = yes. Retraining...")
        retrain = True
    elif rerun == "no":
        retrain = False
    else:
        retrain = ask_yes_no("Retrain model?")

    if retrain:
        import src.train

        src.train.main()


def ask_api_profile(catalog):
    api_key = getpass.getpass("Steam API key (hidden): ")
    steam_id = input("SteamID64 (17 digits): ")

    library = fetch_steam_library(api_key, steam_id)
    profile = profile_from_library(catalog, library)
    print(f"Steam returned {len(library)} games, {profile.stats[0]} in dataset.")
    return profile


def parse_ids(raw):
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]

    if not parts or not all(p.isdigit() for p in parts):
        raise ValueError("Enter numbers separated by commas.")

    return list(dict.fromkeys(int(p) for p in parts))


def ask_picks_profile(catalog):
    raw = input(
        f"Enter {N_PICKS} game_ids separated by commas (e.g. 730, 11230): "
    )
    ids = parse_ids(raw)

    if len(ids) != N_PICKS:
        raise ValueError(f"Need exactly {N_PICKS} different games (got {len(ids)}).")

    profile = profile_from_picks(catalog, ids)
    names = [catalog.names[catalog.id_to_idx[i]] for i in ids]
    print("Selected:", "; ".join(f"{i} ({n})" for i, n in zip(ids, names)))
    return profile


def ask_profile(catalog):
    handlers = {"1": ask_api_profile, "2": ask_picks_profile}

    while True:
        print("\nUser input:\n  1) Steam API\n  2) Pick 5 games")
        handler = handlers.get(input("Choose 1 or 2: ").strip())

        if handler is None:
            continue

        try:
            return handler(catalog)
        except (ValueError, RuntimeError) as error:
            print(f"Error: {error}")


def ask_tags(catalog, profile):
    if profile.auto_tags:
        print("Tags:", ", ".join(catalog.top_tags(profile.weights)))
        return None

    raw = input("Filter tags (comma-separated, Enter to skip): ")
    return [t for t in raw.split(",") if t.strip()]


def recommend_loop(catalog, profile):
    while True:
        print("\nCriteria:")
        for key, (name, _) in RECOMMENDERS.items():
            print(f"  {key}) {name}")

        choice = input("Choose 1, 2 or 3: ").strip()
        if choice not in RECOMMENDERS:
            continue

        name, recommender_class = RECOMMENDERS[choice]
        tags = ask_tags(catalog, profile)

        try:
            result = recommender_class(catalog).recommend(profile, k=5, tags=tags)
        except (ValueError, ImportError, FileNotFoundError) as error:
            print(f"Error: {error}")
            continue

        print(f"\nRecommendations ({name}):")
        print(result.to_string(index=False))

        if not ask_yes_no("\nTry another criterion?", default=True):
            break


def main():
    config = load_config()
    changed = set()

    if ask_yes_no("Edit config?"):
        changed = edit_config(config)

    save_dir = resolve_path(
        config.get("paths", "save_dir", fallback="data/preprocessing")
    )

    try:
        reprocessed = ensure_processed(config, force="min_tag_freq" in changed)
        train_changed = any(k in changed for k in SETTINGS["train"] if k != "rerun")
        rerun = config.get("train", "rerun", fallback="ask").strip().lower()
        maybe_train(save_dir, rerun, force=reprocessed or train_changed)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return

    catalog = Catalog(save_dir)
    profile = ask_profile(catalog)
    recommend_loop(catalog, profile)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")