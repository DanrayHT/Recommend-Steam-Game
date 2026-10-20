import contextlib
from pathlib import Path

import streamlit as st

from main import PROCESSED_FILES, RAW_FILES, SETTINGS
from src.preprocessing import CONFIG_PATH, load_config, resolve_path, get_model_dir
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

st.set_page_config(page_title="Steam Game Recommender", layout="centered")

CRITERIA = {
    "Popularity": PopularityRecommender,
    "Compatibility": CompatibilityRecommender,
    "ML pick": MLRecommender,
}

EDITABLE = {
    section: {key: spec for key, spec in keys.items() if not isinstance(spec[0], tuple)}
    for section, keys in SETTINGS.items()
}


class LiveLog:
    def __init__(self, holder):
        self.text = ""
        self.holder = holder

    def write(self, data):
        self.text += data
        if "\n" in data:
            self.holder.code(self.text[-6000:], language=None)
        return len(data)

    def flush(self):
        pass


def get_paths(config):
    data_dir = resolve_path(config.get("paths", "data_dir", fallback="data"))
    save_dir = resolve_path(config.get("paths", "save_dir", fallback="data/preprocessing"))
    return data_dir, save_dir


def data_state():
    data_dir, save_dir = get_paths(load_config())
    model_dir = get_model_dir(load_config())
    return {
        "raw": all((data_dir / f).exists() for f in RAW_FILES),
        "processed": all((save_dir / f).exists() for f in PROCESSED_FILES),
        "model": all((model_dir / f).exists() for f in ("artifacts.pkl", "ranker.json")),
        "save_dir": save_dir,
        "model_dir": model_dir,
    }


def read_value(config, section, key, typ, default):
    try:
        return typ(config.get(section, key, fallback=str(default)))
    except ValueError:
        return typ(default)


def settings_form(config):
    values = {}

    for section, keys in EDITABLE.items():
        st.markdown(f"**[{section}]**")
        cols = st.columns(3)

        for i, (key, (typ, default, lo, hi)) in enumerate(keys.items()):
            current = read_value(config, section, key, typ, default)
            widget_key = f"cfg_{section}_{key}"

            with cols[i % 3]:
                if typ is str:
                    value = st.text_input(key, value=current, key=widget_key)
                elif typ is int:
                    value = st.number_input(
                        key,
                        min_value=lo,
                        max_value=hi,
                        value=current if lo is None else max(current, lo),
                        step=1,
                        key=widget_key,
                    )
                else:
                    value = st.number_input(
                        key,
                        min_value=None if lo is None else float(lo),
                        max_value=None if hi is None else float(hi),
                        value=float(current),
                        step=0.05 if hi is not None and hi <= 1 else 10.0,
                        format="%g",
                        key=widget_key,
                    )

            values[(section, key)] = value

    return values


def save_config(values):
    config = load_config()

    for (section, key), value in values.items():
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError(f"{key} cannot be empty")
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, key, str(value))

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        config.write(f)


def run_pipeline():
    data_dir, _ = get_paths(load_config())

    if all((data_dir / f).exists() for f in RAW_FILES):
        print("Raw data found. Skipping download.")
    else:
        print("Downloading raw data...")
        try:
            from src import download_data
        except ModuleNotFoundError as error:
            if error.name != "huggingface_hub":
                raise
            raise RuntimeError(
                "Missing huggingface_hub. Run: pip install huggingface_hub"
            ) from None
        download_data.download_user_files()

    from src import preprocessing, train

    print("Preprocessing...")
    preprocessing.main()
    print("Training...")
    train.main()


def train_panel(state):
    with st.container(border=True):
        st.subheader("Train")

        def mark(ok):
            return "found" if ok else "missing"

        st.caption(
            f"Raw data: {mark(state['raw'])} | "
            f"Processed data: {mark(state['processed'])} | "
            f"Model: {mark(state['model'])}"
        )

        message = st.session_state.pop("train_message", None)
        if message:
            st.success(message)

        with st.expander("Settings"):
            values = settings_form(load_config())

        clicked = st.button(
            "Train (download > preprocess > train)", type="primary", key="train_btn"
        )

        if clicked:
            try:
                save_config(values)
            except ValueError as error:
                st.error(str(error))
                return

            with st.status("Running pipeline...", expanded=True) as status:
                log = LiveLog(st.empty())
                try:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        run_pipeline()
                except Exception as error:
                    status.update(label="Pipeline failed", state="error")
                    st.error(f"{type(error).__name__}: {error}")
                    st.session_state["train_log"] = log.text
                    return
                status.update(label="Pipeline finished", state="complete")

            st.session_state["train_log"] = log.text
            st.session_state["train_message"] = "Pipeline finished."
            st.rerun()

        last_log = st.session_state.get("train_log")
        if last_log:
            with st.expander("Last log"):
                st.code(last_log[-8000:], language=None)


@st.cache_resource(show_spinner="Loading model...")
def load_engine(save_dir, stamp):
    catalog = Catalog(Path(save_dir))
    engines, errors = {}, {}

    for name, recommender_class in CRITERIA.items():
        try:
            engines[name] = recommender_class(catalog)
        except (ImportError, FileNotFoundError) as error:
            errors[name] = str(error)

    return catalog, engines, errors


def recommend_all(engine, profile, tags):
    _, engines, errors = engine
    frames = {}

    for name in CRITERIA:
        if name in errors:
            frames[name] = errors[name]
            continue
        try:
            frames[name] = engines[name].recommend(profile, k=5, tags=tags)
        except (ValueError, ImportError, FileNotFoundError) as error:
            frames[name] = str(error)

    return frames


def steam_tab(engine):
    api_key = st.text_input("Steam API key", type="password", key="steam_key")
    steam_id = st.text_input("SteamID64", max_chars=17, key="steam_id")

    steam_id_ok = steam_id.strip().isdigit() and len(steam_id.strip()) == 17
    if steam_id and not steam_id_ok:
        st.caption("SteamID64 must be 17 digits.")

    selected = []
    if engine:
        options = [c[len("tag_"):] for c in engine[0].tag_cols]
        selected = st.multiselect("Filter tags (optional)", options, key="steam_tags")

    st.caption("Press Enter after typing in a field.")

    ready = engine is not None and bool(api_key.strip()) and steam_id_ok
    if st.button("OK", type="primary", key="steam_ok", disabled=not ready):
        try:
            with st.spinner("Fetching library..."):
                library = fetch_steam_library(api_key, steam_id)
                profile = profile_from_library(engine[0], library)
                frames = recommend_all(engine, profile, selected)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
            return

        st.session_state["results"] = {
            "info": f"Steam returned {len(library)} games, {profile.stats[0]} in dataset.",
            "frames": frames,
        }


def add_pick(game_id):
    picks = st.session_state["picks"]
    if game_id not in picks and len(picks) < N_PICKS:
        picks.append(game_id)


def remove_pick(game_id):
    st.session_state["picks"].remove(game_id)


def pick_tab(engine):
    picks = st.session_state.setdefault("picks", [])

    if engine is None:
        st.info("Train first to enable search.")
        catalog = None
    else:
        catalog = engine[0]
        query = st.text_input(
            "Search games", key="pick_query", placeholder="Type a name and press Enter"
        )
        for row in catalog.search(query, limit=8).itertuples():
            left, right = st.columns([5, 1])
            left.write(f"{row.name} ({row.game_id})")
            right.button(
                "Add",
                key=f"add_{row.game_id}",
                on_click=add_pick,
                args=(int(row.game_id),),
                disabled=int(row.game_id) in picks or len(picks) >= N_PICKS,
            )

    st.markdown(f"**Picked {len(picks)}/{N_PICKS}**")
    for game_id in list(picks):
        left, right = st.columns([5, 1])
        name = catalog.names[catalog.id_to_idx[game_id]] if catalog else ""
        left.write(f"{name} ({game_id})")
        right.button("Remove", key=f"rm_{game_id}", on_click=remove_pick, args=(game_id,))

    ready = engine is not None and len(picks) == N_PICKS
    if st.button("OK", type="primary", key="pick_ok", disabled=not ready):
        try:
            with st.spinner("Recommending..."):
                profile = profile_from_picks(catalog, picks)
                frames = recommend_all(engine, profile, None)
        except ValueError as error:
            st.error(str(error))
            return

        st.session_state["results"] = {
            "info": "Tags: " + ", ".join(catalog.top_tags(profile.weights)),
            "frames": frames,
        }


def show_results():
    data = st.session_state.get("results")
    if not data:
        return

    st.divider()
    st.subheader("Recommendations")
    st.caption(data["info"])

    tabs = st.tabs(list(data["frames"]))
    for tab, (name, frame) in zip(tabs, data["frames"].items()):
        with tab:
            if isinstance(frame, str):
                st.error(frame)
            elif frame.empty:
                st.info("No games found.")
            else:
                st.dataframe(frame, hide_index=True)


def main():
    st.title("Steam Game Recommender")

    try:
        state = data_state()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    train_panel(state)

    engine = None
    if state["processed"] and state["model"]:
        stamp = (state["model_dir"] / "artifacts.pkl").stat().st_mtime_ns
        engine = load_engine(str(state["save_dir"]), stamp)
    else:
        st.warning("No trained model yet. Run Train first.")

    steam, pick = st.tabs(["Steam API", "Pick 5"])
    with steam:
        steam_tab(engine)
    with pick:
        pick_tab(engine)

    show_results()


main()