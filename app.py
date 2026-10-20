from pathlib import Path
import streamlit as st
from src.preprocessing import get_model_dir, load_config, resolve_path
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

def data_state():
    config = load_config()
    save_dir = resolve_path(config.get("paths", "save_dir", fallback="data/preprocessing"))
    model_dir = get_model_dir(config)
    ready = (save_dir / "game_info_clean.csv").exists() and all(
        (model_dir / f).exists() for f in ("artifacts.pkl", "ranker.json")
    )
    return ready, save_dir, model_dir

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
    api_key = st.text_input(
        "Steam API key",
        type="password",
        key="steam_key",
    )
    st.markdown(
        "No API Key? "
        "[Get Steam API Key at here](https://steamcommunity.com/dev/apikey)"
    )

    steam_id = st.text_input(
        "SteamID64",
        max_chars=17,
        key="steam_id",
    )
    st.markdown(
        "Don't know SteamID64? "
        "[Get guide to find SteamID at here]"
        "(https://help.steampowered.com/faqs/view/2816-BE67-5B69-0FEC)"
    )

    steam_id_ok = steam_id.strip().isdigit() and len(steam_id.strip()) == 17
    if steam_id and not steam_id_ok:
        st.caption("SteamID64 must be 17 digits.")

    selected = []
    if engine:
        options = [c[len("tag_"):] for c in engine[0].tag_cols]
        selected = st.multiselect(
            "Filter tags (optional)",
            options,
            key="steam_tags",
        )

    st.caption("Press Enter after typing in a field.")

    ready = engine is not None and bool(api_key.strip()) and steam_id_ok
    if st.button(
        "OK",
        type="primary",
        key="steam_ok",
        disabled=not ready,
    ):
        try:
            with st.spinner("Fetching library..."):
                library = fetch_steam_library(api_key, steam_id)
                profile = profile_from_library(engine[0], library)
                frames = recommend_all(engine, profile, selected)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
            return

        st.session_state["results"] = {
            "info": (
                f"Steam returned {len(library)} games, "
                f"{profile.stats[0]} in dataset."
            ),
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
        st.info("Model not available.")
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
    st.markdown("**Results**")
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

def recommend_panel(engine):
    with st.container(border=True):
        if engine is None:
            st.warning("Model not found. Train it locally first (run main.py).")

        steam, pick = st.tabs(["Steam API", "Pick 5"])
        with steam:
            steam_tab(engine)
        with pick:
            pick_tab(engine)

        show_results()

def main():
    st.title("Steam Game Recommender")

    try:
        ready, save_dir, model_dir = data_state()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    engine = None
    if ready:
        stamp = (model_dir / "artifacts.pkl").stat().st_mtime_ns
        engine = load_engine(str(save_dir), stamp)

    recommend_panel(engine)

main()