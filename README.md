# Recommend Steam Game

A personal game recommender for Steam: given your library (via the Steam Web API) or just 5 games you like, it recommends new games using three complementary strategies — popularity, tag/content compatibility, and a trained ML ranker.

- **Data source:** raw data is scraped from Steam in [DanrayHT/Scrape-steam-data](https://github.com/DanrayHT/Scrape-steam-data) (distributed as the HuggingFace dataset `DanrayHT/steam_data_with_playtime`).
- **Live demo:** https://danrayht-recommend-steam-game-app-w7bsws.streamlit.app/

## 1. Goal of the project

Build a recommendation system that suggests Steam games you haven't played yet, based on:

- your owned-games library (fetched from the Steam Web API), or
- as little as **5 game IDs** you pick manually.

Three recommender engines are provided so you can compare results:

| # | Engine | Idea |
|---|--------|------|
| 1 | **Popularity** | Top games by user count / playtime, excluding what you already own and below-minimum review score. |
| 2 | **Compatibility** | Games most similar to your profile via tag multi-hot similarity + SVD (TruncatedSVD) embedding similarity. |
| 3 | **ML pick** | An XGBoost ranker trained on user–game interactions (playtime, likes), scoring candidate games with features such as popularity, review score, tag count, playtime stats, and content/SVD similarity to your history. |

## 2. Project structure

```
Recommend Steam Game/
├── main.py                 # CLI entry point: config → data → train → recommend loop
├── app.py                  # Streamlit web UI (two input tabs, three engines)
├── requirements.txt        # Pinned dependencies
├── src/
│   ├── config.ini          # [paths], [features], [train] settings
│   ├── download_data.py    # Downloads game_info.csv & user_final.csv from HuggingFace
│   ├── preprocessing.py    # Cleans raw CSVs, builds features + tag multi-hot matrix
│   ├── train.py            # Trains the XGBoost ranker (model/artifacts.pkl, model/ranker.json)
│   └── recommend.py        # Catalog + Popularity/Compatibility/ML recommenders, Steam API fetch
├── data/                   # Raw CSVs (auto-downloaded) and data/preprocessing outputs
└── model/                  # Trained artifacts: artifacts.pkl, ranker.json
```

## 3. Installation

Requires Python 3.10+.

```bash
# clone the repo
git clone https://github.com/DanrayHT/Recommend-Steam-Game.git
cd "Recommend Steam Game"

# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# install dependencies
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `requests`, `huggingface_hub`, `streamlit`.

> No API key is required to run the app — raw data (`game_info.csv`, `user_final.csv`) is downloaded automatically from the HuggingFace dataset [`DanrayHT/steam_data_with_playtime`](https://huggingface.co/datasets/DanrayHT/steam_data_with_playtime), which originates from [Scrape-steam-data](https://github.com/DanrayHT/Scrape-steam-data). A Steam Web API key is only needed if you want to use the "Steam API" input mode.

## 4. Usage

### `main.py` — interactive CLI

```bash
python main.py
```

Flow:

1. **Config** — optionally edit `[paths]`, `[features]`, `[train]` values in `src/config.ini`.
2. **Data** — raw CSVs are downloaded automatically if missing, then preprocessed into `data/preprocessing/` (`game_info_clean.csv`, `user_final_clean.csv`, `game_features.csv`, `game_tags_multihot.csv`).
3. **Training** — the XGBoost model is trained (or retrained on request) and saved to `model/artifacts.pkl` + `model/ranker.json`.
4. **Profile** — build your profile either by:
   - entering a Steam Web API key + SteamID64 (your owned games are fetched via `GetOwnedGames`), or
   - picking 5 game IDs you like.
5. **Recommend loop** — choose criterion `1` (Popularity), `2` (Compatibility) or `3` (ML pick), optionally filter by tag, and get a ranked table of recommendations.

### `app.py` — Streamlit web app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser. The UI offers two input tabs:

- **Steam API** — enter an API key + SteamID64 to load your library;
- **Pick 5** — enter 5 game IDs manually.

On submit, all three engines run at once (top 5 each) and results are shown in separate tabs. If the model hasn't been trained yet, the app warns you to train it locally first (`python main.py`). The deployed version is available at https://danrayht-recommend-steam-game-app-w7bsws.streamlit.app/.

## 5. Output

**CLI (`main.py`)** — a ranked table printed in the terminal for each run:

| game_id | name | tags | score | 
|---------|------|------|-------|
| …       | …    | …    | …     |

**Streamlit (`app.py`)** — one tab per engine, each showing the top-5 recommendations with the same columns (game_id, name, tags, score).

**Artifacts on disk:**

```
data/preprocessing/   game_info_clean.csv, user_final_clean.csv,
                      game_features.csv, game_tags_multihot.csv
model/                artifacts.pkl  (catalog + SVD embeddings)
                      ranker.json    (XGBoost model)
```
