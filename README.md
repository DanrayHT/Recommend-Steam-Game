# Recommend Steam Game

A personal game recommender for Steam. Given your library (via the Steam Web API) or just 5 games you like, it recommends new games using three strategies: popularity, tag/content compatibility, and a trained **two-stage ML ranker** (candidate generation + XGBoost learning-to-rank).

- **Live demo:** <https://danrayht-recommend-steam-game-app-w7bsws.streamlit.app/>
- **Data:** scraped and assembled in [DanrayHT/Scrape-steam-data](https://github.com/DanrayHT/Scrape-steam-data), published as the Hugging Face dataset [`DanrayHT/steam_data_with_playtime`](https://huggingface.co/datasets/DanrayHT/steam_data_with_playtime).

## Highlights

- **Two-stage recommender:** candidates from popularity and tag similarity, re-ranked by an XGBoost `XGBRanker` (`rank:ndcg`) on 10 features.
- **Evaluated on held-out users** against popularity, tag and SVD baselines. With a full library, the ranker reaches **Recall@5 0.281 / NDCG@5 0.292 vs. 0.201 / 0.210 for popularity (+40% / +39%)**, while recommending about 20% of the catalog instead of about 1%.
- **Works with little data:** a 5-game profile reaches Recall@5 0.229 (+14% over popularity).
- **Two ways to use it:** an interactive CLI and a Streamlit web app.

## 1. Goal of the project

Build a recommendation system that suggests Steam games you haven't played yet, based on:

- your owned-games library (fetched from the Steam Web API), or
- as little as **5 game IDs** you pick manually.

Three recommender engines are provided so you can compare results:

| # | Engine | Idea |
|---|--------|------|
| 1 | **Popularity** | Top games by recent popularity (log of the 30-day average concurrent players from Steam Charts, with ownership count as a tie-breaker), excluding games you already own and games below the minimum review score. |
| 2 | **Compatibility** | Games most similar to your profile: a z-scored blend (α = 0.5) of tag multi-hot similarity and TruncatedSVD embedding similarity. |
| 3 | **ML pick** | An XGBoost ranker that re-ranks a candidate set using popularity, review score, tag count, game playtime statistics, your history statistics, and content/SVD similarity to your history. |

## 2. Evaluation

All numbers come from `model/metrics.json`, produced at the end of training on **held-out test users** (users are split 80/10/10 into train/validation/test, so no test user is seen during training).

### Setup

- **"Liked" game:** playtime ≥ `min_playtime` (120 minutes). Games owned by fewer than `min_game_users` users (20) are dropped, and users need at least `min_liked` (4) liked games. These values are set in `src/config.ini`.
- **Targets:** for each user, about 20% of their liked games (between 1 and 5) are hidden as targets. All remaining games form the user's history, which is the only input to the recommender.
- **Profiles:** `full` weights the history by `log(1 + playtime)`; `pick5` uses 5 randomly sampled liked games from the history (a simulation of the 5-game cold start). The ranker is trained on a mix of both.
- **Candidate generation:** the union of the top-150 games by popularity and the top-150 by tag similarity, excluding owned games. During training the hidden targets are always added to the candidates so the ranker sees positives; during evaluation they are **not**, so a target that candidate generation misses counts as a miss.
- **Ranker:** `XGBRanker`, objective `rank:ndcg`, up to 500 trees, learning rate 0.05, depth 6, subsample/colsample 0.8, early stopping (30 rounds) on validation NDCG@5.
- **Features (10):** `popularity_log`, `review_score`, `tag_count`, `game_user_count_log`, `game_avg_playtime_log`, `user_hist_count`, `user_total_playtime_log`, `user_avg_playtime_log`, `content_sim`, `svd_sim`.
- **Metrics (K = 5):** *Recall@5* = hidden targets found in the top 5 / number of targets, averaged over users. *NDCG@5* = standard binary-relevance NDCG. *Coverage* = share of the catalog that appears in at least one user's top 5.
- **Leakage control:** TruncatedSVD and game-level playtime statistics are fitted on training users only. Popularity comes from a Steam Charts snapshot, which is a property of the game and independent of which users are in the test set.

### Results

Full library (`full` profile):

| Method | Recall@5 | NDCG@5 | Coverage |
|---|---|---|---|
| Popularity (baseline) | 0.201 | 0.210 | 1.2% |
| Tag similarity only | 0.014 | 0.013 | 14.0% |
| TruncatedSVD only | 0.041 | 0.040 | 31.5% |
| **XGBoost ranker** | **0.281** | **0.292** | 20.6% |

5-game profile (`pick5`):

| Method | Recall@5 | NDCG@5 | Coverage |
|---|---|---|---|
| Popularity (baseline) | 0.201 | 0.210 | 1.2% |
| Tag similarity only | 0.018 | 0.016 | 41.6% |
| TruncatedSVD only | 0.034 | 0.032 | 51.6% |
| **XGBoost ranker** | **0.229** | **0.236** | 23.4% |

Popularity does not depend on the user, so its numbers are identical in both modes.

### Takeaways

- **Popularity is a strong baseline** because the games users play for a long time skew heavily toward popular titles. It also recommends almost the same 1% of the catalog to everyone.
- **Similarity alone is weak at recovering held-out games** (Recall@5 below 0.05), although it reaches much more of the catalog. The two similarity engines are useful for exploration and diversity rather than raw accuracy.
- **Combining the signals in a learned ranker works best:** about +40% Recall@5 and +39% NDCG@5 over popularity with a full library (+14% and +12% in the 5-game mode), while covering about 20% of the catalog.

### Limitations

- **Offline metrics measure how well held-out liked games are recovered**, not real user satisfaction. "Liked" is defined by a playtime threshold, which is a proxy.
- **Candidate generation caps recall:** a hidden game outside the 150 + 150 candidates cannot be recommended.
- **The `pick5` results are somewhat optimistic.** In the simulation, games the user owns (from the full history) are excluded from the candidates, which a real 5-game user cannot provide.
- **Popularity is a snapshot** of the 30-day average concurrent players at scraping time, while playtime records are historical. Only games tracked by Steam Charts are included (about 4.9k of about 34k games in the source dataset), and brand-new games not in the trained catalog cannot be recommended.

## 3. Project structure

```
Recommend-Steam-Game/
├── main.py                 # CLI entry point: config → data → train → recommend loop
├── app.py                  # Streamlit web UI (two input tabs, three engines)
├── requirements.txt        # Pinned dependencies
├── src/
│   ├── config.ini          # [paths], [features], [train] settings
│   ├── download_data.py    # Downloads game_info.csv & user_final.csv from HuggingFace
│   ├── preprocessing.py    # Cleans raw CSVs, builds features + tag multi-hot matrix
│   ├── train.py            # Trains the XGBoost ranker and evaluates it against baselines
│   └── recommend.py        # Catalog + Popularity/Compatibility/ML recommenders, Steam API fetch
├── data/                   # Raw CSVs (auto-downloaded) and data/preprocessing outputs
└── model/                  # Trained artifacts: artifacts.pkl, ranker.json, metrics.json
```

## 4. Installation

Requires Python 3.10+.

```bash
# clone the repo
git clone https://github.com/DanrayHT/Recommend-Steam-Game.git
cd Recommend-Steam-Game

# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# install dependencies
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `requests`, `huggingface_hub`, `streamlit`.

> No API key is required to run the app. Raw data (`game_info.csv`, `user_final.csv`) is downloaded automatically from the Hugging Face dataset [`DanrayHT/steam_data_with_playtime`](https://huggingface.co/datasets/DanrayHT/steam_data_with_playtime), which originates from [Scrape-steam-data](https://github.com/DanrayHT/Scrape-steam-data). A Steam Web API key is only needed for the "Steam API" input mode, and your profile must be public.

## 5. Usage

### `main.py` — interactive CLI

```bash
python main.py
```

Flow:

1. **Config** — optionally edit `[paths]`, `[features]`, `[train]` values in `src/config.ini`.
2. **Data** — raw CSVs are downloaded automatically if missing, then preprocessed into `data/preprocessing/` (`game_info_clean.csv`, `user_final_clean.csv`, `game_features.csv`, `game_tags_multihot.csv`).
3. **Training** — the XGBoost model is trained (or retrained on request), evaluated against the baselines, and saved to `model/artifacts.pkl`, `model/ranker.json` and `model/metrics.json`. The test results are also printed to the console.
4. **Profile** — build your profile either by:
   - entering a Steam Web API key + SteamID64 (your owned games are fetched via `GetOwnedGames`), or
   - picking 5 games by name.
5. **Recommend loop** — choose criterion `1` (Popularity), `2` (Compatibility) or `3` (ML pick), optionally filter by tag, and get a ranked table of recommendations.

### `app.py` — Streamlit web app

```bash
streamlit run app.py
```

Then open <http://localhost:8501> in your browser (or use the [hosted demo](https://danrayht-recommend-steam-game-app-w7bsws.streamlit.app/)). The UI offers two input tabs:

- **Steam API** — enter an API key + SteamID64 to load your library;
- **Pick 5** — enter 5 game IDs manually.

On submit, all three engines run at once (top 5 each) and results are shown in separate tabs. If the model hasn't been trained yet, the app warns you to train it locally first (`python main.py`).

## 6. Output

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
                      metrics.json   (test metrics vs. baselines)
```
