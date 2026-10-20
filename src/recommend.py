import getpass
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from .preprocessing import load_config, resolve_path, get_model_dir

STEAM_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
N_PICKS = 5
MIN_REVIEW_SCORE = 3

def l2norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class Catalog:
    def __init__(self, save_dir):
        self.model_dir = get_model_dir(load_config())

        with open(self.model_dir / "artifacts.pkl", "rb") as f:
            a = pickle.load(f)

        self.features = a["features"]
        self.cfg = a["cfg"]
        self.game_ids = a["game_ids"]
        self.G = len(self.game_ids)
        self.tag_cols = a["tag_cols"]
        self.tag_n = a["tag_n"]
        self.emb = a["emb"]
        self.pop_score = a["pop_score"]
        self.g_pop = a["g_pop"]
        self.g_count_log = a["g_count_log"]
        self.game_cols = {
            "popularity_log": a["g_pop"],
            "review_score": a["g_review"],
            "tag_count": a["g_tagcount"],
            "game_user_count_log": a["g_count_log"],
            "game_avg_playtime_log": a["g_avg_log"],
        }
        self.id_to_idx = {int(g): i for i, g in enumerate(self.game_ids)}

        info = (
            pd.read_csv(save_dir / "game_info_clean.csv")
            .drop_duplicates("game_id")
            .set_index("game_id")
            .reindex(self.game_ids)
        )
        self.names = info["name"].fillna("").astype(str).to_numpy()
        self.tags_text = info["tags"].fillna("").astype(str).to_numpy()

    def search(self, query, limit=10):
        mask = pd.Series(self.names).str.contains(query.strip(), case=False, regex=False)
        idx = np.flatnonzero(mask.to_numpy())
        idx = idx[np.argsort(-np.nan_to_num(self.pop_score[idx]))][:limit]
        return pd.DataFrame({"game_id": self.game_ids[idx], "name": self.names[idx]})

    def similarities(self, weights):
        out = []
        for M in (self.tag_n, self.emb):
            out.append(M @ l2norm(weights @ M))
        return out

    def tag_mask(self, tags):
        if not tags:
            return np.ones(self.G, bool)
        cols = []
        for t in tags:
            name = f"tag_{t.strip().casefold()}"
            if name not in self.tag_cols:
                raise ValueError(f"Unknown tag: {t}")
            cols.append(self.tag_cols.index(name))
        return (self.tag_n[:, cols] > 0).any(axis=1)

    def top_tags(self, weights, n=3):
        score = weights @ (self.tag_n > 0)
        idx = np.argsort(-score)[:n]
        return [self.tag_cols[i][len("tag_"):] for i in idx if score[i] > 0]

    def describe(self, idx, scores):
        return pd.DataFrame({
            "game_id": self.game_ids[idx],
            "name": self.names[idx],
            "tags": [", ".join(self.tags_text[i].split("|")[:5]) for i in idx],
            "score": np.round(scores[idx], 4),
        })


@dataclass
class UserProfile:
    weights: np.ndarray
    owned: np.ndarray
    mode: str
    stats: tuple
    auto_tags: bool = False


def profile_from_library(catalog, playtimes):
    idx, mins = [], []
    for gid, minutes in playtimes.items():
        i = catalog.id_to_idx.get(int(gid))
        if i is not None:
            idx.append(i)
            mins.append(max(float(minutes), 0.0))

    if not idx:
        raise ValueError("None of your games are in the dataset.")

    idx, mins = np.array(idx), np.array(mins)
    owned = np.zeros(catalog.G, bool)
    owned[idx] = True
    weights = np.zeros(catalog.G)
    weights[idx] = np.log1p(mins)

    if weights.sum() == 0:
        weights[idx] = 1.0
        return UserProfile(weights, owned, "pick5", (len(idx), np.nan, np.nan))

    stats = (len(idx), np.log1p(mins.sum()), np.log1p(mins.mean()))
    return UserProfile(weights, owned, "full", stats)


def profile_from_picks(catalog, game_ids):
    game_ids = list(dict.fromkeys(int(g) for g in game_ids))

    if not 1 <= len(game_ids) <= N_PICKS:
        raise ValueError(f"Pick 1 to {N_PICKS} games.")

    unknown = [g for g in game_ids if g not in catalog.id_to_idx]
    if unknown:
        raise ValueError(f"Games not in dataset: {unknown}")

    idx = np.array([catalog.id_to_idx[g] for g in game_ids])
    owned = np.zeros(catalog.G, bool)
    owned[idx] = True
    weights = np.zeros(catalog.G)
    weights[idx] = 1.0

    return UserProfile(weights, owned, "pick5", (len(idx), np.nan, np.nan), auto_tags=True)


def parse_owned_games(payload):
    games = payload.get("response", {}).get("games")

    if not games:
        raise ValueError(
            "Steam returned no games (private profile or wrong SteamID)."
        )

    return {int(g["appid"]): float(g.get("playtime_forever", 0)) for g in games}


def fetch_steam_library(api_key, steam_id, timeout=15):
    steam_id = steam_id.strip()

    if not (steam_id.isdigit() and len(steam_id) == 17):
        raise ValueError("SteamID64 must be 17 digits.")

    try:
        resp = requests.get(
            STEAM_URL,
            params={
                "key": api_key.strip(),
                "steamid": steam_id,
                "format": "json",
                "include_played_free_games": 1,
            },
            timeout=timeout,
        )
    except requests.RequestException:
        raise RuntimeError("Cannot reach Steam API.") from None

    if resp.status_code in (401, 403):
        raise ValueError("Invalid API key or access denied.")
    if resp.status_code != 200:
        raise RuntimeError(f"Steam API error (HTTP {resp.status_code}).")

    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError("Unreadable Steam API response.") from None

    return parse_owned_games(payload)


class BaseRecommender:
    def __init__(self, catalog):
        self.catalog = catalog

    def _allowed(self, profile, tags):
        c = self.catalog
        not_bad = ~(c.game_cols["review_score"] < MIN_REVIEW_SCORE)
        return ~profile.owned & c.tag_mask(tags) & not_bad

    @staticmethod
    def _top(scores, allowed, k):
        s = np.where(allowed, scores, -np.inf)
        idx = np.argsort(-s)[:k]
        return idx[np.isfinite(s[idx])]

    def recommend(self, profile, k=5, tags=None):
        raise NotImplementedError


class PopularityRecommender(BaseRecommender):

    def recommend(self, profile, k=5, tags=None):
        c = self.catalog
        if tags is None and profile.auto_tags:
            tags = c.top_tags(profile.weights)
        allowed = self._allowed(profile, tags)
        scores = np.nan_to_num(c.g_pop, nan=0.0) + 1e-3 * c.g_count_log
        return c.describe(self._top(scores, allowed, k), scores)

class CompatibilityRecommender(BaseRecommender):

    def __init__(self, catalog, alpha=0.5):
        super().__init__(catalog)
        self.alpha = alpha

    def recommend(self, profile, k=5, tags=None):
        c = self.catalog
        allowed = self._allowed(profile, tags)
        if not allowed.any():
            return c.describe(np.array([], int), np.zeros(c.G))

        content, svd = c.similarities(profile.weights)

        def z(x):
            v = x[allowed]
            return (x - v.mean()) / (v.std() + 1e-8)

        scores = self.alpha * z(content) + (1 - self.alpha) * z(svd)
        return c.describe(self._top(scores, allowed, k), scores)


class MLRecommender(BaseRecommender):

    def __init__(self, catalog):
        super().__init__(catalog)
        import xgboost as xgb

        self.model = xgb.XGBRanker()
        self.model.load_model(str(catalog.model_dir / "ranker.json"))

    def recommend(self, profile, k=5, tags=None):
        c = self.catalog
        allowed = self._allowed(profile, tags)
        if not allowed.any():
            return c.describe(np.array([], int), np.zeros(c.G))

        content, svd = c.similarities(profile.weights)

        n = min(c.cfg["n_candidates"], c.G - 1)
        cand = np.zeros(c.G, bool)
        for score in (c.pop_score, content):
            top = self._top(score, allowed, n)
            cand[top] = True
        g = np.flatnonzero(cand)

        cnt, tot, avg = profile.stats
        cols = {name: values[g] for name, values in c.game_cols.items()}
        cols.update({
            "user_hist_count": np.full(len(g), cnt),
            "user_total_playtime_log": np.full(len(g), tot),
            "user_avg_playtime_log": np.full(len(g), avg),
            "content_sim": content[g],
            "svd_sim": svd[g],
        })
        X = np.column_stack([cols[f] for f in c.features]).astype(np.float32)

        scores = np.full(c.G, -np.inf)
        scores[g] = self.model.predict(X)
        return c.describe(self._top(scores, allowed, k), scores)


def ask_steam_profile(catalog):
    api_key = getpass.getpass("Steam API key (hidden): ")
    steam_id = input("SteamID64 (17 digits): ")
    library = fetch_steam_library(api_key, steam_id)
    profile = profile_from_library(catalog, library)
    print(f"Found {len(library)} games, {profile.stats[0]} in dataset.")
    return profile


def ask_picks_profile(catalog):
    picks = []

    while len(picks) < N_PICKS:
        query = input(f"Search game name ({len(picks)}/{N_PICKS} picked): ").strip()
        if not query:
            continue

        found = catalog.search(query)
        if found.empty:
            print("No games found.")
            continue

        print(found.to_string())
        choice = input("Item number to pick (Enter to search again): ").strip()
        if not choice.isdigit() or int(choice) >= len(found):
            continue

        game_id = int(found.iloc[int(choice)]["game_id"])
        if game_id in picks:
            print("Already picked.")
            continue

        picks.append(game_id)
        print(f"Picked: {found.iloc[int(choice)]['name']}")

    return profile_from_picks(catalog, picks)


def main():
    config = load_config()
    save_dir = resolve_path(
        config.get("paths", "save_dir", fallback="data/preprocessing")
    )
    catalog = Catalog(save_dir)

    print("User input:\n  1) Steam API\n  2) Pick 5 games")
    profile = None
    while profile is None:
        choice = input("Choose 1 or 2: ").strip()
        try:
            if choice == "1":
                profile = ask_steam_profile(catalog)
            elif choice == "2":
                profile = ask_picks_profile(catalog)
        except (ValueError, RuntimeError) as error:
            print(f"Error: {error}")

    print("\nCriteria:\n  1) Popularity\n  2) Compatibility\n  3) ML pick")
    recommenders = {
        "1": PopularityRecommender,
        "2": CompatibilityRecommender,
        "3": MLRecommender,
    }
    while True:
        choice = input("Choose 1, 2 or 3: ").strip()
        if choice in recommenders:
            break

    if profile.auto_tags:
        tags = None
        print("Tags:", ", ".join(catalog.top_tags(profile.weights)))
    else:
        raw_tags = input("Filter tags (comma-separated, Enter to skip): ")
        tags = [t for t in raw_tags.split(",") if t.strip()]

    try:
        result = recommenders[choice](catalog).recommend(profile, k=5, tags=tags)
    except ValueError as error:
        print(f"Error: {error}")
        return

    print("\nRecommendations:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()