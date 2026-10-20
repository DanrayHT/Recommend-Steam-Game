"""
Gợi ý game theo 3 tiêu chí:
  - PopularityRecommender   : theo độ phổ biến (không train)
  - CompatibilityRecommender: theo độ tương thích với sở thích (không train)
  - MLRecommender           : máy tự chọn (ranker đã train bằng train.py)

Hai cách lấy thông tin người dùng:
  1. Steam API (API key + SteamID64)  -> profile_from_library
  2. Tự chọn 5 game trong game_info   -> profile_from_picks
"""
import getpass
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from preprocessing import load_config, resolve_path

STEAM_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
N_PICKS = 5
MIN_REVIEW_SCORE = 3

def l2norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


# ----------------------------------------------------------------
# CATALOG: dữ liệu game + artifact đã train
# ----------------------------------------------------------------
class Catalog:
    def __init__(self, save_dir):
        self.model_dir = save_dir / "model"

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
        """Tìm game theo tên (chỉ trong các game hệ thống biết)."""
        mask = pd.Series(self.names).str.contains(query.strip(), case=False, regex=False)
        idx = np.flatnonzero(mask.to_numpy())
        idx = idx[np.argsort(-np.nan_to_num(self.pop_score[idx]))][:limit]
        return pd.DataFrame({"game_id": self.game_ids[idx], "name": self.names[idx]})

    def similarities(self, weights):
        """Độ giống (cosine) giữa hồ sơ user và từng game: theo tag và theo SVD."""
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
                raise ValueError(f"Không có tag: {t}")
            cols.append(self.tag_cols.index(name))
        return (self.tag_n[:, cols] > 0).any(axis=1)

    def top_tags(self, weights, n=3):
        """Các tag nổi bật nhất trong hồ sơ user (theo trọng số game)."""
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


# ----------------------------------------------------------------
# USER PROFILE
# ----------------------------------------------------------------
@dataclass
class UserProfile:
    weights: np.ndarray   # (G,) trọng số sở thích theo từng game
    owned: np.ndarray     # (G,) game đã sở hữu -> loại khỏi gợi ý
    mode: str             # "full" (có playtime thật) | "pick5" (trọng số bằng nhau)
    stats: tuple          # (số game, log tổng playtime, log playtime trung bình)
    auto_tags: bool = False   # True: tự dùng tag của các game đã chọn


def profile_from_library(catalog, playtimes):
    """playtimes: {game_id: phút}. Chỉ giữ game có trong hệ thống."""
    idx, mins = [], []
    for gid, minutes in playtimes.items():
        i = catalog.id_to_idx.get(int(gid))
        if i is not None:
            idx.append(i)
            mins.append(max(float(minutes), 0.0))

    if not idx:
        raise ValueError("Không có game nào của bạn nằm trong dữ liệu của hệ thống.")

    idx, mins = np.array(idx), np.array(mins)
    owned = np.zeros(catalog.G, bool)
    owned[idx] = True
    weights = np.zeros(catalog.G)
    weights[idx] = np.log1p(mins)

    if weights.sum() == 0:
        # Sở hữu nhưng chưa chơi game nào: coi các game ngang nhau.
        weights[idx] = 1.0
        return UserProfile(weights, owned, "pick5", (len(idx), np.nan, np.nan))

    stats = (len(idx), np.log1p(mins.sum()), np.log1p(mins.mean()))
    return UserProfile(weights, owned, "full", stats)


def profile_from_picks(catalog, game_ids):
    """Người dùng tự chọn game; các game có thời gian chơi bằng nhau (30 phút)."""
    game_ids = list(dict.fromkeys(int(g) for g in game_ids))

    if not 1 <= len(game_ids) <= N_PICKS:
        raise ValueError(f"Cần chọn từ 1 đến {N_PICKS} game.")

    unknown = [g for g in game_ids if g not in catalog.id_to_idx]
    if unknown:
        raise ValueError(f"Game không có trong hệ thống: {unknown}")

    idx = np.array([catalog.id_to_idx[g] for g in game_ids])
    owned = np.zeros(catalog.G, bool)
    owned[idx] = True
    weights = np.zeros(catalog.G)
    weights[idx] = 1.0  # bằng nhau (tương đương cùng 30 phút sau khi chuẩn hóa)

    return UserProfile(weights, owned, "pick5", (len(idx), np.nan, np.nan), auto_tags=True)


# ----------------------------------------------------------------
# STEAM API
# ----------------------------------------------------------------
def parse_owned_games(payload):
    games = payload.get("response", {}).get("games")

    if not games:
        raise ValueError(
            "Steam không trả về game nào: hồ sơ/Game details có thể đang để riêng tư, "
            "hoặc SteamID sai."
        )

    return {int(g["appid"]): float(g.get("playtime_forever", 0)) for g in games}


def fetch_steam_library(api_key, steam_id, timeout=15):
    steam_id = steam_id.strip()

    if not (steam_id.isdigit() and len(steam_id) == 17):
        raise ValueError("SteamID64 phải gồm 17 chữ số.")

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
        # Không in lỗi gốc vì nó có thể chứa URL kèm API key.
        raise RuntimeError("Không kết nối được tới Steam API.") from None

    if resp.status_code in (401, 403):
        raise ValueError("API key không hợp lệ hoặc không có quyền truy cập.")
    if resp.status_code != 200:
        raise RuntimeError(f"Steam API lỗi (HTTP {resp.status_code}).")

    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError("Steam API trả về dữ liệu không đọc được.") from None

    return parse_owned_games(payload)


# ----------------------------------------------------------------
# 3 TIÊU CHÍ GỢI Ý
# ----------------------------------------------------------------
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
    """Xếp theo số người chơi trung bình 30 ngày gần đây (last_30d_avg)."""

    def recommend(self, profile, k=5, tags=None):
        c = self.catalog
        if tags is None and profile.auto_tags:
            tags = c.top_tags(profile.weights)
        allowed = self._allowed(profile, tags)
        # last_30d_avg là chính; số user sở hữu chỉ dùng để phá hòa.
        scores = np.nan_to_num(c.g_pop, nan=0.0) + 1e-3 * c.g_count_log
        return c.describe(self._top(scores, allowed, k), scores)

class CompatibilityRecommender(BaseRecommender):
    """Xếp theo độ giống với sở thích: trộn độ giống tag và độ giống hành vi chơi."""

    def __init__(self, catalog, alpha=0.5):
        super().__init__(catalog)
        self.alpha = alpha  # trọng số của tag; phần còn lại là SVD

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
    """Máy tự chọn: sinh ứng viên rồi chấm điểm bằng ranker đã train."""

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

        # Sinh ứng viên giống lúc train: top phổ biến + top giống tag.
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


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def ask_steam_profile(catalog):
    api_key = getpass.getpass("Steam API key (không hiển thị khi gõ): ")
    steam_id = input("SteamID64 (17 chữ số): ")
    library = fetch_steam_library(api_key, steam_id)
    profile = profile_from_library(catalog, library)
    print(f"Tìm thấy {len(library)} game, {profile.stats[0]} game có trong hệ thống.")
    return profile


def ask_picks_profile(catalog):
    picks = []

    while len(picks) < N_PICKS:
        query = input(f"Tìm game theo tên ({len(picks)}/{N_PICKS} đã chọn): ").strip()
        if not query:
            continue

        found = catalog.search(query)
        if found.empty:
            print("Không tìm thấy game nào.")
            continue

        print(found.to_string())
        choice = input("Nhập số thứ tự để chọn (Enter để tìm lại): ").strip()
        if not choice.isdigit() or int(choice) >= len(found):
            continue

        game_id = int(found.iloc[int(choice)]["game_id"])
        if game_id in picks:
            print("Game này đã được chọn.")
            continue

        picks.append(game_id)
        print(f"Đã chọn: {found.iloc[int(choice)]['name']}")

    return profile_from_picks(catalog, picks)


def main():
    config = load_config()
    save_dir = resolve_path(
        config.get("paths", "save_dir", fallback="data/preprocessing")
    )
    catalog = Catalog(save_dir)

    print("Lấy thông tin người dùng:\n  1) Steam API\n  2) Tự chọn 5 game")
    profile = None
    while profile is None:
        choice = input("Chọn 1 hoặc 2: ").strip()
        try:
            if choice == "1":
                profile = ask_steam_profile(catalog)
            elif choice == "2":
                profile = ask_picks_profile(catalog)
        except (ValueError, RuntimeError) as error:
            print(f"Lỗi: {error}")

    print("\nTiêu chí:\n  1) Phổ biến\n  2) Tương thích\n  3) Máy tự chọn")
    recommenders = {
        "1": PopularityRecommender,
        "2": CompatibilityRecommender,
        "3": MLRecommender,
    }
    while True:
        choice = input("Chọn 1, 2 hoặc 3: ").strip()
        if choice in recommenders:
            break

    if profile.auto_tags:
        tags = None
        print("Sở thích suy ra từ tag:", ", ".join(catalog.top_tags(profile.weights)))
    else:
        raw_tags = input("Lọc theo tag (cách nhau bằng dấu phẩy, Enter để bỏ qua): ")
        tags = [t for t in raw_tags.split(",") if t.strip()]

    try:
        result = recommenders[choice](catalog).recommend(profile, k=5, tags=tags)
    except ValueError as error:
        print(f"Lỗi: {error}")
        return

    print("\nGợi ý cho bạn:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()