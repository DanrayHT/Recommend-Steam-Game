import json
import pickle
from types import SimpleNamespace

import xgboost as xgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .preprocessing import load_config, resolve_path, get_model_dir

FEATURES = [
    "popularity_log",
    "review_score",
    "tag_count",
    "game_user_count_log",
    "game_avg_playtime_log",
    "user_hist_count",
    "user_total_playtime_log",
    "user_avg_playtime_log",
    "content_sim",
    "svd_sim",
]

def read_cfg(config):
    s = "train"
    return {
        "min_playtime": config.getfloat(s, "min_playtime", fallback=120.0),
        "min_liked": config.getint(s, "min_liked", fallback=6),
        "min_game_users": config.getint(s, "min_game_users", fallback=20),
        "n_candidates": config.getint(s, "n_candidates", fallback=150),
        "pick5_frac": config.getfloat(s, "pick5_frac", fallback=0.5),
        "valid_frac": config.getfloat(s, "valid_frac", fallback=0.1),
        "test_frac": config.getfloat(s, "test_frac", fallback=0.1),
        "svd_dim": config.getint(s, "svd_dim", fallback=64),
        "batch_size": config.getint(s, "batch_size", fallback=512),
        "max_train_users": config.getint(s, "max_train_users", fallback=0),
        "seed": config.getint(s, "seed", fallback=42),
    }

def l2norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

def split_interactions(users, cfg, rng):
    users = users.copy()
    users["liked"] = users["playtime_forever"] >= cfg["min_playtime"]

    liked_cnt = users.groupby("user_id")["liked"].transform("sum")
    users = users[liked_cnt >= cfg["min_liked"]].copy()
    liked_cnt = users.groupby("user_id")["liked"].transform("sum")

    users["r"] = rng.random(len(users))
    users.loc[~users["liked"], "r"] = np.inf
    rank = users.groupby("user_id")["r"].rank(method="first") - 1
    n_target = (0.2 * liked_cnt).round().clip(1, 5)
    users["is_target"] = users["liked"] & (rank < n_target)

    uniq = rng.permutation(users["user_id"].unique())
    n_valid = int(len(uniq) * cfg["valid_frac"])
    n_test = int(len(uniq) * cfg["test_frac"])
    split = pd.Series(0, index=uniq)
    split.iloc[:n_valid] = 1
    split.iloc[n_valid:n_valid + n_test] = 2
    users["split"] = users["user_id"].map(split)

    return users.drop(columns="r")

def build_context(cfg, save_dir):
    rng = np.random.default_rng(cfg["seed"])

    games = (
        pd.read_csv(save_dir / "game_features.csv")
        .drop_duplicates("game_id")
        .set_index("game_id")
    )
    tags = (
        pd.read_csv(save_dir / "game_tags_multihot.csv")
        .drop_duplicates("game_id")
        .set_index("game_id")
    )
    users = pd.read_csv(save_dir / "user_final_clean.csv")
    users["playtime_forever"] = users["playtime_forever"].fillna(0)

    common = games.index.intersection(tags.index)
    users = users[users["game_id"].isin(common)]
    counts = users["game_id"].value_counts()
    game_ids = np.sort(counts.index[counts >= cfg["min_game_users"]].to_numpy())
    users = users[users["game_id"].isin(game_ids)]
    games, tags = games.loc[game_ids], tags.loc[game_ids]
    G = len(game_ids)

    users = split_interactions(users, cfg, rng)
    users["u"] = pd.factorize(users["user_id"])[0]
    users["g"] = pd.Index(game_ids).get_indexer(users["game_id"])
    U = users["u"].max() + 1

    u_split = users.groupby("u")["split"].first().reindex(range(U)).to_numpy()
    hist = users[~users["is_target"]]
    tgt = users[users["is_target"]]

    def mat(df, values):
        return sparse.csr_matrix(
            (values.astype(np.float32), (df["u"].to_numpy(), df["g"].to_numpy())),
            shape=(U, G),
        )

    ones = lambda df: np.ones(len(df), np.float32)
    OWN = mat(hist, ones(hist))
    TGT = mat(tgt, ones(tgt))
    W_full = mat(hist, np.log1p(hist["playtime_forever"].to_numpy()))

    hl = hist[hist["liked"]].copy()
    hl["rk"] = hl.assign(r=rng.random(len(hl))).groupby("u")["r"].rank(method="first")
    pick = hl[hl["rk"] <= 5]
    W_pick = mat(pick, ones(pick))

    gp = hist.groupby("u")["playtime_forever"]
    idx = range(U)
    u_count = gp.size().reindex(idx, fill_value=0).to_numpy(float)
    u_total = np.log1p(gp.sum().reindex(idx).to_numpy(float))
    u_avg = np.log1p(gp.mean().reindex(idx).to_numpy(float))

    htr = hist[u_split[hist["u"].to_numpy()] == 0]
    gg = htr.groupby("g")["playtime_forever"]
    g_count_log = np.log1p(gg.size().reindex(range(G), fill_value=0).to_numpy(float))
    g_avg_log = np.log1p(gg.mean().reindex(range(G)).to_numpy(float))

    train_rows = np.flatnonzero(u_split == 0)
    X = W_full[train_rows]
    k = int(min(cfg["svd_dim"], G - 1, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=cfg["seed"]).fit(X)
    emb = l2norm(svd.components_.T.astype(np.float32))

    tag_n = l2norm(tags.to_numpy(np.float32))
    pop_log = games["popularity_log"].to_numpy(float)

    return SimpleNamespace(
        cfg=cfg, G=G, game_ids=game_ids, tag_cols=list(tags.columns),
        OWN=OWN, TGT=TGT, W_full=W_full, W_pick=W_pick,
        n_target=np.asarray(TGT.sum(axis=1)).ravel(),
        u_pick_count=np.asarray(W_pick.sum(axis=1)).ravel(),
        u_split=u_split, u_count=u_count, u_total=u_total, u_avg=u_avg,
        g_pop=pop_log,
        g_review=games["review_score"].to_numpy(float),
        g_tagcount=games["tag_count"].to_numpy(float),
        g_count_log=g_count_log, g_avg_log=g_avg_log,
        pop_score=np.nan_to_num(pop_log, nan=0.0) + g_count_log,
        tag_n=tag_n, emb=emb,
    )

def profile_scores(ctx, Wb):
    out = []
    for M in (ctx.tag_n, ctx.emb):
        P = l2norm(np.asarray(Wb @ M))
        out.append(P @ M.T)
    return out

def make_rows(ctx, uidx, mode, add_targets):
    cfg = ctx.cfg
    W = ctx.W_full if mode == "full" else ctx.W_pick
    n = min(cfg["n_candidates"], ctx.G - 1)
    X_parts, y_parts, groups, metas = [], [], [], []

    for s in range(0, len(uidx), cfg["batch_size"]):
        ub = uidx[s:s + cfg["batch_size"]]
        content, svd = profile_scores(ctx, W[ub])
        owned = ctx.OWN[ub].toarray() > 0
        tgt = ctx.TGT[ub].toarray() > 0

        cand = np.zeros_like(owned)
        for score in (ctx.pop_score[None, :], content):
            masked = np.where(owned, -np.inf, score)
            top = np.argpartition(-masked, n, axis=1)[:, :n]
            np.put_along_axis(cand, top, True, axis=1)
        cand &= ~owned
        if add_targets:
            cand |= tgt

        ul, g = np.nonzero(cand)

        if mode == "full":
            cnt, tot, avg = ctx.u_count[ub], ctx.u_total[ub], ctx.u_avg[ub]
        else:
            cnt = ctx.u_pick_count[ub].astype(float)
            tot = avg = np.full(len(ub), np.nan)

        X_parts.append(np.column_stack([
            ctx.g_pop[g], ctx.g_review[g], ctx.g_tagcount[g],
            ctx.g_count_log[g], ctx.g_avg_log[g],
            cnt[ul], tot[ul], avg[ul],
            content[ul, g], svd[ul, g],
        ]).astype(np.float32))
        y_parts.append(tgt[ul, g].astype(np.int8))

        sizes = np.bincount(ul, minlength=len(ub))
        groups.append(sizes[sizes > 0])
        metas.append(pd.DataFrame({"u": ub[ul], "g": g}))

    return (
        np.vstack(X_parts), np.concatenate(y_parts),
        np.concatenate(groups), pd.concat(metas, ignore_index=True),
    )

def evaluate(ctx, model, uidx, mode, k=5):
    X, y, _, meta = make_rows(ctx, uidx, mode, add_targets=False)
    meta["y"] = y.astype(float)
    meta["model"] = model.predict(X)
    meta["popularity"] = ctx.pop_score[meta["g"].to_numpy()]
    meta["content"] = X[:, FEATURES.index("content_sim")]
    meta["svd"] = X[:, FEATURES.index("svd_sim")]

    n_t = ctx.n_target[uidx]
    idcg = np.array([(1 / np.log2(np.arange(2, min(n, k) + 2))).sum() for n in n_t])
    result = {}

    for name in ["popularity", "content", "svd", "model"]:
        rk = meta.groupby("u")[name].rank(ascending=False, method="first")
        top = meta[rk <= k]
        hits = top.groupby("u")["y"].sum().reindex(uidx, fill_value=0).to_numpy()
        dcg = (
            (top["y"] / np.log2(rk[top.index] + 1))
            .groupby(top["u"]).sum()
            .reindex(uidx, fill_value=0).to_numpy()
        )
        result[name] = {
            f"recall@{k}": float((hits / n_t).mean()),
            f"ndcg@{k}": float((dcg / idcg).mean()),
            "coverage": float(top["g"].nunique() / ctx.G),
        }

    return result

def pick_device():
    try:
        model = xgb.XGBRanker(tree_method="hist", device="cuda", n_estimators=1)
        model.fit(
            np.random.rand(20, 2),
            np.random.randint(0, 2, 20),
            qid=np.repeat([0, 1], 10),
        )
        device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        return "cuda" if device.startswith("cuda") else "cpu"
    except Exception:
        return "cpu"

def main():
    config = load_config()
    cfg = read_cfg(config)
    save_dir = resolve_path(
        config.get("paths", "save_dir", fallback="data/preprocessing")
    )
    model_dir = get_model_dir(config)
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    device = pick_device()
    print(f"Training device: {device}")

    ctx = build_context(cfg, save_dir)
    tr = np.flatnonzero(ctx.u_split == 0)
    va = np.flatnonzero(ctx.u_split == 1)
    te = np.flatnonzero(ctx.u_split == 2)
    if cfg["max_train_users"]:
        tr = rng.permutation(tr)[: cfg["max_train_users"]]
        tr.sort()
    print(f"Games: {ctx.G:,} | users train/valid/test: {len(tr):,}/{len(va):,}/{len(te):,}")

    def build(uidx, pick_frac):
        sets = [make_rows(ctx, uidx, "full", True)]
        sub = np.sort(rng.choice(uidx, int(len(uidx) * pick_frac), replace=False))
        if len(sub):
            sets.append(make_rows(ctx, sub, "pick5", True))
        return (
            np.vstack([s[0] for s in sets]),
            np.concatenate([s[1] for s in sets]),
            np.concatenate([s[2] for s in sets]),
        )

    X_tr, y_tr, g_tr = build(tr, cfg["pick5_frac"])
    X_va, y_va, g_va = build(va, 1.0)
    print(f"Train rows: {len(y_tr):,} (positive: {y_tr.sum():,}) | valid rows: {len(y_va):,}")

    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        tree_method="hist",
        device=device,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="ndcg@5",
        early_stopping_rounds=30,
        importance_type="gain",
        random_state=cfg["seed"],
    )
    ranker.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_va, y_va)], eval_group=[g_va],
        verbose=50,
    )
    ranker.get_booster().set_param({"device": "cpu"})

    metrics = {mode: evaluate(ctx, ranker, te, mode) for mode in ("full", "pick5")}
    print("\nTest results (vs baselines):")
    for mode, res in metrics.items():
        print(f"\n[{mode}]")
        print(pd.DataFrame(res).T.round(4).to_string())

    print("\nFeature importance (gain):")
    imp = pd.Series(ranker.feature_importances_, index=FEATURES)
    print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())

    ranker.save_model(str(model_dir / "ranker.json"))
    with open(model_dir / "artifacts.pkl", "wb") as f:
        pickle.dump({
            "features": FEATURES,
            "cfg": cfg,
            "game_ids": ctx.game_ids,
            "tag_cols": ctx.tag_cols,
            "tag_n": ctx.tag_n,
            "emb": ctx.emb,
            "pop_score": ctx.pop_score,
            "g_pop": ctx.g_pop,
            "g_review": ctx.g_review,
            "g_tagcount": ctx.g_tagcount,
            "g_count_log": ctx.g_count_log,
            "g_avg_log": ctx.g_avg_log,
        }, f)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nModel saved to: {model_dir}")

if __name__ == "__main__":
    main()