import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
rng = np.random.default_rng()

POSITIONS = ["GK","DEF","MID","FWD"]
PLAYSTYLES = ["defense_first","possession","counter","pressing","all_round"]

def main(n_users, n_items, seed):
    global rng
    rng = np.random.default_rng(seed)
    root = Path("data/processed")
    root.mkdir(parents=True, exist_ok=True)

    #users with a preferred playstyle and position bias
    users = pd.DataFrame({
        "user_id": np.arange(n_users),
        "playstyle": rng.choice(PLAYSTYLES, n_users, p=[0.18,0.24,0.22,0.18,0.18]),
        "pref_position": rng.choice(POSITIONS, n_users, p=[0.1,0.35,0.35,0.2]),
    })

    #items with position + rating-like attribute
    items = pd.DataFrame({
        "item_id": np.arange(n_items),
        "position": rng.choice(POSITIONS, n_items, p=[0.08,0.42,0.32,0.18]),
        "ovr": np.clip(rng.normal(78, 6, size=n_items).round(), 60, 95).astype(int)
    })

    #implicit interactions: higher prob if position matches user pref and ovr high
    edges = []
    for u in users.itertuples(index=False):
        # each user considers a random subset of items
        candidate_items = rng.choice(items.item_id.values, size=rng.integers(80,160), replace=False)
        for it in candidate_items:
            pos = items.loc[items.item_id.eq(it), "position"].item()
            ovr = items.loc[items.item_id.eq(it), "ovr"].item()
            base = 0.05
            if pos == u.pref_position: base += 0.10
            base += (ovr - 75) * 0.004  # higher ovr => more likely
            interacted = rng.random() < min(max(base, 0.01), 0.6)
            if interacted:
                ts = rng.integers(1_700_000_000, 1_735_000_000)  # pseudo timestamps
                edges.append((u.user_id, it, ts))

    inter = pd.DataFrame(edges, columns=["user_id","item_id","timestamp"]).sort_values("timestamp")
    #simple chronological split (80/10/10)
    n = len(inter)
    train = inter.iloc[: int(n*0.8)].reset_index(drop=True)
    val   = inter.iloc[int(n*0.8): int(n*0.9)].reset_index(drop=True)
    test  = inter.iloc[int(n*0.9):].reset_index(drop=True)

    users.to_parquet(root/"users.parquet", index=False)
    items.to_parquet(root/"items.parquet", index=False)
    train.to_parquet(root/"train.parquet", index=False)
    val.to_parquet(root/"val.parquet", index=False)
    test.to_parquet(root/"test.parquet", index=False)

    print(f"Generated users:{len(users)} items:{len(items)} interactions:{len(inter)}")
    print("Saved to data/processed/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=1000)
    ap.add_argument("--n_items", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n_users, args.n_items, args.seed)
