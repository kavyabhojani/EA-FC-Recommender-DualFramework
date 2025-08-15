import pandas as pd
import numpy as np
import tensorflow as tf
from src.tensorflow.wide_deep import build_wide_deep
from src.utils.metrics import precision_recall_at_k, ndcg_at_k

DATA_DIR = "data/processed"

def make_ds(df, batch_size=1024, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices({
        "user_id": df["user_id"].astype("int32").values,
        "item_id": df["item_id"].astype("int32").values,
        "label":   np.ones(len(df), dtype="float32")
    })
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def evaluate(model, users, items, holdout, K=10, max_users=200):
    metrics = {"precision":[], "recall":[], "ndcg":[]}
    ulist = holdout["user_id"].unique()[:max_users]
    for u in ulist:
        gt = list(holdout.loc[holdout.user_id.eq(u),"item_id"].unique())
        user_vec = np.full((len(items),), u, dtype=np.int32)
        item_vec = items.astype(np.int32)
        scores = model.predict({"user_id":user_vec, "item_id":item_vec}, verbose=0).reshape(-1)
        ranked = [int(items[i]) for i in np.argsort(-scores)]
        p,r = precision_recall_at_k(ranked, gt, K)
        n = ndcg_at_k(ranked, gt, K)
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["ndcg"].append(n)
    return np.mean(metrics["precision"]), np.mean(metrics["recall"]), np.mean(metrics["ndcg"])

def main():
    users = pd.read_parquet(f"{DATA_DIR}/users.parquet")
    items = pd.read_parquet(f"{DATA_DIR}/items.parquet")
    train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    val   = pd.read_parquet(f"{DATA_DIR}/val.parquet")

    n_users = users.user_id.max()+1
    n_items = items.item_id.max()+1
    model = build_wide_deep(n_users, n_items, emb_dim=32)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss)

    #create negatives on the fly by pairing random items (simple demo)
    pos = train[["user_id","item_id"]].copy()
    neg = pos.copy()
    neg["item_id"] = np.random.randint(0, n_items, size=len(neg))
    neg = neg[neg["item_id"] != pos["item_id"]]
    pos["label"] = 1.0; neg["label"] = 0.0
    train_df = pd.concat([pos, neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    ds = tf.data.Dataset.from_tensor_slices((
        {"user_id": train_df["user_id"].astype("int32"),
         "item_id": train_df["item_id"].astype("int32")},
        train_df["label"].astype("float32")
    )).shuffle(len(train_df)).batch(1024).prefetch(tf.data.AUTOTUNE)

    model.fit(ds, epochs=3, verbose=1)

    p,r,n = evaluate(model, users["user_id"].values, items["item_id"].values, val, K=10)
    print(f"TF Val@10: precision={p:.4f} recall={r:.4f} ndcg={n:.4f}")
    model.save("wide_deep.keras")
    print("Saved model -> wide_deep.keras")

if __name__ == "__main__":
    main()
