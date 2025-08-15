import streamlit as st
import pandas as pd
import numpy as np
import torch
from src.pytorch.ncf import NCF

st.set_page_config(page_title="GameSense FC Demo", layout="centered")
st.title("GameSense FC — Personalized Content (Demo)")

users = pd.read_parquet("data/processed/users.parquet")
items = pd.read_parquet("data/processed/items.parquet")

user_id = st.number_input("User ID", min_value=0, max_value=int(users.user_id.max()), value=0, step=1)
topk = st.slider("Top-K", min_value=5, max_value=20, value=10, step=1)

#quick-load a small NCF if available; else show message
try:
    n_users = users.user_id.max()+1
    n_items = items.item_id.max()+1
    model = NCF(n_users, n_items, emb_dim=64)
    model.load_state_dict(torch.load("ncf.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        u = torch.full((n_items,), int(user_id), dtype=torch.long)
        it = torch.arange(0, n_items, dtype=torch.long)
        scores = model(u, it).numpy()
        ranked = np.argsort(-scores)[:topk]
    recs = items.iloc[ranked][["item_id","position","ovr"]]
    st.subheader("Recommended Items")
    st.dataframe(recs.reset_index(drop=True))
except Exception as e:
    st.info("Train a PyTorch model first (run: `python train_pytorch.py`).")
    st.caption(str(e))
