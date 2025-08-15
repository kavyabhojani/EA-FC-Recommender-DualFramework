import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.pytorch.ncf import NCF, bce_loss
from src.utils.metrics import precision_recall_at_k, ndcg_at_k

DATA_DIR = "data/processed"

class ImplicitDataset(Dataset):
    def __init__(self, df, n_items, neg_per_pos=4, seed=42):
        self.df = df[["user_id","item_id"]].values
        self.n_items = n_items
        self.neg_per_pos = neg_per_pos
        self.rng = np.random.default_rng(seed)
        # build user->set(items) for negative sampling
        self.user_pos = {}
        for u,i in self.df:
            self.user_pos.setdefault(int(u), set()).add(int(i))
        self.samples = []
        for u,i in self.df:
            self.samples.append((int(u), int(i), 1))
            for _ in range(neg_per_pos):
                j = int(self.rng.integers(0, n_items))
                while j in self.user_pos[int(u)]:
                    j = int(self.rng.integers(0, n_items))
                self.samples.append((int(u), j, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u,i,y = self.samples[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(float(y))

def build_user_catalog(train):
    user2items = {}
    for r in train.itertuples(index=False):
        user2items.setdefault(int(r.user_id), set()).add(int(r.item_id))
    return user2items

def evaluate(model, user2items, all_items, holdout, device, K=10, max_users=200):
    model.eval()
    users = list(holdout["user_id"].unique())[:max_users]
    precs, recs, ndcgs = [], [], []
    with torch.no_grad():
        for u in users:
            gt = list(holdout.loc[holdout.user_id.eq(u),"item_id"].unique())
            # score all items (small demo; for real scale use sampling)
            u_tensor = torch.full((len(all_items),), u, dtype=torch.long, device=device)
            i_tensor = torch.tensor(all_items, dtype=torch.long, device=device)
            scores = model(u_tensor, i_tensor).cpu().numpy()
            ranked = [all_items[i] for i in np.argsort(-scores)]
            p,r = precision_recall_at_k(ranked, gt, K)
            n = ndcg_at_k(ranked, gt, K)
            precs.append(p); recs.append(r); ndcgs.append(n)
    return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(ndcgs))

def main():
    users = pd.read_parquet(f"{DATA_DIR}/users.parquet")
    items = pd.read_parquet(f"{DATA_DIR}/items.parquet")
    train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    val   = pd.read_parquet(f"{DATA_DIR}/val.parquet")

    n_users = users.user_id.max()+1
    n_items = items.item_id.max()+1

    ds = ImplicitDataset(train, n_items)
    dl = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(n_users, n_items, emb_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        losses = []
        for u,i,y in tqdm(dl, desc=f"epoch {epoch+1}"):
            u,i,y = u.to(device), i.to(device), y.to(device)
            logits = model(u,i)
            loss = bce_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch+1} loss={np.mean(losses):.4f}")

    user2items = build_user_catalog(train)
    all_items = list(range(n_items))
    p,r,n = evaluate(model, user2items, all_items, val, device, K=10)
    print(f"Val@10: precision={p:.4f} recall={r:.4f} ndcg={n:.4f}")

    torch.save(model.state_dict(), "ncf.pt")
    print("Saved model -> ncf.pt")

if __name__ == "__main__":
    main()
