import numpy as np

def precision_recall_at_k(ranked_items, ground_truth, k=10):
    ranked_k = ranked_items[:k]
    hits = len(set(ranked_k) & set(ground_truth))
    precision = hits / k
    recall = hits / max(len(ground_truth), 1)
    return precision, recall

def ndcg_at_k(ranked_items, ground_truth, k=10):
    ranked_k = ranked_items[:k]
    dcg = 0.0
    for i, it in enumerate(ranked_k, start=1):
        if it in ground_truth:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(ground_truth), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0
