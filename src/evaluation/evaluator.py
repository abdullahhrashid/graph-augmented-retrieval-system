import numpy as np

def recall_at_k(ranked_ids, gold_ids, k):
    top_k = set(ranked_ids[:k])
    found = len(top_k & set(gold_ids))
    return found / len(gold_ids)


def em_at_k(ranked_ids, gold_ids, k):
    top_k = set(ranked_ids[:k])
    return 1.0 if set(gold_ids).issubset(top_k) else 0.0


def ndcg_at_k(ranked_ids, gold_ids, k):
    gold_set = set(gold_ids)
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in gold_set:
            dcg += 1.0 / np.log2(i + 2)  

    n_rel = min(len(gold_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(ranked_ids, gold_ids):
    gold_set = set(gold_ids)
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in gold_set:
            hits += 1
            sum_precision += hits / (i + 1)
    if len(gold_set) == 0:
        return 0.0
    return sum_precision / len(gold_set)


def evaluate_system(all_ranked_ids, all_gold_ids, k_values):
    n = len(all_ranked_ids)
    results = {}

    for k in k_values:
        recalls = [recall_at_k(all_ranked_ids[i], all_gold_ids[i], k) for i in range(n)]
        ems = [em_at_k(all_ranked_ids[i], all_gold_ids[i], k) for i in range(n)]
        ndcgs = [ndcg_at_k(all_ranked_ids[i], all_gold_ids[i], k) for i in range(n)]

        results[f'Recall@{k}'] = np.mean(recalls)
        results[f'EM@{k}'] = np.mean(ems)
        results[f'nDCG@{k}'] = np.mean(ndcgs)

    aps = [average_precision(all_ranked_ids[i], all_gold_ids[i]) for i in range(n)]
    results['MAP'] = np.mean(aps)

    return results
