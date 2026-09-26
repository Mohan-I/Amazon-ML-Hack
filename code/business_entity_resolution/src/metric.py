"""
metric.py — the official F_0.5 scorer, computed exactly as the challenge
describes: per Source-1 entity, macro-averaged, singletons included
(empty-empty = 1.0, any false positive on a true singleton = 0.0).
"""


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(pred: set, true: set) -> float:
    if not true and not pred:
        return 1.0  # correctly predicted singleton
    if not pred:
        return 0.0  # missed everything
    tp = len(pred & true)
    precision = tp / len(pred)
    recall = tp / len(true) if true else 0.0
    return f_beta(precision, recall)


def macro_f05(predictions: dict, ground_truth: dict) -> float:
    """predictions, ground_truth: dict[source1_entity_id] -> set(matched_ids).
    Every key in ground_truth must have an entry in predictions (missing
    entities count as an empty prediction)."""
    total = 0.0
    n = 0
    for sid, true in ground_truth.items():
        pred = predictions.get(sid, set())
        total += score_entity(pred, true)
        n += 1
    return total / n if n else 0.0
