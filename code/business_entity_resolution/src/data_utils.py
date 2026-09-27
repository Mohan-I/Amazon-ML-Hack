"""
data_utils.py — Ground truth labeling and leak-free dataset splitting (Person 2).

Provides:
  - Robust ground truth loading (handles empty matches, multiple comma-separated IDs, singletons)
  - Candidate pair parsing and binary target assignment
  - Entity-level train/validation splitting on Source 1 entities (strictly leak-free)
  - Dataset statistics reporting
"""

import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Loads ground truth TSV into a dict mapping source1_entity_id -> set of matched_entity_ids."""
    log(f"loading ground truth from {gt_path.name} ...")
    gt = pl.read_csv(gt_path, separator="\t", infer_schema_length=0)
    gt_map: Dict[str, Set[str]] = {}
    total_matches = 0

    for sid, matched in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        if matched and matched.strip():
            cands = set(x.strip() for x in matched.split(",") if x.strip())
            gt_map[sid] = cands
            total_matches += len(cands)
        else:
            gt_map[sid] = set()

    singletons = sum(1 for s in gt_map.values() if len(s) == 0)
    log(f"  ground truth loaded: {len(gt_map):,} S1 entities, {total_matches:,} true match pairs, {singletons:,} singletons")
    return gt_map


def load_candidate_pairs(cand_path: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Loads candidate pairs TSV.
    Returns:
      pairs: list of (source1_entity_id, candidate_entity_id)
      universe_s1: list of all source1_entity_id rows in the candidate file
    """
    log(f"loading candidates from {cand_path.name} ...")
    cand_df = pl.read_csv(cand_path, separator="\t", infer_schema_length=0)
    universe_s1 = cand_df["source1_entity_id"].to_list()

    pairs: List[Tuple[str, str]] = []
    for sid, cands in zip(universe_s1, cand_df["candidate_entity_ids"].to_list()):
        if not cands or not cands.strip():
            continue
        for cid in cands.split(","):
            cid_clean = cid.strip()
            if cid_clean:
                pairs.append((sid, cid_clean))

    log(f"  candidate file loaded: {len(universe_s1):,} S1 rows, {len(pairs):,} total candidate pairs")
    return pairs, universe_s1


def create_training_labels(
    pairs: List[Tuple[str, str]],
    gt_map: Dict[str, Set[str]],
) -> np.ndarray:
    """Assigns binary match labels for candidate pairs:
    label = 1 if candidate_entity_id in gt_map[source1_entity_id], else 0.
    """
    n_pairs = len(pairs)
    labels = np.zeros(n_pairs, dtype=np.int32)

    unique_s1 = set()
    unique_cand = set()
    n_pos = 0

    for i, (sid, cid) in enumerate(pairs):
        unique_s1.add(sid)
        unique_cand.add(cid)
        if cid in gt_map.get(sid, set()):
            labels[i] = 1
            n_pos += 1

    n_neg = n_pairs - n_pos
    pos_pct = (n_pos / n_pairs * 100.0) if n_pairs > 0 else 0.0

    log("=" * 60)
    log("TRAINING LABELS SUMMARY")
    log("=" * 60)
    log(f"Total candidate pairs        : {n_pairs:,}")
    log(f"Positive pairs (label = 1)   : {n_pos:,}")
    log(f"Negative pairs (label = 0)   : {n_neg:,}")
    log(f"Positive percentage          : {pos_pct:.2f}%")
    log(f"Source 1 entities in pairs   : {len(unique_s1):,}")
    log(f"Candidate entities in pairs  : {len(unique_cand):,}")
    log("=" * 60)

    return labels


def entity_level_train_val_split(
    pairs: List[Tuple[str, str]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Set[str], Set[str]]:
    """Partitions dataset at the Source 1 entity level to prevent data leakage.
    Returns:
      train_indices: 1D numpy array of row indices in `pairs`
      val_indices: 1D numpy array of row indices in `pairs`
      train_s1_set: set of Source 1 IDs in train split
      val_s1_set: set of Source 1 IDs in val split
    """
    s1_list = [sid for sid, _ in pairs]
    unique_s1 = sorted(list(set(s1_list)))

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_s1)

    n_val_s1 = int(len(unique_s1) * val_ratio)
    val_s1_set = set(unique_s1[:n_val_s1])
    train_s1_set = set(unique_s1[n_val_s1:])

    # Assert no overlap between train and validation S1 entities
    assert len(train_s1_set & val_s1_set) == 0, "Leakage detected: train and val S1 entities overlap!"

    train_indices = []
    val_indices = []

    for i, sid in enumerate(s1_list):
        if sid in val_s1_set:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_idx_arr = np.array(train_indices, dtype=np.int64)
    val_idx_arr = np.array(val_indices, dtype=np.int64)

    log("=" * 60)
    log("ENTITY-LEVEL TRAIN / VALIDATION SPLIT (LEAK-FREE)")
    log("=" * 60)
    log(f"Random seed                  : {seed}")
    log(f"Training S1 entities         : {len(train_s1_set):,}")
    log(f"Validation S1 entities       : {len(val_s1_set):,}")
    log(f"Training candidate pairs     : {len(train_idx_arr):,}")
    log(f"Validation candidate pairs   : {len(val_idx_arr):,}")
    log("=" * 60)

    return train_idx_arr, val_idx_arr, train_s1_set, val_s1_set
