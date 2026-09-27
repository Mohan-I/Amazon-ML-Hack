# ML SQUAD — Business Entity Resolution

**Amazon ML Challenge 2026 · Team Submission**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-success)
![Polars](https://img.shields.io/badge/Polars-1.x-CD792C)
![Metric](https://img.shields.io/badge/Macro%20F%E2%82%80.5-0.86-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**A scalable, precision-tuned hybrid pipeline that resolves noisy multi-source business records — built to survive the unseen French test set.**

</div>

---

## 📌 Overview

Business identity data arrives from independent, noisy sources that share **no common identifiers**. Our task is to determine, for every **Source 1** entity, which records in **Source 2** and **Source 3** refer to the **same real-world business** — including the many cases where the correct answer is *"no match at all"* (singletons).

We built a **hybrid blocking + classifier** pipeline:

1. **Country-aware two-pass blocking** reduces the comparison space from *O(N²)* to ~20 candidates per S1 entity.
2. **45 pairwise similarity features** encode name, address, PIN, city, and house-number agreement.
3. **LightGBM** scores every candidate pair.
4. **Threshold tuning** maximizes the official **macro F₀.₅** metric (precision-weighted, β = 0.5).

> **Why F₀.₅ matters:** merging two distinct businesses (a false merge) is more damaging than missing a link. F₀.₅ weights **precision 2× over recall** — so our entire pipeline is tuned to *avoid* false positives.

---

## 📊 Results

### Validation Performance (held-out, entity-level, leak-free split)

| Metric | Value |
|---|---|
| **Macro F₀.₅ (official)** | **0.86** |
| Mean Precision | 0.976 |
| Mean Recall | 0.942 |
| Chosen Threshold | 0.60 |
| Singleton Precision | 1.00 |

### Threshold Sweep (validation)

| Threshold | Macro F₀.₅ | Mean Precision | Mean Recall | Notes |
|:---------:|:----------:|:--------------:|:-----------:|:------|
| 0.50 | 0.84 | 0.95 | 0.96 | more merges |
| **0.60** | **0.86** | **0.976** | **0.942** | ✅ **chosen** |
| 0.70 | 0.85 | 0.99 | 0.88 | conservative alternative |
| 0.80 | 0.82 | 0.99 | 0.79 | precision-only |

### Blocking Recall (upper bound on final F₀.₅)

| K (candidates / S1) | Blocking Recall | Total Candidate Pairs |
|:---:|:---:|:---:|
| 10 | 91.2% | 0.8 M |
| 20 | 95.1% | 1.6 M |
| **30** | **96.8%** | **2.4 M** |
| 50 | 98.2% | 4.0 M |

---

## 🏗 Architecture

```
+-------------------+      +-------------------+      +-------------------+
|  dataset/test/    | ---> |   blocking.py     | ---> | candidate_pairs   |
| (1.73M Entities)  |      | (Inverted Index)  |      |    (1.58M Pairs)  |
+-------------------+      +-------------------+      +-------------------+
                                                                |
                                                                v
+-------------------+      +-------------------+      +-------------------+
|  matching_results | <--- |   final_match.py  | <--- |   score_ml.py     |
|   (PASS Verified) |      |  (Threshold F0.5) |      | (LightGBM Model)  |
+-------------------+      +-------------------+      +-------------------+
```

### Pipeline Stages

| # | Stage | Script | Input → Output |
|---|-------|--------|----------------|
| 1 | Clean & normalize | `normalize.py` | raw TSVs → cleaned name/address/PIN/city |
| 2 | Blocking (train + test) | `blocking.py` | cleaned records → `candidate_pairs_{train,test}.tsv` |
| 3 | Feature engineering | `features.py` | candidate pairs → 45-dim feature matrix |
| 4 | Train LightGBM | `train_ml.py` | train features + GT → `models/lgbm_matcher.txt` |
| 5 | Tune threshold | `threshold_tune.py` | train scores + GT → `best_threshold.json` |
| 6 | Score test pairs | `score_ml.py` | test candidates + model → `pair_scores_test.tsv` |
| 7 | Final match generation | `final_match.py` | test scores + threshold → `matching_results.tsv` |
| 8 | Official validation | `utils/validate_submission.py` | outputs + test dir → `PASS` / `FAIL` |

### Key Design Decisions

- **🔒 Country-aware two-pass blocking**
  A partitioned pass for records with a valid country string, **plus a fallback non-partitioned pass** for records with missing or dirty country fields. This is what keeps French test entities from being silently dropped during candidate generation.

- **🧮 45 pairwise features**
  RapidFuzz (`token_sort`, `token_set`, `partial`, `Levenshtein`, `Jaro-Winkler`), Jaccard over tokens and character 3-grams, PIN / city / house-number match flags, country agreement, and length/token-count deltas — all computed on **normalized** text.

- **🎯 Precision-heavy threshold selection**
  Threshold chosen by maximizing the **official macro F₀.₅** on a leak-free validation split, with a **conservative tie-break** that prefers the higher threshold when two operating points are within 0.001 F₀.₅.

- **👤 Explicit singleton handling**
  S1 entities with no candidate above the threshold receive an **empty** `matched_entity_ids` list. Correctly predicting a singleton earns a full **1.0** on that entity.

- **🔁 Leak-free validation**
  Train / validation split is performed at the **Source 1 entity level** — no S1 entity appears in both splits. This prevents the model from memorizing candidate identities.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce the full pipeline

```bash
cd code/business_entity_resolution/src
bash run_all.sh
```

This runs blocking → training → threshold tuning → scoring → final match generation → **official validator**.

### 3. Or reproduce just the final submission

If `models/lgbm_matcher.txt` and `output/pair_scores_test.tsv` already exist:

```bash
python final_match.py                  # uses output/best_threshold.json
python final_match.py --threshold 0.62 # manual override
python final_match.py --conservative   # higher tie-break threshold
```

---

## 📂 Project Structure

```
Amazon-ML-Hack/
├── .gitignore
├── Documentation_template.md          ✅ top-level (spec)
├── LICENSE                            ✅ MIT
├── README.md                          ✅ this file - repo readme
├── code/
│   └── business_entity_resolution/
│       ├── README.md                  ✅ reproduction instructions
│       ├── models/
│       │   └── lgbm_matcher.txt       ✅ trained LightGBM model
│       ├── requirements.txt           ✅ pinned deps
│       └── src/
│           ├── blocking.py            # two-pass country-aware inverted-index blocking
│           ├── data_utils.py          # GT loading, entity-level split
│           ├── features.py            # 45 pairwise features
│           ├── final_match.py         # threshold → matching_results.tsv
│           ├── metric.py              # official macro F₀.₅ scorer
│           ├── normalize.py           # text cleaning, unidecode, PIN/city extract
│           ├── run_all.sh             ✅ end-to-end entry point 
│           ├── score_ml.py            # test candidate scoring
│           ├── threshold_tune.py      # threshold sweep + curve
│           └── train_ml.py            # LightGBM training
│           └── legacy/                (optional)
│               └── match_baseline.py
├── output/
│   ├── matching_results.tsv           🔴 MUST GENERATE # ✅ scored on leaderboard
│   ├── candidate_pairs.tsv            🔴 MUST GENERATE # blocking audit set
│   └── README.md                      (optional — can stay or be removed)
└── utils/
    └── validate_submission.py         ✅
```

---

## 🔁 Reproducing the Submission

### Full end-to-end (recommended)

```bash
cd code/business_entity_resolution/src
bash run_all.sh
```

### Manual step-by-step

```bash
# 1. Blocking — train candidates (for training + threshold tuning)
python blocking.py --split train --max-candidates 20

# 2. Blocking — test candidates (for submission)
python blocking.py --split test  --max-candidates 20

# 3. Train LightGBM
python train_ml.py

# 4. Tune threshold on validation
python threshold_tune.py

# 5. Score test candidates
python score_ml.py

# 6. Apply threshold → matching_results.tsv + candidate_pairs.tsv
python final_match.py

# 7. Official validator
cd ../../..
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Expected final output:** `PASS — no blocking issues found. Safe to submit.`

---

## 📐 Model & Feature Details

### Model

| Property | Value |
|---|---|
| Algorithm | LightGBM (GBDT, binary classification) |
| Trees | 300 |
| Learning rate | 0.05 |
| Num leaves | 31 |
| Max depth | 6 |
| Subsample / Colsample | 0.8 / 0.8 |
| Early stopping | on validation macro F₀.₅ |
| License | **MIT** |
| Parameters | << 8 B |

### Top Feature Importances

| Rank | Feature | Importance |
|---:|:---|:---:|
| 1 | Name token_set ratio | 0.142 |
| 2 | Name token_sort ratio | 0.115 |
| 3 | PIN exact match | 0.098 |
| 4 | Address token Jaccard | 0.087 |
| 5 | House number match | 0.076 |
| 6 | Combined name+address ratio | 0.068 |
| 7 | Name char-3-gram Jaccard | 0.054 |
| 8 | Country match flag | 0.043 |
| 9 | Name Levenshtein | 0.032 |
| 10 | Address token_set ratio | 0.028 |

---

## ⚖️ Fair Play & Compliance

| Rule | Compliance |
|---|---|
| External data lookup (APIs, govt DBs, geocoding) | ❌ **Not used** — training data only |
| Model license | ✅ **MIT** (LightGBM) |
| Model size | ✅ Well under 8 B parameters |
| Output format | ✅ Verified by `utils/validate_submission.py` |
| Country hard-coding | ✅ Open-set handling with fallback blocking pass |
| Every S1 test entity present | ✅ Guaranteed by `final_match.py` |

> **Fair Play:** This challenge was solved using **only the provided training data**. No external databases, commercial ER APIs, geocoding services, or internet-sourced augmentation were used at any stage.

---

## 📄 Documentation

The full methodology write-up is available at the **top level of the submission zip**:

```
Documentation_template.md
```

It covers:
- Problem analysis and EDA findings
- Blocking / candidate-generation strategy (two-pass, country-aware)
- Model architecture and 45-feature engineering
- Threshold tuning, results, and error analysis
- Code artefacts and reproduction entry points

---

## 👥 Team ML SQUAD

| Member | Role |
|---|---|
| **Dodda SriLatha** | Data cleaning, blocking, candidate generation |
| **Saiteja Gadu** | Feature engineering, model training, scoring |
| **Mohan Yadav** | Threshold tuning, final matching, validation, documentation |

---

## 📜 License

This project is released under the **MIT License** — see [`LICENSE`](./LICENSE) for details.

---

<div align="center">

**Built for Amazon ML Challenge 2026 · 25–27 September 2026**

*Precision-heavy by design. Country-agnostic by necessity. Reproducible by construction.*

</div>
