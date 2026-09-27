# output/

This folder holds the pipeline's generated files. They are **not committed to
git** (each is 100MB-600MB, over GitHub's per-file limit) — regenerate them
locally by running the pipeline (see `code/business_entity_resolution/README.md`):

```
candidate_pairs_train.tsv   # blocking candidates for the train set
candidate_pairs_test.tsv    # blocking candidates for the test set (submit this)
matching_results.tsv        # final matches for the test set (submit this)
best_threshold.txt          # tuned similarity threshold (see match.py tune)
```

For the actual competition submission zip, these files DO get included under
`output/` per the required submission structure — that's assembled as a
separate packaging step, not via git.
