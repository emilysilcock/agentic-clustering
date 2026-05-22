# `benchmarking/data_processing/`

Loaders for the 7 benchmark datasets defined in `paper/SPEC.md` §5.1. Each loader
normalises its source into a unified schema and writes to `data/derived/<name>/`.
Loaders are deterministic — every method in `benchmarking/experiments/` reads from
these files; no method should call `load_dataset` itself.

## Running

```powershell
# Process all 7
uv run --native-tls python -m benchmarking.data_processing.process_all

# Process a subset
uv run --native-tls python -m benchmarking.data_processing.process_all --only banking77 clinc150

# Per-dataset (for debugging one loader)
uv run --native-tls python -m benchmarking.data_processing.banking77
```

The `--native-tls` flag and an unset `SSL_CERT_FILE` are required on the Windows
machine this project was developed on — see the project memory file
`feedback-windows-tls-stack` for why. `truststore` is injected automatically by
`benchmarking/__init__.py` on win32.

## Output layout

For each dataset:

```
data/derived/<dataset>/
  documents.jsonl    # one JSON object per document (schema below)
  taxonomy.json      # {label_id: label_name}, includes "-1": "__none__" iff has_none
  meta.json          # provenance: HF handle, config, splits used, k, n_none, …
```

### `documents.jsonl` schema

```json
{
  "doc_id": "banking77-test-000042",
  "text": "How do I activate my new card?",
  "gold_label_id": 3,
  "gold_label_name": "activate_my_card",
  "is_none": false,
  "source_split": "test",
  "source_row_index": 42
}
```

- `gold_label_id` is `-1` and `gold_label_name` is `"__none__"` when `is_none=true`.
  In-scope ids are contiguous `0..k-1`.
- `doc_id` format is `<dataset>-<source_split>-<6-digit row index>`.
- `source_split` + `source_row_index` enable traceability back to the source row.

## The 7 datasets at a glance

| Module | n_docs | k | has_none | Source notes |
|---|---:|---:|:---:|---|
| `banking77.py` | 3,080 | 77 | no | HF `PolyAI/banking77` test, via `refs/convert/parquet` (the canonical revision is a legacy dataset script) |
| `clinc150.py` | 5,500 | 150 | yes (1,000) | HF `clinc/clinc_oos` config `plus`, test. OOS class mapped to `-1`. Loaded from the parquet conversion branch directly (configs are subdirectories there). |
| `massive_intent.py` | 2,974 | 60 | no | HF `mteb/amazon_massive_intent` EN test. Taxonomy derived from train+val+test (test alone is missing `cooking_query`, but k=60 stays per prior work). |
| `massive_domain.py` | 2,974 | 18 | no | HF `mteb/amazon_massive_scenario` EN test. `process_all` verifies the texts align row-by-row with `massive_intent`. |
| `goemotions.py` | 45,446 | 27 | yes (16,021) | HF `google-research-datasets/go_emotions` config `simplified`, train+val+test concatenated. Neutral mapped to `-1`. Multi-label rows in `simplified` (8,817) are skipped. |
| `twenty_newsgroups.py` | 18,331 | 20 | no | sklearn `fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))`. 515 rows dropped because the text was empty after stripping. |
| `stackexchange.py` | 4,156 | 121 | no | ClusterLLM-small (`small.jsonl` from the ClusterLLM data zip). Auto-downloaded to `data/raw/clusterllm/` on first run. **Not** the MTEB `stackexchange-clustering` HF dataset — see SPEC §5.1.1. |

Numbers above are what `meta.json` reports after the first run; treat them as
load-bearing assertions in tests.

## Validation

`base.validate()` is called for every dataset by `process_all`. It hard-fails on:

- Empty text fields
- Duplicate `doc_id`s
- Out-of-range `gold_label_id`s (must be `0..k-1` for in-scope, `-1` for none)
- Taxonomy size mismatch with the expected `k`
- `n_none == 0` on datasets where `expects_none=True` (CLINC, GoEmotions)
- For `massive_domain`: text-by-text alignment with `massive_intent` (cross-corpus check)

## Adding a new dataset

1. Create `benchmarking/data_processing/<name>.py` exporting `load() -> ProcessedDataset`.
   Use `base.format_doc_id`, `base.build_meta`. For datasets where source label ids are
   sparse or filtered, call `base.remap_in_scope_label_ids` to compact ids to `0..k-1`.
2. Register `(load_fn, expected_k, expects_none)` in `process_all.REGISTRY`.
3. Add a one-liner to `SPEC.md` §5.1.
4. Run `python -m benchmarking.data_processing.process_all --only <name>` to verify.
