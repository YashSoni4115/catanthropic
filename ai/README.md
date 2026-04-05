# AI Data Pipeline

## Modules

- `board_state_extractor.py`: extracts canonical game-state payloads.
- `vectorizer_v1.py`: observed-state vectorizer (`vec_v1.1.0`).
- `dataset_builder_v1.py`: outcome-aware streaming row builder.
- `trajectory_dataset_builder_v1.py`: trajectory-aware builder with per-game row indexing.
- `split_manifest_v1.py`: generates deterministic game-level split manifests.
- `train_win_model_v1.py`: trains first baseline winner predictor (`P(current player eventually wins)`).

## Recommended Corpus Workflow

1. Generate a split manifest once (by `game_id`).
2. Build rows using that manifest (flat or trajectory builder).
3. Keep split assignment stable across all experiments.

## 1) Generate Split Manifest

```zsh
/Users/yash/Desktop/catanthropic/.venv/bin/python -m ai.split_manifest_v1 \
  --input /path/to/states.jsonl \
  --output /path/to/split_manifest.json \
  --split-seed seed_v1 \
  --split-ratios 80,10,10
```

## 2) Build Flat Rows Using Manifest

```zsh
/Users/yash/Desktop/catanthropic/.venv/bin/python -m ai.dataset_builder_v1 \
  --input /path/to/states.jsonl \
  --output /path/to/rows.jsonl \
  --split-manifest /path/to/split_manifest.json
```

## 3) Build Trajectory Rows Using Manifest

```zsh
/Users/yash/Desktop/catanthropic/.venv/bin/python -m ai.trajectory_dataset_builder_v1 \
  --input /path/to/states.jsonl \
  --output /path/to/trajectory_rows.jsonl \
  --split-manifest /path/to/split_manifest.json
```

## Tests

```zsh
/Users/yash/Desktop/catanthropic/.venv/bin/python -m unittest -v \
  /Users/yash/Desktop/catanthropic/ai/tests/test_vectorizer_v1.py \
  /Users/yash/Desktop/catanthropic/ai/tests/test_dataset_builder_v1.py \
  /Users/yash/Desktop/catanthropic/ai/tests/test_trajectory_dataset_builder_v1.py \
  /Users/yash/Desktop/catanthropic/ai/tests/test_split_manifest_v1.py
```

## Winner Model V1 Training

Train directly from self-play snapshots (recommended first pass):

```zsh
/Users/yash/Desktop/catanthropic/.venv/bin/python -m ai.train_win_model_v1 \
  --input /Users/yash/Desktop/catanthropic/ai/selfplay/data/selfplay_1000games_v4.jsonl \
  --input-format selfplay_raw \
  --output-dir /Users/yash/Desktop/catanthropic/selfplay/models/win_model_v1 \
  --winner-only \
  --max-rows-per-game 128 \
  --models hgb,logreg \
  --balance-classes \
  --seed 42
```

Stage-aware (harder early-game emphasis) + low-resource mode:

```zsh
/Users/yash/Desktop/catanthropic/.venv/bin/python -m ai.train_win_model_v1 \
  --input /Users/yash/Desktop/catanthropic/ai/selfplay/data/selfplay_1000games_v4.jsonl \
  --input-format selfplay_raw \
  --output-dir /Users/yash/Desktop/catanthropic/ai/selfplay/models/win_model_v1_stageaware \
  --winner-only \
  --max-rows-per-game 64 \
  --models hgb,logreg \
  --balance-classes \
  --stage-weights 0.5,0.3,0.2 \
  --skip-feature-importance \
  --seed 42
```

Notes:
- `stage_balanced_log_loss` is written per split in `metrics.json`.
- `--stage-weights` is `early,mid,late`; increase early weight if you care more about opening quality.
- `--skip-feature-importance` avoids expensive permutation importance on smaller laptops.

Artifacts:
- `config.json`
- `split_summary.json`
- `metrics.json`
- `model_*.pkl`
- `feature_importance_*.json`
