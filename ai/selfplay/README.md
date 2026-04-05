# Self-Play Trajectory Generation

This folder contains a Python wrapper that runs the Java legal self-play runner repeatedly and writes merged JSONL trajectories.

## Files

- `selfplay_trajectory_generator.py`: Runs `game.SelfPlayGameRunner` for `N` games and appends JSONL rows to one output file.

## Quick Start

From repo root:

```sh
python ai/selfplay/selfplay_trajectory_generator.py --num-games 5 --output data/selfplay_5games.jsonl
```

This command compiles Java runner classes first, then simulates 5 full games.

## Faster Re-runs

If classes are already compiled, skip compile:

```sh
python ai/selfplay/selfplay_trajectory_generator.py --num-games 20 --output data/selfplay_20games.jsonl --skip-compile
```

## Debug One Game (Forensics)

```sh
python ai/selfplay/selfplay_trajectory_generator.py \
	--num-games 1 \
	--output data/selfplay_debug_1game.jsonl \
	--java-debug \
	--java-max-turns 80 \
	--seed 42
```

Optional: relax strict invariant aborts while inspecting logs.

```sh
python ai/selfplay/selfplay_trajectory_generator.py \
	--num-games 1 \
	--output data/selfplay_debug_1game.jsonl \
	--java-debug \
	--java-no-strict
```

## Downstream Dataset Pipeline

```sh
python ai/split_manifest_v1.py --input data/selfplay_20games.jsonl --output data/split_manifest.json
python ai/trajectory_dataset_builder_v1.py --input data/selfplay_20games.jsonl --output data/trajectory_dataset.jsonl --split-manifest data/split_manifest.json
python ai/dataset_builder_v1.py --input data/selfplay_20games.jsonl --output data/flat_dataset.jsonl --split-manifest data/split_manifest.json
```

## Guided Generation (Greedy)

Use the trained win model at every decision step in Java self-play (greedy action choice):

```sh
python ai/selfplay/selfplay_trajectory_generator.py \
	--num-games 20 \
	--output ai/selfplay/data/selfplay_guided_20games.jsonl \
	--java-guided \
	--java-guided-model-path ai/selfplay/models/win_model_v1_stageaware/model_hgb.pkl \
	--java-guided-python-exec /Users/yash/Desktop/catanthropic/.venv/bin/python \
	--java-max-turns 1200 \
	--seed 100
```

Guided options forwarded to Java:
- `--java-guided`
- `--java-guided-model-path`
- `--java-guided-python-exec`

## Notes

- Generated games come from your Java game logic, so states are legally reachable.
- Each output line is one trajectory snapshot row in JSON format.
