---
license: bsd-3-clause
pretty_name: ShowNow Multi-Future Dataset
task_categories:
- image-to-video
- video-to-video
- reinforcement-learning
tags:
- world-model
- interactive-video
- counterfactual
- action-conditioned
- vizdoom
size_categories:
- 10K<n<100K
---

# ShowNow Multi-Future Dataset

Matched multi-future video data for action-conditioned, interactive world models, released with
**ShowNow: Fine-Grained Speculative Streaming for Responsive World Models**.

Each sample is a **fork**: a short first-person history followed by **eight futures**. All eight futures start from the
*same* restored game state and differ only in the action taken. The futures are matched counterfactuals: any difference
between them is caused by the action, not by a different starting point.

| subset | forks | futures per fork | action schedule in the future | use in the paper |
|---|---|---|---|---|
| `held` | 10,241 | 8 | one key held for the whole 3 s future | training |
| `switch` | 11,471 | 8 | each branch starts with its own key, then switches among the 8 keys every 0.25-1.5 s (25 % held throughout) | training |
| `eval` | 252 | 8 | as `switch`, new game states (different generator seed) | validation / test (`eval_split.json`) |

Actions (one per branch): `W` forward, `A` strafe left, `S` backward, `D` strafe right, `TURN_L`, `TURN_R`, `ATTACK`, `JUMP`.

## Format

WebDataset-style tar shards (`<subset>/<subset>-NNNNN.tar`, 1000 forks per shard). Each fork is a folder `<fork_id>/`
containing:

- `history.mp4`: the history before the fork.
- `future_<ACTION>.mp4` x 8: the 3 s future (96 frames) for each action.
- `meta.json`, which holds:
  - engine and map: ViZDoom 1.3.1 / GZDoom, `freedoom2.wad`, map, skill, seed;
  - camera: intrinsics, plus a camera-to-world pose per frame (OpenCV convention, metres);
  - per frame: the button vector and the game state;
  - the fork state (position, yaw, velocity, health, weapon, ammo), and nearby and visible monsters;
  - per-future captions and effect summaries;
  - consistency checks of the state restoration.
- `preview.jpg`, where present.

Video: 1280x720, 32 fps (frames sampled at the nearest engine tic of t = n/32 s; the engine runs at 35 tics/s), horizontal
FOV 130 degrees. The paper trains at 16 fps and 864x480 (temporal and spatial resampling of these files).

`manifest.json` lists the forks and the SHA-256 of every shard. `eval_split.json` gives the evaluation splits:
`e2_val` (48 forks), `test_e2` (96), and one-minute rollout scenes `scenes` (10) / `test_e3` (20).

## How the data were generated

An exploring agent plays Freedoom2 maps MAP01-MAP28 in ViZDoom with god mode. At each sampled fork point the game state
is saved. Each of the eight action branches is then rendered from the restored state. Restoration is verified: each
branch must reproduce the saved snapshot; forks that fail are discarded. Forks with damage in the history are excluded,
and so are forks facing a wall or with a too-dark history (thresholds are in `meta.json` under `checks`). The `held` and
`switch` subsets were used, mixed, for all training stages in the paper. `eval` uses training maps with a different
generator seed and was used only for evaluation.

## Loading

```python
from huggingface_hub import hf_hub_download
import tarfile
p = hf_hub_download("songyiren/ShowNow-MultiFuture", "held/held-00000.tar", repo_type="dataset")
tarfile.open(p).extractall("shownow_mf")  # -> shownow_mf/<fork_id>/{history.mp4, future_*.mp4, meta.json}
```

## License

The videos are rendered with ViZDoom from the Freedoom assets. Freedoom is distributed under a BSD-style license, and
this dataset is released under BSD-3-Clause.

## Citation

```bibtex
@article{shownow2026,
  title   = {ShowNow: Fine-Grained Speculative Streaming for Responsive World Models},
  author  = {Song, Yiren and Fang, Guian and Shou, Mike Zheng},
  year    = {2026}
}
```
