# ShowNow: Fine-Grained Speculative Streaming for Responsive World Models

**Yiren Song · Guian Fang · Mike Zheng Shou** · Show Lab, National University of Singapore

ShowNow is an action-conditioned, autoregressive video world model that responds to key presses in real time on a
single H100. At every step (one Wan2.2 latent = 4 frames = 0.25 s at 16 fps) it generates the next latent for **every
supported action at once**, in one forward pass over a shared-history tree, and reads the held key only just before
the frames on screen run out. Only the chosen candidate is committed to the K/V cache and decoded (TAEHV). The
student is distilled to **3 SGF denoising steps** (timesteps 1000, 667, 333).

This repository contains the training, inference and evaluation code of the **8-action** model on ViZDoom
(Freedoom 2): W, A, S, D, TURN_L, TURN_R, ATTACK, JUMP (action ids 1-8, 0 = no key).

- Paper: [arXiv:2611.34693](https://arxiv.org/abs/2611.34693)
- Model: [ShowNow-5B-8F](https://huggingface.co/songyiren/ShowNow-5B-8F) — eight-future BF16 inference weights, approximately 10 GB
- Dataset: [ShowNow Multi-Future Dataset](https://huggingface.co/datasets/songyiren/ShowNow-MultiFuture) (Hugging Face;
  21,964 forks, each with eight futures; held-action, switching-action, and evaluation subsets)
- License: [Apache 2.0](LICENSE)

## Layout

```
sim/gen_vizdoom_cf8.py            ViZDoom counterfactual fork generator (hold-key and key-switch modes)
shownow/data/                     pre-encoding to Wan2.2 latents + camera poses (true / nominal)
shownow/tree/                     tree teacher, Stage 1 (AnyFlow flow map) and Stage 2 (SGF distillation)
shownow/train/                    model builders shared by the stages
shownow/latency/tree_fast.py      real-time engine (shared-prefix K/V ring, cascade attention)
shownow/latency/tree_opt.py       optimised real-time path (fused kernels, FA3, CUDA graphs, fused commit)
shownow/demo/                     browser demo (MJPEG stream + key events)
shownow/eval/                     evaluation: E1 latency harness and statistics, E2 short-horizon and E3 60 s quality,
                                  IDM, compute benchmarks, external-model latency adapters
sim/record_human_trace.py         records human key timing in ViZDoom for latency replay
tests/tree/                       tree-attention / tree K/V equivalence tests (GPU)
scripts/train.sh                  torchrun launcher (single / multi node)
scripts/gen_data_8a.sh            parallel data generation
```

## Installation

Python 3.10 and an H100 80 GB GPU are recommended. The release smoke test used PyTorch 2.14.1+cu130 and
FlashAttention 3; training used the separate PyTorch 2.5.1 / CUDA 12.4 environment. Use a driver compatible with the
PyTorch CUDA build. Training uses FSDP over 8 GPUs per node.

```bash
git clone https://github.com/showlab/ShowNow && cd ShowNow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-fast.txt       # optimized H100 inference
# For training / reference inference, use a separate environment:
#   pip install -r requirements.txt
# optional, data generation (CPU):
#   pip install -r requirements-sim.txt

# external code, placed at the paths the scripts expect (relative to the repo root)
git clone https://github.com/Junchao-cs/SolarWM SolarWM
git -C SolarWM checkout ce1da4e7705391eda8eeda6016c0fd3f614b975e
git clone https://github.com/madebyollin/taehv third_party/taehv
git -C third_party/taehv checkout 9055eee0391548d1787938d040a161cbe396bf98
# taew2_2.pth is included in the upstream TAEHV repository.
```

## Download and run

```bash
python scripts/download_model.py
export SHOWNOW_N_ACTIONS=9 PYTHONPATH=$PWD
python -m shownow.demo.generate --out runs/demo
```

This downloads the eight-future EMA student, inference configuration, nominal camera curves and three portable
initialization clips. It verifies the asset checksums and installs the small architecture/camera files at the paths
expected by the runtime. The optimized single-GPU demo prepares all eight candidates at every step, selects actions
W/A/S/D/turn/fire/jump, and writes `runs/demo/rollout.mp4` plus a smoke-test report. The default rollout is 8 seconds
at **864 × 480, 16 fps**, using three denoising steps. The first run compiles kernels and captures CUDA graphs.
`--scene 0`, `1`, or `2` selects a different initialization; `--steps` sets the number of generated four-frame blocks.

The quick start uses precomputed prompt embeddings and the TAEHV decoder. It needs no teacher checkpoint, full
training dataset, text-encoder weights, or full Wan VAE. The released `.safetensors` contains only the inference EMA;
the original training `.pt` format remains supported by the loader.

| Asset | Location |
|---|---|
| Eight-future student and demo inputs | [songyiren/ShowNow-5B-8F](https://huggingface.co/songyiren/ShowNow-5B-8F) |
| Eight-future videos, metadata and fixed evaluation split | [songyiren/ShowNow-MultiFuture](https://huggingface.co/datasets/songyiren/ShowNow-MultiFuture) |
| Streaming decoder | `third_party/taehv/taew2_2.pth` from the pinned TAEHV repository |
| Additional assets for data encoding / training | SolarWM-5B-base, following the [SolarWM instructions](https://github.com/Junchao-cs/SolarWM); place under `models/SolarWM-5B-base/` |

All eight-action commands use `SHOWNOW_N_ACTIONS=9`: eight actions plus the no-action embedding. Interactive idle
pauses the stream. Simultaneous keys use the most recently pressed action.

## Data

The generated forks used in the paper are released as the
[ShowNow Multi-Future Dataset](https://huggingface.co/datasets/songyiren/ShowNow-MultiFuture) (mp4 + per-fork metadata;
`shownow/data/hf_release_multifuture.py` builds the shards).

| Subset | Forks | Contents |
|---|---:|---|
| `held` | 10,241 | One held action per branch |
| `switch` | 11,471 | Action changes within each branch |
| `eval` | 252 | Separate-seed evaluation forks and fixed split |

The full release is approximately 304 GB across 24 tar shards. Each fork has `history.mp4`, eight
`future_<ACTION>.mp4` files, and `meta.json`. Raw videos are 1280 × 720 at 32 fps; encoding crops the history and
resamples to the model's 864 × 480, 16 fps input. Download only the required subset with the Hugging Face CLI:

```bash
hf download songyiren/ShowNow-MultiFuture --repo-type dataset --include "eval/*" "manifest.json" "eval_split.json" --local-dir data/multifuture
mkdir -p data/cf_vizdoom_sw8i_32fps
tar -xf data/multifuture/eval/eval-00000.tar -C data/cf_vizdoom_sw8i_32fps
```

`manifest.json` records sample IDs, sizes and SHA256 for every shard. `eval_split.json` gives portable `.pt`
filenames; after encoding the evaluation videos, copy it to `data/cf_vizdoom_sw8i_lat/split.json`.
The existing inference demos use the small model-package examples and do not require downloading these shards.
To regenerate the dataset, or to make more:

1. **Generate forks** (CPU). A fork = 3 s of real history + 8 counterfactual 3 s futures from the same engine state, one
   per branch. Hold-key forks hold the branch key for 3 s; key-switch forks start with the branch key and switch to a
   different random key every 1-6 latents (0.25-1.5 s, on the latent grid; 25 % of branches hold).

   ```bash
   # hold-key training forks, MAP01-32 (MAP29-32 = map-disjoint validation)
   scripts/gen_data_8a.sh data/cf_vizdoom_8a_32fps 0 32 400 --prefix c8_ --no-damage-flash
   # key-switch training forks, training maps, seed 0
   scripts/gen_data_8a.sh data/cf_vizdoom_sw8t_32fps 0 32 400 --prefix t8_ --mode switch --no-damage-flash \
       --maps "$(printf 'MAP%02d,' {1..28} | sed 's/,$//')"
   # key-switch evaluation forks, training maps, a different seed
   scripts/gen_data_8a.sh data/cf_vizdoom_sw8i_32fps 0 32 50 --prefix e8_ --mode switch --no-damage-flash --seed 1 \
       --maps "$(printf 'MAP%02d,' {1..28} | sed 's/,$//')"
   ```

2. **Pre-encode** to 16 fps Wan2.2 latents (21 per branch: 9 history + 12 future, 864x480) and camera poses
   (`true` = engine camera, `nominal` = the key's median free-motion trajectory, what inference provides):

   ```bash
   python -m shownow.data.vizdoom_preencode_8a curves                 # nominal per-key motion curves (once)
   python -m shownow.data.vizdoom_preencode_8a encode --forks <list.json> --shard i --num-shards n
   python -m shownow.data.vizdoom_preencode_sw8a --src data/cf_vizdoom_sw8t_32fps --out data/cf_vizdoom_sw8t_lat \
       --cam-out data/cf_vizdoom_sw8t_cam --shard i --num-shards n
   python -m shownow.data.vizdoom_preencode_sw8a --src data/cf_vizdoom_sw8i_32fps --out data/cf_vizdoom_sw8i_lat \
       --cam-out data/cf_vizdoom_sw8i_cam --shard i --num-shards n
   python -m shownow.eval.sw8a_split                                   # fixed evaluation split of the sw8i forks
   ```

   Forks with any damage / health change are skipped (red damage flash). An optional `keep.json` (list of sample ids) in
   a latent directory restricts the training split to those forks.

## Training (recipe of the paper's 8-action model)

Training code covers teacher fine-tuning, AnyFlow distillation and Self-Forcing refinement. This release distributes
the final inference student. The recipe below records the intermediate teacher/student initializations used by the
paper; those checkpoints are not part of the inference package.

Environment for every stage:

```bash
export SHOWNOW_N_ACTIONS=9 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# mixed hold-key + key-switch data for the --data-v3 stages; no SolarWM-Data mixing
export SHOWNOW_V3_DOOM_ROOTS=data/cf_vizdoom_8a_lat,data/cf_vizdoom_sw8t_lat SHOWNOW_V3_SWM_ROOT=none
```

`scripts/train.sh <module> [args]` runs one stage with torchrun (8 GPUs per node; one fork per GPU per step, so the
global batch is the number of GPUs; `--accum` reproduces a larger batch on fewer nodes). GPU counts below are those
used for the paper.

| stage | command (module + key arguments) | GPUs |
|---|---|---|
| 1. tree teacher, hold keys | `shownow.tree.teacher_tree_ft --data data/cf_vizdoom_8a_lat --cam-dir data/cf_vizdoom_8a_cam --init <4-action tree teacher> --steps 3000 --lr 1e-5 --lr-action 3e-4 --lam 0.25 --cam-scale 0.5` | 8 |
| 2. tree teacher, key switches | `shownow.tree.teacher_tree_ft --data-v3 --init <stage 1 @3000> --steps 2000 --lr 1e-5 --lr-action 3e-4 --lam 0.25 --cam-scale 0.5` | 16 |
| 3. Stage 1 (AnyFlow, B1), hold keys | `shownow.tree.student_stage1_tree --data data/cf_vizdoom_8a_lat --cam-dir data/cf_vizdoom_8a_cam --init <stage 1 teacher @3000> --steps 2000 --block 1 --history 15 --branches 4 --lam-cf 0.25` then `--resume <@2000> --steps 4000` | 8, then 16 |
| 4. Stage 1, key switches | `shownow.tree.student_stage1_tree --data-v3 --init <stage 2 teacher @2000> --resume <stage 3 @4000> --steps 6000 --block 1 --history 15 --branches 4 --lam-cf 0.25` (the paper's run stopped at step 5750) | 16 |
| 5. Stage 2 (SGF, 3 steps) | `shownow.tree.student_stage2_tree --data-v3 --student <stage 4 @5750> --teacher <stage 2 teacher @2000> --denoising-steps 1000,667,333 --steps 1500 --ratio 5 --lr 1e-6 --lr-action 5e-6 --critic-lr 4e-7 --guidance 3.0 --ema-decay 0.99 --ema-start 39 --block 1 --history 15 --branches 4 --p-true 0.3 --p-static 0.1 --save-every 250` | 16 |

Every stage also takes `--out <dir>`. Stage 2 can be continued with `--resume <its checkpoint>` (generator, EMA and
critic weights and the step counters are restored; optimizer moments restart). The released model is Stage 2 @1000
(best E2 action accuracy on the in-map validation forks).

## Inference

**Browser demo** (reference FastTree engine, approximately 270 ms per eight-candidate step on an H100).
The optimized video-generation quick start above uses the faster `RTStepper` engine.

```bash
SHOWNOW_N_ACTIONS=9 SOLARWM_COMPILE_FLEX=1 PYTHONPATH=$PWD python -m shownow.demo.play_server \
    --student checkpoints/ShowNow-5B-8F/shownow_8f.safetensors \
    --scene-dir checkpoints/ShowNow-5B-8F/examples/lat --denoising-steps 1000,667,333 --port 8765
# open http://127.0.0.1:8765 : W/A/S/D move, Q/E turn, F fire, Space jump, R restart, N next scene
```

**Optimised real-time path** (`shownow/latency/tree_opt.py`, environment of `requirements-fast.txt`): per step all 8
candidates in one tree pass (about 228 ms incl. commit on an H100, option set `exact` = lossless fused kernels + fused
commit), CUDA graphs, FA3. Key-press latency with scripted Poisson key traces:

```bash
SHOWNOW_N_ACTIONS=9 PYTHONPATH=$PWD python -m shownow.eval.latency_harness --student <ckpt> --data sw8a \
    --denoising-steps 1000,667,333 --policies shownow_opt --opts exact --lead 1 \
    --traces poisson0.5,poisson1,poisson2,poisson4 --seeds 0,1,2 --seconds 60 --out runs/e1
```

**60 s rollouts** with scripted keys (real-time path, then normal inference replaying the same keys and cameras):

```bash
SHOWNOW_N_ACTIONS=9 PYTHONPATH=$PWD python -m shownow.eval.long_eval_8a gen --kind s2 --mode realtime_opt --opts exact \
    --student <ckpt> --out runs/e3/rt
SHOWNOW_N_ACTIONS=9 PYTHONPATH=$PWD python -m shownow.eval.long_eval_8a gen --kind s2 --student <ckpt> \
    --from runs/e3/rt --out runs/e3/ref
```

`--rank i --world n` shards any of these over GPUs. `SHOWNOW_SPLIT=test` selects the held-out test scenes.

## Paper experiments

All latency numbers come from one H100 per run with nothing else running on the machine (the harness checks this;
`--allow-shared` skips the check, not for reported numbers). Fixed settings of the 8-action main table: 16 fps,
`--key-set 1,3,2,4,5,6,7,8`, Poisson key traces at 0.5 / 1 / 2 / 4 presses per second, 60 s per trace, seeds 0-5.

**E1, key-press latency** (`shownow/eval/latency_harness.py`, `requirements-fast.txt` environment). Configurations:

| paper name | harness arguments |
|---|---|
| ShowNow-8F (deployed) | `--policies shownow_opt --opts exact --lead 1` |
| ShowNow-4F | `--policies shownow_opt --opts exact --lead 1 --k 4` |
| ShowNow-1F | `--policies one_future_opt --opts exact --lead 1` |
| ShowNow-Reactive | `--policies reactive_opt --opts exact --reactive-lead 1` |
| SolarWM-FT (B3 student, `requirements.txt` environment) | `--block 3 --policies reactive --reactive-lead -1` |

```bash
SHOWNOW_N_ACTIONS=9 SHOWNOW_OPT_DYNAMIC=0 SOLARWM_COMPILE_FLEX=1 PYTHONPATH=$PWD python -m shownow.eval.latency_harness \
    --student <ckpt> --data sw8a --denoising-steps 1000,667,333 --fps 16 --key-set 1,3,2,4,5,6,7,8 \
    --traces poisson0.5,poisson1,poisson2,poisson4 --seeds 0,1,2,3,4,5 --seconds 60 \
    --policies shownow_opt --opts exact --lead 1 --out runs/e1/shownow_8f
# statistics: equal-rate mixture P50 / P90 / P99, paired seed-cluster bootstrap CIs; presses in the last 10 s of a
# trace are not scored
python -m shownow.eval.main_table_stats --run ShowNow-Reactive=runs/e1/reactive --run ShowNow-1F=runs/e1/one_future \
    --run ShowNow-8F=runs/e1/shownow_8f --ref ShowNow-Reactive --end-margin 10 --metric presentation --out runs/e1/stats.md
```

`--opts none` turns the deployment fusions off (per-frame kernels, fused QKV, fused GELU, fused commit);
`--traces human:<file.json>` replays a trace recorded with `sim/record_human_trace.py`.
`shownow/eval/press_breakdown.py` and `shownow/eval/perceived_latency.py` (runs saved with `--save-lat`) break the
latency down per press and measure action-to-effect latency with the IDM.

**E2, short-horizon action following** (fork evaluation scenes, IDM v2 trained with `shownow/eval/idm_8a_v2.py`):

```bash
SHOWNOW_N_ACTIONS=9 PYTHONPATH=$PWD python -m shownow.eval.e2_8a --kind s2 --ckpt <ckpt> --split test_e2 \
    --mode realtime_opt --opts exact --out runs/e2/rt --rank i --world n
SHOWNOW_N_ACTIONS=9 PYTHONPATH=$PWD python -m shownow.eval.e2_8a_score runs/e2/rt --idm <idm_v2.pt> --tag v2
```

**E3, 60 s quality** (20 test scenes; generation as in *Inference* above with `--seed <s>`, then scoring):

```bash
SHOWNOW_SPLIT=test python -m shownow.eval.long_eval_8a score runs/e3/rt --idm <idm_v2.pt> --metrics-name long_metrics_v2.json
python -m shownow.eval.long_eval_8a table runs/e3/* --metrics-name long_metrics_v2.json
```

`shownow/eval/e3_equal_steps.py` compares models at an equal number of autoregressive steps.

**Compute benchmarks.** `shownow/eval/compute_bench.py` (cache-free vs cached paths, tree vs serial independent
candidates), `shownow/eval/batched_indep_bench.py` (the tree vs eight independent sequences in one batch with a
duplicated history cache), `shownow/eval/unfused_budget_bench.py` (per-step cost with the deployment fusions off).

**External models.** `shownow/eval/ext_latency.py --model {mg2,mg3,yume,reworld,diamond}` and
`shownow/eval/hy_nomas_latency.py` measure other interactive world models under the same key-press protocol; each runs
in that model's own environment with its code and weights installed separately.

## Release scope

This repository releases the eight-action ViZDoom training and inference code, the final eight-future student,
and the matched multi-future dataset. Long rollouts and revisits can lose visual consistency; combinations of
simultaneous actions and the separate 33B model are outside this checkpoint's action interface.

## Citation

```bibtex
@article{shownow2026,
  title   = {ShowNow: Fine-Grained Speculative Streaming for Responsive World Models},
  author  = {Song, Yiren and Fang, Guian and Shou, Mike Zheng},
  journal = {arXiv preprint arXiv:2611.34693},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.34693}
}
```

## Acknowledgements

ShowNow builds on [SolarWM](https://github.com/Junchao-cs/SolarWM) and Wan2.2, uses the
[TAEHV](https://github.com/madebyollin/taehv) decoder, and generates data with [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
and Freedoom.

