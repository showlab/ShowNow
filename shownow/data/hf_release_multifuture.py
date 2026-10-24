"""Package the ShowNow Multi-Future Dataset for the Hugging Face Hub (WebDataset-style tar shards).

Subsets (exactly the forks used by the paper):
  held     data/cf_vizdoom_8a_32fps, forks listed in data/cf_vizdoom_8a_lat_forks.json   (training, held keys)
  switch   data/cf_vizdoom_sw8t_32fps, forks listed in data/cf_vizdoom_sw8t_lat/keep.json (training, key switches)
  eval     data/cf_vizdoom_sw8i_32fps, every fork, plus data/cf_vizdoom_sw8i_lat/split.json   (validation / test)
Each fork -> one sample in a shard: <id>/meta.json, history.mp4, future_<KEY>.mp4 x 8 (preview.jpg if present).

  python -m shownow.data.hf_release_multifuture pack --out <staging dir> [--per-shard 1000]
  python -m shownow.data.hf_release_multifuture upload --out <staging dir> --repo songyiren/ShowNow-MultiFuture [--private]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBSETS = {
    "held": (REPO / "data/cf_vizdoom_8a_32fps", REPO / "data/cf_vizdoom_8a_lat_forks.json"),
    "switch": (REPO / "data/cf_vizdoom_sw8t_32fps", REPO / "data/cf_vizdoom_sw8t_lat/keep.json"),
    "eval": (REPO / "data/cf_vizdoom_sw8i_32fps", None),
}
KEYS = ["W", "A", "S", "D", "TURN_L", "TURN_R", "ATTACK", "JUMP"]
FILES = ["meta.json", "history.mp4"] + [f"future_{k}.mp4" for k in KEYS]


def forks(name):
    root, lst = SUBSETS[name]
    ids = json.load(open(lst)) if lst else sorted(p.name for p in root.iterdir() if p.is_dir() and p.name[:3] == "e8_")
    for i in ids:
        missing = [f for f in FILES if not (root / i / f).exists()]
        assert not missing, (name, i, missing)
    return root, sorted(ids)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def pack(out: Path, per: int):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in SUBSETS:
        root, ids = forks(name)
        (out / name).mkdir(exist_ok=True)
        shards = []
        for s in range(0, len(ids), per):
            tp = out / name / f"{name}-{s // per:05d}.tar"
            if not tp.exists():
                tmp = tp.with_suffix(".tar.part")
                with tarfile.open(tmp, "w") as tar:
                    for i in ids[s:s + per]:
                        for f in FILES + (["preview.jpg"] if (root / i / "preview.jpg").exists() else []):
                            tar.add(root / i / f, arcname=f"{i}/{f}")
                tmp.rename(tp)
            shards.append({"file": f"{name}/{tp.name}", "forks": ids[s:s + per], "sha256": sha256(tp),
                           "bytes": tp.stat().st_size})
            print(name, tp.name, flush=True)
        manifest[name] = {"n_forks": len(ids), "shards": shards}
    split = json.load(open(REPO / "data/cf_vizdoom_sw8i_lat/split.json"))
    (out / "eval_split.json").write_text(json.dumps(split, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "README.md").write_text((Path(__file__).parent / "hf_multifuture_README.md").read_text())
    print({k: v["n_forks"] for k, v in manifest.items()})


def upload(out: Path, repo: str, private: bool):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(out),
                            allow_patterns=["*.tar", "*.json", "README.md"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "upload"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-shard", type=int, default=1000)
    ap.add_argument("--repo", default="songyiren/ShowNow-MultiFuture")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(Path(a.out), a.per_shard)
    else:
        upload(Path(a.out), a.repo, a.private)


if __name__ == "__main__":
    main()
