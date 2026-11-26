"""Download the eight-future inference weights and three portable demo scenes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-id', default='songyiren/ShowNow-5B-8F')
    ap.add_argument('--revision', default='main')
    ap.add_argument('--local-dir', type=Path, default=ROOT/'checkpoints/ShowNow-5B-8F')
    args = ap.parse_args()
    snapshot_download(args.repo_id, revision=args.revision, local_dir=args.local_dir)
    manifest = json.loads((args.local_dir/'release_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        p = args.local_dir/name
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16*1024*1024), b''):
                digest.update(chunk)
        if p.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise RuntimeError(f'Checksum failed: {name}')
    for src, dst in [('base_config/transformer/config.json', 'models/SolarWM-5B-base/transformer/config.json'),
                     ('nominal_curves.npy', 'data/cf_vizdoom_8a_cam/nominal_curves.npy')]:
        dest = ROOT/dst
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = (args.local_dir/src).read_bytes()
        if dest.exists() and dest.read_bytes() != content:
            raise RuntimeError(f'{dest} already contains different data; keep a backup before replacing it')
        shutil.copy2(args.local_dir/src, dest)
    print(f'Verified model and demo inputs: {args.local_dir}')


if __name__ == '__main__':
    main()
