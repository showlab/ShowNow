"""fp32 check: tree KV sampler == tree cache-free window sampler (same noise), per future block."""
import os
import sys
from pathlib import Path

os.environ.setdefault("SOLARWM_COMPILE_FLEX", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "SolarWM/src"))
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from shownow.train.student_stage2 import build_stage2_student  # noqa: E402
from shownow.train.teacher_action_ft import camera_tokens  # noqa: E402
from shownow.tree import tree_kv, tree_samplers  # noqa: E402
from shownow.tree.tree import enable_tree  # noqa: E402

dev = torch.device("cuda")
d, _, _ = build_stage2_student(str(REPO / "runs/student_stage1/smoke/student_stage1_step000020.pt"))
d.module.to(dev, torch.float32).eval()
enable_tree(d.module)
fork = torch.load(REPO / "data/cf_vizdoom_10k_lat/cf_00009.pt", weights_only=False)
cam = camera_tokens(torch.load(REPO / "data/cf_vizdoom_10k_cam/cf_00009.pt", weights_only=True)["nominal"], 4, dev, 0.5)
clean, acts, prompt = fork["latents"].to(dev, torch.float32), fork["actions"].to(dev), fork["prompt"].to(dev, torch.float32)[None]
steps = [1000.0]  # single step: isolates attention/caching from sampler noise
# disable autocast inside both samplers
import contextlib  # noqa: E402
torch.autocast = lambda *a, **k: contextlib.nullcontext()
with torch.no_grad():
    torch.manual_seed(0)
    a = tree_kv.TreeKVSampler(d, 4).generate(clean, prompt, cam, acts, steps, generator=torch.Generator(device=dev).manual_seed(0))
    torch.manual_seed(0)
    b = tree_samplers.student_tree_sample(d, d.module, clean, prompt, cam, acts, solver="fewstep", fewstep=steps, seed=0,
                                          x0_of=d.flow_to_x0)
for s in (9, 12, 15, 18):
    print("block", s // 3, ((a[:, s:s + 3] - b[:, s:s + 3]).norm() / b[:, s:s + 3].norm()).item())
