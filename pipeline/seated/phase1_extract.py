"""PHASE 1 (standalone subprocess -> auto-frees VRAM on exit):
HMR2 on the image -> SMPL body_pose -> smplx22 axis-angle -> GROUND (feet at Y=0,
pelvis at seat height) -> write an all-frames pose-lock constraint.

argv: <image> <out_constraint_json> [<T frames>] [<lock_mode>]

lock_mode:
  sitting — end-effector pin Hips+LeftFoot+RightFoot (upper body free)
  lying   — fullbody lock of all joint rotations (skeleton locked to extract)
"""
import sys
import os
import json
import subprocess
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
IMG = sys.argv[1]
OUT = sys.argv[2]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 90
LOCK_MODE = (sys.argv[4] if len(sys.argv) > 4 else "sitting").strip().lower()
if LOCK_MODE not in ("sitting", "lying"):
    sys.exit(f"unknown lock_mode {LOCK_MODE!r}; use sitting|lying")

MD = r"C:/Users/AIBOX/dev/ComfyUI-scail/custom_nodes/ComfyUI-MotionDiff"
KIM = r"C:/Users/AIBOX/dev/ComfyUI-scail/custom_nodes/ComfyUI-Kimodo/kimodo"
sys.path.insert(0, MD)
sys.path.insert(0, KIM)
import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def vram():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()

print("[phase1] VRAM before:", vram(), "lock_mode=", LOCK_MODE, flush=True)

_orig = torch.load


def _load(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)


torch.load = _load
from motiondiff_modules.hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from motiondiff_modules.hmr2.datasets.vitdet_dataset import ViTDetDataset
from motiondiff_modules.hmr2.configs import CACHE_DIR_4DHUMANS
from ultralytics import YOLO

device = torch.device("cuda")
model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
model = model.to(device).eval()
det = YOLO(str(Path(CACHE_DIR_4DHUMANS) / "person_yolov8m-seg.pt"))
img_bgr = cv2.imread(IMG)
if img_bgr is None:
    sys.exit(f"cannot read image: {IMG}")
rgb = img_bgr[:, :, ::-1]
boxes = det.predict([rgb], classes=[0], conf=0.25, iou=0.7, verbose=False)[0].boxes.xyxy.cpu().numpy()
if len(boxes) == 0:
    sys.exit("no person detected in image")
ds = ViTDetDataset(cfg, img_bgr, boxes)
batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
with torch.no_grad():
    out = model(batch)
bp = out["pred_smpl_params"]["body_pose"][0].detach().cpu().numpy()  # [23,3,3]
print("[phase1] body_pose", bp.shape, "VRAM after HMR:", vram(), flush=True)

# smplx22 axis-angle: joint0=root identity, joints1-21 = SMPL body_pose[0:21]
local = np.zeros((22, 3))
local[1:22] = Rotation.from_matrix(bp[:21]).as_rotvec()

# ground: FK at origin -> raise so lowest foot at Y=0
from kimodo.skeleton.definitions import SMPLXSkeleton22
from kimodo.constraints import axis_angle_to_matrix

skel = SMPLXSkeleton22()
g = skel.fk(axis_angle_to_matrix(torch.tensor(local)[None]), torch.zeros(1, 3))[1][0].numpy()
H = float(-g[[10, 11, 7, 8], 1].min())
print(f"[phase1] ground H={H:.3f} (pelvis seat height)", flush=True)

frames = list(range(T))
rots = [local.tolist()] * T
roots = [[0.0, H, 0.0]] * T

if LOCK_MODE == "lying":
    # Full skeleton lock: all joint rotations pinned every frame.
    constraints = [{
        "type": "fullbody",
        "frame_indices": frames,
        "local_joints_rot": rots,
        "root_positions": roots,
    }]
else:
    # Sitting recipe: pin pelvis + feet; hips/knees/spine/arms free.
    constraints = [{
        "type": "end-effector",
        "joint_names": ["Hips", "LeftFoot", "RightFoot"],
        "frame_indices": frames,
        "local_joints_rot": rots,
        "root_positions": roots,
    }]

Path(OUT).write_text(json.dumps(constraints), encoding="utf-8")
print("[phase1] wrote", OUT, flush=True)
