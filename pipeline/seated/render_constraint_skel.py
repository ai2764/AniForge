"""Render a short skeleton mp4 from a Kimodo constraint JSON (FK once, hold).

argv: <constraint.json> <out.mp4> [n_frames=45]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

KIM = r"C:/Users/AIBOX/dev/ComfyUI-scail/custom_nodes/ComfyUI-Kimodo/kimodo"
sys.path.insert(0, KIM)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kimodo.constraints import axis_angle_to_matrix
from kimodo.skeleton.definitions import SMPLXSkeleton22
from pipeline.seated.generate_anchored import render_smplx_guide, skeleton_camera_from_joints

cjson = Path(sys.argv[1])
out = Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 45
c0 = json.loads(cjson.read_text(encoding="utf-8"))[0]
local = np.array(c0["local_joints_rot"][0], dtype=np.float64)
root = np.array(c0["root_positions"][0], dtype=np.float64)
mats = axis_angle_to_matrix(torch.tensor(local)[None].float())
P = SMPLXSkeleton22().fk(mats, torch.tensor(root)[None].float())[1][0].numpy()
# Single-frame extract pose for idle/action anchoring (sitting/lying/standing).
pose_path = out.parent / "extract_pose.npy"
np.save(pose_path, P.astype(np.float64))
cam = skeleton_camera_from_joints(P)
clip = np.stack([P] * n, axis=0)
render_smplx_guide(clip, out, camera=cam)
print("wrote", out, "and", pose_path, flush=True)
