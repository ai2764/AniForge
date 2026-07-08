"""PHASE 1 (standalone subprocess -> auto-frees VRAM on exit):
HMR2 on the image -> SMPL body_pose -> smplx22 axis-angle -> GROUND (feet at Y=0,
pelvis at seat height) -> write an all-frames butt+feet pin constraint (end-effector
Hips+LeftFoot+RightFoot). argv: <image> <out_constraint_json> [<T frames>]"""
import sys, os, json, subprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
IMG = sys.argv[1]
OUT = sys.argv[2]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 90
MD = r"C:/Users/AIBOX/dev/ComfyUI-scail/custom_nodes/ComfyUI-MotionDiff"
KIM = r"C:/Users/AIBOX/dev/ComfyUI-scail/custom_nodes/ComfyUI-Kimodo/kimodo"
sys.path.insert(0, MD); sys.path.insert(0, KIM)
import torch, numpy as np, cv2
from scipy.spatial.transform import Rotation

def vram(): return subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
print("[phase1] VRAM before:", vram(), flush=True)

_orig = torch.load
def _load(*a, **k): k["weights_only"] = False; return _orig(*a, **k)
torch.load = _load
from motiondiff_modules.hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from motiondiff_modules.hmr2.datasets.vitdet_dataset import ViTDetDataset
from motiondiff_modules.hmr2.configs import CACHE_DIR_4DHUMANS
from ultralytics import YOLO
from pathlib import Path

device = torch.device("cuda")
model, cfg = load_hmr2(DEFAULT_CHECKPOINT); model = model.to(device).eval()
det = YOLO(str(Path(CACHE_DIR_4DHUMANS) / "person_yolov8m-seg.pt"))
img_bgr = cv2.imread(IMG); rgb = img_bgr[:, :, ::-1]
boxes = det.predict([rgb], classes=[0], conf=0.25, iou=0.7, verbose=False)[0].boxes.xyxy.cpu().numpy()
ds = ViTDetDataset(cfg, img_bgr, boxes)
batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))
batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
with torch.no_grad(): out = model(batch)
bp = out["pred_smpl_params"]["body_pose"][0].detach().cpu().numpy()  # [23,3,3]
print("[phase1] body_pose", bp.shape, "VRAM after HMR:", vram(), flush=True)

# smplx22 axis-angle: joint0=root identity, joints1-21 = SMPL body_pose[0:21]
local = np.zeros((22,3)); local[1:22] = Rotation.from_matrix(bp[:21]).as_rotvec()

# ground: FK at origin -> raise so lowest foot at Y=0
from kimodo.skeleton.definitions import SMPLXSkeleton22
from kimodo.constraints import axis_angle_to_matrix
skel = SMPLXSkeleton22()
g = skel.fk(axis_angle_to_matrix(torch.tensor(local)[None]), torch.zeros(1,3))[1][0].numpy()
H = float(-g[[10,11,7,8],1].min())
print(f"[phase1] ground H={H:.3f} (pelvis seat height)", flush=True)

constraints = [{
    "type": "end-effector",
    "joint_names": ["Hips","LeftFoot","RightFoot"],
    "frame_indices": list(range(T)),
    "local_joints_rot": [local.tolist()]*T,
    "root_positions": [[0.0,H,0.0]]*T,
}]
Path(OUT).write_text(json.dumps(constraints)); print("[phase1] wrote", OUT, flush=True)
