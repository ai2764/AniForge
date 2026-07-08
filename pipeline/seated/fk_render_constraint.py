"""FK-render the extracted sitting constraint through Kimodo's own smplx22 path.

Uses load_constraints_lst -> axis_angle_to_matrix -> SMPLXSkeleton22.fk (the exact
code path Kimodo uses to read the constraint), so no reimplementation drift.
Renders rest pose (identity) + constraint pose, front (XY) + side (ZY), and prints
quantitative sit-metrics.
"""
import sys, os, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, cv2

KIM = r"C:/Users/AIBOX/dev/ComfyUI-scail/custom_nodes/ComfyUI-Kimodo/kimodo"
sys.path.insert(0, KIM)
from kimodo.skeleton.definitions import SMPLXSkeleton22
from kimodo.constraints import load_constraints_lst

CJSON = r"C:/Users/AIBOX/.claude/jobs/4b35e22a/tmp/sitting_constraints.json"
OUT   = r"C:/Users/AIBOX/AppData/Local/Temp/claude/C--Users-AIBOX-dev-motion-portrait/a89c1fb2-249e-404d-b1e4-bb41d0c47b8f/scratchpad/constraint_fk.png"

NAMES = ["pelvis","L_hip","R_hip","spine1","L_knee","R_knee","spine2","L_ankle",
         "R_ankle","spine3","L_foot","R_foot","neck","L_collar","R_collar","head",
         "L_shoulder","R_shoulder","L_elbow","R_elbow","L_wrist","R_wrist"]

skel = SMPLXSkeleton22()
parents = skel.joint_parents.cpu().numpy().tolist()
print("parents:", parents)

# ---- rest pose (identity rotations) ----
ident = torch.eye(3).repeat(1, 22, 1, 1)          # [1,22,3,3]
root0 = torch.zeros(1, 3)
_, rest_pos, _ = skel.fk(ident, root0)
rest = rest_pos[0].cpu().numpy()                   # [22,3]

# ---- constraint pose via Kimodo's own loader ----
cons = load_constraints_lst(CJSON, skel)
c = cons[0]
pose = c.global_joints_positions[0].cpu().numpy()  # [22,3]

# ---- determine up axis from rest pose (head vs feet) ----
head_i, foot_i = NAMES.index("head"), NAMES.index("R_foot")
span = rest[head_i] - rest[foot_i]
up_axis = int(np.argmax(np.abs(span)))
print("rest head-foot vector:", np.round(span,3), "=> up axis =", "XYZ"[up_axis],
      "(sign", np.sign(span[up_axis]), ")")

def metrics(P, tag):
    print(f"\n--- {tag} sit-metrics ---")
    up = up_axis
    for side, hip, knee, ankle in [("L",1,4,7),("R",2,5,8)]:
        thigh = P[knee]-P[hip]; shin = P[ankle]-P[knee]
        tl=np.linalg.norm(thigh)+1e-9; sl=np.linalg.norm(shin)+1e-9
        # downward fraction of thigh along up-axis (neg = points down)
        thigh_down = -np.sign(span[up])*thigh[up]/tl
        knee_flex = np.degrees(np.arccos(np.clip(np.dot(thigh,shin)/(tl*sl),-1,1)))
        # torso up-fraction
        print(f"  {side}: thigh_downfrac={thigh_down:+.2f} (1=straight down/standing, ~0=horizontal/sitting)"
              f"  knee_bend={knee_flex:5.1f} deg (0=straight leg)")
    torso = P[NAMES.index("spine3")]-P[0]
    print(f"  pelvis->spine3 up-frac={np.sign(span[up])*torso[up]/(np.linalg.norm(torso)+1e-9):+.2f} (torso uprightness)")

metrics(rest, "REST(identity)")
metrics(pose, "CONSTRAINT")

# ---- render front (XY) + side (ZY), rest row + constraint row ----
def draw(P, ax_h, ax_v, size=360, flip_v=True, title=""):
    img = np.full((size, size, 3), 30, np.uint8)
    xy = P[:, [ax_h, ax_v]].astype(np.float64)
    mn, mx = xy.min(0), xy.max(0); span2=(mx-mn).max()+1e-6
    m=size*0.12; eff=size-2*m
    px=((xy[:,0]-mn[0])/span2*eff+m).astype(int)
    py=(xy[:,1]-mn[1])/span2*eff+m
    py=(size-py).astype(int) if flip_v else py.astype(int)
    for j,p in enumerate(parents):
        if p>=0: cv2.line(img,(px[j],py[j]),(px[p],py[p]),(90,170,255),2)
    for j in range(22):
        col=(0,220,120)
        if j in (4,5): col=(60,60,255)      # knees red
        if j in (1,2): col=(255,120,0)      # hips blue
        cv2.circle(img,(px[j],py[j]),4,col,-1)
    cv2.putText(img,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1)
    return img

H,V = None,None
hh=[i for i in range(3) if i!=up_axis]  # the two ground axes
front_h = hh[0]  # left-right
side_h  = hh[1]  # front-back (depth)
r1=np.hstack([draw(rest,front_h,up_axis,title="REST front"),
              draw(rest,side_h,up_axis,title="REST side")])
r2=np.hstack([draw(pose,front_h,up_axis,title="CONSTRAINT front"),
              draw(pose,side_h,up_axis,title="CONSTRAINT side")])
cv2.imwrite(OUT, np.vstack([r1,r2]))
print("\nwrote", OUT)
