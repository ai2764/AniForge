"""Bone/joint tables for the SOMASkeleton77 body subset used by the scail guide.

The Kimodo NPZ `posed_joints` array is `[T, 77, 3]` in SOMASkeleton77 order.
We keep a 23-joint body subset (torso/limb chain) and drop individual finger,
eye, jaw, head-end, and toe-end joints, which clutter the guide and are not
needed by scail.
"""
from __future__ import annotations
import numpy as np

# new-index -> original SOMASkeleton77 index, in kept order.
BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 39, 40, 41, 42, 67, 68, 69, 70, 72, 73, 74, 75]
# 0  Hips           7  LeftShoulder    15 LeftLeg
# 1  Spine1         8  LeftArm         16 LeftShin
# 2  Spine2         9  LeftForeArm     17 LeftFoot
# 3  Chest          10 LeftHand        18 LeftToeBase
# 4  Neck1          11 RightShoulder   19 RightLeg
# 5  Neck2          12 RightArm        20 RightShin
# 6  Head           13 RightForeArm    21 RightFoot
#                   14 RightHand       22 RightToeBase

N_JOINTS = len(BODY_JOINTS)

# (child_new_index, parent_new_index) pairs, indices within BODY_JOINTS.
BONES = [
    (1, 0), (2, 1), (3, 2),                  # spine: Hips->Spine1->Spine2->Chest
    (4, 3), (5, 4), (6, 5),                  # neck+head: Chest->Neck1->Neck2->Head
    (7, 3), (8, 7), (9, 8), (10, 9),         # left arm: Chest->LShoulder->LArm->LForeArm->LHand
    (11, 3), (12, 11), (13, 12), (14, 13),   # right arm
    (15, 0), (16, 15), (17, 16), (18, 17),   # left leg: Hips->LLeg->LShin->LFoot->LToe
    (19, 0), (20, 19), (21, 20), (22, 21),   # right leg
]

# BGR color per bone, grouped by limb for readability.
COLORS = [
    (100, 200, 100), (100, 200, 100), (100, 200, 100),   # spine - green
    (100, 200, 100), (100, 200, 100), (100, 200, 100),   # neck/head - green
    (50, 150, 255), (50, 150, 255), (50, 150, 255), (50, 150, 255),      # left arm - orange
    (255, 100, 100), (255, 100, 100), (255, 100, 100), (255, 100, 100),  # right arm - blue
    (100, 100, 255), (100, 100, 255), (100, 100, 255), (100, 100, 255),  # left leg - red
    (255, 50, 200), (255, 50, 200), (255, 50, 200), (255, 50, 200),      # right leg - purple
]

# Per-joint softness, 0.0 rigid core -> 1.0 softest distal.
SOFT = np.array([
    0.0,               # 0 Hips
    0.0, 0.0, 0.0,     # 1-3 Spine1, Spine2, Chest
    0.0, 0.0,          # 4-5 Neck1, Neck2
    0.7,               # 6 Head (subtle bob)
    0.35, 0.35, 0.7, 1.0,   # 7-10 LShoulder, LArm, LForeArm, LHand
    0.35, 0.35, 0.7, 1.0,   # 11-14 RShoulder, RArm, RForeArm, RHand
    0.0, 0.35, 0.7, 1.0,    # 15-18 LLeg, LShin, LFoot, LToe
    0.0, 0.35, 0.7, 1.0,    # 19-22 RLeg, RShin, RFoot, RToe
], dtype=np.float32)


def load_posed_joints(npz_path):
    """Load `posed_joints` from a Kimodo NPZ and gather the body subset.

    Returns ndarray[T, N_JOINTS, 3].
    """
    data = np.load(npz_path)
    posed = data["posed_joints"]
    return posed[:, BODY_JOINTS, :]
