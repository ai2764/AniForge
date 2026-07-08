"""Automated seated pipeline with 3-phase VRAM strategy (NO ComfyUI restart):
  Phase 1  HMR2 extract (subprocess) -> grounded butt-pin constraint; exit frees VRAM
  Phase 2  Kimodo load-once, gen idle+action (subprocess);            exit frees VRAM
  Phase 3  SCAIL x2 via ComfyUI (only SCAIL ever lives in ComfyUI)
Each heavy model runs in its own process -> auto-unload on exit. ComfyUI holds only
SCAIL (comfy-managed, /free-able between requests)."""
import sys, os, time, json, shutil, subprocess, urllib.request
sys.path.insert(0, r"C:/Users/AIBOX/dev/motion-portrait")
import numpy as np, cv2
from pathlib import Path
from pipeline.comfy import ComfyClient
from pipeline.scail import drive_character
from pipeline.generate import _pad_to_aspect, _output_size, align_4k1

PYEXE = os.environ.get("MP_PYEXE", sys.executable)
HERE = Path(__file__).resolve().parent            # phase scripts live next to this file
REF_SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:/Users/AIBOX/dev/ComfyUI-scail/input/mp_sitting.png")
RUN = Path(os.environ.get("MP_RUN_DIR", str(HERE/"runs"))); RUN.mkdir(parents=True, exist_ok=True)
CJSON = str(RUN/"constraint.json")
IDLE_PREFIX, ACTION_PREFIX = "kimodo_pl_idle", "kimodo_pl_action"
OUTDIR = r"C:/Users/AIBOX/dev/ComfyUI-scail/output"

def vram():
    return subprocess.run(["nvidia-smi","--query-gpu=memory.used,power.draw","--format=csv,noheader,nounits"],
                          capture_output=True,text=True).stdout.strip()
def free_comfy():
    try:
        req=urllib.request.Request("http://127.0.0.1:8188/free",
            data=json.dumps({"unload_models":True,"free_memory":True}).encode(),
            headers={"Content-Type":"application/json"})
        urllib.request.urlopen(req,timeout=30).read()
    except Exception as e: print("  /free err:",e)
def run_phase(name, args):
    print(f"\n=== {name} (subprocess) === VRAM before: {vram()}", flush=True)
    t=time.time()
    r=subprocess.run([PYEXE]+args, env={**os.environ,"PYTHONIOENCODING":"utf-8"})
    time.sleep(3)  # let CUDA context tear down
    print(f"=== {name} done in {time.time()-t:.0f}s, exit {r.returncode}, VRAM AFTER EXIT: {vram()}", flush=True)
    if r.returncode: sys.exit(f"{name} failed")

# smplx22 guide renderer (front XY, mirrors skeleton_spring.render)
PAR=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
SP=(100,200,100);LA=(50,150,255);RA=(255,100,100);LL=(100,100,255);RL=(255,50,200)
GRP={3:SP,6:SP,9:SP,12:SP,15:SP,13:LA,16:LA,18:LA,20:LA,14:RA,17:RA,19:RA,21:RA,1:LL,4:LL,7:LL,10:LL,2:RL,5:RL,8:RL,11:RL}
BON=[(i,PAR[i]) for i in range(1,22)]; COL=[GRP[i] for i in range(1,22)]
def render_guide(P, path, size=512):
    x,y=P[:,:,0],P[:,:,1]; cx=.5*(x.min()+x.max()); cy=.5*(y.min()+y.max()); sc=max(x.max()-x.min(),y.max()-y.min(),.1)*1.3
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),30,(size,size))
    for f in range(P.shape[0]):
        img=np.ones((size,size,3),np.uint8)*240
        px=lambda p:(int((p[0]-cx)/sc*size+size/2),int(size/2-(p[1]-cy)/sc*size))
        for (a,b),c in zip(BON,COL): cv2.line(img,px(P[f,a]),px(P[f,b]),c,5,cv2.LINE_AA)
        for j in range(22): cv2.circle(img,px(P[f,j]),4,(50,50,50),-1,cv2.LINE_AA)
        w.write(img)
    w.release()
def newest(prefix):
    import glob; return sorted(glob.glob(os.path.join(OUTDIR,f"{prefix}_*.npz")),key=os.path.getmtime)[-1]

T0=time.time()
print("PIPELINE START. VRAM:", vram(), flush=True)
print("free ComfyUI (unload any prior SCAIL)..."); free_comfy(); time.sleep(4); print("  VRAM:", vram())

# PHASE 1 + 2 (standalone subprocesses)
run_phase("PHASE1 HMR extract", [str(HERE/"phase1_extract.py"), str(REF_SRC), CJSON, "90"])
run_phase("PHASE2 Kimodo idle+action", [str(HERE/"gen_kimodo_standalone.py"), CJSON, IDLE_PREFIX, ACTION_PREFIX])

# PHASE 3: SCAIL x2 via ComfyUI
REF = RUN/"ref.png"; shutil.copyfile(REF_SRC, REF)
out_w,out_h=_output_size(REF); print(f"\n=== PHASE3 SCAIL === output {out_w}x{out_h}, VRAM before: {vram()}", flush=True)
c=ComfyClient()
for prefix,outname,pos in [(IDLE_PREFIX,"idle","a seated character in a calm idle pose, full body, consistent identity"),
                           (ACTION_PREFIX,"action","a seated character gesturing expressively, full body, consistent identity")]:
    P=np.load(newest(prefix))["posed_joints"]
    skel=RUN/f"{outname}_skel.mp4"; guide=RUN/f"{outname}_guide.mp4"
    render_guide(P,skel); _pad_to_aspect(skel,guide,out_w,out_h)
    t=time.time()
    out=drive_character(c,guide,REF,RUN/f"{outname}.mp4",length=align_4k1(P.shape[0]),
                        width=out_w,height=out_h,prefix="mp_pl_"+outname,seed=42,positive=pos)
    print(f"  SCAIL {outname}: {time.time()-t:.0f}s -> {out}  VRAM: {vram()}", flush=True)

print(f"\nPIPELINE DONE in {time.time()-T0:.0f}s. outputs in {RUN}", flush=True)
