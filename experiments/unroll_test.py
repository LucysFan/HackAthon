"""Проверка идеи: сеть + K дифференцируемых GS-шагов (unrolled refinement)."""
import sys, math, warnings, time
sys.path.insert(0,'/home/user/HackAthon')
import numpy as np, torch
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, losses as L, corpus as CO, gs as G
from hdoe.targets import load_targets
from hdoe.model import build_net, UnrolledRefine
T=load_targets(); q=CO.equalize_energy_t(torch.as_tensor(T,dtype=torch.float32))
def score(phi):
    v=np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(np.asarray(phi.detach() if torch.is_tensor(phi) else phi,dtype=np.float32)))))
    ph=torch.as_tensor(P.decode8bit(v))[None,None]
    I=torch.cat([P.propagate(ph,z) for z in C.ZS],1); pred=L.normalize_batch(I)
    return (round(float(np.mean([M.pcc_torch(pred[:,i:i+1],q[i:i+1]) for i in range(4)])),3),
            round(float(np.mean([M.ssim_torch(pred[:,i:i+1],q[i:i+1]) for i in range(4)])),3))
import os
for ck in ("outputs/best.pt","outputs/distilled.pt"):
    if not os.path.exists(ck): continue
    net=build_net({}); net.load_state_dict(torch.load(ck,map_location='cpu')['model']); net.eval()
    with torch.no_grad(): ph=net(q[None], noise=False)
    print(ck,"net only",score(ph[0,0]), flush=True)
    for K in (1,2,3,6,12):
        t0=time.time()
        ur=UnrolledRefine(steps=K, gain=1.0)
        with torch.no_grad(): out=ur(ph, q[None])
        print(f"   +unroll K={K}: {score(out[0,0])} ({time.time()-t0:.2f}s)", flush=True)
