import sys; sys.path.insert(0,'/home/user/HackAthon')
import torch, numpy as np, warnings, time
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, losses as L, corpus as CO, gs as G
from hdoe.targets import load_targets
from hdoe.model import build_net
T=load_targets(); q=CO.equalize_energy_t(torch.as_tensor(T,dtype=torch.float32))
def score(phi):
    phi_q=torch.as_tensor(P.decode8bit(np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(phi))))))[None,None]
    I=torch.cat([P.propagate(phi_q,z) for z in C.ZS],1); pred=L.normalize_batch(I)
    pc=np.mean([float(M.pcc_torch(pred[:,i:i+1],q[i:i+1])) for i in range(4)])
    ss=np.mean([float(M.ssim_torch(pred[:,i:i+1],q[i:i+1])) for i in range(4)])
    return round(pc,3), round(ss,3)
# net init
for ck in ("outputs/distilled.pt","outputs/best.pt"):
    import os
    if not os.path.exists(ck): continue
    net=build_net({}); net.load_state_dict(torch.load(ck,map_location='cpu')['model']); net.eval()
    with torch.no_grad(): ph=net(q[None], noise=False)
    print(ck, "net-only", score(ph[0,0]))
    for it in (20,40,80):
        t0=time.time(); r=G.gs_run(T, iters=it, seed=0, init=ph); 
        print(f"  warm GS {it}: {score(r['phi'][0,0])} {time.time()-t0:.1f}s", flush=True)
for it in (20,40,80,150):
    t0=time.time(); r=G.gs_run(T, iters=it, seed=0)
    print(f"random GS {it}: {score(r['phi'][0,0])} {time.time()-t0:.1f}s", flush=True)
