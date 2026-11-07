import sys; sys.path.insert(0,'/home/user/HackAthon')
import torch, numpy as np, warnings, time
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, losses as L, corpus as CO, gs as G, refine as R
from hdoe.targets import load_targets
T=load_targets()
q=CO.equalize_energy_t(torch.as_tensor(T,dtype=torch.float32))
# start 1: GS
r=G.gs_solve(q.numpy(), iters=100, restarts=1, verbose=False)
# start 2: random small
g=torch.Generator().manual_seed(0)
for name, init in (("gs", r["phi"][0,0].numpy()), ("zero", np.zeros((C.N,C.N))), ("rand", (torch.rand(1,1,C.N,C.N,generator=g)*2*np.pi-np.pi)[0,0].numpy())):
    t0=time.time()
    out=R.refine_phase(T, init, steps=400, lr=5e-2, verbose=False)
    phi_q=torch.as_tensor(P.decode8bit(np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(out["phi"]))))))[None,None]
    I=torch.cat([P.propagate(phi_q,z) for z in C.ZS],1); pred=L.normalize_batch(I)
    ss=[round(float(M.ssim_torch(pred[:,i:i+1],q[i:i+1])),3) for i in range(4)]
    pc=[round(float(M.pcc_torch(pred[:,i:i+1],q[i:i+1])),3) for i in range(4)]
    print(name, "SSIM", ss, "mean", round(np.mean(ss),3), "PCC", pc, "mean", round(np.mean(pc),3), f"{time.time()-t0:.0f}s", flush=True)
