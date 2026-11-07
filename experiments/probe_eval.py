import sys; sys.path.insert(0,'/home/user/HackAthon')
import torch, warnings, numpy as np
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, gs as G
from hdoe.targets import load_targets
T=load_targets()
q=torch.as_tensor(T/np.maximum(T.max(),1e-9),dtype=torch.float32)
q=q/q.flatten(1).sum(-1)[:,None,None]*q[0].numel()*0.16
r=G.gs_solve(q.numpy(), iters=100, restarts=1, verbose=False)
I=torch.cat([P.propagate(r["phi"],z) for z in C.ZS],1)
pred=P.normalize01(I)
print("GS mean ssim", round(float(np.mean([M.ssim_torch(pred[:,i:i+1], q[i:i+1]) for i in range(4)])),4),
      "mean pcc", round(float(np.mean([M.pcc_torch(pred[:,i:i+1], q[i:i+1]) for i in range(4)])),4))
print("raw intensity range", [(round(float(I[0,i].min()),3), round(float(I[0,i].max()),3)) for i in range(4)])
print("target max", [round(float(q[i].max()),3) for i in range(4)])
