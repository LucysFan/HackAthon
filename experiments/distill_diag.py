import sys; sys.path.insert(0,'/home/user/HackAthon')
import torch, numpy as np, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, gs as G, corpus as CO
from hdoe.targets import load_targets
from hdoe.model import build_net
T=load_targets()
q=CO.equalize_energy_t(torch.as_tensor(T,dtype=torch.float32))
r=G.gs_solve(q.numpy(), iters=150, restarts=1, verbose=False)
phi=r["phi"]
print("GS phi stats", float(phi.min()), float(phi.max()), float(phi.std()))
# check recons of teacher phi
I=torch.cat([P.propagate(phi,z) for z in C.ZS],1)
pred=P.normalize01(I)
print("teacher recons PCC/SSIM", [(round(float(M.pcc_torch(pred[:,i:i+1],q[i:i+1])),3), round(float(M.ssim_torch(pred[:,i:i+1],q[i:i+1])),3)) for i in range(4)])
# quantized teacher
v=np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(phi))))
phiq=torch.as_tensor(P.decode8bit(v))[None,None]
Iq=torch.cat([P.propagate(phiq,z) for z in C.ZS],1); predq=P.normalize01(Iq)
print("teacher QUANT recons", [(round(float(M.pcc_torch(predq[:,i:i+1],q[i:i+1])),3)) for i in range(4)])
# 3) can we predict the phase at all? simple conv net overfit test on 1 sample, big lr
from hdoe import losses as L
net=build_net({})
opt=torch.optim.Adam(net.parameters(),3e-3)
X=q[None].repeat(4,1,1,1)
tgt=torch.stack([torch.cos(phi[0,0]), torch.sin(phi[0,0])])[None].repeat(4,1,1,1)
for s in range(200):
    ph=net(X, noise=False)
    pr=torch.cat([torch.cos(ph),torch.sin(ph)],1)
    loss=((pr-tgt)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if s%40==0: print("overfit step",s,"loss",round(float(loss),4), "phi std", round(float(ph.std()),3))
