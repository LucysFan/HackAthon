import sys; sys.path.insert(0,'/home/user/HackAthon')
import torch, numpy as np, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, losses as L, corpus as CO
from hdoe.targets import load_targets
from hdoe.model import build_net
T=torch.as_tensor(load_targets(),dtype=torch.float32)
q=CO.equalize_energy_t(T)[None]
tgt=torch.load('/home/user/HackAthon/outputs/teacher_masks.pt')
print("teacher shape", tgt.shape, "range", float(tgt.min()), float(tgt.max()))
# how well does target phi reproduce its own intensities?
phi=torch.atan2(tgt[0:1,1], tgt[0:1,0])
I=torch.cat([P.propagate(phi,z) for z in C.ZS],1)
pred=L.normalize_batch(I)
print("teacher recon ss", [round(float(M.ssim_torch(pred[:,i:i+1],q[:,i:i+1])),3) for i in range(4)],
      "pcc", [round(float(M.pcc_torch(pred[:,i:i+1],q[:,i:i+1])),3) for i in range(4)])
# baseline: what do we get for a *random* phi? (upper bound on "generic" behaviour)
g=torch.Generator().manual_seed(0)
rnd=torch.rand(1,1,C.N,C.N,generator=g)*2*np.pi-np.pi
Ir=torch.cat([P.propagate(rnd,z) for z in C.ZS],1); predr=L.normalize_batch(Ir)
print("random recon ss", [round(float(M.ssim_torch(predr[:,i:i+1],q[:,i:i+1])),3) for i in range(4)],
      "pcc", [round(float(M.pcc_torch(predr[:,i:i+1],q[:,i:i+1])),3) for i in range(4)])
# net at step ~ (best.pt.distill)
net=build_net({}); net.load_state_dict(torch.load('/home/user/HackAthon/outputs/best.pt.distill')['model'])
with torch.no_grad(): ph=net(q, noise=False)
print("net phi std", float(ph.std()), "teacher phi std", float(phi.std()))
In=torch.cat([P.propagate(ph,z) for z in C.ZS],1); predn=L.normalize_batch(In)
print("net recon ss", [round(float(M.ssim_torch(predn[:,i:i+1],q[:,i:i+1])),3) for i in range(4)],
      "pcc", [round(float(M.pcc_torch(predn[:,i:i+1],q[:,i:i+1])),3) for i in range(4)])
# cos/sin mse
pr=torch.cat([torch.cos(ph),torch.sin(ph)],1)
print("cos/sin mse", float(((pr-tgt[0:1])**2).mean()))
# correlation of phase maps
print("corr phase", float(torch.corrcoef(torch.stack([ph.flatten(), phi.flatten()]))[0,1]))
