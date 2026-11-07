import sys; sys.path.insert(0,'/home/user/HackAthon')
import torch, numpy as np, warnings, time
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, losses as L
from hdoe.model import build_net
pack=torch.load('/home/user/HackAthon/outputs/teacher_masks.pt', weights_only=False)
Q=pack["quads"]; K=len(pack["cos"])
T_=torch.cat([pack["cos"],pack["sin"]],1)
print("teachers", tuple(Q.shape), tuple(T_.shape), "K",K)
net=build_net({})
opt=torch.optim.Adam(net.parameters(),2e-3)
rng=torch.Generator().manual_seed(0)
t0=time.time()
for s in range(1201):
    idx=[int(torch.randint(0,K,(1,),generator=rng)) for _ in range(8)]
    q=Q[idx]; tgt=T_[idx]
    phi=net(q)
    pred=torch.cat([torch.cos(phi),torch.sin(phi)],1)
    cm=pred[:,:1].mean(dim=(1,2,3)); sm=pred[:,1:].mean(dim=(1,2,3))
    pa=torch.cat([pred[:,:1]+(tgt[:,:1].mean(dim=(1,2,3))-cm)[:,None,None,None],
                  pred[:,1:]+(tgt[:,1:].mean(dim=(1,2,3))-sm)[:,None,None,None]],1)
    n=pa/(pa.pow(2).sum(1,keepdim=True).sqrt()+1e-8); t=tgt/(tgt.pow(2).sum(1,keepdim=True).sqrt()+1e-8)
    cosd=(n*t).sum(1).clamp(-1,1)
    loss=(1-cosd).mean()+0.5*((pred-tgt)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if s%100==0:
        print(s,"loss",round(float(loss),4),"cos",round(float(cosd.mean()),4),f"{time.time()-t0:.0f}s",flush=True)
from hdoe.targets import load_targets
from hdoe import corpus as CO
q=CO.equalize_energy_t(torch.as_tensor(load_targets(),dtype=torch.float32))[None]
with torch.no_grad(): ph=net(q, noise=False)
I=torch.cat([P.propagate(ph,z) for z in C.ZS],1); pred=L.normalize_batch(I)
print("eval ss", [round(float(M.ssim_torch(pred[:,i:i+1],q[:,i:i+1])),3) for i in range(4)])
print("eval pcc",[round(float(M.pcc_torch(pred[:,i:i+1],q[:,i:i+1])),3) for i in range(4)])
