import sys; sys.path.insert(0,'/home/user/HackAthon')
import torch, numpy as np, warnings, time
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, corpus as CO, gs as G
from hdoe.targets import load_targets
from hdoe.model import build_net
from scipy.ndimage import gaussian_filter
T=load_targets(); q=CO.equalize_energy_t(torch.as_tensor(T,dtype=torch.float32))
r=G.gs_solve(q.numpy(), iters=150, restarts=1, verbose=False)
phi=r["phi"][0,0].numpy()
targets={"gs_sharp":phi, "gs_smooth":gaussian_filter(phi,3), "random":np.random.uniform(-np.pi,np.pi,(128,128))}
for name,ph in targets.items():
    net=build_net(dict(width=32,depth=4))
    opt=torch.optim.Adam(net.parameters(),3e-3)
    X=q[None]
    tgt=torch.cat([torch.cos(torch.as_tensor(ph))[None,None],torch.sin(torch.as_tensor(ph))[None,None]],1)
    t0=time.time(); cosv=0
    for s in range(300):
        pr=net(X, noise=False)
        p=torch.cat([torch.cos(pr),torch.sin(pr)],1)
        cosd=(p*tgt).sum()/ (p.norm()*tgt.norm()+1e-8)
        loss=1-cosd
        opt.zero_grad(); loss.backward(); opt.step()
        cosv=float(cosd)
        if (s+1)%100==0: print(f"  {name} step {s+1} cos {cosv:.3f} ({time.time()-t0:.0f}s)", flush=True)
