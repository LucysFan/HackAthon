"""Сеть как инициализация + физическая доводка (п. 5.6.2): сколько шагов нужно."""
import sys, time, warnings
sys.path.insert(0,'/home/user/HackAthon')
import numpy as np, torch
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, losses as L, corpus as CO, gs as G, refine as R
from hdoe.targets import load_targets
from hdoe.model import build_net
T=load_targets(); Tt=torch.as_tensor(T,dtype=torch.float32)
def rep(phi):
    v=np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(np.asarray(phi,dtype=np.float32)))))
    ph=torch.as_tensor(P.decode8bit(v))[None,None]
    pcc=[];ssim=[]
    for i,z in enumerate(C.ZS):
        I01=P.normalize01(P.propagate(ph,z)[0,0].numpy())
        pcc.append(M.pcc(T[i],I01)); ssim.append(M.ssim(T[i],I01))
    return np.mean(pcc), np.mean(ssim), [round(x,3) for x in pcc], [round(x,3) for x in ssim]
net=build_net({}); net.load_state_dict(torch.load('outputs/best.pt',map_location='cpu')['model']); net.eval()
with torch.no_grad(): ph=net(Tt[None], noise=False)
print("net only", rep(ph[0,0]), flush=True)
for steps in (25, 50, 100, 200):
    t0=time.time()
    out=R.refine_phase(T, ph[0,0], steps=steps, lr=3e-2, w_tv=0.0, quant_noise=False)
    sc=rep(out["phi"].numpy())
    print(f"net + polish {steps}: PCC {sc[0]:.4f} SSIM {sc[1]:.4f} {time.time()-t0:.1f}s", sc[2], sc[3], flush=True)
# blend of net & gs as init
r=G.gs_solve(T, iters=200, restarts=2, verbose=False)
bl=R.blend_phases(ph[0,0].numpy(), r["phi"][0,0].numpy(), 0.5)
t0=time.time(); out=R.refine_phase(T, bl, steps=150, lr=3e-2, w_tv=0.0, quant_noise=False)
sc=rep(out["phi"].numpy()); print(f"blend(net,gs) + polish 150: PCC {sc[0]:.4f} SSIM {sc[1]:.4f} {time.time()-t0:.1f}s", flush=True)
