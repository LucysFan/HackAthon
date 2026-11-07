import sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,'/home/user/HackAthon')
from scipy.ndimage import gaussian_filter
from hdoe import config as C, physics as P, metrics as M, gs as G
torch.set_num_threads(2)
n=np.arange(C.N)-C.N//2; Y,X=np.meshgrid(n,n,indexing='ij'); R=np.hypot(X,Y); PHI=np.arctan2(Y,X)
def norm4(q,e=0.16):
    q=np.stack(q); q=q/np.maximum(q.max(),1e-9)
    return q/q.reshape(len(q),-1).sum(-1)[:,None,None]*q[0].size*e
def ev(q,iters=200,restarts=1):
    q=norm4(q); r=G.gs_solve(q,iters=iters,restarts=restarts,verbose=False)
    pcc=[float(M.pcc(torch.as_tensor(q[i]),P.normalize01(P.propagate(r["phi"],z)[0,0]))) for i,z in enumerate(C.ZS)]
    ssim=[float(M.ssim(q[i],P.normalize01(P.propagate(r["phi"],z)[0,0]))) for i,z in enumerate(C.ZS)]
    return dict(pcc=[round(x,2) for x in pcc], pcc_mean=round(float(np.mean(pcc)),3), ssim_mean=round(float(np.mean(ssim)),3))
def small(sig):
    o=[]
    o.append(((R>12)&(R<20)&~((PHI>1.0)&(PHI<2.2))).astype(float))
    o.append((((np.abs(X+2)<6)&(np.abs(Y)<20))|((np.abs(Y+4)<6)&(np.abs(X)<15))).astype(float))
    t=np.zeros_like(X)
    for k in range(6):
        a=k*1.05; r=4+k*3.2
        t[((X-r*np.cos(a))**2+(Y-r*np.sin(a))**2)<4.5**2]=1.0
    o.append(t)
    t=np.zeros_like(X); t[((np.abs(X-Y*0.5)<5)&(np.abs(Y+2)<16))]=1
    t[((np.abs(X+Y*0.5+1)<5)&(np.abs(Y+2)<16))]=1; o.append(np.clip(t,0,1))
    return [gaussian_filter(x,sig) if sig>0 else x for x in o]
for sig in (0.0,1.5,3.0):
    print("small sig",sig, ev(small(sig)), flush=True)
