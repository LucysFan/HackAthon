"""Диагностика: сглаживание маски GS и доводка фазовым лоссом."""
import sys
sys.path.insert(0, '/home/user/HackAthon')
import warnings
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
from hdoe import config as C, physics as P, metrics as M, losses as L, corpus as CO, gs as G, refine as R
from hdoe.targets import load_targets

T = load_targets()
q = CO.equalize_energy_t(torch.as_tensor(T, dtype=torch.float32))


def score(phi):
    v = np.squeeze(np.asarray(P.encode8bit(torch.as_tensor(np.asarray(phi, dtype=np.float32)))))
    phi_q = torch.as_tensor(P.decode8bit(v))[None, None]
    I = torch.cat([P.propagate(phi_q, z) for z in C.ZS], 1)
    pred = L.normalize_batch(I)
    pc = [round(float(M.pcc_torch(pred[:, i:i + 1], q[i:i + 1])), 3) for i in range(4)]
    ss = [round(float(M.ssim_torch(pred[:, i:i + 1], q[i:i + 1])), 3) for i in range(4)]
    return pc, ss, round(float(np.mean(pc)), 3), round(float(np.mean(ss)), 3)


r = G.gs_solve(q.numpy(), iters=150, restarts=2, verbose=False)
phi = r["phi"][0, 0].numpy()
print("GS raw", score(phi), flush=True)
for sig in (1, 2, 3, 4):
    print("GS smooth", sig, score(gaussian_filter(phi, sig)), flush=True)
import time
t0 = time.time()
out = R.refine_phase(T, phi, steps=250, lr=2e-2, verbose=False)
print("GS + refine(250)", score(out["phi"].numpy()), f"{time.time() - t0:.0f}s", flush=True)
np.save('/home/user/HackAthon/outputs/_diag_gs_phi.npy', phi)
