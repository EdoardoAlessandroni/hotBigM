"""Does the penalty weight M control whether the circuit sees the objective?

M* is set by a Gibbs model the trained circuit demonstrably outperforms (it predicts
feasibility should fail at T_norm ~ 5, yet P(feas) = 0.998). If M* is far larger than
needed, the objective's relative weight in the trained QUBO is correspondingly suppressed.

Sweeps M by hand over many orders of magnitude, holding everything else fixed, and measures
  P(feasible)                     -- does the constraint still hold at small M?
  mean quantile of feasible mass  -- does the circuit prefer GOOD feasible states?
A window where P(feas) stays high AND mean quantile drops well below 0.5 would show the
M-method is simply over-conservative for QAOA. No such window = the effect is not about M.
"""
import itertools
import numpy as np
from qaoa_sampling_oneshot import (run_ma_qaoa_with_solution, get_QUBO_PO, get_PO_obj,
                                   evaluate_energy, state_inttobin, N_stocks, w)

N_idx, LAYERS = 2, 3
N = N_stocks[N_idx]; n_bits = N*w
VSEEDS = (42, 142, 242)
MS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

b = 2**w-1
X = np.array([[int(x) for s in c for x in format(s,f"0{w}b")]
              for c in itertools.product(range(b+1),repeat=N) if sum(c)==b], float)

print(f"n_bits={n_bits} p={LAYERS} seed=442  (sweep's M* ~ 1.8e5)\n")
print(f"{'vseed':>6} {'M':>9} {'obj/pen':>9} {'P(feas)':>9} {'mean quant':>11} {'eff #st':>8}")
print("-"*58)
for vseed in VSEEDS:
    Q_pen,c_pen = get_QUBO_PO(N,w,1,"seed",("PO_small",vseed),penalization=True,objective=False)
    Q_obj,c_obj = get_QUBO_PO(N,w,1,"seed",("PO_small",vseed),penalization=False,objective=True)
    Qo,Lo = get_PO_obj(n_bits,("PO_small",vseed))
    E_feas = np.sort(np.einsum("ij,jk,ik->i",X,Qo+np.diag(Lo),X))
    fm = np.zeros(2**n_bits,bool); Ea = np.zeros(2**n_bits)
    for i in range(2**n_bits):
        x = state_inttobin(i,n_bits)
        fm[i] = np.isclose(evaluate_energy(x,Q_pen,c_pen),0); Ea[i] = evaluate_energy(x,Q_obj,c_obj)
    for M in MS:
        Q, const = get_QUBO_PO(N,w,M,"seed",("PO_small",vseed))
        Qn = Q/np.max(np.abs(Q))
        ratio = np.max(np.abs(Q_obj))/np.max(np.abs(M*Q_pen))
        Qd = {ix:v for ix,v in np.ndenumerate(Qn) if not np.isclose(v,0)}
        _, _, probs, _ = run_ma_qaoa_with_solution(Qd, n_bits, p=LAYERS, steps=500,
                                                   n_samples=1000, verbose=False,
                                                   draw_samples=True, seed=442)
        probs = np.asarray(probs,float); pt = probs[fm].sum()
        pf = probs[fm]/pt
        q = np.searchsorted(E_feas,Ea[fm],side="right")/len(E_feas)
        mq = float((pf*q).sum()); eff = float(np.exp(-(pf[pf>0]*np.log(pf[pf>0])).sum()))
        print(f"{vseed:>6} {M:>9.0e} {ratio:>9.2e} {pt:>9.4f} {mq:>11.3f} {eff:>8.1f}", flush=True)
