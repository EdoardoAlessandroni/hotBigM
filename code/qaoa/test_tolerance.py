"""Test: is the p=3 'blindness' an artefact of an ill-scaled stopping tolerance?

tolerance_opt = 1e-4 is 23x-2900x larger than the objective's ENTIRE range in normalized
cost units (dE_obj/max|Q|), so the convergence test fires when the penalty flattens and
objective-driven descent never counts as progress. Prediction: with a tolerance well below
the objective scale, p=3 should stop landing on arbitrary feasible states and start
preferring low-objective ones (mean quantile << 0.5).
"""
import itertools
import numpy as np
from qaoa_sampling_oneshot import (maqaoa_sampling_oneshot, get_QUBO_PO, get_PO_obj,
                                   evaluate_energy, state_inttobin, N_stocks, w)

N_idx = 2; N = N_stocks[N_idx]; n_bits = N*w
b = 2**w-1
X = np.array([[int(x) for s in c for x in format(s,f"0{w}b")]
              for c in itertools.product(range(b+1),repeat=N) if sum(c)==b], float)

print(f"n_bits={n_bits}, p=3, seed=442 -- old tolerance 1e-4 vs tight 1e-10\n")
print(f"{'vseed':>6} {'tol':>8} {'P(feas)':>9} {'eta_eff':>9} {'F(E_f)':>8} {'mean quant':>11} "
      f"{'eff #st':>8}")
print("-"*66)
for vseed in (42, 142, 242):
    Q_pen,c_pen = get_QUBO_PO(N,w,1,"seed",("PO_small",vseed),penalization=True,objective=False)
    Q_obj,c_obj = get_QUBO_PO(N,w,1,"seed",("PO_small",vseed),penalization=False,objective=True)
    Qo,Lo = get_PO_obj(n_bits,("PO_small",vseed))
    E_feas = np.sort(np.einsum("ij,jk,ik->i",X,Qo+np.diag(Lo),X))
    fm = np.zeros(2**n_bits,bool); Ea = np.zeros(2**n_bits)
    for i in range(2**n_bits):
        x = state_inttobin(i,n_bits)
        fm[i] = np.isclose(evaluate_energy(x,Q_pen,c_pen),0); Ea[i] = evaluate_energy(x,Q_obj,c_obj)
    for tol in (1e-4, 1e-10):
        r = maqaoa_sampling_oneshot("PO",N_idx,("PO_small",vseed),"optimality",0.5,3,
                                    qaoa_samples=1000,qaoa_steps=500,
                                    Ef_from_percentile=True,Ef_percentile=75,
                                    qaoa_seed=442,tolerance_opt=tol,print_time=False)
        pr = np.asarray(r["probability_states"],float); pf_tot = pr[fm].sum()
        pf = pr[fm]/pf_tot
        q = np.searchsorted(E_feas,Ea[fm],side="right")/len(E_feas)
        mq = float((pf*q).sum()); eff = float(np.exp(-(pf[pf>0]*np.log(pf[pf>0])).sum()))
        print(f"{vseed:>6} {tol:>8.0e} {pf_tot:>9.4f} {r['eta_eff_ex']:>9.4f} "
              f"{float(np.mean(E_feas<=r['E_f'])):>8.3f} {mq:>11.3f} {eff:>8.1f}", flush=True)
