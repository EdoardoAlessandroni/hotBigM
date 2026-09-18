"""Is the M*/span_proxy loop solvable self-consistently, and where does the overshoot come from?

The deployed QUBO is Q(M) = Q_obj + M*Q_pen, whose span is  Delta(M) = A + M*B  with
A = UB_obj - E_LB_obj and B = UB_pen. The QAOA temperature in raw units is Delta(M)*T_norm, so the
*self-consistent* penalty weight is the fixed point of

    M  =  F( 1 / [ (A + M*B) * T_norm ] ),        F = M_method_opt (or its L1 closed form)

`qaoa_sampling_oneshot` does not solve this: it evaluates the span once at a reference M0 =
L1_norm_hot(..., T_norm, eta), which is a penalty weight in its own right, and stops. This script
asks whether the fixed point exists at all, and what M* would be under M-independent alternatives.

For the L1-shaped F(beta) = L1 + (1/beta)*c_n, with c_n = n*ln2 - ln(1-eta), the equation is LINEAR
in M and can be solved in closed form:

    M = L1 + (A + M*B)*T_norm*c_n     =>     M = (L1 + A*T_norm*c_n) / (1 - B*T_norm*c_n)

so the loop gain is  G = B*T_norm*c_n , and a positive solution exists only when G < 1. Everything
below measures G.
"""

import numpy as np

import qaoa_sampling_oneshot as qso
from myalgo import M_method_opt

W, DIRECTORY, ETA, PEAK_MAX = 3, "PO_small", 0.5, 25


def ingredients(n_bits, vseed, layers):
    """A, B, T_norm, c_n and the pipeline's own M0 / s_conv / M*, for one cell."""
    N = n_bits // W
    info = (DIRECTORY, vseed)
    Q_obj, c_obj = qso.get_QUBO_PO(N, W, 1, "seed", info, penalization=False, objective=True)
    Q_pen, c_pen = qso.get_QUBO_PO(N, W, 1, "seed", info, penalization=True, objective=False)
    E_LB = qso.lower_bound_obj_PO(N, W, info)

    T_norm = qso.temperature_from_layers(layers, n_bits)
    c_n = n_bits * np.log(2) - np.log(1 - ETA)
    L1 = qso.L1_norm(Q_obj, c_obj)

    UB_obj = qso.sdp_bound(Q_obj, c_obj, sense="max")
    UB_pen = qso.sdp_bound(Q_pen, c_pen, sense="max")
    A, B = UB_obj - E_LB, UB_pen

    M0 = qso.L1_norm_hot(Q_obj, c_obj, n_bits, T_norm, ETA)
    s_conv = A + M0 * B
    return dict(N=N, info=info, Q_obj=Q_obj, c_obj=c_obj, E_LB=E_LB, T_norm=T_norm, c_n=c_n,
                L1=L1, A=A, B=B, M0=M0, s_conv=s_conv)


def F_exact(M, g, n_bits, layers):
    """One turn of the loop with the real algorithm: M -> M_method_opt at the span evaluated at M."""
    beta = 1.0 / ((g["A"] + M * g["B"]) * g["T_norm"])
    M_new, _ = M_method_opt((g["N"], W), "PO", g["info"], "seed", beta, PEAK_MAX, ETA,
                            None, g["E_LB"], log_stable=True,
                            Ef_from_percentile=True, Ef_percentile=75)
    return M_new


def F_l1(M, g, *_):
    """Same turn under the L1 closed form, which is what makes the algebra transparent."""
    T = (g["A"] + M * g["B"]) * g["T_norm"]
    return g["L1"] + T * g["c_n"]


if __name__ == "__main__":
    print("Loop gain G = B*T_norm*c_n  (fixed point exists only if G < 1)\n")
    print(f"{'n':>3} {'p':>2} | {'A=UBobj-ELB':>12} {'B=UBpen':>9} {'T_norm':>8} {'c_n':>7} "
          f"| {'G':>10} | {'M0':>9} {'s_conv':>10} {'M* (pipeline)':>14}")
    for n_bits in (6, 9, 12, 15):
        for layers in (1, 2, 3):
            g = ingredients(n_bits, 42, layers)
            G = g["B"] * g["T_norm"] * g["c_n"]
            beta_pipe = 1.0 / (g["s_conv"] * g["T_norm"])
            M_star, _ = M_method_opt((g["N"], W), "PO", g["info"], "seed", beta_pipe, PEAK_MAX, ETA,
                                     None, g["E_LB"], log_stable=True,
                                     Ef_from_percentile=True, Ef_percentile=75)
            print(f"{n_bits:3d} {layers:2d} | {g['A']:12.4g} {g['B']:9.4g} {g['T_norm']:8.4g} "
                  f"{g['c_n']:7.3f} | {G:10.3e} | {g['M0']:9.4g} {g['s_conv']:10.4g} "
                  f"{M_star:14.4g}", flush=True)

    print("\n\nIterating the real loop from the pipeline's own M*, n=12 p=1 vseed 42:")
    g = ingredients(12, 42, 1)
    M = 1.0 / (g["s_conv"] * g["T_norm"])
    M, _ = M_method_opt((g["N"], W), "PO", g["info"], "seed", M, PEAK_MAX, ETA, None, g["E_LB"],
                        log_stable=True, Ef_from_percentile=True, Ef_percentile=75)
    for it in range(6):
        M_next = F_exact(M, g, 12, 1)
        print(f"   iter {it}:  M = {M:12.5g}  ->  F(M) = {M_next:12.5g}   "
              f"(ratio {M_next / M:.3g})", flush=True)
        M = M_next

    print("\n\nM-independent alternatives for the temperature scale (n, p -> M*):")
    print(f"{'n':>3} {'p':>2} | {'pipeline M*':>12} {'obj-span M*':>12} {'maxQpen M*':>12}")
    for n_bits in (6, 9, 12, 15):
        for layers in (1, 2, 3):
            g = ingredients(n_bits, 42, layers)
            Q_pen, _ = qso.get_QUBO_PO(g["N"], W, 1, "seed", g["info"],
                                       penalization=True, objective=False)
            out = []
            for scale in (g["s_conv"], g["A"], np.abs(Q_pen).max()):
                beta = 1.0 / (scale * g["T_norm"])
                m, _ = M_method_opt((g["N"], W), "PO", g["info"], "seed", beta, PEAK_MAX, ETA,
                                    None, g["E_LB"], log_stable=True,
                                    Ef_from_percentile=True, Ef_percentile=75)
                out.append(m)
            print(f"{n_bits:3d} {layers:2d} | {out[0]:12.4g} {out[1]:12.4g} {out[2]:12.4g}",
                  flush=True)
