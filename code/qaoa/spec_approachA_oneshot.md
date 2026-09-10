# Spec — Approach A: one-shot span-proxy `qaoa_sampling_A`

## Goal
Reimplement `qaoa_sampling` (in `qaoa_playground.ipynb`) as a new function
`qaoa_sampling_A`, replacing the `select_normalization` / `norm_guess` +
`temperature_unnormalized = norm_guess * temperature_normalized` logic with a
principled **spectral-span proxy** `s_conv ≈ ΔE = E_max^full − E_min^full` of the
**full** QUBO `E = E_obj + M·E_pen`.

Keep everything else (QUBO construction, QAOA call, energy evaluation, η_eff
computation, results dict) identical to `qaoa_sampling`. Same input/output style.
Print the important quantities when `verbose=True`.

## Why (context for the implementer — do not re-derive, just implement)
The coldness reported by Díez-Valle, `β_p = κ·p·n^{-3/2}`, is referenced to
energies min–max-normalized to `[0,1]`, i.e. to the **full-Hamiltonian span**
`ΔE = E_max^full − E_min^full`. To place that coldness on the original (un-scaled)
energy axis on which the M-algorithm operates, we feed the algorithm
`β = β_p / s_conv`, with `s_conv` an efficiently computable estimate of `ΔE`.

The full landscape splits by feasibility:
- `E_min^full = E_min^obj` (feasible ground state; penalty = 0), lower-bounded by
  the objective SDP `E_LB_obj`.
- `E_max^full ≈ UB_obj + M·v_max`, where `v_max = max_x E_pen(x)` and `UB_obj`
  upper-bounds the objective. We do NOT have `v_max` in closed form, so we
  **upper-bound `E_pen`** the same way we upper-bound the objective.

Hence:
```
s_conv(M) = (UB_obj − E_LB_obj) + M · UB_pen
```
This is **linear in M**; the one-shot approximation evaluates it at a fixed
reference `M0` (default `M_L1`, guaranteed sufficient). One-shot is exact when the
objective span dominates (TSP/NPP) and conservative/over-penalizing — hence
feasibility-safe — when the penalty term dominates (PO).

## New/changed helpers to implement

### `sdp_bound(Q, const, sense="min")`
Generalize `lower_bound_obj_PO` to an arbitrary QUBO matrix `Q` (n×n, may be
asymmetric — symmetrize as `0.5*(Q+Q.T)` for the SDP) and scalar `const`, working
directly from the in-memory matrix (NOT reading an `.lp` file). Return a bound on
`min_x (x^T Q x + const)` when `sense="min"`, and on `max_x` when `sense="max"`.

Implementation (reuse the MOSEK SDP relaxation pattern already in
`lower_bound_obj_PO`):
- Build the lifted `Q_tilde` ((n+1)×(n+1)) with the linear part taken from the
  diagonal of `Q` (binary vars: `x_i^2 = x_i`, so diagonal entries are linear
  coefficients). Concretely: let `Qm = 0.5*(Q+Q.T)`; linear `L = diag(Qm)`;
  quadratic part `Qq = Qm − diag(diag(Qm))`. Then `Q_tilde[1:,1:]=Qq`,
  `Q_tilde[0,1:]=0.5*L`, `Q_tilde[1:,0]=0.5*L`, `Q_tilde[0,0]=0`.
- Same constraints as `lower_bound_obj_PO`: `X >> 0`, `X[i,i]==X[0,i]`, `X[0,0]==1`.
- `sense="min"` → `cp.Minimize(cp.trace(Q_tilde @ X))`;
  `sense="max"` → `cp.Maximize(...)`  (equivalently minimize `−Q_tilde`).
- Return `prob.value + const`.
- Solver: MOSEK, matching existing code.

Add a cheap fallback `use_sdp_pen=False` path for the **penalty** upper bound only:
`UB_pen_cheap = sum of positive entries of the symmetrized full penalty QUBO`
(i.e. an ℓ1-style upper bound `Σ_{ij: Qm_ij>0} Qm_ij + max(const,0)`), so large
instances can skip the penalty SDP. Default `use_sdp_pen=True` (tighter; loose UB
on the span is the unsafe direction because it lowers β → too hot).

### `span_proxy(Q_obj, const_obj, Q_pen, const_pen, M0, problem_type, E_LB_obj, use_sdp_pen=True)`
Return `s_conv = (UB_obj − E_LB_obj) + M0 * UB_pen`, where:
- `E_LB_obj`: reuse existing `E_LB` (0 for NPP/TSP, `lower_bound_obj_PO` for PO).
- `UB_obj = sdp_bound(Q_obj, const_obj, sense="max")`. For NPP/TSP the objective
  is small/structured; still compute it (cheap), or allow an ℓ1 fallback.
- `UB_pen`: if `use_sdp_pen`, `sdp_bound(Q_pen, const_pen, sense="max")`; else the
  cheap ℓ1 bound above. (`E_pen` min is 0 by construction — feasible points — so
  no penalty lower-bound SDP is needed.)

## `qaoa_sampling_A` — signature
Mirror `qaoa_sampling` exactly, add two kwargs:
```python
def qaoa_sampling_A(problem_type, N_idx, info_instance, Mstrategy, eta_required,
                    layers, qaoa_samples=1000, qaoa_steps=100,
                    M0_ref="L1", use_sdp_pen=True,
                    print_time=True, verbose_qaoa=False, verbose=False):
```
- `M0_ref`: reference weight for evaluating the span. `"L1"` → use `M_L1`
  (compute it first, as in the original). Allow a float override.
- `use_sdp_pen`: passed to `span_proxy`.

## Step-by-step body (diff against `qaoa_sampling`)

1. **Block 1 (Get LCBO):** unchanged.

2. **Block 2 — replace the normalization/temperature logic:**
   - `temperature_normalized = temperature_from_layers(layers, n_bits)`  *(keep)*
   - Compute `E_LB` exactly as now (0 for NPP/TSP; `lower_bound_obj_PO` for PO).
   - Compute `M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits, temperature_unnormalized_placeholder, min_pfeas)`
     — **note the chicken-and-egg:** `L1_norm_hot` currently takes an
     unnormalized temperature. Resolve by computing `M_L1` at the *normalized*
     temperature scale first is wrong; instead compute the span at `M0`, then β,
     then everything else. Concretely reorder:
       a. Pick `M0`: if `M0_ref=="L1"`, compute a provisional `M_L1` using the
          **normalized** temperature times a provisional span. To avoid recursion,
          for the *reference only* use the cheap ℓ1 objective bound to get a
          provisional span `s0 = (UB_obj_cheap − E_LB) + M_L1_provisional·UB_pen_cheap`.
          **Simpler and preferred:** set `M0` directly to the cheap closed form
          `M_L1(β_norm)` where `β_norm = 1/temperature_normalized` (i.e. evaluate
          Eq. 12 at the normalized temperature). This is only a reference scale for
          the span; its exact value is not critical when the objective span
          dominates, and when the penalty dominates any O(1)-consistent choice is
          reabsorbed. Document this choice in a comment.
       b. `s_conv = span_proxy(Q_obj, const_obj, Q_pen, const_pen, M0, problem_type, E_LB, use_sdp_pen)`
       c. `temperature_unnormalized = s_conv * temperature_normalized`
          `beta = 1 / temperature_unnormalized`
   - `min_pfeas`, `peak_max` (4 NPP/TSP, 25 PO): unchanged.
   - Call `M_method_feas` / `M_method_opt` exactly as now with this `beta`.
   - Recompute the *reported* `M_L1 = L1_norm_hot(Q_obj, const_obj, n_bits,
     temperature_unnormalized, min_pfeas)` for logging/comparison.

3. **Block 3 (Build QUBO(M*) + QAOA):**
   - Build `Q, const` with `M_star` as now.
   - **Compile normalization (unchanged in spirit):** divide by
     `norm_final = np.max(np.abs(Q))` ONLY. Remove the `Q/norm_guess` step —
     there is no `norm_guess` anymore; the span already lives in `beta`, and the
     compile normalization is a separate trainability gauge that does not re-enter
     the temperature logic. (Mathematically inert at the optimum; kept for Adam.)
   - Feed to `run_qaoa_with_solution` as now.

4. **Blocks 4–5 (energies, η_eff, η_eff_exact):** unchanged.

## Verbose printing (when `verbose=True`)
Print, clearly labelled:
- `n_bits`, `layers`
- `temperature_normalized`, `s_conv`, `temperature_unnormalized`, `beta`
- `M0` (reference used for span), `M_star`, `M_L1`
- `E_LB (obj)`, `UB_obj`, `UB_pen`, and which of {SDP, ℓ1} produced each
- `norm_final`
- the existing η line: `req / gua / eff_ex`

## Results dict
Same keys as `qaoa_sampling`, plus:
- `results["s_conv"] = s_conv`
- `results["M0_ref"] = M0`
- `results["UB_obj"] = UB_obj`, `results["UB_pen"] = UB_pen`
- drop `results["norm_guess"]` (no longer defined) or set it to `s_conv` for
  backward-compat with downstream plotting — pick one and note it.

## Do NOT
- Do not reintroduce the `T_norm ↔ T_unnorm` round-trip beyond the single
  `temperature_unnormalized = s_conv * temperature_normalized` line.
- Do not let `norm_final` (compile gauge) feed back into `beta` or `M_star`.
