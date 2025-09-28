#!/usr/bin/env python3
# === GA + BFGS refinement → final_submission.npy (improved) ===
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import random
np.random.seed(42)
random.seed(42)

# headless plotting (faster)
import matplotlib
matplotlib.use('Agg')

# GA (PyMOO)
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.optimize import minimize as moo_minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.indicators.hv import HV

# Local toolkit
from LINKS.CP import make_empty_submission, evaluate_submission
from LINKS.Optimization import Tools, DifferentiableTools, MechanismRandomizer

# SciPy optimizer for BFGS
from scipy.optimize import minimize as sci_minimize


# -------------------------------------------------------------------
# Data & kernels
# -------------------------------------------------------------------
TARGET_CURVES_PATH = "target_curves.npy"
if not os.path.exists(TARGET_CURVES_PATH):
    raise FileNotFoundError("target_curves.npy not found in the working directory.")
target_curves = np.load(TARGET_CURVES_PATH)

PROBLEM_TOOLS = Tools(device='cpu');               PROBLEM_TOOLS.compile()
DIFF_TOOLS    = DifferentiableTools(device='cpu'); DIFF_TOOLS.compile()


# -------------------------------------------------------------------
# GA problem (matches teammate’s set-up)
# -------------------------------------------------------------------
class AdvancedMechanismOptimization(ElementwiseProblem):
    def __init__(self, target_curve, N=5):
        self.N = N
        variables = {f"X0_{i}": Real(bounds=(0.0, 1.0)) for i in range(2*N)}
        variables["target"] = Integer(bounds=(2, N-1))
        super().__init__(vars=variables, n_obj=2, n_constr=2)

        self.target_curve = target_curve
        # Fixed 5-bar topology
        self.edges = np.array([[0,2], [1,3], [2,3], [2,4], [3,4]])
        self.fixed_joints = np.array([0, 1])
        self.motor = np.array([0, 2])

    def _evaluate(self, x, out, *args, **kwargs):
        x0 = np.zeros((self.N, 2))
        for i in range(self.N):
            x0[i, 0] = x[f"X0_{2*i}"]
            x0[i, 1] = x[f"X0_{2*i + 1}"]
        target_idx = int(x["target"])
        try:
            dist, mat = PROBLEM_TOOLS(
                x0, self.edges, self.fixed_joints, self.motor,
                self.target_curve, target_idx=target_idx
            )
            if np.isnan(dist) or np.isinf(dist):
                out["F"] = [np.inf, np.inf]
                out["G"] = [np.inf, np.inf]
            else:
                out["F"] = [dist, mat]
                out["G"] = [dist - 0.75, mat - 10.0]
        except Exception:
            out["F"] = [np.inf, np.inf]
            out["G"] = [np.inf, np.inf]


def generate_diverse_initial_population(n_samples=200):
    """Match teammate’s scale to keep GA behavior comparable."""
    pop = []
    # Try randomized mechanisms
    try:
        rnd = MechanismRandomizer(min_size=5, max_size=5, device='cpu')
        for _ in range(n_samples // 2):
            try:
                mech = rnd(n=5)
                pop.append(mech['x0'])
            except Exception:
                pass
    except Exception:
        pass
    # Fill remaining with perturbed templates
    base_configs = [
        np.array([[0.3,0.2],[0.6,0.2],[0.3,0.3],[0.6,0.4],[0.4,0.5]]),
        np.array([[0.2,0.3],[0.8,0.3],[0.3,0.5],[0.7,0.5],[0.5,0.7]]),
        np.array([[0.4,0.4],[0.6,0.4],[0.4,0.5],[0.6,0.5],[0.5,0.6]]),
    ]
    i = 0
    while len(pop) < n_samples:
        base = base_configs[i % len(base_configs)]
        pert = np.clip(base + 0.1*np.random.randn(*base.shape), 0.05, 0.95)
        pop.append(pert)
        i += 1
    return pop[:n_samples]


def run_ga_for_curve(curve_idx, target_curve, init_pop):
    from pymoo.core.sampling import Sampling
    class InitialSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            X = []
            for i in range(n_samples):
                x = {}
                x0 = init_pop[i % len(init_pop)]
                for j in range(problem.N):
                    x[f"X0_{2*j}"]   = float(x0[j,0])
                    x[f"X0_{2*j+1}"] = float(x0[j,1])
                x["target"] = np.random.choice([2,3,4])
                X.append(x)
            return X

    prob = AdvancedMechanismOptimization(target_curve, N=5)
    algo = NSGA2(
        pop_size=400,
        sampling=InitialSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=0.1, eta=20),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination()
    )
    res = moo_minimize(prob, algo, ('n_gen', 100), verbose=False, seed=42 + curve_idx)

    sols = []
    if res.X is not None:
        if hasattr(res.X, '__len__') and not isinstance(res.X, dict):
            take = min(len(res.X), 50)   # like teammate's flow
            for i in range(take):
                x0 = np.zeros((5,2))
                for j in range(5):
                    x0[j,0] = res.X[i][f"X0_{2*j}"]
                    x0[j,1] = res.X[i][f"X0_{2*j+1}"]
                sols.append({
                    'x0': x0,
                    'edges': prob.edges,
                    'fixed_joints': prob.fixed_joints,
                    'motor': prob.motor,
                    'target_joint': int(res.X[i]["target"])
                })
        else:
            x0 = np.zeros((5,2))
            for j in range(5):
                x0[j,0] = res.X[f"X0_{2*j}"]
                x0[j,1] = res.X[f"X0_{2*j+1}"]
            sols.append({
                'x0': x0,
                'edges': prob.edges,
                'fixed_joints': prob.fixed_joints,
                'motor': prob.motor,
                'target_joint': int(res.X["target"])
            })
    return sols


# -------------------------------------------------------------------
# BFGS refinement (with symmetric cap penalties + validity filtering)
# -------------------------------------------------------------------
np.random.seed(0)  # for refinement jitters

# caps & weights
d_cap_stages = [2.0, 1.2, 0.9, 0.75]
m_cap        = 10.0
weight_sets  = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]

# penalties (augmented-Lagrangian style)
rho_d = 5.0   # distance cap penalty weight
rho_m = 5.0   # material cap penalty weight
lam_d = 0.0
lam_m = 0.0

# BFGS + fallback settings
bfgs_opts    = dict(maxiter=300, gtol=1e-6, disp=False)
jitter_sigma = 1e-2
fb_alpha0, fb_outer, fb_backtracks = 1e-2, 10, 5

def make_obj(shape, Ei, FJi, Mi, Ti, tgt_curve, wd, wm, d_cap):
    """Return fun(z)->(J, grad) with penalties if d>d_cap or m>m_cap."""
    def fun(z):
        Xi = z.reshape(shape)
        d, m, gd, gm = DIFF_TOOLS([Xi], [Ei], [FJi], [Mi], tgt_curve, [Ti])
        d = float(d[0]); m = float(m[0])
        if not (np.isfinite(d) and np.isfinite(m)): return 1e12, np.zeros_like(z)
        gd, gm = gd[0], gm[0]
        if not (np.all(np.isfinite(gd)) and np.all(np.isfinite(gm))): return 1e12, np.zeros_like(z)

        d_norm, m_norm = d/d_cap, m/m_cap
        # base scalarization
        J  = wd*d_norm + wm*m_norm
        gJ = wd*(gd/d_cap) + wm*(gm/m_cap)

        # distance penalty (only above cap)
        if d_norm > 1.0:
            vio = d_norm - 1.0
            J  += lam_d*vio + 0.5*rho_d*(vio**2)
            gJ += (lam_d + rho_d*vio)*(gd/d_cap)

        # material penalty (only above cap)
        if m_norm > 1.0:
            vio = m_norm - 1.0
            J  += lam_m*vio + 0.5*rho_m*(vio**2)
            gJ += (lam_m + rho_m*vio)*(gm/m_cap)

        return float(J), gJ.reshape(-1)
    return fun


def refine_member_once(xi, Ei, FJi, Mi, Ti, tgt_curve, wd, wm):
    """Continuation over d_cap stages; multi-start BFGS with tiny backtracking fallback."""
    x = xi.copy(); shape = x.shape
    for d_cap in d_cap_stages:
        obj = make_obj(shape, Ei, FJi, Mi, Ti, tgt_curve, wd, wm, d_cap)

        # baseline J for fallback
        d0, m0, *_ = DIFF_TOOLS([x], [Ei], [FJi], [Mi], tgt_curve, [Ti])
        J0 = 0.5*(float(d0[0])/d_cap) + 0.5*(float(m0[0])/m_cap)

        # multi-start BFGS
        z_base = x.reshape(-1)
        starts = [z_base,
                  z_base + jitter_sigma*np.random.randn(*z_base.shape),
                  z_base - jitter_sigma*np.random.randn(*z_base.shape)]
        bestJ, bestz = None, None
        for z0 in starts:
            res = sci_minimize(obj, z0, method="BFGS", jac=True, options=bfgs_opts)
            J_try, _ = obj(res.x)
            if bestJ is None or (np.isfinite(J_try) and J_try < bestJ):
                bestJ, bestz = J_try, res.x
        x_opt = bestz.reshape(shape)

        # fallback: tiny backtracking GD if no improvement
        d1, m1, *_ = DIFF_TOOLS([x_opt], [Ei], [FJi], [Mi], tgt_curve, [Ti])
        J1 = 0.5*(float(d1[0])/d_cap) + 0.5*(float(m1[0])/m_cap)
        if (not np.isfinite(J1)) or (J1 >= J0 - 1e-12):
            Xi = x.copy()
            for _ in range(fb_outer):
                J, g = obj(Xi.reshape(-1))
                if not np.isfinite(J) or not np.all(np.isfinite(g)): break
                gnorm = float(np.linalg.norm(g))
                if gnorm <= 0 or not np.isfinite(gnorm): break
                a = fb_alpha0 / (1e-8 + gnorm)
                ok = False
                for __ in range(fb_backtracks):
                    Xtry = Xi.reshape(-1) - a*g
                    J2, _ = obj(Xtry)
                    if np.isfinite(J2) and (J2 < J - 1e-4*a*gnorm*gnorm):
                        Xi = Xtry.reshape(shape); ok = True; break
                    a *= 0.5
                if not ok: break
            x = Xi
        else:
            x = x_opt
    return x


# -------------------------------------------------------------------
# Driver: GA → BFGS refinement → select valid & best → save final_submission.npy
# -------------------------------------------------------------------
print("\n============================")
print(" GA + BFGS REFINEMENT START ")
print("============================")

init_pop = generate_diverse_initial_population(200)
submission = make_empty_submission()

# optional cap on how many per curve to write (keep best after filtering)
MAX_SAVE_PER_CURVE = 120

ref_point = np.array([0.75, 10.0])
hv_ind = HV(ref_point)

for curve_idx in range(6):
    print(f"\n--- Curve {curve_idx} ---")
    tgt_curve = np.asarray(target_curves[curve_idx], float)

    # 1) GA
    ga_solutions = run_ga_for_curve(curve_idx, tgt_curve, init_pop)
    if not ga_solutions:
        print("No GA solutions; skipping.")
        continue

    # 2) Evaluate originals and keep valid; collect candidates
    candidates = []
    for s in ga_solutions:
        x0 = np.asarray(s['x0'], float)
        E  = np.asarray(s['edges'], int)
        FJ = np.asarray(s['fixed_joints'], int)
        M  = np.asarray(s['motor'], int)
        T  = int(s['target_joint'])

        d0, m0 = PROBLEM_TOOLS(x0, E, FJ, M, tgt_curve, T)
        if np.isfinite(d0) and np.isfinite(m0) and (d0 <= 0.75) and (m0 <= 10.0):
            candidates.append({'x0': x0, 'edges': E, 'fixed_joints': FJ, 'motor': M, 'target_joint': T})

        # 3) Refine for each weight and add valid ones
        for (wd, wm) in weight_sets:
            x_ref = refine_member_once(x0, E, FJ, M, T, tgt_curve, wd, wm)
            d, m = PROBLEM_TOOLS(x_ref, E, FJ, M, tgt_curve, T)
            if np.isfinite(d) and np.isfinite(m) and (d <= 0.75) and (m <= 10.0):
                candidates.append({'x0': x_ref, 'edges': E, 'fixed_joints': FJ, 'motor': M, 'target_joint': T})

    # 4) If too many, sort by simple scalarization and keep the best K
    if len(candidates) > MAX_SAVE_PER_CURVE:
        scored = []
        for c in candidates:
            d, m = PROBLEM_TOOLS(c['x0'], c['edges'], c['fixed_joints'], c['motor'], tgt_curve, c['target_joint'])
            J = 0.5*(d/0.75) + 0.5*(m/10.0)
            scored.append((J, c))
        scored.sort(key=lambda t: t[0])
        candidates = [c for _, c in scored[:MAX_SAVE_PER_CURVE]]

    # 5) Save into submission for this curve
    for c in candidates:
        submission[f'Problem {curve_idx+1}'].append(c)

    # 6) Report HV for this curve (just for feedback)
    F = []
    for c in candidates:
        d, m = PROBLEM_TOOLS(c['x0'], c['edges'], c['fixed_joints'], c['motor'], tgt_curve, c['target_joint'])
        F.append([d, m])
    F = np.array(F) if len(F) else np.empty((0,2))
    hv = hv_ind(F) if len(F) else 0.0
    print(f"Saved {len(candidates)} valid designs | Curve HV = {hv:.4f}")

# Save & evaluate
out_path = 'submission.npy'
np.save(out_path, submission)
print(f"\nSaved refined submission to: {out_path}")

print("\nEvaluating final submission...")
try:
    evaluate_submission(submission)
except Exception as e:
    print("evaluate_submission raised an exception:", e)

print("\n✅ Done.")
