import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
np.random.seed(42)



from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.optimize import minimize as moo_minimize
from scipy.optimize import minimize as sci_minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.indicators.hv import HV


from LINKS.CP import make_empty_submission, evaluate_submission
from LINKS.Optimization import Tools, DifferentiableTools, MechanismRandomizer



target_curves = np.load("target_curves.npy")

PROBLEM_TOOLS = Tools(device='cpu');               
PROBLEM_TOOLS.compile()

DIFF_TOOLS    = DifferentiableTools(device='cpu')
DIFF_TOOLS.compile()



# GA IMPLEMENTATION


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
    # Fill remaining 
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






# BFGS IMPLEMENTATION


d_cap = 0.75
m_cap = 10.0

opt_opts = dict(maxiter=300, gtol=1e-6, disp=False)

# backtracking GD fallback
fb_initial = 1e-2
fb_iterations = 10
fb_halvings    = 5

def build_objective(shape, edges, fixed_joints, motor, target_idx, target_curve):
    def fun(z):
        x = z.reshape(shape)
        d, m, gd, gm = DIFF_TOOLS([x], [edges], [fixed_joints], [motor], target_curve, [target_idx])
        d = float(d[0])
        m = float(m[0])
        if not (np.isfinite(d) and np.isfinite(m)):
            return 1e12, np.zeros_like(z)

        gd = gd[0]; gm = gm[0]
        if not (np.all(np.isfinite(gd)) and np.all(np.isfinite(gm))):
            return 1e12, np.zeros_like(z)

        J     = 0.5*(d/d_cap) + 0.5*(m/m_cap)
        gradJ = 0.5*(gd/d_cap) + 0.5*(gm/m_cap)
        return float(J), gradJ.reshape(-1)
    return fun

def refine_design(x0, edges, fixed_joints, motor, target_idx, target_curve):
    x = x0.copy()
    shape = x.shape
    obj = build_objective(shape, edges, fixed_joints, motor, target_idx, target_curve)

    # starting objective
    d0, m0, *_ = DIFF_TOOLS([x], [edges], [fixed_joints], [motor], target_curve, [target_idx])
    J0 = 0.5*(float(d0[0])/d_cap) + 0.5*(float(m0[0])/m_cap)

    # BFGS
    res = sci_minimize(obj, x.reshape(-1), method="BFGS", jac=True, options=opt_opts)
    x_try = res.x.reshape(shape)

    # check improvement
    d1, m1, *_ = DIFF_TOOLS([x_try], [edges], [fixed_joints], [motor], target_curve, [target_idx])
    J1 = 0.5*(float(d1[0])/d_cap) + 0.5*(float(m1[0])/m_cap)

    if (not np.isfinite(J1)) or (J1 >= J0 - 1e-12):
        # backtracking GD fallback
        x_fb = x.copy()
        for _ in range(fb_iterations):
            J, g = obj(x_fb.reshape(-1))
            if not np.isfinite(J) or not np.all(np.isfinite(g)):
                break
            gnorm = float(np.linalg.norm(g))
            if gnorm <= 0 or not np.isfinite(gnorm):
                break

            step = fb_initial / (1e-8 + gnorm)
            improved = False
            for __ in range(fb_halvings):
                x_trial = x_fb.reshape(-1) - step * g
                J2, grad= obj(x_trial)
                # Armijo-like check
                if np.isfinite(J2) and (J2 < J - 1e-4 * step * gnorm * gnorm):
                    x_fb = x_trial.reshape(shape)
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
        return x_fb
    else:
        return x_try







# MAIN SUBMISSION


init_pop = generate_diverse_initial_population(200)
submission = make_empty_submission()

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

    # 2) Evaluate originals and keep valid
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
        x_ref = refine_design(x0, E, FJ, M, T, tgt_curve)
        d, m = PROBLEM_TOOLS(x_ref, E, FJ, M, tgt_curve, T)
        if np.isfinite(d) and np.isfinite(m) and (d <= 0.75) and (m <= 10.0):
            candidates.append({
                'x0': x_ref,
                'edges': E,
                'fixed_joints': FJ,
                'motor': M,
                'target_joint': T
            })

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


# Save result
np.save('submission7.npy', submission)

