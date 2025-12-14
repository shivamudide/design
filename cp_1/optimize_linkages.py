#!/usr/bin/env python3
"""
Advanced Linkage Mechanism Optimization
Combines GA with gradient refinement for better results
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent plots from popping up
import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm, trange
import time

# Set seeds
np.random.seed(42)
random.seed(42)

# Imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

# LINKS imports
from LINKS.CP import make_empty_submission, evaluate_submission
from LINKS.Optimization import Tools, DifferentiableTools, MechanismRandomizer
from LINKS.Kinematics import MechanismSolver

print("Loading target curves...")
target_curves = np.load('target_curves.npy')

# Initialize tools
PROBLEM_TOOLS = Tools(device='cpu')
PROBLEM_TOOLS.compile()

GRADIENT_TOOLS = DifferentiableTools(device='cpu')
GRADIENT_TOOLS.compile()

class AdvancedMechanismOptimization(ElementwiseProblem):
    def __init__(self, target_curve, N=5):
        self.N = N
        variables = dict()
        
        # Position variables for all joints
        for i in range(2*N):
            variables[f"X0_{i}"] = Real(bounds=(0.0, 1.0))
        
        # Target joint
        variables["target"] = Integer(bounds=(2, N-1))
        
        super().__init__(vars=variables, n_obj=2, n_constr=2)
        
        self.target_curve = target_curve
        
        # Fixed 5-bar topology
        self.edges = np.array([[0,2], [1,3], [2,3], [2,4], [3,4]])
        self.fixed_joints = np.array([0, 1])
        self.motor = np.array([0, 2])
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Reconstruct x0 from variables
        x0 = np.zeros((self.N, 2))
        for i in range(self.N):
            x0[i, 0] = x[f"X0_{2*i}"]
            x0[i, 1] = x[f"X0_{2*i + 1}"]
        
        target_idx = x["target"]
        
        try:
            distance, material = PROBLEM_TOOLS(
                x0, self.edges, self.fixed_joints, self.motor,
                self.target_curve, target_idx=target_idx
            )
            
            if np.isnan(distance) or np.isinf(distance):
                out["F"] = [np.inf, np.inf]
                out["G"] = [np.inf, np.inf]
            else:
                out["F"] = [distance, material]
                out["G"] = [distance - 0.75, material - 10.0]
        except:
            out["F"] = [np.inf, np.inf]
            out["G"] = [np.inf, np.inf]


def generate_diverse_initial_population(n_samples=200):
    """Generate diverse initial mechanisms using multiple strategies"""
    population = []
    
    # Try to use MechanismRandomizer
    try:
        randomizer = MechanismRandomizer(min_size=5, max_size=5, device='cpu')
        for _ in range(n_samples // 2):
            try:
                mech = randomizer(n=5)
                population.append(mech['x0'])
            except:
                pass
    except:
        pass
    
    # Add variations of known good configurations
    base_configs = [
        # Standard 5-bar
        np.array([[0.3, 0.2], [0.6, 0.2], [0.3, 0.3], [0.6, 0.4], [0.4, 0.5]]),
        # Wide configuration
        np.array([[0.2, 0.3], [0.8, 0.3], [0.3, 0.5], [0.7, 0.5], [0.5, 0.7]]),
        # Compact configuration
        np.array([[0.4, 0.4], [0.6, 0.4], [0.4, 0.5], [0.6, 0.5], [0.5, 0.6]]),
    ]
    
    while len(population) < n_samples:
        base = base_configs[len(population) % len(base_configs)]
        # Add random perturbation
        perturbed = base + np.random.randn(*base.shape) * 0.1
        perturbed = np.clip(perturbed, 0.05, 0.95)
        population.append(perturbed)
    
    return population[:n_samples]


def gradient_refinement(x0_batch, edges, fixed_joints, motor, target_curve, target_idxs, 
                        n_steps=200, step_size=5e-4):
    """Refine solutions using gradient descent"""
    x = [x0.copy() for x0 in x0_batch]
    best_x = [x0.copy() for x0 in x0_batch]
    best_distances = [np.inf] * len(x)
    
    for step in range(n_steps):
        try:
            distances, materials, distance_grads, material_grads = GRADIENT_TOOLS(
                x, [edges]*len(x), [fixed_joints]*len(x), [motor]*len(x),
                target_curve, target_idxs
            )
            
            for i in range(len(x)):
                # Only update if valid
                if distances[i] <= 0.75 and materials[i] <= 10.0:
                    if distances[i] < best_distances[i]:
                        best_distances[i] = distances[i]
                        best_x[i] = x[i].copy()
                    
                    # Gradient step
                    if not np.any(np.isnan(distance_grads[i])) and not np.any(np.isinf(distance_grads[i])):
                        x[i] = x[i] - step_size * distance_grads[i]
                        x[i] = np.clip(x[i], 0.0, 1.0)
        except:
            break
    
    return best_x, best_distances


def optimize_curve_advanced(target_curve, curve_idx, initial_population):
    """Advanced optimization combining GA and gradient refinement"""
    
    print(f"\n{'='*60}")
    print(f"Optimizing Curve {curve_idx + 1} (Advanced)")
    
    problem = AdvancedMechanismOptimization(target_curve, N=5)
    
    # Create initial sampling from diverse population
    from pymoo.core.sampling import Sampling
    
    class InitialSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            X = []
            for i in range(n_samples):
                x = {}
                x0 = initial_population[i % len(initial_population)]
                
                for j in range(problem.N):
                    x[f"X0_{2*j}"] = x0[j, 0]
                    x[f"X0_{2*j + 1}"] = x0[j, 1]
                
                x["target"] = np.random.choice([2, 3, 4])  # Try different target joints
                X.append(x)
            return X
    
    # Run GA with larger population and more generations
    algorithm = NSGA2(
        pop_size=400,
        sampling=InitialSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=0.1, eta=20),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination()
    )
    
    print(f"  Phase 1: Genetic Algorithm (100 generations)...")
    results_ga = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        verbose=False,
        seed=42 + curve_idx
    )
    
    # Extract solutions for gradient refinement
    if results_ga.X is not None and results_ga.F is not None:
        print(f"  Phase 2: Gradient refinement...")
        
        # Prepare batch for gradient optimization
        x0_batch = []
        target_batch = []
        
        if hasattr(results_ga.X, '__len__') and not isinstance(results_ga.X, dict):
            for i in range(min(len(results_ga.X), 50)):  # Limit to top 50 solutions
                x0 = np.zeros((5, 2))
                for j in range(5):
                    x0[j, 0] = results_ga.X[i][f"X0_{2*j}"]
                    x0[j, 1] = results_ga.X[i][f"X0_{2*j + 1}"]
                x0_batch.append(x0)
                target_batch.append(int(results_ga.X[i]["target"]))
        else:
            x0 = np.zeros((5, 2))
            for j in range(5):
                x0[j, 0] = results_ga.X[f"X0_{2*j}"]
                x0[j, 1] = results_ga.X[f"X0_{2*j + 1}"]
            x0_batch.append(x0)
            target_batch.append(int(results_ga.X["target"]))
        
        # Apply gradient refinement
        refined_x0, refined_distances = gradient_refinement(
            x0_batch, problem.edges, problem.fixed_joints, problem.motor,
            target_curve, target_batch, n_steps=150
        )
        
        # Combine original and refined solutions
        all_solutions = []
        
        # Add refined solutions
        for x0, target_idx in zip(refined_x0, target_batch):
            all_solutions.append({
                'x0': x0,
                'edges': problem.edges,
                'fixed_joints': problem.fixed_joints,
                'motor': problem.motor,
                'target_joint': target_idx
            })
        
        # Add original GA solutions
        if hasattr(results_ga.X, '__len__') and not isinstance(results_ga.X, dict):
            for i in range(len(results_ga.X)):
                x0 = np.zeros((5, 2))
                for j in range(5):
                    x0[j, 0] = results_ga.X[i][f"X0_{2*j}"]
                    x0[j, 1] = results_ga.X[i][f"X0_{2*j + 1}"]
                all_solutions.append({
                    'x0': x0,
                    'edges': problem.edges,
                    'fixed_joints': problem.fixed_joints,
                    'motor': problem.motor,
                    'target_joint': int(results_ga.X[i]["target"])
                })
        
        # Evaluate all solutions
        all_F = []
        for sol in all_solutions:
            try:
                dist, mat = PROBLEM_TOOLS(
                    sol['x0'], sol['edges'], sol['fixed_joints'], 
                    sol['motor'], target_curve, sol['target_joint']
                )
                if dist <= 0.75 and mat <= 10.0:
                    all_F.append([dist, mat])
            except:
                pass
        
        if all_F:
            all_F = np.array(all_F)
            ref_point = np.array([0.75, 10.0])
            ind = HV(ref_point)
            hv = ind(all_F)
            print(f"  Final hypervolume: {hv:.4f}")
            print(f"  Valid solutions: {len(all_F)}")
            return all_solutions[:len(all_F)], hv
        
    print(f"  No valid solutions found")
    return [], 0.0


# Main execution
print("\n" + "="*60)
print("ADVANCED LINKAGE OPTIMIZATION")
print("="*60)

# Generate diverse initial population
print("\nGenerating diverse initial population...")
initial_population = generate_diverse_initial_population(300)
print(f"Created {len(initial_population)} initial configurations")

# Optimize each curve
submission = make_empty_submission()
all_hypervolumes = []
start_time = time.time()

for curve_idx in range(6):
    solutions, hv = optimize_curve_advanced(
        target_curves[curve_idx], 
        curve_idx, 
        initial_population
    )
    
    # Add to submission
    for sol in solutions:
        submission[f'Problem {curve_idx + 1}'].append(sol)
    
    all_hypervolumes.append(hv)

# Calculate and display final results
elapsed_time = time.time() - start_time

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)

# Save submission
np.save('advanced_submission.npy', submission)
print(f"\nSubmission saved to: advanced_submission.npy")

# Display detailed scores
print("\n" + "="*60)
print("DETAILED RESULTS")
print("="*60)
print("\nIndividual Hypervolumes:")
for i, hv in enumerate(all_hypervolumes):
    print(f"  Curve {i+1}: {hv:.4f}")

overall_score = np.mean(all_hypervolumes)
print(f"\n{'='*60}")
print(f"OVERALL SCORE: {overall_score:.4f}")
print(f"{'='*60}")

print(f"\nOptimization time: {elapsed_time/60:.2f} minutes")
print(f"Average time per curve: {elapsed_time/6/60:.2f} minutes")

# Try to evaluate with the built-in function
print("\nVerifying with evaluate_submission...")
try:
    evaluate_submission(submission)
except:
    pass

print(f"\n✅ Done! Your overall score is {overall_score:.4f}")
print("Upload 'advanced_submission.npy' to the leaderboard")