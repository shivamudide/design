#!/usr/bin/env python3
"""
Advanced Mechanism Synthesis Optimization
It converts the functionality from the Advanced_Starter_Notebook.ipynb into a modular,
reusable Python structure.

Key Features:
- Multi-objective optimization (distance and material usage)
- Mixed variable genetic algorithm with NSGA-II
- Gradient-based post-processing
- Mechanism visualization and analysis
- Hypervolume calculation and comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from tqdm.auto import tqdm, trange

# Optimization imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer, Binary
from pymoo.core.mixed import (
    MixedVariableMating, 
    MixedVariableGA, 
    MixedVariableSampling, 
    MixedVariableDuplicateElimination
)
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import Sampling
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# LINKS imports
from LINKS.Optimization import DifferentiableTools, Tools, MechanismRandomizer
from LINKS.Visualization import MechanismVisualizer, GAVisualizer
from LINKS.Kinematics import MechanismSolver
from LINKS.Geometry import CurveEngine
from LINKS.CP import make_empty_submission, evaluate_submission


@dataclass
class OptimizationConfig:
    """Configuration class for optimization parameters."""
    
    # Environment settings
    device: str = 'cpu'
    random_seed: int = 0
    
    # GA parameters
    population_size: int = 100
    num_generations: int = 100
    mutation_probability: float = 0.5
    
    # Mechanism parameters
    mechanism_size: int = 7
    min_mechanism_size: int = 6
    max_mechanism_size: int = 14
    
    # Constraints
    max_distance: float = 0.75
    max_material: float = 10.0
    
    # Gradient optimization parameters
    gradient_step_size: float = 4e-4
    gradient_steps: int = 1000
    
    def __post_init__(self):
        """Set up environment after initialization."""
        if self.device == 'cpu':
            os.environ["JAX_PLATFORMS"] = "cpu"
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)


class MechanismSynthesisProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for mechanism synthesis.
    
    This class defines the optimization problem with mixed variable types
    including connectivity matrix, node positions, fixed nodes, and target node.
    """
    
    def __init__(self, target_curve: np.ndarray, config: OptimizationConfig):
        """
        Initialize the mechanism synthesis problem.
        
        Args:
            target_curve: Target curve to match (N x 2 array of points)
            config: Optimization configuration
        """
        self.N = config.mechanism_size
        self.target_curve = target_curve
        self.config = config
        
        # Initialize optimization tools
        self.tools = Tools(device=config.device)
        self.tools.compile()
        
        variables = self._create_variables()
        super().__init__(vars=variables, n_obj=2, n_constr=2)
    
    def _create_variables(self) -> Dict[str, Any]:
        """Create the variable dictionary for the optimization problem."""
        variables = {}
        N = self.N
        
        # Connectivity matrix variables (upper triangular, excluding diagonal)
        for i in range(N):
            for j in range(i):
                variables[f"C{j}_{i}"] = Binary()
        
        # Remove C0_1 since we know node 1 is connected to the motor
        if "C0_1" in variables:
            del variables["C0_1"]
        
        # Position variables (N x 2 real coordinates between 0 and 1)
        for i in range(2 * N):
            variables[f"X0{i}"] = Real(bounds=(0.0, 1.0))
        
        # Node type variables (fixed vs non-fixed)
        for i in range(N):
            variables[f"fixed_nodes{i}"] = Binary()
        
        # Target node (any node except motor node)
        variables["target"] = Integer(bounds=(1, N-1))
        
        return variables
    
    def convert_1D_to_mech(self, x: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Convert 1D optimization variables to mechanism representation.
        
        Args:
            x: Dictionary of optimization variables
            
        Returns:
            Tuple of (positions, edges, fixed_joints, motor, target_idx)
        """
        N = self.N
        target_idx = x["target"]
        
        # Build connectivity matrix
        C = np.zeros((N, N))
        x["C0_1"] = 1  # Motor connection is fixed
        
        for i in range(N):
            for j in range(i):
                C[j, i] = x[f"C{j}_{i}"]
        
        edges = np.array(np.where(C == 1)).T
        
        # Reshape position matrix
        x0 = np.array([x[f"X0{i}"] for i in range(2 * N)]).reshape([N, 2])
        
        # Extract fixed joints
        fixed_joints = np.where(np.array([x[f"fixed_nodes{i}"] for i in range(N)]))[0].astype(int)
        
        # Motor is fixed as [0, 1]
        motor = np.array([0, 1])
        
        return x0, edges, fixed_joints, motor, target_idx
    
    def convert_mech_to_1D(self, x0: np.ndarray, edges: np.ndarray, 
                          fixed_joints: np.ndarray, target_idx: Optional[int] = None, 
                          **kwargs) -> Dict[str, Any]:
        """
        Convert mechanism representation to 1D optimization variables.
        
        Args:
            x0: Node positions (N x 2)
            edges: Edge connectivity (E x 2)
            fixed_joints: Array of fixed joint indices
            target_idx: Target joint index
            
        Returns:
            Dictionary of optimization variables
        """
        N = self.N
        x = {}
        
        # Target node
        if target_idx is None:
            target_idx = x0.shape[0] - 1
        x["target"] = target_idx
        
        # Connectivity matrix
        C = np.zeros((N, N), dtype=bool)
        C[edges[:, 0], edges[:, 1]] = 1
        C[edges[:, 1], edges[:, 0]] = 1
        
        for i in range(N):
            for j in range(i):
                x[f"C{j}_{i}"] = C[i, j]
        
        if "C0_1" in x:
            del x["C0_1"]
        
        # Position matrix
        if x0.shape[0] != N:
            x0 = np.pad(x0, ((0, N - x0.shape[0]), (0, 0)), 'constant', constant_values=0)
        
        for i in range(2 * N):
            x[f"X0{i}"] = x0.flatten()[i]
        
        # Fixed nodes
        for i in range(N):
            x[f"fixed_nodes{i}"] = (i in fixed_joints) or (i >= N)
        
        return x
    
    def _evaluate(self, x: Dict[str, Any], out: Dict[str, Any], *args, **kwargs):
        """Evaluate the optimization objectives and constraints."""
        # Convert to mechanism representation
        x0, edges, fixed_joints, motor, target_idx = self.convert_1D_to_mech(x)
        
        # Simulate mechanism
        distance, material = self.tools(
            x0, edges, fixed_joints, motor, 
            self.target_curve, target_idx=target_idx
        )
        
        # Set objectives and constraints
        out["F"] = np.array([distance, material])
        out["G"] = out["F"] - np.array([self.config.max_distance, self.config.max_material])


class CustomSampling(Sampling):
    """Custom sampling class for initializing GA with pre-generated mechanisms."""
    
    def __init__(self, initial_population: List[Dict[str, Any]]):
        """Initialize with a list of mechanism dictionaries."""
        super().__init__()
        self.initial_population = initial_population
    
    def _do(self, problem: MechanismSynthesisProblem, n_samples: int, **kwargs) -> np.ndarray:
        """Sample from the initial population."""
        return np.array([
            self.initial_population[i % len(self.initial_population)] 
            for i in range(n_samples)
        ])


class AdvancedMechanismSynthesis:
    """
    Main class for advanced mechanism synthesis optimization.
    
    This class orchestrates the entire optimization process including:
    - GA optimization with mixed variables
    - Gradient-based post-processing
    - Visualization and analysis
    - Submission generation
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize the synthesis framework.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.target_curves = None
        self.results = None
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all necessary components."""
        self.randomizer = MechanismRandomizer(
            min_size=self.config.min_mechanism_size,
            max_size=self.config.max_mechanism_size,
            device=self.config.device
        )
        
        self.visualizer = MechanismVisualizer()
        self.ga_visualizer = GAVisualizer()
        self.solver = MechanismSolver(device=self.config.device)
        self.curve_engine = CurveEngine(device=self.config.device)
        
        self.diff_tools = DifferentiableTools(device=self.config.device)
        self.diff_tools.compile()
    
    def load_target_curves(self, filepath: str = 'target_curves.npy') -> np.ndarray:
        """
        Load target curves from file.
        
        Args:
            filepath: Path to the target curves file
            
        Returns:
            Array of target curves
        """
        self.target_curves = np.load(filepath)
        return self.target_curves
    
    def visualize_target_curves(self, save_path: Optional[str] = None):
        """
        Visualize all target curves in a 2x3 subplot.
        
        Args:
            save_path: Optional path to save the figure
        """
        if self.target_curves is None:
            raise ValueError("Target curves not loaded. Call load_target_curves() first.")
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        
        for i in range(6):
            x_coords = self.target_curves[i][:, 0]
            y_coords = self.target_curves[i][:, 1]
            
            row, col = i // 3, i % 3
            axs[row, col].plot(x_coords, y_coords, color='black', linewidth=3)
            axs[row, col].set_title(f'Target Curve {i + 1}')
            axs[row, col].axis('equal')
            axs[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_initial_population(self, size: int) -> List[Dict[str, Any]]:
        """
        Generate initial population of mechanisms.
        
        Args:
            size: Number of mechanisms to generate
            
        Returns:
            List of mechanism dictionaries
        """
        print(f"Generating {size} initial mechanisms...")
        mechanisms = []
        
        for _ in trange(size, desc="Generating mechanisms"):
            mech = self.randomizer(n=self.config.mechanism_size)
            mechanisms.append(mech)
        
        return mechanisms
    
    def evaluate_population_performance(self, mechanisms: List[Dict[str, Any]], 
                                      target_curve: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the performance of a population.
        
        Args:
            mechanisms: List of mechanism dictionaries
            target_curve: Target curve to evaluate against
            
        Returns:
            Tuple of (best_distance, best_material)
        """
        problem = MechanismSynthesisProblem(target_curve, self.config)
        population_1D = [problem.convert_mech_to_1D(**mech) for mech in mechanisms]
        
        F = problem.evaluate(np.array(population_1D))[0]
        
        best_distance = F[:, 0].min()
        best_material = F[:, 1].min()
        
        return best_distance, best_material
    
    def run_ga_optimization(self, target_curve_idx: int, 
                           initial_mechanisms: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Run genetic algorithm optimization.
        
        Args:
            target_curve_idx: Index of target curve to optimize for
            initial_mechanisms: Optional initial population
            
        Returns:
            Optimization results
        """
        if self.target_curves is None:
            raise ValueError("Target curves not loaded. Call load_target_curves() first.")
        
        target_curve = self.target_curves[target_curve_idx]
        problem = MechanismSynthesisProblem(target_curve, self.config)
        
        # Setup algorithm
        if initial_mechanisms is not None:
            initial_population = [problem.convert_mech_to_1D(**mech) for mech in initial_mechanisms]
            sampling = CustomSampling(initial_population)
            
            # Evaluate initial population
            best_dist, best_mat = self.evaluate_population_performance(
                initial_mechanisms, target_curve
            )
            print(f'Initial population - Best Distance: {best_dist:.4f}, Best Material: {best_mat:.4f}')
        else:
            sampling = MixedVariableSampling()
        
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            sampling=sampling,
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            mutation=PolynomialMutation(prob=self.config.mutation_probability),
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )
        
        print(f"Running GA optimization for target curve {target_curve_idx + 1}...")
        
        results = minimize(
            problem,
            algorithm,
            ('n_gen', self.config.num_generations),
            verbose=True,
            save_history=True,
            seed=self.config.random_seed
        )
        
        self.results = results
        return results
    
    def analyze_ga_results(self) -> Optional[float]:
        """
        Analyze GA optimization results and calculate hypervolume.
        
        Returns:
            Hypervolume value if solutions found, None otherwise
        """
        if self.results is None or self.results.X is None:
            print("No feasible solutions found!")
            return None
        
        ref_point = np.array([self.config.max_distance, self.config.max_material])
        ind = HV(ref_point)
        hypervolume = ind(self.results.F)
        
        print(f'Hypervolume: {hypervolume:.6f}')
        
        # Plot results
        self.ga_visualizer.plot_HV(
            self.results.F, 
            ref_point, 
            objective_labels=['Distance', 'Material']
        )
        
        return hypervolume
    
    def visualize_best_solutions(self, target_curve_idx: int):
        """
        Visualize the best solutions for both objectives.
        
        Args:
            target_curve_idx: Index of target curve
        """
        if self.results is None or self.results.X is None:
            print("No solutions to visualize!")
            return
        
        if self.target_curves is None:
            raise ValueError("Target curves not loaded.")
        
        target_curve = self.target_curves[target_curve_idx]
        problem = MechanismSynthesisProblem(target_curve, self.config)
        
        # Best distance solution
        best_dist_idx = np.argmin(self.results.F[:, 0])
        x0, edges, fixed_joints, motor, target_idx = problem.convert_1D_to_mech(
            self.results.X[best_dist_idx]
        )
        
        print(f"\nBest Distance Solution:")
        print(f"Distance: {self.results.F[best_dist_idx, 0]:.4f}, Material: {self.results.F[best_dist_idx, 1]:.4f}")
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Best Distance Solution")
        self.visualizer(x0, edges, fixed_joints, motor, highlight=target_idx, ax=plt.gca())
        
        traced_curve = self.solver(x0, edges, fixed_joints, motor)[target_idx]
        self.curve_engine.visualize_comparison(traced_curve, target_curve)
        
        # Best material solution
        best_mat_idx = np.argmin(self.results.F[:, 1])
        x0, edges, fixed_joints, motor, target_idx = problem.convert_1D_to_mech(
            self.results.X[best_mat_idx]
        )
        
        print(f"\nBest Material Solution:")
        print(f"Distance: {self.results.F[best_mat_idx, 0]:.4f}, Material: {self.results.F[best_mat_idx, 1]:.4f}")
        
        plt.subplot(1, 2, 2)
        plt.title("Best Material Solution")
        self.visualizer(x0, edges, fixed_joints, motor, ax=plt.gca())
        
        traced_curve = self.solver(x0, edges, fixed_joints, motor)[target_idx]
        self.curve_engine.visualize_comparison(traced_curve, target_curve)
        
        plt.tight_layout()
        plt.show()
    
    def run_gradient_optimization(self, target_curve_idx: int) -> Tuple[List[np.ndarray], float]:
        """
        Run gradient-based optimization on GA results.
        
        Args:
            target_curve_idx: Index of target curve
            
        Returns:
            Tuple of (optimized_positions, new_hypervolume)
        """
        if self.results is None or self.results.X is None:
            raise ValueError("No GA results available for gradient optimization.")
        
        if self.target_curves is None:
            raise ValueError("Target curves not loaded.")
        
        target_curve = self.target_curves[target_curve_idx]
        problem = MechanismSynthesisProblem(target_curve, self.config)
        
        # Extract mechanism data from GA results
        x0s, edges, fixed_joints, motors, target_idxs = [], [], [], [], []
        
        for i in range(self.results.X.shape[0]):
            x0, edge, fixed, motor, target_idx = problem.convert_1D_to_mech(self.results.X[i])
            x0s.append(x0)
            edges.append(edge)
            fixed_joints.append(fixed)
            motors.append(motor)
            target_idxs.append(target_idx)
        
        print("Running gradient-based optimization...")
        
        # Gradient optimization loop
        x = x0s.copy()
        done_optimizing = np.zeros(len(x), dtype=bool)
        x_last = x.copy()
        
        for step in trange(self.config.gradient_steps, desc="Gradient optimization"):
            # Get gradients
            distances, materials, distance_grads, material_grads = self.diff_tools(
                x, edges, fixed_joints, motors, target_curve, target_idxs
            )
            
            # Check validity
            valid_mask = np.logical_and(distances <= self.config.max_distance, 
                                      materials <= self.config.max_material)
            valids = np.where(valid_mask)[0]
            invalids = np.where(~valid_mask)[0]
            
            # Revert invalid solutions
            for i in invalids:
                done_optimizing[i] = True
                x[i] = x_last[i]
            
            x_last = x.copy()
            
            # Update valid solutions
            for i in valids:
                if not done_optimizing[i]:
                    x[i] = x[i] - self.config.gradient_step_size * distance_grads[i]
            
            if np.all(done_optimizing):
                print(f'All members done optimizing at step {step}')
                break
        
        # Calculate combined hypervolume
        combined_x0s = x0s + x
        combined_edges = edges + edges
        combined_fixed_joints = fixed_joints + fixed_joints
        combined_motors = motors + motors
        combined_target_idxs = target_idxs + target_idxs
        
        F_combined = np.array(problem.tools(
            combined_x0s, combined_edges, combined_fixed_joints, 
            combined_motors, target_curve, combined_target_idxs
        )).T
        
        F_original = np.array(problem.tools(
            x0s, edges, fixed_joints, motors, target_curve, target_idxs
        )).T
        
        ref_point = np.array([self.config.max_distance, self.config.max_material])
        ind = HV(ref_point)
        
        hv_original = ind(F_original)
        hv_combined = ind(F_combined)
        
        print(f'Hypervolume before gradient opt: {hv_original:.4f}, after: {hv_combined:.4f}')
        
        # Visualize comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title('Before Gradient Optimization')
        self.ga_visualizer.plot_HV(F_original, ref_point, 
                                  objective_labels=['Distance', 'Material'], ax=plt.gca())
        
        plt.subplot(1, 2, 2)
        plt.title('After Gradient Optimization')
        self.ga_visualizer.plot_HV(F_combined, ref_point, 
                                  objective_labels=['Distance', 'Material'], ax=plt.gca())
        
        plt.tight_layout()
        plt.show()
        
        return x, hv_combined
    
    def create_submission(self, target_curve_idx: int) -> Dict[str, Any]:
        """
        Create a submission dictionary from optimization results.
        
        Args:
            target_curve_idx: Index of target curve
            
        Returns:
            Submission dictionary
        """
        if self.results is None or self.results.X is None:
            raise ValueError("No optimization results available for submission.")
        
        target_curve = self.target_curves[target_curve_idx]
        problem = MechanismSynthesisProblem(target_curve, self.config)
        
        submission = make_empty_submission()
        
        for i in range(self.results.X.shape[0]):
            x0, edges, fixed_joints, motor, target_idx = problem.convert_1D_to_mech(
                self.results.X[i]
            )
            
            mech = {
                'x0': x0,
                'edges': edges,
                'fixed_joints': fixed_joints,
                'motor': motor,
                'target_joint': target_idx
            }
            
            submission['Problem 2'].append(mech)
        
        return submission
    
    def run_complete_optimization(self, target_curve_idx: int, 
                                 use_initial_population: bool = True,
                                 run_gradient_opt: bool = True) -> Dict[str, Any]:
        """
        Run the complete optimization pipeline.
        
        Args:
            target_curve_idx: Index of target curve to optimize for
            use_initial_population: Whether to generate initial population
            run_gradient_opt: Whether to run gradient optimization
            
        Returns:
            Dictionary with optimization results and metrics
        """
        results_summary = {}
        
        # Load target curves if not already loaded
        if self.target_curves is None:
            self.load_target_curves()
        
        # Generate initial population if requested
        initial_mechanisms = None
        if use_initial_population:
            initial_mechanisms = self.generate_initial_population(self.config.population_size)
            results_summary['initial_population_size'] = len(initial_mechanisms)
        
        # Run GA optimization
        ga_results = self.run_ga_optimization(target_curve_idx, initial_mechanisms)
        ga_hypervolume = self.analyze_ga_results()
        results_summary['ga_hypervolume'] = ga_hypervolume
        
        if ga_results.X is not None:
            # Visualize best solutions
            self.visualize_best_solutions(target_curve_idx)
            
            # Run gradient optimization if requested
            if run_gradient_opt:
                optimized_positions, final_hypervolume = self.run_gradient_optimization(target_curve_idx)
                results_summary['final_hypervolume'] = final_hypervolume
                results_summary['hypervolume_improvement'] = final_hypervolume - (ga_hypervolume or 0)
            
            # Create submission
            submission = self.create_submission(target_curve_idx)
            results_summary['submission'] = submission
            
            print("\nOptimization completed successfully!")
            print(f"Final results: {results_summary}")
        else:
            print("Optimization failed to find feasible solutions.")
            results_summary['status'] = 'failed'
        
        return results_summary


def main():
    """
    Main function demonstrating the usage of the AdvancedMechanismSynthesis class.
    """
    # Create configuration
    config = OptimizationConfig(
        device='cpu',
        population_size=100,
        num_generations=100,
        mechanism_size=7,
        random_seed=123
    )
    
    # Initialize synthesis framework
    synthesis = AdvancedMechanismSynthesis(config)
    
    # Load and visualize target curves
    synthesis.load_target_curves()
    synthesis.visualize_target_curves()
    
    # Run optimization for target curve 1 (index 1)
    results = synthesis.run_complete_optimization(
        target_curve_idx=1,
        use_initial_population=True,
        run_gradient_opt=True
    )
    
    # Evaluate submission
    if 'submission' in results:
        print("\nEvaluating submission...")
        evaluate_submission(results['submission'])
    
    print("\nOptimization pipeline completed!")


if __name__ == "__main__":
    main()
