#!/usr/bin/env python3
"""
Example usage of the Advanced Mechanism Synthesis framework.
"""

from advanced_mechanism_synthesis import AdvancedMechanismSynthesis, OptimizationConfig
import numpy as np


def example_basic_optimization():
    """
    Example 1: Basic optimization with default settings.
    """
    print("=== Example 1: Basic Optimization ===")
    
    # Create configuration with default settings
    config = OptimizationConfig(
        device='cpu',
        population_size=50,  # Smaller for faster demo
        num_generations=50,
        mechanism_size=6,
        random_seed=42
    )
    
    # Initialize synthesis framework
    synthesis = AdvancedMechanismSynthesis(config)
    
    # Load target curves
    synthesis.load_target_curves()
    
    # Run optimization for the first target curve
    results = synthesis.run_complete_optimization(
        target_curve_idx=0,
        use_initial_population=True,
        run_gradient_opt=False  # Skip gradient opt for faster demo
    )
    
    print(f"Results: {results}")


def example_custom_configuration():
    """
    Example 2: Custom configuration for specific requirements.
    """
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create custom configuration
    config = OptimizationConfig(
        device='cpu',
        population_size=80,
        num_generations=75,
        mechanism_size=8,  # Larger mechanism
        mutation_probability=0.7,  # Higher mutation rate
        max_distance=0.5,  # Stricter distance constraint
        max_material=8.0,  # Stricter material constraint
        gradient_step_size=2e-4,  # Smaller gradient steps
        random_seed=123
    )
    
    # Initialize synthesis framework
    synthesis = AdvancedMechanismSynthesis(config)
    
    # Load target curves
    synthesis.load_target_curves()
    
    # Visualize target curves
    synthesis.visualize_target_curves()
    
    # Run optimization for target curve 2
    results = synthesis.run_complete_optimization(
        target_curve_idx=2,
        use_initial_population=True,
        run_gradient_opt=True
    )
    
    return results


def example_batch_optimization():
    """
    Example 3: Batch optimization for multiple target curves.
    """
    print("\n=== Example 3: Batch Optimization ===")
    
    config = OptimizationConfig(
        device='cpu',
        population_size=60,
        num_generations=40,
        mechanism_size=7,
        random_seed=456
    )
    
    synthesis = AdvancedMechanismSynthesis(config)
    synthesis.load_target_curves()
    
    batch_results = {}
    
    # Optimize for multiple target curves
    for curve_idx in [0, 1, 3]:  # Optimize for curves 1, 2, and 4
        print(f"\nOptimizing for target curve {curve_idx + 1}...")
        
        results = synthesis.run_complete_optimization(
            target_curve_idx=curve_idx,
            use_initial_population=True,
            run_gradient_opt=True
        )
        
        batch_results[f'curve_{curve_idx + 1}'] = results
    
    # Print summary
    print("\n=== Batch Results Summary ===")
    for curve_name, results in batch_results.items():
        if 'final_hypervolume' in results:
            hv = results['final_hypervolume']
            improvement = results.get('hypervolume_improvement', 0)
            print(f"{curve_name}: HV={hv:.4f}, Improvement={improvement:.4f}")
        else:
            print(f"{curve_name}: Optimization failed")
    
    return batch_results


def example_analysis_only():
    """
    Example 4: Analysis and visualization without optimization.
    """
    print("\n=== Example 4: Analysis Only ===")
    
    config = OptimizationConfig(device='cpu')
    synthesis = AdvancedMechanismSynthesis(config)
    
    # Load and visualize target curves
    curves = synthesis.load_target_curves()
    print(f"Loaded {len(curves)} target curves")
    
    # Generate some random mechanisms for analysis
    mechanisms = synthesis.generate_initial_population(20)
    
    # Evaluate their performance
    for i, curve in enumerate(curves[:3]):  # Analyze first 3 curves
        best_dist, best_mat = synthesis.evaluate_population_performance(mechanisms, curve)
        print(f"Curve {i+1}: Best Distance={best_dist:.4f}, Best Material={best_mat:.4f}")


def main():
    """Run all examples."""
    try:
        # Run basic example
        example_basic_optimization()
        
        # Run custom configuration example
        results = example_custom_configuration()
        
        # Run batch optimization
        batch_results = example_batch_optimization()
        
        # Run analysis only
        example_analysis_only()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all required dependencies are installed and target_curves.npy exists.")


if __name__ == "__main__":
    main()
