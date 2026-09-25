#!/usr/bin/env python3
"""
PyAPX - Python Toolkit for Atomic Configuration Pattern Exploration

A toolkit that integrates DFT codes with Bayesian optimization to explore stable atomic configurations in materials.
"""

import os
import sys
from .utils import read_card, read_energy_evaluator, read_sampling_sequence, get_current_sample_id, apx_print


def main():
    """
    Main execution function
    """
    apx_print("=== PyAPX - Python Toolkit for Atomic Configuration Pattern Exploration ===")
    
    # Read energy evaluator
    energy_evaluator = read_energy_evaluator()
    apx_print(f"Energy evaluator: {energy_evaluator}")
    
    # Read optimizer
    from .utils import read_optimizer
    optimizer = read_optimizer()
    apx_print(f"Optimizer: {optimizer}")
    if optimizer.lower() == "physbo":
        from .utils import read_physbo_centering_setting
        apx_print(f"PHYSBO centering: {read_physbo_centering_setting()}")
    
    # Read encoding setting
    from .utils import read_encoding_setting
    encoding = read_encoding_setting()
    apx_print(f"Encoding: {encoding}")
    
    # Read sampling sequence
    tasks = read_sampling_sequence()
    
    if not tasks:
        apx_print("Sampling sequence not found. Please check apx.in.")
        return
    
    apx_print("Scheduled tasks:")
    for i, (task_type, num_iterations) in enumerate(tasks, 1):
        apx_print(f"  {i}. {task_type}: {num_iterations} iterations")
    
    # Initialize sample ID
    current_sample_id = get_current_sample_id()
    
    # Check if any sampling is scheduled
    total_iterations = sum(num_iterations for _, num_iterations in tasks)
    
    # Perform encoding if requested
    if encoding:
        from .sequential_explorer import run_encoding
        success = run_encoding()
        if not success:
            apx_print("Encoding failed. Exiting.")
            return
    
    # If no sampling is scheduled, exit after encoding
    if total_iterations == 0:
        apx_print("No sampling scheduled. Exiting after encoding.")
        return
    
    # Execute sampling tasks sequentially
    for task_type, num_iterations in tasks:
        if num_iterations == 0:
            continue  # Skip tasks with 0 iterations
            
        print()
        apx_print(f"=== Executing {task_type} ===")
        apx_print(f"Iterations: {num_iterations}")
        
        if task_type == "RANDOM_SAMPLING":
            sampling_type = "random"
        elif task_type == "BAYES_SAMPLING":
            sampling_type = "bayes"
        else:
            apx_print(f"Unsupported task type: {task_type}")
            continue
        
        # Execute sampling loop
        from .sequential_explorer import run_sampling_loop
        run_sampling_loop(num_iterations, sampling_type, current_sample_id, energy_evaluator)
        current_sample_id += num_iterations
    
    apx_print("=== All tasks completed ===")

if __name__ == "__main__":
    main() 