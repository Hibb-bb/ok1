#!/usr/bin/env python3
"""
Test script to check:
1. Whether real-world data uses Exp map to hyperbolic space
2. Whether MHN requires normalized patterns
"""

import numpy as np
from scipy.special import softmax
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_recall import update_mhn

def check_real_world_exp_map():
    """
    Check if real-world data uses Exp map to hyperbolic space.
    """
    print("=" * 70)
    print("QUESTION 1: Checking if real-world data uses Exp map to hyperbolic space")
    print("=" * 70)
    
    print("\nCode Analysis:")
    print("\n1. In baseline_recall.py (run_recall_mhn and run_recall_dam):")
    print("   - Real-world data (MNIST/CIFAR10) uses sample_images_from_dataset()")
    print("   - Returns Euclidean vectors (normalized to [0,1], optionally PCA-reduced)")
    print("   - NO exp map is applied")
    print("   - Used directly in Euclidean space for MHN/DAM")
    
    print("\n2. In memory_recall.py (run_recall_hyperbolic):")
    print("   - Only uses synthetic hyperbolic data via sample_hyperboloid_points_from_tangent_ball()")
    print("   - Does NOT support real-world datasets (mnist/cifar10)")
    print("   - Uses exp map only for synthetic hyperbolic data")
    
    print("\n✓ CONCLUSION: Real-world data is NOT mapped to hyperbolic space via Exp map.")
    print("  It is used directly in Euclidean space for MHN/DAM models.\n")

def test_mhn_normalization():
    """
    Test if MHN requires normalized patterns by comparing performance
    with normalized vs unnormalized memory patterns.
    """
    print("=" * 70)
    print("QUESTION 2: Testing if MHN requires normalized patterns")
    print("=" * 70)
    
    rng = np.random.default_rng(42)
    n_trials = 20
    M = 20  # number of memory patterns
    dim = 30  # dimension
    max_steps = 15
    tol = 0.05
    
    results_normalized = []
    results_unnormalized = []
    
    print(f"\nTesting with {n_trials} trials, M={M}, dim={dim}")
    print(f"Convergence tolerance: {tol}, Max steps: {max_steps}\n")
    
    for trial in range(n_trials):
        # Create memory patterns with varying norms (not normalized)
        memory_unnorm = rng.normal(scale=2.0, size=(M, dim))
        # Scale each pattern by a random factor (1x to 10x) to create varying norms
        scales = rng.uniform(1.0, 10.0, size=M)
        memory_unnorm = memory_unnorm * scales[:, None]
        
        # Normalized version (each pattern has unit norm)
        norms_unnorm = np.linalg.norm(memory_unnorm, axis=1, keepdims=True)
        memory_norm = memory_unnorm / norms_unnorm
        
        # Check norm statistics
        if trial == 0:
            print(f"Sample norm statistics (trial 0):")
            print(f"  Unnormalized: mean={np.mean(norms_unnorm):.2f}, "
                  f"std={np.std(norms_unnorm):.2f}, "
                  f"min={np.min(norms_unnorm):.2f}, max={np.max(norms_unnorm):.2f}")
            norms_norm = np.linalg.norm(memory_norm, axis=1)
            print(f"  Normalized: mean={np.mean(norms_norm):.2f}, "
                  f"std={np.std(norms_norm):.6f}, "
                  f"min={np.min(norms_norm):.2f}, max={np.max(norms_norm):.2f}\n")
        
        # Test with normalized patterns
        target_idx = 0
        target_norm = memory_norm[target_idx].copy()
        query_norm = target_norm + rng.normal(scale=0.3, size=target_norm.shape)
        target_norm_norm = np.linalg.norm(target_norm)
        
        # Run MHN updates
        for step in range(max_steps):
            query_norm = update_mhn(query_norm, memory_norm, beta=1.0)
            dist = np.linalg.norm(query_norm - target_norm)
            if dist < tol:
                break
        rel_dist_norm = dist / target_norm_norm if target_norm_norm > 0 else dist
        results_normalized.append((dist, rel_dist_norm))
        
        # Test with unnormalized patterns
        target_unnorm = memory_unnorm[target_idx].copy()
        query_unnorm = target_unnorm + rng.normal(scale=0.3, size=target_unnorm.shape)
        target_unnorm_norm = np.linalg.norm(target_unnorm)
        
        # Run MHN updates
        for step in range(max_steps):
            query_unnorm = update_mhn(query_unnorm, memory_unnorm, beta=1.0)
            dist = np.linalg.norm(query_unnorm - target_unnorm)
            if dist < tol:
                break
        rel_dist_unnorm = dist / target_unnorm_norm if target_unnorm_norm > 0 else dist
        results_unnormalized.append((dist, rel_dist_unnorm))
    
    # Unpack results
    dists_norm = [r[0] for r in results_normalized]
    rel_dists_norm = [r[1] for r in results_normalized]
    dists_unnorm = [r[0] for r in results_unnormalized]
    rel_dists_unnorm = [r[1] for r in results_unnormalized]
    
    print("\nResults Summary:")
    print("-" * 70)
    print(f"Normalized patterns - Absolute distance to target:")
    print(f"  Mean: {np.mean(dists_norm):.6f}, Std: {np.std(dists_norm):.6f}")
    print(f"  Min: {np.min(dists_norm):.6f}, Max: {np.max(dists_norm):.6f}")
    print(f"Normalized patterns - Relative distance (dist/norm):")
    print(f"  Mean: {np.mean(rel_dists_norm):.6f}, Std: {np.std(rel_dists_norm):.6f}")
    print(f"  Min: {np.min(rel_dists_norm):.6f}, Max: {np.max(rel_dists_norm):.6f}")
    
    print(f"\nUnnormalized patterns - Absolute distance to target:")
    print(f"  Mean: {np.mean(dists_unnorm):.6f}, Std: {np.std(dists_unnorm):.6f}")
    print(f"  Min: {np.min(dists_unnorm):.6f}, Max: {np.max(dists_unnorm):.6f}")
    print(f"Unnormalized patterns - Relative distance (dist/norm):")
    print(f"  Mean: {np.mean(rel_dists_unnorm):.6f}, Std: {np.std(rel_dists_unnorm):.6f}")
    print(f"  Min: {np.min(rel_dists_unnorm):.6f}, Max: {np.max(rel_dists_unnorm):.6f}")
    
    ratio_abs = np.mean(dists_unnorm) / np.mean(dists_norm)
    ratio_rel = np.mean(rel_dists_unnorm) / np.mean(rel_dists_norm)
    print(f"\nPerformance ratio (unnormalized/normalized):")
    print(f"  Absolute distance: {ratio_abs:.3f}")
    print(f"  Relative distance: {ratio_rel:.3f}")
    
    # Check convergence (using relative distance for fair comparison)
    rel_tol = 0.1  # 10% relative error
    success_norm = sum(1 for r in rel_dists_norm if r < rel_tol)
    success_unnorm = sum(1 for r in rel_dists_unnorm if r < rel_tol)
    
    print(f"\nConvergence analysis (relative distance < {rel_tol} = {100*rel_tol:.0f}%):")
    print(f"  Normalized: {success_norm}/{n_trials} ({100*success_norm/n_trials:.1f}% success rate)")
    print(f"  Unnormalized: {success_unnorm}/{n_trials} ({100*success_unnorm/n_trials:.1f}% success rate)")
    
    print("\n" + "=" * 70)
    if ratio_rel > 2.0:
        print("✓ CONCLUSION: MHN performs SIGNIFICANTLY BETTER with normalized patterns")
        print("  Normalization appears to be important for MHN to work effectively.")
    elif ratio_rel > 1.5:
        print("⚠ CONCLUSION: MHN works but performs better with normalized patterns")
        print("  Normalization improves performance but is not strictly required.")
    elif ratio_rel < 0.7 or success_unnorm > success_norm * 1.5:
        print("ℹ CONCLUSION: MHN works WITHOUT normalization - may even perform better!")
        print("  This simulation shows MHN converges well with unnormalized patterns.")
        print("  Normalization is NOT strictly required for MHN to function.")
        print("  (Note: normalization may still be beneficial for other reasons,")
        print("   such as numerical stability, theoretical guarantees, etc.)")
    else:
        print("✗ CONCLUSION: MHN works reasonably well even without normalization")
        print("  Normalization may not be strictly necessary for MHN.")
    print("=" * 70)

if __name__ == "__main__":
    check_real_world_exp_map()
    print("\n")
    test_mhn_normalization()
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
