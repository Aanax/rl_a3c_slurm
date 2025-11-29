#!/usr/bin/env python3
"""
Script to plot s vector analysis for a single monitoring file.
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_s_analysis(s_values, game_num, rank, save_path=None):
    """Plot the analysis of s values for a game."""
    if not s_values:
        return

    s_tensors = [s.cpu().numpy() for s in s_values]
    s_flat = np.concatenate([s.flatten() for s in s_tensors])

    plt.figure(figsize=(15, 10))

    # 1. Time series of means through game
    plt.subplot(2, 3, 1)
    step_means = [s.mean().item() for s in s_tensors]
    plt.plot(step_means, linewidth=1)
    plt.title(f'Mean S through Game {game_num}, Rank {rank}')
    plt.xlabel('Step')
    plt.ylabel('Mean S')
    plt.grid(True, alpha=0.3)

    # 2. Time series of stds through game
    plt.subplot(2, 3, 2)
    step_stds = [s.std().item() for s in s_tensors]
    plt.plot(step_stds, linewidth=1, color='orange')
    plt.title(f'Std S through Game {game_num}, Rank {rank}')
    plt.xlabel('Step')
    plt.ylabel('Std S')
    plt.grid(True, alpha=0.3)

    # 3. S distribution (all values)
    plt.subplot(2, 3, 3)
    plt.hist(s_flat, bins=100, alpha=0.7, edgecolor='black', density=True)
    plt.title(f'S Distribution - Game {game_num}, Rank {rank}')
    plt.xlabel('S Value')
    plt.ylabel('Density')

    # 4. Random s vector distribution at random step
    plt.subplot(2, 3, 4)
    if s_tensors:
        random_idx = np.random.randint(len(s_tensors))
        random_s = s_tensors[random_idx]
        plt.hist(random_s.flatten(), bins=50, alpha=0.7, edgecolor='black', color='green', density=True)
        plt.title(f'Random S Vector (Step {random_idx})')
        plt.xlabel('S Value')
        plt.ylabel('Density')

    # 5. Box plot of s values per step (sample of steps)
    plt.subplot(2, 3, 5)
    sample_size = min(100, len(s_tensors))
    sample_indices = np.random.choice(len(s_tensors), sample_size, replace=False)
    sample_s = [s_tensors[i].flatten() for i in sorted(sample_indices)]
    plt.boxplot(sample_s, showfliers=False)
    plt.title('S Values Distribution per Step (Sample)')
    plt.xlabel('Step Index')
    plt.ylabel('S Value')
    plt.xticks([])  # Too many ticks

    # 6. Correlation between consecutive s vectors (first 100 dims for visualization)
    plt.subplot(2, 3, 6)
    if len(s_tensors) > 1:
        s_flat_list = [s.flatten()[:100] for s in s_tensors[:min(1000, len(s_tensors))]]  # First 100 dims, first 1000 steps
        s_array = np.array(s_flat_list)
        corr_matrix = np.corrcoef(s_array.T)
        plt.imshow(corr_matrix, cmap='viridis', aspect='auto')
        plt.title('S Vector Correlation (First 100 dims)')
        plt.colorbar()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot s vector analysis for a single file')
    parser.add_argument('--file', required=True, help='Path to the pickle file')
    parser.add_argument('--save', help='Save plot to file instead of showing')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File {args.file} does not exist!")
        return

    print(f"Loading s monitoring data from {args.file}")

    try:
        with open(args.file, 'rb') as f:
            game_data = pickle.load(f)

        game = game_data.get('game', 'unknown')
        rank = game_data.get('rank', 'unknown')
        s_values = game_data.get('s_values', [])

        print(f"Game: {game}, Rank: {rank}")
        print(f"Number of steps: {len(s_values)}")
        if s_values:
            print(f"S vector shape: {s_values[0].shape}")

        plot_s_analysis(s_values, game, rank, save_path=args.save)

    except Exception as e:
        print(f"Error loading or plotting {args.file}: {e}")

if __name__ == '__main__':
    main()
