#!/usr/bin/env python3
"""
Script to analyze the s monitoring data collected during training.
This script loads the pickled s values and provides basic statistics and visualizations.
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from collections import defaultdict

def load_s_data(log_dir, env_name):
    """Load all s monitoring data for a given environment."""
    pattern = f"{log_dir}/s_monitor_{env_name}_rank*_game*.pkl"
    files = glob.glob(pattern)

    data = []
    for file in sorted(files):
        try:
            with open(file, 'rb') as f:
                game_data = pickle.load(f)
                # Parse rank and game from filename
                basename = os.path.basename(file)
                match = re.search(r'rank(\d+)_game(\d+)', basename)
                if match:
                    rank = int(match.group(1))
                    game = int(match.group(2))
                    game_data['rank'] = rank
                    game_data['game'] = game
                    data.append(game_data)
                else:
                    print(f"Could not parse rank and game from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return data

def analyze_s_distribution(s_values):
    """Analyze the distribution of s values."""
    if not s_values:
        return {}

    # Convert to numpy for easier analysis
    s_tensors = [s.cpu().numpy() for s in s_values]
    s_flat = np.concatenate([s.flatten() for s in s_tensors])

    stats = {
        'mean': np.mean(s_flat),
        'std': np.std(s_flat),
        'min': np.min(s_flat),
        'max': np.max(s_flat),
        'median': np.median(s_flat),
        'q25': np.percentile(s_flat, 25),
        'q75': np.percentile(s_flat, 75),
        'total_samples': len(s_flat),
        'total_steps': len(s_values)
    }

    return stats

def plot_s_distribution(s_values, game_num, rank, save_path=None):
    """Plot the distribution of s values for a game."""
    if not s_values:
        return

    s_tensors = [s.cpu().numpy() for s in s_values]
    s_flat = np.concatenate([s.flatten() for s in s_tensors])

    plt.figure(figsize=(12, 8))

    # Histogram of all s values
    plt.subplot(2, 2, 1)
    plt.hist(s_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'S Distribution - Game {game_num}, Rank {rank}')
    plt.xlabel('S Value')
    plt.ylabel('Frequency')

    # Time series of means
    plt.subplot(2, 2, 2)
    step_means = [s.mean().item() for s in s_tensors]
    plt.plot(step_means)
    plt.title('Mean S per Step')
    plt.xlabel('Step')
    plt.ylabel('Mean S')

    # Time series of stds
    plt.subplot(2, 2, 3)
    step_stds = [s.std().item() for s in s_tensors]
    plt.plot(step_stds)
    plt.title('Std S per Step')
    plt.xlabel('Step')
    plt.ylabel('Std S')

    # Distribution of random s vector
    plt.subplot(2, 2, 4)
    if s_tensors:
        random_idx = np.random.randint(len(s_tensors))
        random_s = s_tensors[random_idx]
        plt.hist(random_s.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.title(f'Random S Vector (Step {random_idx})')
        plt.xlabel('S Value')
        plt.ylabel('Frequency')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze s monitoring data')
    parser.add_argument('--log_dir', default='logs/', help='Log directory')
    parser.add_argument('--env', default='PongNoFrameskip-v4', help='Environment name')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--outfile', help='Output file for statistics')
    args = parser.parse_args()

    print(f"Loading s monitoring data for {args.env} from {args.log_dir}")

    data = load_s_data(args.log_dir, args.env)

    if not data:
        print("No s monitoring data found!")
        return

    print(f"Found {len(data)} monitoring files")

    # Open outfile if specified
    outfile = None
    if args.outfile:
        outfile = open(args.outfile, 'w')
        print(f"Writing statistics to {args.outfile}")

    # Analyze each game
    for game_data in data:
        game = game_data['game']
        rank = game_data['rank']
        s_values = game_data['s_values']

        # Write to outfile or stdout
        output_target = outfile if outfile else None
        if output_target:
            output_target.write(f"\nGame {game}, Rank {rank}:\n")
        else:
            print(f"\nGame {game}, Rank {rank}:")

        stats = analyze_s_distribution(s_values)
        for key, value in stats.items():
            if output_target:
                output_target.write(f"{key} : {value}\n")
            else:
                print(f"{key} : {value}")

        if args.plot:
            plot_path = f"{args.log_dir}/s_analysis_game{game}_rank{rank}.png"
            plot_s_distribution(s_values, game, rank, save_path=plot_path)
            print(f"Plot saved to {plot_path}")

    # Close outfile if opened
    if outfile:
        outfile.close()
        print(f"Statistics written to {args.outfile}")

if __name__ == '__main__':
    main()
