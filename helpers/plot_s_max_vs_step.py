#!/usr/bin/env python3
"""
Script to plot max s values vs step for each game in the specified model directory.
Also reports the number of s components > 0.5.
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from collections import defaultdict

def load_game_data(model_dir, specified_rank):
    """Load s monitoring data grouped by game for specified rank."""
    pattern = f"{model_dir}/s_monitor_PongNoFrameskip-v4_rank{specified_rank}_game*.pkl"
    files = glob.glob(pattern)

    game_data = defaultdict(dict)  # game -> {rank: s_values}

    for file in sorted(files):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)

                # Parse rank and game from filename
                basename = os.path.basename(file)
                match = re.search(r'rank(\d+)_game(\d+)', basename)
                if match:
                    rank = int(match.group(1))
                    game = int(match.group(2))
                    s_values = data.get('s_values', [])
                    game_data[game][rank] = s_values
                    print(f"Loaded game {game}, rank {rank}: {len(s_values)} steps")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return game_data

def compute_s_max_vs_step(game_data, specified_rank):
    """For each game (for specified rank), compute the max s value at each step."""
    game_stats = {}

    for game, ranks in game_data.items():
        if specified_rank not in ranks:
            continue

        s_values = ranks[specified_rank]

        s_max_per_step = [torch.abs(s).max().item() for s in s_values]
        components_above_05_per_step = [(torch.abs(s) > 0.5).sum().item() for s in s_values]

        game_stats[game] = {
            's_max': s_max_per_step,
            'components_above_05': components_above_05_per_step,
            'final_above_05': components_above_05_per_step[-1] if components_above_05_per_step else 0
        }

    return game_stats

def plot_s_max(game_stats, model_name, rank):
    """Plot s_max vs step for each game, grouped by ranges."""
    # Group games into ranges: 0-100, 100-200, 200-300, 300-inf
    ranges = [(0, 100), (100, 200), (200, 300), (300, float('inf'))]
    for min_game, max_game_orig in ranges:
        max_game = max_game_orig if max_game_orig != float('inf') else max(game_stats.keys())
        subset_stats = {g: s for g, s in game_stats.items() if min_game < g <= max_game}
        if not subset_stats:
            continue

        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(subset_stats)))

        for i, (game, stats) in enumerate(sorted(subset_stats.items())):
            steps = np.arange(len(stats['s_max']))
            plt.plot(steps, stats['s_max'], label=f'Game {game}', color=colors[i])

        plt.xlabel('Step')
        plt.ylabel('Max |S| Value')
        range_str = f'{min_game}-{max_game}' if max_game_orig != float('inf') else f'{min_game}+'
        plt.title(f'Max |S| Values vs Step - {model_name} (Rank {rank}) Games {range_str}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f's_max_vs_step_rank{rank}_games{min_game}-{max_game if max_game_orig != float("inf") else "inf"}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {filename}")

def plot_components(game_stats, model_name, rank):
    """Plot number of components |s|>0.5 vs step for each game, grouped by ranges."""
    # Group games into ranges: 0-100, 100-200, 200-300, 300-inf
    ranges = [(0, 100), (100, 200), (200, 300), (300, float('inf'))]
    for min_game, max_game_orig in ranges:
        max_game = max_game_orig if max_game_orig != float('inf') else max(game_stats.keys())
        subset_stats = {g: s for g, s in game_stats.items() if min_game < g <= max_game}
        if not subset_stats:
            continue

        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(subset_stats)))

        for i, (game, stats) in enumerate(sorted(subset_stats.items())):
            steps = np.arange(len(stats['components_above_05']))
            plt.plot(steps, stats['components_above_05'], label=f'Game {game}', color=colors[i])

        plt.xlabel('Step')
        plt.ylabel('Number of Components |S| > 0.5')
        range_str = f'{min_game}-{max_game}' if max_game_orig != float('inf') else f'{min_game}+'
        plt.title(f'Number of Components |S| > 0.5 vs Step - {model_name} (Rank {rank}) Games {range_str}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f's_num_components_rank{rank}_games{min_game}-{max_game if max_game_orig != float("inf") else "inf"}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {filename}")

def main():
    model_dir = "logs/Our_32w_g099_Concat1frameNormXUnnormDiffOnNormed_noReluLastConv_1024fc_same_re2"
    model_name = "Our_32w_g099_Concat1frameNormXUnnormDiffOnNormed_noReluLastConv_1024fc_same_re2"
    specified_rank = 0

    print(f"Analyzing data from {model_dir} for rank {specified_rank}")

    game_data = load_game_data(model_dir,specified_rank)

    if not game_data:
        print("No data found!")
        return

    print(f"Found {len(game_data)} games")

    game_stats = compute_s_max_vs_step(game_data, specified_rank)

    print(f"Games with rank {specified_rank}: {len(game_stats)}")

    # Print final components >0.5 for each game
    print(f"\nFinal number of s components >0.5 (Rank {specified_rank}):")
    for game, stats in sorted(game_stats.items()):
        final_above = stats['final_above_05']
        print(f"Game {game}: {final_above}")

    plot_s_max(game_stats, model_name, specified_rank)

    plot_components(game_stats, model_name, specified_rank)

if __name__ == '__main__':
    main()
