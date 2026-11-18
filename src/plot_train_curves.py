"""Plot Pong training curves: mean reward and the spread across runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_REWARD = -21.0
ROLLING_WINDOW = 10
NUM_POINTS = 500
BASELINE_COLOR = "blue"
BASELINE_BAND_ALPHA = 0.5
SERIES_BAND_ALPHA = 0.2
SERIES_COLORS = (
    "green",
    "red",
    "darkorange",
    "purple",
    "black",
    "magenta",
    "teal",
    "olive",
    "navy",
    "crimson",
)


def main() -> None:
    """Load local experiment logs and save the comparison figure."""
    args = _parse_args()
    baseline_dir = args.baseline.expanduser().resolve()
    experiment_dirs = _experiment_dirs(args.experiments, baseline_dir)

    plt.rcParams.update({"font.size": 20})
    figure, axis = plt.subplots(figsize=(20, 10))

    _draw_folder(
        axis,
        baseline_dir,
        _legend_label(baseline_dir),
        BASELINE_COLOR,
        BASELINE_BAND_ALPHA,
        args.min_frames,
        args.num_points,
    )
    for index, experiment_dir in enumerate(experiment_dirs):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        _draw_folder(
            axis,
            experiment_dir,
            _legend_label(experiment_dir),
            color,
            SERIES_BAND_ALPHA,
            args.min_frames,
            args.num_points,
        )

    _style_axis(axis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight")
    print(args.out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean Pong reward curves with a per-run spread band.",
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="Local logs/<experiment> directories to draw, besides the baseline.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Local logs directory drawn as the comparison line.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Where to write the png.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=2_000_000,
        help="Skip a run whose last logged frame is below this (default: 2000000).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=NUM_POINTS,
        help="Points on the shared frame grid (default: 500).",
    )
    args = parser.parse_args()
    if args.min_frames < 0:
        parser.error("--min-frames must be >= 0")
    if args.num_points < 2:
        parser.error("--num-points must be >= 2")
    return args


def _experiment_dirs(paths: list[Path], baseline_dir: Path) -> list[Path]:
    directories = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved == baseline_dir:
            print(f"Skipping {resolved.name}: it is the baseline", file=sys.stderr)
            continue
        directories.append(resolved)
    if not directories:
        _fail("Pass at least one experiment folder besides the baseline")
    return directories


def _draw_folder(
    axis,
    log_dir: Path,
    label: str,
    color: str,
    band_alpha: float,
    min_frames: int,
    num_points: int,
) -> None:
    curves = _usable_curves(log_dir, min_frames)
    grid, mean, runs = _mean_and_runs(curves, num_points)
    axis.plot(grid, mean, lw=3, color=color, label=label)
    spread = np.std(runs, axis=0)
    axis.fill_between(
        grid,
        mean - spread,
        mean + spread,
        color=color,
        alpha=band_alpha,
    )
    print(
        f"{label}: {len(curves)} runs, frames {int(grid[0])}..{int(grid[-1])}",
        file=sys.stderr,
    )


def _style_axis(axis) -> None:
    axis.tick_params(axis="both", which="major", labelsize=10)
    axis.tick_params(axis="both", which="minor", labelsize=8)
    axis.set_xlabel("Общее число кадров", fontsize=18)
    axis.set_ylabel("Средняя награда за последние 10 тестов", fontsize=18)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        fancybox=True,
        shadow=True,
        prop={"size": 15},
    )


def _usable_curves(log_dir: Path, min_frames: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if not log_dir.is_dir():
        _fail(f"Log folder does not exist: {log_dir}")
    log_files = _log_files(log_dir)
    if not log_files:
        _fail(f"No log files in {log_dir}")

    curves = []
    skipped = 0
    for path in log_files:
        frames, rewards = _read_episode_curve(path)
        if len(frames) < 2 or frames[-1] < min_frames:
            skipped += 1
            continue
        averaged = _reward_mean_10(rewards)
        filled = np.where(np.isnan(averaged), DEFAULT_REWARD, averaged)
        curves.append((np.asarray(frames, dtype=float), filled))
    if not curves:
        _fail(
            f"No logs in {log_dir.name} reached {min_frames} frames "
            f"({skipped} skipped). Pass --min-frames with a smaller cutoff."
        )
    if skipped:
        print(
            f"{log_dir.name}: skipped {skipped} logs shorter than {min_frames} frames",
            file=sys.stderr,
        )
    return curves


def _log_files(log_dir: Path) -> list[Path]:
    files = []
    for path in log_dir.iterdir():
        if path.is_file() and "log" in path.name:
            files.append(path)
    files.sort(key=lambda path: path.name)
    return files


def _legend_label(log_dir: Path) -> str:
    """Folder name plus gamor_<gamma_oracle2> when the log records it."""
    value = _gamma_oracle2(log_dir)
    if value is None:
        return log_dir.name
    return f"{log_dir.name} gamor_{format(value, 'g')}"


def _gamma_oracle2(log_dir: Path) -> float | None:
    for path in _log_files(log_dir):
        value = _gamma_oracle2_in_file(path)
        if value is not None:
            return value
    return None


def _gamma_oracle2_in_file(path: Path) -> float | None:
    marker = "gamma_oracle2: "
    with path.open() as handle:
        for line in handle:
            if "episode reward" in line:
                return None
            start = line.find(marker)
            if start < 0:
                continue
            token = line[start + len(marker):].strip().split()[0]
            try:
                return float(token)
            except ValueError:
                return None
    return None


def _read_episode_curve(log_path: Path) -> tuple[list[int], list[float]]:
    frames = []
    rewards = []
    with log_path.open() as handle:
        for line in handle:
            reward = _episode_reward(line)
            frame = _frames_total(line)
            if reward is None or frame is None:
                continue
            frames.append(frame)
            rewards.append(reward)
    return _dedupe_frames(frames, rewards)


def _episode_reward(line: str) -> float | None:
    return _float_after(line, "episode reward ")


def _frames_total(line: str) -> int | None:
    value = _float_after(line, "frames_total ")
    if value is None:
        return None
    return int(value)


def _float_after(line: str, marker: str) -> float | None:
    start = line.find(marker)
    if start < 0:
        return None
    tail = line[start + len(marker) :]
    token = tail.split(",", 1)[0].split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def _dedupe_frames(
    frames: list[int],
    rewards: list[float],
) -> tuple[list[int], list[float]]:
    last_reward: dict[int, float] = {}
    ordered: list[int] = []
    for frame, reward in zip(frames, rewards):
        if frame not in last_reward:
            ordered.append(frame)
        last_reward[frame] = reward
    if _goes_backwards(ordered):
        ordered = sorted(last_reward)
    curve_frames = []
    curve_rewards = []
    for frame in ordered:
        curve_frames.append(frame)
        curve_rewards.append(last_reward[frame])
    return curve_frames, curve_rewards


def _goes_backwards(frames: list[int]) -> bool:
    previous = None
    for frame in frames:
        if previous is not None and frame < previous:
            return True
        previous = frame
    return False


def _reward_mean_10(rewards: list[float]) -> np.ndarray:
    """Rolling mean of episode reward, same window as the notebook."""
    values = np.asarray(rewards, dtype=float)
    averaged = np.full(values.shape, np.nan)
    window = ROLLING_WINDOW
    if len(values) < window:
        return averaged
    cumulative = np.cumsum(values)
    cumulative[window:] = cumulative[window:] - cumulative[:-window]
    averaged[window - 1 :] = cumulative[window - 1 :] / window
    return averaged


def _mean_and_runs(
    curves: list[tuple[np.ndarray, np.ndarray]],
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = min(frames[0] for frames, _rewards in curves)
    # The notebook stops at the shortest run, so every curve covers the grid.
    stop = min(frames[-1] for frames, _rewards in curves)
    if stop <= start:
        _fail("Frame grid is empty: the shortest run ends before the others start")
    grid = np.linspace(start, stop, num_points)
    runs = [_interpolate(frames, rewards, grid) for frames, rewards in curves]
    stacked = np.vstack(runs)
    return grid, stacked.mean(axis=0), stacked


def _interpolate(frames: np.ndarray, rewards: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation, extrapolated outside a run with its end segment."""
    values = np.interp(grid, frames, rewards)
    left = grid < frames[0]
    right = grid > frames[-1]
    if np.any(left):
        slope = (rewards[1] - rewards[0]) / (frames[1] - frames[0])
        values[left] = rewards[0] + slope * (grid[left] - frames[0])
    if np.any(right):
        slope = (rewards[-1] - rewards[-2]) / (frames[-1] - frames[-2])
        values[right] = rewards[-1] + slope * (grid[right] - frames[-1])
    return values


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
