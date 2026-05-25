"""Replot FID vs NFE from a saved ``fid_sweep.json``."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.metrics_io import load_fid_sweep
from src.evaluation.point_cloud_viz import plot_fid_vs_nfe


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FID vs NFE from eval JSON.")
    parser.add_argument("fid_json", type=Path, help="Path to fid_sweep.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Save figure path")
    parser.add_argument("--log-y", action="store_true", help="Log-scale FID axis (paper style)")
    args = parser.parse_args()

    result = load_fid_sweep(args.fid_json)
    plot_fid_vs_nfe(
        result,
        save_path=args.output,
        show=args.output is None,
        log_y=args.log_y,
    )


if __name__ == "__main__":
    main()
