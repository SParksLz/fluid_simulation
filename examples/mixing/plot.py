"""Plot two mixing runs; requires the optional matplotlib package."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.mixing.color import colors_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path, help="reference output directory")
    parser.add_argument("diffusing", type=Path, help="comparison output directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("Install optional plotting dependency: python -m pip install matplotlib")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, directory in enumerate((args.control, args.diffusing)):
        report = json.loads((directory / "run.json").read_text())
        state = np.load(directory / "final_state.npz")
        x, c = state["positions"], state["concentration"]
        lo, hi = report["apic_config"]["bound_lo"], report["apic_config"]["bound_hi"]
        is_tank = report.get("scene", "vortex") == "tank"
        if is_tank:
            mask = np.abs(x[:, 1]-0.5*(lo[1]+hi[1])) < 1.5*report["spacing_m"]
            vertical, title = 2, "Center y slice"
        else:
            mask = x[:, 2] > hi[2]-report["spacing_m"]
            vertical, title = 1, "Top particle layer"
        # Older reports predate the color-model option and used RGB blending.
        model = report.get("color_model", "rgb")
        colors = colors_numpy(c[mask], model)
        D = report["diffusivity_m2_s"]
        label = f"{model.upper()}, D = {D:g}"
        axes[0, column].scatter(x[mask, 0], x[mask, vertical], c=colors, s=34)
        axes[0, column].set(xlim=(lo[0], hi[0]), ylim=(lo[vertical], hi[vertical]), aspect="equal",
                            title=f"{title}, {label} m²/s ({report['final']['time_s']:g} s)",
                            xlabel="x [m]", ylabel="z [m]" if is_tank else "y [m]")
        with (directory / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        t = [float(row["time_s"]) for row in rows]
        axes[1, 0].plot(t, [float(row["mixing_index"]) for row in rows], label=label)
        axes[1, 1].plot(t, [100*float(row["dye_relative_error"]) for row in rows], label=label)
    axes[1, 0].set(xlabel="Time [s]", ylabel="Particle concentration mixing index")
    axes[1, 1].set(xlabel="Time [s]", ylabel="Dye integral relative error [%]")
    for ax in axes[1]:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output = args.output or args.diffusing / "comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
