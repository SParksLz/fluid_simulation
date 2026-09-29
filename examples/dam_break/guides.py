"""Static water-height and rest-volume rulers shared by GL and USD."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Guide:
    name: str
    starts: np.ndarray
    ends: np.ndarray
    color: tuple
    width: float
    volume_fraction: float | None = None


def _digit_segments(text, x, y, z, height=0.026):
    """Small line labels stay readable in GL and travel with exported USD."""
    nodes = [(0, 1), (0.55, 1), (0.55, 0.5), (0.55, 0), (0, 0), (0, 0.5)]
    segments = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 2)]
    digits = {"0": "012345", "1": "12", "2": "01643", "3": "01236", "4": "5612",
              "5": "05623", "6": "054326", "7": "012", "8": "0123456", "9": "012356"}
    lines = []
    for character in text:
        if character in digits:
            for index in digits[character]:
                a, b = segments[int(index)]
                lines.append(((x + nodes[a][0] * height, y, z + nodes[a][1] * height),
                              (x + nodes[b][0] * height, y, z + nodes[b][1] * height)))
        elif character == "%":
            lines.append(((x, y, z), (x + 0.6 * height, y, z + height)))
            for dx, dz in [(0.05, 0.8), (0.45, 0.05)]:
                corners = [(dx, dz), (dx + 0.15, dz), (dx + 0.15, dz + 0.15), (dx, dz + 0.15)]
                for i in range(4):
                    a, b = corners[i], corners[(i + 1) % 4]
                    lines.append(((x + a[0] * height, y, z + a[1] * height),
                                  (x + b[0] * height, y, z + b[1] * height)))
        x += height * 0.85
    return lines


def height_guides(scene):
    lo, hi = scene.spec.bound_lo, scene.spec.bound_hi
    front = lo[1] - 0.012
    ticks = []
    # Physical-height ticks on the two front tank posts, at 0.1 m intervals.
    for z in np.arange(lo[2], hi[2] + 1.0e-6, 0.1):
        for x in (lo[0], hi[0]):
            ticks.append(((x - 0.025, front, z), (x + 0.025, front, z)))
    result = [Guide("HeightScale", np.asarray([a for a, _ in ticks], np.float32),
                    np.asarray([b for _, b in ticks], np.float32), (0.65, 0.65, 0.70), 0.002)]
    for fraction, color in [(1.0, (1.0, 0.35, 0.10)), (0.9, (1.0, 0.70, 0.15)),
                            (0.8, (1.0, 0.90, 0.35))]:
        z = lo[2] + scene.ideal_height * fraction
        lines = []
        if fraction == 1.0:
            corners = [(lo[0], front, z), (hi[0], front, z),
                       (hi[0], hi[1], z), (lo[0], hi[1], z)]
            lines.extend((corners[i], corners[(i + 1) % 4]) for i in range(4))
        else:
            # Dashed water-level marks leave the water column visible.
            for x in np.arange(lo[0], hi[0], 0.12):
                lines.append(((x, front, z), (min(x + 0.06, hi[0]), front, z)))
        label_height = min(0.026, scene.ideal_height * 0.075)
        lines += _digit_segments(f"{int(fraction * 100)}%", hi[0] + 0.045, front,
                                 z - label_height * 0.5, height=label_height)
        result.append(Guide(f"RestVolume{int(fraction * 100)}",
                            np.asarray([a for a, _ in lines], np.float32),
                            np.asarray([b for _, b in lines], np.float32), color,
                            0.003 if fraction == 1.0 else 0.0015, fraction))
    return result
