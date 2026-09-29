"""PBF dam-break example with the same particles as APIC."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.dam_break.run import main

if __name__ == "__main__":
    main("pbf")
