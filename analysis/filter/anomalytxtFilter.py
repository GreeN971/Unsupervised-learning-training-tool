"""Extract float anomaly scores from anomalyScoresRaw.txt into anomalyScoresGMM.csv.

The raw file is a repeating block of lines: sample id, dataset name, score, label.
Only lines that are a standalone float (e.g. "0.76") are kept; the output CSV
has a single "Score" column, in the same order as the raw file.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "datasets" / "anomalyScoresRaw.txt"
OUTPUT = ROOT / "datasets" / "anomalyScoresGMM.csv"

# Whole-line float: optional sign, digits with a decimal point, optional exponent.
FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$")


def extract_floats(path: Path) -> list[str]:
    """Return float lines as their original text, so "0.70" stays "0.70"."""
    scores = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if FLOAT_RE.match(token):
                scores.append(token)
    return scores


def main() -> None:
    scores = extract_floats(INPUT)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Score"])
        writer.writerows([s] for s in scores)
    print(f"Wrote {len(scores)} scores to {OUTPUT}")


if __name__ == "__main__":
    main()
