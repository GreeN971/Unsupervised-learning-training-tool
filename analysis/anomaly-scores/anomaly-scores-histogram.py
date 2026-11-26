"""Histogram and descriptive statistics of the GMM anomaly scores.

Reads ``datasets/anomalyScoresGMM.csv`` (one column ``Score``), draws a
histogram with the basic statistics in the corner and writes the same
statistics to ``output/summary.txt``. Run from the repo root:

    .venv/bin/python analysis/anomaly-scores/anomaly-scores-histogram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = _REPO_ROOT / "datasets" / "anomalyScoresGMM.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = OUT_DIR / "figures"

BIN_WIDTH = 1

# Reference palette shared with the other analysis scripts (light, for print).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_DATA = "#2a78d6"
C_FLAG = "#eb6834"
C_REF = "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def num(x, d=3):
    """Czech decimal comma."""
    return f"{x:.{d}f}".replace(".", ",")


def describe(s):
    q = s.quantile([0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    iqr = q[0.75] - q[0.25]
    return {
        "n": len(s),
        "min": s.min(),
        "max": s.max(),
        "rozpětí": s.max() - s.min(),
        "průměr": s.mean(),
        "medián": q[0.5],
        "směr. odchylka": s.std(),
        "variační koeficient": s.std() / s.mean(),
        "P5": q[0.05],
        "P10": q[0.10],
        "Q1": q[0.25],
        "Q3": q[0.75],
        "P90": q[0.90],
        "P95": q[0.95],
        "P99": q[0.99],
        "IQR": iqr,
        "horní plot (Q3 + 1,5·IQR)": q[0.75] + 1.5 * iqr,
        "šikmost": stats.skew(s, bias=False),
    }


def write_summary(s, d, outliers):
    lines = [f"n = {d['n']}"] + [f"{k} = {num(v)}" for k, v in d.items() if k != "n"]
    sw = stats.shapiro(s)
    lines += [
        "",
        f"Shapiro-Wilk: W = {num(sw.statistic)}, p = {sw.pvalue:.2e}",
        f"odlehlé hodnoty nad horním plotem ({len(outliers)}): "
        + ", ".join(num(v, 2) for v in outliers),
    ]
    text = "=== Popisná statistika: anomalyScoresGMM ===\n" + "\n".join(lines) + "\n"
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def plot(s, d, outliers):
    fence = d["horní plot (Q3 + 1,5·IQR)"]
    fig, (ax_box, ax) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True, height_ratios=[1, 4],
        gridspec_kw={"hspace": 0.05},
    )

    # Top: box plot with every score as a jittered point.
    ax_box.boxplot(
        s, orientation="horizontal", widths=0.55, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="#cde2fb", edgecolor=C_DATA),
        medianprops=dict(color=C_DATA, linewidth=2),
        whiskerprops=dict(color=C_DATA), capprops=dict(color=C_DATA),
    )
    rng = np.random.default_rng(0)
    y = 1 + rng.uniform(-0.18, 0.18, len(s))
    is_out = s > fence
    ax_box.scatter(s[~is_out], y[~is_out], s=14, color=INK_2, alpha=0.55, zorder=3,
                   linewidths=0)
    ax_box.scatter(s[is_out], y[is_out], s=22, color=C_FLAG, zorder=3,
                   edgecolors=SURFACE, linewidths=0.5)
    ax_box.set_yticks([])
    ax_box.grid(axis="y", visible=False)
    ax_box.spines["left"].set_visible(False)
    ax_box.set_title("Rozdělení anomálních skóre (GMM)")

    # Bottom: histogram with counts, KDE scaled to counts, rug and reference lines.
    lo = np.floor(s.min() / BIN_WIDTH) * BIN_WIDTH
    hi = (np.floor(s.max() / BIN_WIDTH) + 1) * BIN_WIDTH
    bins = np.arange(lo, hi + 1e-9, BIN_WIDTH)
    counts, edges, patches = ax.hist(
        s, bins=bins, color=C_DATA, alpha=0.85, edgecolor=SURFACE, linewidth=0.6,
        label=f"histogram (šířka třídy {num(BIN_WIDTH, 2)})",
    )
    for c, left, patch in zip(counts, edges[:-1], patches):
        if left >= fence:
            patch.set_facecolor(C_FLAG)
        if c:
            ax.text(left + BIN_WIDTH / 2, c + 0.15, f"{int(c)}", ha="center",
                    va="bottom", fontsize=8, color=INK_2)

    x = np.linspace(0, hi, 600)
    kde = stats.gaussian_kde(s)(x) * len(s) * BIN_WIDTH
    ax.plot(x, kde, color=INK_2, label="jádrový odhad hustoty (přepočten na počty)")
    ax.plot(s, np.zeros(len(s)), "|", color=INK_2, markersize=8, alpha=0.6,
            clip_on=False)

    ax.axvspan(d["Q1"], d["Q3"], color=C_REF, alpha=0.08,
               label=f"IQR: {num(d['Q1'])} – {num(d['Q3'])}")
    ax.axvline(d["medián"], color=C_REF, linestyle="--",
               label=f"medián = {num(d['medián'])}")
    ax.axvline(d["průměr"], color=C_FLAG, label=f"průměr = {num(d['průměr'])}")
    ax.axvline(d["P95"], color=MUTED, linestyle=":", linewidth=1.5,
               label=f"P95 = {num(d['P95'])}")
    ax.axvline(fence, color=C_FLAG, linestyle="-.", linewidth=1.2,
               label=f"Q3 + 1,5·IQR = {num(fence)} ({len(outliers)} odlehlých)")
    ax_box.axvline(fence, color=C_FLAG, linestyle="-.", linewidth=1.2)

    # Label each outlier with its value next to its point in the box-plot panel.
    for v, yy in zip(s[is_out], y[is_out]):
        ax_box.annotate(num(v, 2), xy=(v, yy), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=C_FLAG)

    ax.set_xlim(lo - 0.1, hi)
    ax.set_ylim(-counts.max() * 0.015, counts.max() * 1.1)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(BIN_WIDTH))
    ax.set(xlabel="anomální skóre", ylabel="počet vzorků")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)

    box = "\n".join([
        f"n           = {d['n']}",
        f"min         = {num(d['min'])}",
        f"max         = {num(d['max'])}",
        f"rozpětí     = {num(d['rozpětí'])}",
        f"průměr      = {num(d['průměr'])}",
        f"medián      = {num(d['medián'])}",
        f"směr. odch. = {num(d['směr. odchylka'])}",
        f"Q1 / Q3     = {num(d['Q1'])} / {num(d['Q3'])}",
        f"IQR         = {num(d['IQR'])}",
        f"P90 / P95   = {num(d['P90'])} / {num(d['P95'])}",
        f"P99         = {num(d['P99'])}",
        f"šikmost     = {num(d['šikmost'])}",
    ])
    ax.text(0.985, 0.97, box, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK_2, family="monospace", multialignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=AXIS))

    fig.savefig(FIG_DIR / "histogram.png")
    plt.close(fig)


def main():
    s = pd.read_csv(CSV_PATH)["Score"].dropna().reset_index(drop=True)
    d = describe(s)
    outliers = sorted(s[s > d["horní plot (Q3 + 1,5·IQR)"]])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_summary(s, d, outliers)
    plot(s, d, outliers)


if __name__ == "__main__":
    main()
