"""Exploratory data analysis (EDA) of the Cheb 10-minute air temperature series.

Follows the thesis chapter 7.3 methodology: univariate inspection of the
distribution (normality, 3-sigma rule, ECDF, MAD, box plot and histogram),
then formal clustering (k-means with the elbow rule, DBSCAN) and a short
iterative removal loop showing why it must stop after only a few iterations.

Only the VALUE column of the raw CHMI export is kept and renamed to
``temperature``; the cleaned single-column dataset is written next to the
figures. Run from the repo root:

    .venv/bin/python analysis/analysis.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

_REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = _REPO_ROOT / "datasets" / "10m-0-20000-0-11406-T-202507.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = OUT_DIR / "figures"

# DBSCAN's min_samples; the k-distance graph used to pick eps uses the same K,
# so it describes exactly DBSCAN's core condition.
K = 10
# Modified z-score cut-off recommended by Iglewicz & Hoaglin for MAD.
MAD_THRESHOLD = 3.5
# Upper bound for the iterative loops; the chapter recommends only a few
# iterations, the extra ones are run purely to show the cascade effect.
MAX_ITER = 10
# Largest number of k-means clusters tried by the elbow rule.
MAX_CLUSTERS = 10

# Reference palette (dataviz skill) - light surface only, figures go to print.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_DATA = "#2a78d6"   # slot 1 blue - the data
C_FLAG = "#eb6834"   # slot 2 orange - flagged values / thresholds
C_REF = "#1baf7a"    # slot 3 aqua - reference (normal) distribution

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

# SAS ODS Graphics look (HTMLBlue style) for the DBSCAN figure: white wall with
# a full frame, no grid, bold centred titles, Arial (or its metric-compatible
# Liberation Sans), open circle markers and the GraphData1-3 colours.
_FONTS = {f.name for f in font_manager.fontManager.ttflist}
SAS_FONT = next((f for f in ("Arial", "Liberation Sans") if f in _FONTS), "DejaVu Sans")
SAS_BLUE = "#445694"   # GraphData1
SAS_RED = "#A23A2E"    # GraphData2
SAS_TEAL = "#01665E"   # GraphData3
SAS_REF = "#808080"    # reference lines
SAS_BORDER = "#C1C1C1"  # graph and legend border
SAS_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [SAS_FONT, "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": "black",
    "axes.titlepad": 8,
    "axes.labelsize": 10,
    "axes.labelcolor": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 1,
    "legend.edgecolor": SAS_BORDER,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
    # The graph border is drawn at the figure edge, so keep the full canvas.
    "savefig.bbox": "standard",
}


def load_temperature():
    """Keep only VALUE from the raw export and rename it to temperature."""
    raw = pd.read_csv(CSV_PATH)
    return raw[["VALUE"]].rename(columns={"VALUE": "temperature"})


class CommaFormatter(ScalarFormatter):
    """Tick labels with a decimal comma, as the figures go into Czech text."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def num(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def save(fig, name):
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(CommaFormatter())
        ax.yaxis.set_major_formatter(CommaFormatter())
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)


def section(title):
    print(f"\n=== {title} ===")


def describe(t):
    section("Popisná statistika")
    q = t.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"n = {t.size}, chybějící = {t.isna().sum()}, unikátních = {t.nunique()}")
    print(f"průměr = {t.mean():.2f}, směr. odchylka = {t.std():.2f}")
    print(f"min = {t.min():.1f}, max = {t.max():.1f}, rozpětí = {t.max() - t.min():.1f}")
    print("kvantily:", ", ".join(f"P{int(p * 100)}={v:.1f}" for p, v in q.items()))
    steps = np.diff(np.sort(t.unique()))
    print(f"rozlišení (nejmenší krok mezi hodnotami) = {steps.min():.1f} °C")
    print(f"šikmost = {stats.skew(t):.3f}, špičatost (exces) = {stats.kurtosis(t):.3f}")


def normality(t):
    section("Test normality")
    sw = stats.shapiro(t)
    k2 = stats.normaltest(t)
    ks = stats.kstest(t, stats.norm(t.mean(), t.std()).cdf)
    print(f"Shapiro-Wilk W = {sw.statistic:.4f}, p = {sw.pvalue:.2e}")
    print(f"D'Agostino-Pearson K2 = {k2.statistic:.1f}, p = {k2.pvalue:.2e}")
    print(f"Kolmogorov-Smirnov D = {ks.statistic:.4f}, p = {ks.pvalue:.2e}")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    (osm, osr), (slope, intercept, _) = stats.probplot(t, dist="norm")
    ax.plot(osm, osr, "o", ms=3, color=C_DATA, alpha=0.5, label="naměřené hodnoty")
    ax.plot(osm, slope * osm + intercept, color=C_REF, label="normální rozdělení")
    ax.set(title="Q-Q graf vůči normálnímu rozdělení",
           xlabel="teoretické kvantily N(0, 1)", ylabel="teplota [°C]")
    ax.legend()
    save(fig, "02_qq_plot")


def three_sigma(t):
    section("Pravidlo 3σ")
    m, s = t.mean(), t.std()
    lo, hi = m - 3 * s, m + 3 * s
    mask = (t < lo) | (t > hi)
    print(f"meze = [{lo:.2f}; {hi:.2f}] °C, mimo mez: {mask.sum()} "
          f"({mask.mean() * 100:.2f} %), pod: {(t < lo).sum()}, nad: {(t > hi).sum()}")
    print(f"očekáváno při normalitě: {2 * stats.norm.sf(3) * t.size:.1f}")
    return mask, (lo, hi)


def mad_rule(t):
    section("MAD (modifikované z-skóre)")
    med = t.median()
    mad = stats.median_abs_deviation(t)
    z = 0.6745 * (t - med) / mad
    mask = z.abs() > MAD_THRESHOLD
    lo, hi = med - MAD_THRESHOLD * mad / 0.6745, med + MAD_THRESHOLD * mad / 0.6745
    print(f"medián = {med:.2f}, MAD = {mad:.2f} (škálovaná = {mad / 0.6745:.2f})")
    print(f"meze |z*| > {MAD_THRESHOLD}: [{lo:.2f}; {hi:.2f}] °C, mimo mez: {mask.sum()} "
          f"({mask.mean() * 100:.2f} %)")
    return mask, (lo, hi)


def iqr_rule(t):
    section("Krabicový graf (IQR)")
    q1, q3 = t.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (t < lo) | (t > hi)
    print(f"Q1 = {q1:.1f}, Q3 = {q3:.1f}, IQR = {iqr:.1f}, ploty = [{lo:.2f}; {hi:.2f}] °C, "
          f"mimo: {mask.sum()} ({mask.mean() * 100:.2f} %)")
    return mask, (lo, hi)


def plot_histogram(t, mad_bounds):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(t, bins=np.arange(t.min() - 0.25, t.max() + 0.75, 0.5), density=True,
            color=C_DATA, alpha=0.85, edgecolor=SURFACE, linewidth=0.8,
            label="histogram (šířka třídy 0,5 °C)")
    x = np.linspace(t.min() - 3, t.max() + 3, 400)
    ax.plot(x, stats.gaussian_kde(t)(x), color=INK_2, label="jádrový odhad hustoty")
    ax.plot(x, stats.norm.pdf(x, t.mean(), t.std()), color=C_REF, linestyle="--",
            label="normální rozdělení N(x̄, s)")
    for b in mad_bounds:
        ax.axvline(b, color=C_FLAG, linewidth=1.5)
    ax.plot([], [], color=C_FLAG,
            label=f"meze MAD: {num(mad_bounds[0], 2)} – {num(mad_bounds[1], 2)} °C")
    ax.set(title="Rozdělení teploty", xlabel="teplota [°C]", ylabel="hustota")
    # Opaque legend so the upper MAD line does not run through its text.
    ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=SURFACE,
              framealpha=1)
    save(fig, "01_histogram")


def plot_boxplot(t, iqr_bounds):
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.boxplot(t, orientation="horizontal", widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#cde2fb", edgecolor=C_DATA),
               medianprops=dict(color=C_DATA, linewidth=2),
               whiskerprops=dict(color=C_DATA), capprops=dict(color=C_DATA),
               flierprops=dict(marker="o", markersize=4, markerfacecolor=C_FLAG,
                               markeredgecolor=SURFACE, alpha=0.7))
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set(title=f"Krabicový graf (body mimo vousy: > {num(iqr_bounds[1], 2)} °C)",
           xlabel="teplota [°C]")
    save(fig, "03_boxplot")


def ecdf(t):
    section("ECDF")
    x = np.sort(t.to_numpy())
    y = np.arange(1, x.size + 1) / x.size
    unique = np.unique(x)
    gaps = np.diff(unique)
    order = np.argsort(gaps)[::-1][:5]
    print("největší mezery mezi sousedními hodnotami:",
          ", ".join(f"{unique[i]:.1f}->{unique[i + 1]:.1f} ({gaps[i]:.1f})" for i in order))
    for v in [t.quantile(0.005), t.quantile(0.995)]:
        print(f"kvantil ECDF {v:.1f} °C")
    for lim in [10, 11, 12]:
        print(f"P(T < {lim} °C) empiricky = {(x < lim).mean() * 100:.2f} %, "
              f"normálně = {stats.norm.cdf(lim, t.mean(), t.std()) * 100:.2f} %")
    for lim in [25, 28, 30]:
        print(f"P(T > {lim} °C) empiricky = {(x > lim).mean() * 100:.2f} %, "
              f"normálně = {stats.norm.sf(lim, t.mean(), t.std()) * 100:.2f} %")

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25)
    panels = [
        (fig.add_subplot(gs[0, :]), None, None, "ECDF teploty"),
        (fig.add_subplot(gs[1, 0]), (9, 12), (-0.0005, 0.05), "Detail dolního chvostu"),
        (fig.add_subplot(gs[1, 1]), (28, 35), (0.95, 1.0005), "Detail horního chvostu"),
    ]
    grid = np.linspace(x.min() - 4, x.max() + 2, 400)
    # Leading zero so the step starts at F = 0 left of the minimum.
    x_step, y_step = np.r_[grid[0], x], np.r_[0, y]
    for ax, xlim, ylim, title in panels:
        ax.step(x_step, y_step, where="post", color=C_DATA, label="empirická (ECDF)")
        ax.plot(grid, stats.norm.cdf(grid, t.mean(), t.std()), color=C_REF,
                linestyle="--", label="normální CDF")
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        ax.set(title=title, xlabel="teplota [°C]", ylabel="F(x)")
    panels[0][0].legend(loc="lower right")
    save(fig, "04_ecdf")


def run_dbscan(X, eps):
    """DBSCAN labels with a float tolerance on eps: readings are stored with
    0.1 °C resolution, so e.g. 15.2 - 15.1 is not exactly 0.1 and would
    otherwise fall just outside an eps that is a multiple of the resolution."""
    return DBSCAN(eps=eps + 1e-9, min_samples=K).fit_predict(X)


def kneedle(values):
    """Index of the point farthest from the chord of the sorted, normalised
    curve - the usual 'elbow' heuristic for the DBSCAN k-distance graph."""
    y = np.sort(values)
    x = np.linspace(0, 1, y.size)
    yn = (y - y[0]) / (y[-1] - y[0])
    return int(np.argmax(x - yn)), y


def kmeans_elbow(t, flagged):
    """Number of k-means clusters by the elbow rule on the within-cluster sum
    of squares (WCSS), then the clusters for the chosen k."""
    section(f"k-means: pravidlo lokte (k = 1–{MAX_CLUSTERS})")
    X = t.to_numpy().reshape(-1, 1)
    ks = np.arange(1, MAX_CLUSTERS + 1)
    # 20 restarts per k: k-means depends on the initial centres, keep the best.
    models = [KMeans(k, n_init=20, random_state=0).fit(X) for k in ks]
    wcss = np.array([m.inertia_ for m in models])
    # Elbow = the k farthest below the straight line joining the first and last
    # point of the normalised curve.
    x_n = (ks - ks[0]) / (ks[-1] - ks[0])
    y_n = (wcss - wcss[-1]) / (wcss[0] - wcss[-1])
    gap = (1 - x_n) - y_n
    best = int(ks[np.argmax(gap)])
    for k, w, g in zip(ks, wcss, gap):
        print(f"k = {k:2d}: WCSS = {w:9.0f} ({w / wcss[0] * 100:5.1f} % z k = 1), "
              f"odstup od spojnice = {g:.3f}")
    print(f"loket: k = {best}")

    labels = models[best - 1].labels_
    # Number clusters from the coldest to the warmest.
    order = np.argsort(models[best - 1].cluster_centers_.ravel())
    labels = np.argsort(order)[labels]
    x = X[:, 0]
    for c in range(best):
        m = labels == c
        print(f"  shluk {c + 1}: [{x[m].min():.1f}; {x[m].max():.1f}] °C, n = {m.sum()}, "
              f"střed = {x[m].mean():.2f} °C, nad mezí MAD: {(m & flagged.to_numpy()).sum()}")
    plot_elbow(ks, wcss, best, x, labels)


def plot_elbow(ks, wcss, best, x, labels):
    """Elbow curve and the resulting clusters, SAS look."""
    colors = [SAS_BLUE, SAS_RED, SAS_TEAL, "#543005", "#9D3CDB", "#7F8E1F"]
    # Two points larger text than the SAS defaults, as requested for this figure.
    larger = {key: SAS_RC[key] + 2 for key in ("axes.titlesize", "axes.labelsize",
              "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
    with plt.rc_context({**SAS_RC, **larger}):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), layout="constrained",
                                 height_ratios=[1.2, 1])
        fig.get_layout_engine().set(h_pad=0.15, w_pad=0.15)

        ax = axes[0]
        ax.plot(ks, wcss, color=SAS_BLUE, marker="o", ms=7, markerfacecolor="white",
                markeredgecolor=SAS_BLUE, markeredgewidth=1)
        ax.plot(best, wcss[best - 1], marker="o", ms=10, color=SAS_RED, zorder=3)
        ax.annotate(f"loket: k = {best}", (best, wcss[best - 1]), xytext=(12, 12),
                    textcoords="offset points", fontsize=11)
        ax.set(title="Pravidlo lokte (k-means)", xlabel="Počet shluků k",
               ylabel="Vnitroshlukový součet čtverců", xticks=ks)
        ax.set_ylim(bottom=0)

        ax = axes[1]
        # Vertical jitter only spreads the points out; the y position carries no value.
        jitter = np.random.default_rng(3).uniform(-1, 1, x.size)
        for c in range(best):
            m = labels == c
            ax.scatter(x[m], jitter[m], s=14, marker="o", facecolors="none",
                       edgecolors=colors[c], linewidths=0.6,
                       label=f"Shluk {c + 1}: {num(x[m].min())}–{num(x[m].max())} °C")
            ax.plot(x[m].mean(), 0, marker="X", ms=12, color="black", zorder=3)
        ax.set_yticks([])
        ax.set_ylim(-1.25, 1.25)
        ax.set(title=f"Shluky k-means pro k = {best} (× = střed shluku)", xlabel="Teplota [°C]")
        handles = [Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                          markeredgecolor=colors[c], markersize=8,
                          label=h.get_label()) for c, h in enumerate(ax.collections[:best])]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                  ncol=best, borderaxespad=0, columnspacing=1.5)

        fig.patches.append(Rectangle((0.001, 0.001), 0.998, 0.998, transform=fig.transFigure,
                                     fill=False, edgecolor=SAS_BORDER, linewidth=1))
        save(fig, "05_kmeans_loket")


def dbscan(t):
    section(f"DBSCAN (min_samples = {K})")
    X = t.to_numpy().reshape(-1, 1)
    # n_neighbors counts the point itself, like min_samples: K-th column = (K-1)-th neighbour.
    kth = NearestNeighbors(n_neighbors=K).fit(X).kneighbors(X)[0][:, -1]
    knee, kth_sorted = kneedle(kth)
    eps_knee = max(round(kth_sorted[knee], 1), 0.1)
    print(f"koleno k-distance grafu: eps = {eps_knee:.2f} °C "
          f"(bod {knee} z {kth.size}, P99 k-vzdálenosti = {np.quantile(kth, 0.99):.2f})")

    grid = np.round(np.arange(0.1, 1.01, 0.1), 1)
    sensitivity = []
    for e in grid:
        lab = run_dbscan(X, e)
        sensitivity.append((len(set(lab) - {-1}), int((lab == -1).sum())))
        print(f"  eps = {e:.1f}: shluků = {sensitivity[-1][0]}, šum = {sensitivity[-1][1]}")
    # Smallest eps at which the tail fragments merge into one cluster; below it
    # the "clusters" are artefacts of the 0.1 °C sensor resolution.
    eps_stable = next(e for e, (c, _) in zip(grid, sensitivity) if c == 1)

    results = {}
    for e in [eps_knee, eps_stable]:
        labels = run_dbscan(X, e)
        noise = labels == -1
        print(f"eps = {e:.1f}: šum ({noise.sum()}) = {np.sort(X[noise, 0])}")
        for c in sorted(set(labels) - {-1}):
            v = X[labels == c, 0]
            print(f"  shluk {c}: n = {v.size}, rozsah = [{v.min():.1f}; {v.max():.1f}] °C")
        results[e] = labels

    plot_dbscan(X, kth_sorted, eps_knee, eps_stable, grid, sensitivity, results)
    return eps_stable


def plot_dbscan(X, kth_sorted, eps_knee, eps_stable, grid, sensitivity, results):
    """DBSCAN figure in the SAS ODS Graphics (HTMLBlue) look, see SAS_RC."""
    with plt.rc_context(SAS_RC):
        fig = plt.figure(figsize=(10, 9.5), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.12, w_pad=0.15, hspace=0.06, wspace=0.06)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(np.arange(kth_sorted.size), kth_sorted, color=SAS_BLUE)
        for e, style, label in [(eps_knee, "--", "koleno"), (eps_stable, "-", "stabilní eps")]:
            ax.axhline(e, color=SAS_REF, linestyle=style, linewidth=0.9)
            ax.text(0.02, e, f"{label} = {num(e)}", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=9)
        ax.set(title=f"k-distance graf (k = {K})",
               xlabel="Pořadí bodu (seřazeno podle vzdálenosti)",
               ylabel=f"Vzdálenost k {K - 1}. sousedovi [°C]")

        ax = fig.add_subplot(gs[0, 1])
        noise_counts = [n for _, n in sensitivity]
        ax.plot(grid, noise_counts, color=SAS_BLUE, marker="o", ms=6,
                markerfacecolor="white", markeredgecolor=SAS_BLUE, markeredgewidth=1)
        # SAS DATALABEL= position: upper right of the marker.
        for e, (c, n) in zip(grid, sensitivity):
            ax.annotate(str(c), (e, n), xytext=(4, 4), textcoords="offset points",
                        fontsize=9)
        ax.set(title="Citlivost na eps (popisek = počet shluků)", xlabel="eps [°C]",
               ylabel="Počet bodů šumu", xticks=grid)
        ax.set_ylim(0, max(noise_counts) * 1.12)

        # Vertical jitter only spreads the points out; the y position carries no value.
        jitter = np.random.default_rng(3).uniform(-1, 1, X.shape[0])
        open_circle = dict(marker="o", facecolors="none", linewidths=0.6)
        for row, e in enumerate([eps_knee, eps_stable], start=1):
            labels = results[e]
            noise = labels == -1
            main = labels == np.bincount(labels[~noise]).argmax()
            minor = ~noise & ~main
            n_clusters = len(set(labels) - {-1})
            ax = fig.add_subplot(gs[row, :])
            ax.scatter(X[main, 0], jitter[main], s=14, edgecolors=SAS_BLUE, **open_circle)
            ax.scatter(X[minor, 0], jitter[minor], s=14, edgecolors=SAS_TEAL, **open_circle)
            ax.scatter(X[noise, 0], jitter[noise], s=26, color=SAS_RED, zorder=3)
            ax.set_yticks([])
            ax.set_ylim(-1.25, 1.25)
            ax.set_xlim(8.5, 35)
            ax.set(title=f"DBSCAN, eps = {num(e)} °C (shluky: {n_clusters}, šum: {noise.sum()})",
                   xlabel="Teplota [°C]")
            # Fixed attribute map, so a group keeps its colour in both panels.
            handles = [Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                              markeredgecolor=SAS_BLUE, label="Hlavní shluk")]
            if minor.any():
                handles.append(Line2D([], [], linestyle="none", marker="o",
                                      markerfacecolor="none", markeredgecolor=SAS_TEAL,
                                      label=f"Malé shluky ({n_clusters - 1})"))
            handles.append(Line2D([], [], linestyle="none", marker="o", color=SAS_RED,
                                  label="Šum"))
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.26),
                      ncol=len(handles), borderaxespad=0, columnspacing=2)

        fig.patches.append(Rectangle((0.001, 0.001), 0.998, 0.998, transform=fig.transFigure,
                                     fill=False, edgecolor=SAS_BORDER, linewidth=1))
        save(fig, "06_dbscan")


def iterative_removal(t, eps):
    """Repeat each rule on the data left after removing its previous flags."""
    section("Iterativní odstraňování")

    def sigma(x):
        return np.abs(x - x.mean()) > 3 * x.std(ddof=1)

    def mad(x):
        return np.abs(0.6745 * (x - np.median(x)) / stats.median_abs_deviation(x)) > MAD_THRESHOLD

    def density(x):
        return run_dbscan(x.reshape(-1, 1), eps) == -1

    history = {}
    for name, rule in [("3σ", sigma), ("MAD", mad), ("DBSCAN", density)]:
        x = t.to_numpy()
        removed = []
        for i in range(MAX_ITER):
            mask = rule(x)
            if not mask.any():
                break
            removed.append(int(mask.sum()))
            print(f"{name} iterace {i + 1}: odstraněno {mask.sum():3d}, "
                  f"rozsah zbytku = [{x[~mask].min():.1f}; {x[~mask].max():.1f}] °C, "
                  f"celkem {sum(removed)} ({sum(removed) / t.size * 100:.2f} %)")
            x = x[~mask]
        if len(removed) == MAX_ITER:
            print(f"{name}: po {MAX_ITER} iteracích stále nekonverguje")
        history[name] = np.cumsum(removed)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (name, cum), color in zip(history.items(), [C_FLAG, C_DATA, C_REF]):
        it = np.arange(1, cum.size + 1)
        ax.plot(it, cum, marker="o", ms=5, color=color, label=name)
        ax.annotate(name, (it[-1], cum[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set(title="Kumulativní počet odstraněných hodnot", xlabel="iterace",
           ylabel="odstraněno hodnot", xticks=range(1, MAX_ITER + 1))
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    save(fig, "07_iterative_removal")


def sequence_check(t):
    """The rows are consecutive 10-minute readings, so the row order still
    carries the time axis even after DT was dropped."""
    section("Doplňkově: pořadí měření")
    d = t.diff()
    top = d.abs().sort_values(ascending=False).head(6)
    print("největší změny mezi sousedními měřeními:",
          ", ".join(f"řádek {i}: {t[i - 1]:.1f} -> {t[i]:.1f} ({d[i]:+.1f})" for i in top.index))
    print(f"směr. odchylka změny za 10 min = {d.std():.2f} °C, "
          f"|změna| > 3 °C: {(d.abs() > 3).sum()}")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t.index, t, color=C_DATA, linewidth=0.8)
    jumps = top.index[:2]
    ax.scatter(jumps, t[jumps], s=40, color=C_FLAG, edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="největší skokové poklesy")
    ax.set(title="Teplota v pořadí měření (krok 10 min)",
           xlabel="pořadí měření", ylabel="teplota [°C]", xlim=(0, t.size))
    ax.legend(loc="upper right")
    save(fig, "08_sequence")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_temperature()
    df.to_csv(OUT_DIR / "temperature.csv", index=False)
    t = df["temperature"]

    describe(t)
    normality(t)
    three_sigma(t)
    mad_mask, mad_bounds = mad_rule(t)
    _, iqr_bounds = iqr_rule(t)
    plot_histogram(t, mad_bounds)
    plot_boxplot(t, iqr_bounds)
    ecdf(t)
    kmeans_elbow(t, mad_mask)
    eps = dbscan(t)
    iterative_removal(t, eps)
    sequence_check(t)
    print(f"\nGrafy uloženy do {FIG_DIR}, data do {OUT_DIR / 'temperature.csv'}")


if __name__ == "__main__":
    main()
