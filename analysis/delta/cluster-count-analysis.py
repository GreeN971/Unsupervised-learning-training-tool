"""Number of k-means clusters for the Cheb temperature series with the delta column.

Input is ``datasets/10m-0-20000-0-11406-T-202507-delta.csv``: the 10-minute air
temperature (VALUE) and its change from the previous reading (DELTA). Each
reading is a point in the (VALUE, DELTA) plane; both features are standardised
(z-score) so the 0.1 degC step of DELTA weighs as much as the much wider range
of VALUE. The first reading has no DELTA and is dropped.

k = 1..MAX_CLUSTERS is scored by several criteria, because no single one is
reliable on its own:

* elbow rule on the within-cluster sum of squares (WCSS), point farthest from
  the chord as in ``analysis.py``,
* silhouette coefficient (max is best),
* Calinski-Harabasz index (max is best),
* Davies-Bouldin index (min is best),
* bootstrap stability: mean adjusted Rand index between the clustering of the
  full data and of bootstrap resamples (high is best),
* BIC of a Gaussian mixture with full covariance (min is best), for comparison.

Run from the repo root; the printed report is the summary:

    .venv/bin/python analysis/delta/cluster-count-analysis.py > analysis/delta/output/summary.txt
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = _REPO_ROOT / "datasets" / "10m-0-20000-0-11406-T-202507-delta.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = OUT_DIR / "figures"
LOCAL_TZ = "Europe/Prague"

MAX_CLUSTERS = 10
N_INIT = 20
# Bootstrap resamples per k for the stability criterion.
N_BOOT = 30
# A cluster with fewer points than this is an outlier, not a regime.
MIN_CLUSTER = 20

# Same palette as the other analysis scripts.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_DATA = "#2a78d6"
C_FLAG = "#eb6834"
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9d3cdb", "#c79a1c", "#52514e",
                  "#d6457a", "#5aa6e8", "#7f8e1f", "#898781"]

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


class CommaFormatter(ScalarFormatter):
    """Tick labels with a decimal comma, as the figures go into Czech text."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def save(fig, name):
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(CommaFormatter())
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)


def section(title):
    print(f"\n=== {title} ===")


def load():
    df = pd.read_csv(CSV_PATH, parse_dates=["DT"]).dropna(subset=["DELTA"])
    df["local"] = df["DT"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return df.reset_index(drop=True)


def describe(df):
    section("Data")
    print(f"n = {len(df)} (první měření bez DELTA vynecháno)")
    for col in ("VALUE", "DELTA"):
        s = df[col]
        print(f"{col}: průměr = {s.mean():.2f}, směr. odchylka = {s.std():.2f}, "
              f"min = {s.min():.1f}, max = {s.max():.1f}, exces = {s.kurt():.1f}")
    print(f"korelace VALUE–DELTA = {df['VALUE'].corr(df['DELTA']):.3f}")
    for _, r in df.nsmallest(2, "DELTA").iterrows():
        print(f"  extrémní pokles: {r['local']:%d. %m. %H:%M} místního času, "
              f"DELTA = {r['DELTA']:.1f} °C na {r['VALUE']:.1f} °C")


def elbow(ks, wcss):
    x_n = (ks - ks[0]) / (ks[-1] - ks[0])
    y_n = (wcss - wcss[-1]) / (wcss[0] - wcss[-1])
    return int(ks[np.argmax((1 - x_n) - y_n)])


def stability(X, k, ref_labels, rng):
    aris = []
    for b in range(N_BOOT):
        idx = rng.choice(len(X), len(X), replace=True)
        model = KMeans(k, n_init=5, random_state=b).fit(X[idx])
        aris.append(adjusted_rand_score(ref_labels, model.predict(X)))
    return np.mean(aris), np.min(aris)


def score(X):
    section(f"Kritéria pro k = 1–{MAX_CLUSTERS} (k-means na standardizovaných VALUE, DELTA)")
    ks = np.arange(1, MAX_CLUSTERS + 1)
    rng = np.random.default_rng(0)
    rows, models = [], {}
    for k in ks:
        km = KMeans(k, n_init=N_INIT, random_state=0).fit(X)
        models[k] = km
        bic = GaussianMixture(k, n_init=3, random_state=0).fit(X).bic(X)
        row = {"k": k, "wcss": km.inertia_, "bic": bic,
               "smallest": np.bincount(km.labels_).min()}
        if k > 1:
            row["silhouette"] = silhouette_score(X, km.labels_)
            row["ch"] = calinski_harabasz_score(X, km.labels_)
            row["db"] = davies_bouldin_score(X, km.labels_)
            row["ari_mean"], row["ari_min"] = stability(X, k, km.labels_, rng)
        rows.append(row)
    res = pd.DataFrame(rows).set_index("k")

    print(f"{'k':>2} {'WCSS':>8} {'silueta':>8} {'C-H':>7} {'D-B':>6} "
          f"{'ARI':>6} {'ARI min':>8} {'BIC GMM':>9} {'nejmenší':>9}")
    for k, r in res.iterrows():
        f = lambda key, fmt: format(r[key], fmt) if pd.notna(r.get(key)) else "–"
        print(f"{k:>2} {r['wcss']:8.1f} {f('silhouette', '8.3f'):>8} {f('ch', '7.0f'):>7} "
              f"{f('db', '6.3f'):>6} {f('ari_mean', '6.3f'):>6} {f('ari_min', '8.3f'):>8} "
              f"{r['bic']:9.0f} {int(r['smallest']):9d}")

    valid = res.loc[2:]
    picks = {
        "loket (WCSS)": elbow(ks, res["wcss"].to_numpy()),
        "silueta (max)": int(valid["silhouette"].idxmax()),
        "Calinski–Harabasz (max)": int(valid["ch"].idxmax()),
        "Davies–Bouldin (min)": int(valid["db"].idxmin()),
        "stabilita ARI (max, k ≥ 3)": int(valid.loc[3:, "ari_mean"].idxmax()),
        "BIC GMM (min)": int(res["bic"].idxmin()),
    }
    section("Doporučení jednotlivých kritérií")
    for name, k in picks.items():
        tiny = " – obsahuje shluk z odlehlých bodů" if res.loc[k, "smallest"] < MIN_CLUSTER else ""
        print(f"{name:28s} k = {k}{tiny}")
    first_tiny = res.index[(res["smallest"] < MIN_CLUSTER)].min()
    print(f"od k = {first_tiny} vzniká shluk s méně než {MIN_CLUSTER} body "
          f"(extrémní poklesy teploty), nejde o samostatný režim")
    return res, models, picks


def recommend(res, picks):
    """Majority vote among the criteria, ignoring k that only isolate outliers.
    k = 2 is excluded from the stability vote since it is trivially stable."""
    votes = pd.Series([k for k in picks.values() if res.loc[k, "smallest"] >= MIN_CLUSTER])
    best = int(votes.mode().min())
    section(f"Doporučený počet shluků: k = {best}")
    print(f"hlasy kritérií: {', '.join(f'k={k}' for k in picks.values())}")
    r = res.loc[best]
    print(f"silueta = {r['silhouette']:.3f} (max), stabilita ARI = {r['ari_mean']:.3f}, "
          f"loket WCSS: {r['wcss'] / res.loc[1, 'wcss'] * 100:.0f} % z k = 1")
    return best


def describe_clusters(df, X, km, k):
    section(f"Shluky pro k = {k} (seřazeny podle DELTA)")
    order = np.argsort(km.cluster_centers_[:, 1])
    labels = np.argsort(order)[km.labels_]
    hours = df["local"].dt.hour.to_numpy()
    for c in range(k):
        m = labels == c
        v, d = df["VALUE"][m], df["DELTA"][m]
        print(f"shluk {c + 1}: n = {m.sum():4d} ({m.mean() * 100:4.1f} %), "
              f"T = {v.mean():5.1f} °C [{v.min():.1f}; {v.max():.1f}], "
              f"DELTA = {d.mean():+.2f} °C [{d.min():+.1f}; {d.max():+.1f}], "
              f"medián hodiny = {np.median(hours[m]):.0f} h")
    return labels


def plot_criteria(res, picks, best):
    ks = res.index.to_numpy()
    panels = [
        ("wcss", "Vnitroshlukový součet čtverců (loket)", picks["loket (WCSS)"]),
        ("silhouette", "Silueta (max)", picks["silueta (max)"]),
        ("ch", "Calinski–Harabasz (max)", picks["Calinski–Harabasz (max)"]),
        ("db", "Davies–Bouldin (min)", picks["Davies–Bouldin (min)"]),
        ("ari_mean", "Stabilita – průměrné ARI (max)", picks["stabilita ARI (max, k ≥ 3)"]),
        ("bic", "BIC Gaussovy směsi (min)", picks["BIC GMM (min)"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), layout="constrained")
    for ax, (key, title, pick) in zip(axes.ravel(), panels):
        s = res[key].dropna()
        ax.plot(s.index, s.values, color=C_DATA, marker="o", ms=5)
        ax.axvline(best, color=MUTED, linewidth=1, linestyle="--", zorder=0)
        ax.plot(pick, s[pick], marker="o", ms=10, color=C_FLAG, zorder=3)
        ax.set(title=title, xlabel="počet shluků k", xticks=ks)
    fig.suptitle(f"Volba počtu shluků k-means (VALUE, DELTA); čárkovaně doporučené k = {best}",
                 color=INK, fontsize=12)
    handle = Line2D([], [], linestyle="none", marker="o", ms=9, color=C_FLAG,
                    label="volba daného kritéria")
    fig.legend(handles=[handle], loc="outside lower center")
    save(fig, "01_kriteria")


def plot_clusters(df, labels, k):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for c in range(k):
        m = labels == c
        ax.scatter(df["VALUE"][m], df["DELTA"][m], s=9, alpha=0.6, color=CLUSTER_COLORS[c],
                   linewidths=0, label=f"shluk {c + 1} (n = {m.sum()})")
    ax.set(title=f"Shluky k-means pro k = {k}", xlabel="teplota VALUE [°C]",
           ylabel="změna za 10 min DELTA [°C]")
    ax.legend(loc="lower right", markerscale=2)
    save(fig, "02_shluky")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    describe(df)
    X = StandardScaler().fit_transform(df[["VALUE", "DELTA"]])
    res, models, picks = score(X)
    best = recommend(res, picks)
    labels = describe_clusters(df, X, models[best], best)
    plot_criteria(res, picks, best)
    plot_clusters(df, labels, best)
    df.assign(CLUSTER=labels + 1).drop(columns="local").to_csv(
        OUT_DIR / "clusters.csv", index=False)
    print(f"\nGrafy uloženy do {FIG_DIR}, přiřazení shluků do {OUT_DIR / 'clusters.csv'}")


if __name__ == "__main__":
    main()
