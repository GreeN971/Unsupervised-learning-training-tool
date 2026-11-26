"""Time-series EDA of the Cheb 10-minute air temperature series.

Same methodology as ``analysis/analysis.py`` (thesis chapter 7.3): description,
normality, 3-sigma rule, MAD, box plot, histogram, ECDF, k-means with the elbow
rule, DBSCAN, iterative removal and the sequence check. The difference is that
the timestamp DT is kept: the series is treated as a time series of
temperature, not as an unordered sample.

Air temperature has a strong daily cycle, so a value is only unusual relative
to what is typical at that time of day. Each reading is therefore compared with
the median of all readings taken in the same local hour of the day during the
month; the univariate rules are applied to this deviation from the hourly
median. Global rules on the raw values and a rolling (Hampel) filter are shown
alongside for comparison.

Timestamps in the export are UTC; plots and hour-of-day statistics use local
time (Europe/Prague, CEST = UTC+2 in July). The cleaned two-column dataset
(time in UTC, temperature) is written next to the figures. Run from the repo
root:

    .venv/bin/python analysis/time-series/time-series-analysis.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
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
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

_REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = _REPO_ROOT / "datasets" / "10m-0-20000-0-11406-T-202507.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = OUT_DIR / "figures"
LOCAL_TZ = "Europe/Prague"

# Readings per day (10-minute step).
PERIOD = 144
# DBSCAN's min_samples; the k-distance graph uses the same K.
K = 10
# Modified z-score cut-off recommended by Iglewicz & Hoaglin for MAD.
MAD_THRESHOLD = 3.5
# Hampel filter: centred window of 13 readings = +-1 hour around each point.
HAMPEL_WINDOW = 13
# Upper bound for the iterative loops.
MAX_ITER = 10
# Largest number of k-means clusters tried by the elbow rule.
MAX_CLUSTERS = 10
# ACF is shown up to three days of lags.
ACF_LAGS = 3 * PERIOD

DEV_LABEL = "odchylka od mediánu hodiny [°C]"

# Reference palette (dataviz skill) - light surface only, figures go to print.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_DATA = "#2a78d6"   # slot 1 blue - the data
C_FLAG = "#eb6834"   # slot 2 orange - flagged values / thresholds
C_REF = "#1baf7a"    # slot 3 aqua - reference (normal distribution, typical value)

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

# SAS ODS Graphics look (HTMLBlue style) for the k-means and DBSCAN figures,
# identical to analysis.py.
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
    "savefig.bbox": "standard",
}


def load_series():
    """Keep DT and VALUE from the raw export as a temperature time series."""
    raw = pd.read_csv(CSV_PATH, parse_dates=["DT"])
    df = raw[["DT", "VALUE"]].rename(columns={"DT": "time", "VALUE": "temperature"})
    return df.sort_values("time").reset_index(drop=True)


class CommaFormatter(ScalarFormatter):
    """Tick labels with a decimal comma, as the figures go into Czech text."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def num(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def stamp(ts):
    return ts.strftime("%d. %m. %H:%M")


def date_axis(ax):
    """Day ticks in Czech format on a local-time axis."""
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 5, 10, 15, 20, 25, 30]))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d. %-m."))


def save(fig, name):
    # Decimal comma on numeric axes only; date and categorical axes keep theirs.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(CommaFormatter())
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)


def section(title):
    print(f"\n=== {title} ===")


def describe(t, local):
    section("Popisná statistika")
    step = local.diff().dropna()
    full = pd.date_range(local.iloc[0], local.iloc[-1], freq="10min")
    print(f"období: {stamp(local.iloc[0])} – {stamp(local.iloc[-1])} (místní čas), "
          f"krok = {step.mode()[0]}, chybějících časů = {full.size - local.size}, "
          f"nepravidelných kroků = {(step != step.mode()[0]).sum()}")
    q = t.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"n = {t.size}, chybějící = {t.isna().sum()}, unikátních = {t.nunique()}")
    print(f"průměr = {t.mean():.2f}, směr. odchylka = {t.std():.2f}")
    print(f"min = {t.min():.1f} ({stamp(local[t.idxmin()])}), "
          f"max = {t.max():.1f} ({stamp(local[t.idxmax()])}), rozpětí = {t.max() - t.min():.1f}")
    print("kvantily:", ", ".join(f"P{int(p * 100)}={v:.1f}" for p, v in q.items()))
    print(f"šikmost = {stats.skew(t):.3f}, špičatost (exces) = {stats.kurtosis(t):.3f}")

    # Local days only when complete: the UTC export starts and ends at 02:00 local.
    daily = t.groupby(local.dt.date).agg(["min", "mean", "max", "size"])
    daily = daily[daily["size"] == PERIOD]
    amp = daily["max"] - daily["min"]
    print(f"úplných místních dní = {len(daily)}")
    print(f"denní amplituda: průměr = {amp.mean():.1f}, min = {amp.min():.1f} "
          f"({amp.idxmin():%d. %m.}), max = {amp.max():.1f} ({amp.idxmax():%d. %m.}) °C")
    print(f"denní průměr: {daily['mean'].min():.1f} – {daily['mean'].max():.1f} °C, "
          f"nejteplejší den {daily['mean'].idxmax():%d. %m.}, "
          f"nejchladnější {daily['mean'].idxmin():%d. %m.}")
    hourly = t.groupby(local.dt.hour).mean()
    print(f"průměrný denní chod: minimum v {hourly.idxmin()}:00 ({hourly.min():.1f} °C), "
          f"maximum v {hourly.idxmax()}:00 ({hourly.max():.1f} °C)")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(local, t, color=C_DATA, linewidth=0.7, label="teplota (10 min)")
    ax.plot(local, t.rolling(PERIOD, center=True, min_periods=PERIOD // 2).mean(),
            color=C_REF, linewidth=1.8, label="klouzavý průměr 24 h")
    ax.set(title="Teplota v čase (Cheb, červenec 2025)", xlabel="datum (místní čas)",
           ylabel="teplota [°C]", xlim=(local.iloc[0], local.iloc[-1]))
    date_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "01_time_series")


def hourly_deviation(t, local):
    """Deviation of each reading from the median of its local hour of day."""
    section("Odchylka od mediánu hodiny dne")
    hours = local.dt.hour
    profile = t.groupby(hours).median()
    dev = t - hours.map(profile)
    print("medián podle hodiny:",
          ", ".join(f"{h}h={v:.1f}" for h, v in profile.items()))
    print(f"rozpětí typického denního chodu = {profile.max() - profile.min():.1f} °C")
    print(f"rozptyl: teplota = {t.var():.2f}, odchylka = {dev.var():.2f} "
          f"(denní chod vysvětluje {(1 - dev.var() / t.var()) * 100:.0f} %)")
    print(f"odchylka: průměr = {dev.mean():.3f}, směr. odchylka = {dev.std():.2f}, "
          f"min = {dev.min():.2f} ({stamp(local[dev.idxmin()])}), "
          f"max = {dev.max():.2f} ({stamp(local[dev.idxmax()])})")
    return dev


def autocorrelation(t, dev):
    """The rules below assume independent observations; the ACF shows how far
    from that the raw series is and how much removing the daily cycle helps."""
    section("Autokorelace")
    a_raw = acf(t, nlags=ACF_LAGS, fft=True)
    a_dev = acf(dev, nlags=ACF_LAGS, fft=True)
    for lag in [1, 6, 36, 72, PERIOD]:
        print(f"ACF lag {lag:3d} ({lag * 10 / 60:4.1f} h): teplota = {a_raw[lag]:.3f}, "
              f"odchylka = {a_dev[lag]:.3f}")
    for name, x in [("teplota", t), ("odchylka", dev)]:
        lb = acorr_ljungbox(x, lags=[PERIOD])
        print(f"Ljung-Box ({name}, lag {PERIOD}): Q = {lb['lb_stat'].iloc[0]:.0f}, "
              f"p = {lb['lb_pvalue'].iloc[0]:.2e}")
    # Effective sample size for the mean under AR(1): n (1 - rho) / (1 + rho).
    rho = a_raw[1]
    print(f"efektivní velikost výběru (AR(1), rho = {rho:.3f}): "
          f"{t.size * (1 - rho) / (1 + rho):.0f} z {t.size}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    lags_h = np.arange(ACF_LAGS + 1) * 10 / 60
    band = 1.96 / np.sqrt(t.size)
    for ax, a, title in [(axes[0], a_raw, "ACF teploty"),
                         (axes[1], a_dev, "ACF odchylky od mediánu hodiny")]:
        ax.fill_between(lags_h, -band, band, color=GRID, label="95% pásmo bílého šumu")
        ax.vlines(lags_h, 0, a, color=C_DATA, linewidth=0.8)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set(title=title, ylabel="autokorelace")
    axes[0].legend(loc="upper right")
    axes[1].set(xlabel="zpoždění [h]", xticks=np.arange(0, 73, 12), xlim=(0, 72))
    save(fig, "02_acf")


def normality(t, dev):
    section("Test normality")
    for name, x in [("teplota", t), ("odchylka", dev)]:
        sw = stats.shapiro(x)
        k2 = stats.normaltest(x)
        ks = stats.kstest(x, stats.norm(x.mean(), x.std()).cdf)
        print(f"[{name}] Shapiro-Wilk W = {sw.statistic:.4f}, p = {sw.pvalue:.2e}; "
              f"K2 = {k2.statistic:.1f}, p = {k2.pvalue:.2e}; "
              f"KS D = {ks.statistic:.4f}, p = {ks.pvalue:.2e}; "
              f"šikmost = {stats.skew(x):.2f}, exces = {stats.kurtosis(x):.2f}")
    print("(p-hodnoty předpokládají nezávislá pozorování, viz ACF - jsou jen orientační)")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, x, title, unit in [(axes[0], t, "teplota", "teplota [°C]"),
                               (axes[1], dev, "odchylka od mediánu hodiny", DEV_LABEL)]:
        (osm, osr), (slope, intercept, _) = stats.probplot(x, dist="norm")
        ax.plot(osm, osr, "o", ms=3, color=C_DATA, alpha=0.5, label="naměřené hodnoty")
        ax.plot(osm, slope * osm + intercept, color=C_REF, label="normální rozdělení")
        ax.set(title=f"Q-Q graf: {title}", xlabel="teoretické kvantily N(0, 1)", ylabel=unit)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "03_qq_plot")


def three_sigma(x, label):
    m, s = x.mean(), x.std()
    lo, hi = m - 3 * s, m + 3 * s
    mask = (x < lo) | (x > hi)
    print(f"[{label}] meze = [{lo:.2f}; {hi:.2f}] °C, mimo mez: {mask.sum()} "
          f"({mask.mean() * 100:.2f} %), pod: {(x < lo).sum()}, nad: {(x > hi).sum()}")
    return mask, (lo, hi)


def mad_rule(x, label):
    med = x.median()
    mad = stats.median_abs_deviation(x)
    z = 0.6745 * (x - med) / mad
    mask = z.abs() > MAD_THRESHOLD
    lo, hi = med - MAD_THRESHOLD * mad / 0.6745, med + MAD_THRESHOLD * mad / 0.6745
    print(f"[{label}] medián = {med:.2f}, MAD = {mad:.2f} (škálovaná = {mad / 0.6745:.2f}), "
          f"meze = [{lo:.2f}; {hi:.2f}] °C, mimo mez: {mask.sum()} ({mask.mean() * 100:.2f} %)")
    return mask, (lo, hi)


def iqr_rule(x, label):
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (x < lo) | (x > hi)
    print(f"[{label}] Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}, "
          f"ploty = [{lo:.2f}; {hi:.2f}] °C, mimo: {mask.sum()} ({mask.mean() * 100:.2f} %)")
    return mask, (lo, hi)


def hampel(t):
    """Rolling MAD rule: each reading against the median of its +-1 h window."""
    med = t.rolling(HAMPEL_WINDOW, center=True, min_periods=1).median()
    mad = (t - med).abs().rolling(HAMPEL_WINDOW, center=True, min_periods=1).median()
    # A window with identical readings has MAD = 0; use the sensor resolution.
    scale = np.maximum(mad / 0.6745, 0.1)
    mask = (t - med).abs() > MAD_THRESHOLD * scale
    print(f"[Hampel, okno {HAMPEL_WINDOW} měření = ±{(HAMPEL_WINDOW // 2) * 10} min] "
          f"mimo mez: {mask.sum()} ({mask.mean() * 100:.2f} %)")
    return mask


def rules(t, dev, local):
    """3-sigma, MAD and IQR globally on the raw values (as in analysis.py) and on
    the deviation from the hourly median, plus the Hampel filter."""
    section("Pravidlo 3σ")
    s_raw, _ = three_sigma(t, "teplota")
    s_dev, _ = three_sigma(dev, "odchylka")
    print(f"očekáváno při normalitě: {2 * stats.norm.sf(3) * t.size:.1f}")
    section("MAD (modifikované z-skóre)")
    m_raw, _ = mad_rule(t, "teplota")
    m_dev, m_bounds = mad_rule(dev, "odchylka")
    h_mask = hampel(t)
    section("Krabicový graf (IQR)")
    i_raw, _ = iqr_rule(t, "teplota")
    i_dev, i_bounds = iqr_rule(dev, "odchylka")

    section("Kdy pravidla označují (místní hodina)")
    hours = local.dt.hour
    flags = {"3σ teplota": s_raw, "MAD teplota": m_raw, "IQR teplota": i_raw,
             "3σ odchylka": s_dev, "MAD odchylka": m_dev, "IQR odchylka": i_dev,
             "Hampel": h_mask}
    for name, mask in flags.items():
        h = hours[mask]
        days = local[mask].dt.date.nunique()
        span = f"{h.min():2d}–{h.max():2d} h, medián {h.median():4.1f} h" if mask.any() else "-"
        print(f"{name:13s}: {mask.sum():4d} bodů, {days:2d} dní, hodiny {span}")
    both = m_dev & h_mask
    print(f"MAD odchylka ∩ Hampel: {both.sum()} bodů: "
          + ", ".join(f"{stamp(local[i])} {t[i]:.1f} °C" for i in both[both].index[:12]))
    print(f"MAD teplota ∩ MAD odchylka: {(m_raw & m_dev).sum()} bodů")

    plot_flags(t, local, [("3σ na teplotě", s_raw),
                          ("MAD na teplotě", m_raw),
                          ("MAD na odchylce od mediánu hodiny", m_dev),
                          (f"Hampelův filtr (okno ±{(HAMPEL_WINDOW // 2) * 10} min)", h_mask)])
    return m_dev, m_bounds, i_bounds


def plot_flags(t, local, methods):
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 2.3 * len(methods)),
                             sharex=True, sharey=True)
    for ax, (title, mask) in zip(axes, methods):
        ax.plot(local, t, color=C_DATA, linewidth=0.6)
        ax.scatter(local[mask], t[mask], s=12, color=C_FLAG, zorder=3)
        ax.set(title=f"{title}: označeno {mask.sum()} hodnot", ylabel="teplota [°C]")
    axes[-1].set(xlabel="datum (místní čas)", xlim=(local.iloc[0], local.iloc[-1]))
    date_axis(axes[-1])
    fig.tight_layout()
    save(fig, "04_flags_in_time")


def plot_histogram(dev, mad_bounds):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(dev, bins=np.arange(np.floor(dev.min()), dev.max() + 0.5, 0.5),
            density=True, color=C_DATA, alpha=0.85, edgecolor=SURFACE, linewidth=0.6,
            label="histogram (šířka třídy 0,5 °C)")
    x = np.linspace(dev.min() - 1, dev.max() + 1, 600)
    ax.plot(x, stats.gaussian_kde(dev)(x), color=INK_2, label="jádrový odhad hustoty")
    ax.plot(x, stats.norm.pdf(x, dev.mean(), dev.std()), color=C_REF, linestyle="--",
            label="normální rozdělení N(x̄, s)")
    for b in mad_bounds:
        ax.axvline(b, color=C_FLAG, linewidth=1.5)
    ax.plot([], [], color=C_FLAG,
            label=f"meze MAD: {num(mad_bounds[0], 2)} – {num(mad_bounds[1], 2)} °C")
    ax.set(title="Rozdělení odchylky od mediánu hodiny", xlabel=DEV_LABEL, ylabel="hustota")
    ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=SURFACE,
              framealpha=1)
    save(fig, "05_histogram")


def plot_boxplot(t, dev, local, iqr_bounds):
    """Hour-of-day box plots (the medians are the reference the deviation is
    measured from) and the box plot of the deviation."""
    flier = dict(marker="o", markersize=3, markerfacecolor=C_FLAG,
                 markeredgecolor=SURFACE, alpha=0.7)
    box = dict(patch_artist=True, boxprops=dict(facecolor="#cde2fb", edgecolor=C_DATA),
               medianprops=dict(color=C_DATA, linewidth=2),
               whiskerprops=dict(color=C_DATA), capprops=dict(color=C_DATA),
               flierprops=flier)
    hours = local.dt.hour
    groups = [t[hours == h].to_numpy() for h in range(24)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), height_ratios=[2.2, 1])
    ax = axes[0]
    ax.boxplot(groups, widths=0.6, tick_labels=[str(h) for h in range(24)], **box)
    ax.grid(axis="x", visible=False)
    ax.set(title="Teplota podle hodiny dne (čára = medián hodiny)",
           xlabel="hodina (místní čas)", ylabel="teplota [°C]")
    ax = axes[1]
    ax.boxplot(dev, orientation="horizontal", widths=0.5, **box)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set(title=f"Krabicový graf odchylky (ploty: {num(iqr_bounds[0], 2)} a "
                 f"{num(iqr_bounds[1], 2)} °C)", xlabel=DEV_LABEL)
    fig.tight_layout()
    save(fig, "06_boxplot")

    out = 0
    for g in groups:
        q1, q3 = np.quantile(g, [0.25, 0.75])
        out += ((g < q1 - 1.5 * (q3 - q1)) | (g > q3 + 1.5 * (q3 - q1))).sum()
    print(f"[IQR po hodinách] mimo vousy svých hodinových krabic: {out} "
          f"({out / t.size * 100:.2f} %)")


def ecdf(dev):
    section("ECDF odchylky")
    x = np.sort(dev.to_numpy())
    y = np.arange(1, x.size + 1) / x.size
    unique = np.unique(x)
    gaps = np.diff(unique)
    order = np.argsort(gaps)[::-1][:5]
    print("největší mezery mezi sousedními hodnotami:",
          ", ".join(f"{unique[i]:.2f}->{unique[i + 1]:.2f} ({gaps[i]:.2f})" for i in order))
    for v in [dev.quantile(0.005), dev.quantile(0.995)]:
        print(f"kvantil ECDF {v:.2f} °C")
    m, s = dev.mean(), dev.std()
    for lim in [-4, -6, -8]:
        print(f"P(d < {lim} °C) empiricky = {(x < lim).mean() * 100:.2f} %, "
              f"normálně = {stats.norm.cdf(lim, m, s) * 100:.2f} %")
    for lim in [4, 6, 8]:
        print(f"P(d > {lim} °C) empiricky = {(x > lim).mean() * 100:.2f} %, "
              f"normálně = {stats.norm.sf(lim, m, s) * 100:.2f} %")

    lo_tail, hi_tail = np.quantile(x, [0.05, 0.95])
    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25)
    panels = [
        (fig.add_subplot(gs[0, :]), None, None, "ECDF odchylky od mediánu hodiny"),
        (fig.add_subplot(gs[1, 0]), (x.min() - 0.5, lo_tail), (-0.0005, 0.05),
         "Detail dolního chvostu"),
        (fig.add_subplot(gs[1, 1]), (hi_tail, x.max() + 0.5), (0.95, 1.0005),
         "Detail horního chvostu"),
    ]
    grid = np.linspace(x.min() - 2, x.max() + 2, 600)
    x_step, y_step = np.r_[grid[0], x], np.r_[0, y]
    for ax, xlim, ylim, title in panels:
        ax.step(x_step, y_step, where="post", color=C_DATA, label="empirická (ECDF)")
        ax.plot(grid, stats.norm.cdf(grid, m, s), color=C_REF, linestyle="--",
                label="normální CDF")
        if xlim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        ax.set(title=title, xlabel=DEV_LABEL, ylabel="F(x)")
    panels[0][0].legend(loc="lower right")
    save(fig, "07_ecdf")


def run_dbscan(X, eps):
    """DBSCAN with a tiny float tolerance on eps, as in analysis.py."""
    return DBSCAN(eps=eps + 1e-9, min_samples=K).fit_predict(X)


def kneedle(values):
    """Index of the point farthest from the chord of the sorted, normalised curve."""
    y = np.sort(values)
    x = np.linspace(0, 1, y.size)
    yn = (y - y[0]) / (y[-1] - y[0])
    return int(np.argmax(x - yn)), y


def sas_frame(fig):
    fig.patches.append(Rectangle((0.001, 0.001), 0.998, 0.998, transform=fig.transFigure,
                                 fill=False, edgecolor=SAS_BORDER, linewidth=1))


def kmeans_elbow(dev, local, flagged):
    section(f"k-means na odchylce: pravidlo lokte (k = 1–{MAX_CLUSTERS})")
    X = dev.to_numpy().reshape(-1, 1)
    ks = np.arange(1, MAX_CLUSTERS + 1)
    models = [KMeans(k, n_init=20, random_state=0).fit(X) for k in ks]
    wcss = np.array([m.inertia_ for m in models])
    x_n = (ks - ks[0]) / (ks[-1] - ks[0])
    y_n = (wcss - wcss[-1]) / (wcss[0] - wcss[-1])
    gap = (1 - x_n) - y_n
    best = int(ks[np.argmax(gap)])
    for k, w, g in zip(ks, wcss, gap):
        print(f"k = {k:2d}: WCSS = {w:9.1f} ({w / wcss[0] * 100:5.1f} % z k = 1), "
              f"odstup od spojnice = {g:.3f}")
    print(f"loket: k = {best}")

    labels = models[best - 1].labels_
    order = np.argsort(models[best - 1].cluster_centers_.ravel())
    labels = np.argsort(order)[labels]
    x = X[:, 0]
    hours = local.dt.hour.to_numpy()
    days = local.dt.date.to_numpy()
    for c in range(best):
        m = labels == c
        print(f"  shluk {c + 1}: [{x[m].min():.2f}; {x[m].max():.2f}] °C, n = {m.sum()}, "
              f"střed = {x[m].mean():.2f} °C, mimo mez MAD: {(m & flagged.to_numpy()).sum()}, "
              f"dní = {len(set(days[m]))}, medián hodiny = {np.median(hours[m]):.0f} h")
    plot_elbow(ks, wcss, best, x, labels, local)


def plot_elbow(ks, wcss, best, x, labels, local):
    colors = [SAS_BLUE, SAS_RED, SAS_TEAL, "#543005", "#9D3CDB", "#7F8E1F"]
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
        ax.set(title="Pravidlo lokte (k-means na odchylce od mediánu hodiny)",
               xlabel="Počet shluků k", ylabel="Vnitroshlukový součet čtverců", xticks=ks)
        ax.set_ylim(bottom=0)

        ax = axes[1]
        for c in range(best):
            m = labels == c
            ax.scatter(local[m], x[m], s=10, marker="o", facecolors="none",
                       edgecolors=colors[c], linewidths=0.5,
                       label=f"Shluk {c + 1}: {num(x[m].min(), 2)} až {num(x[m].max(), 2)} °C")
        ax.set(title=f"Shluky k-means pro k = {best} v čase", xlabel="Datum (místní čas)",
               ylabel="Odchylka [°C]", xlim=(local.iloc[0], local.iloc[-1]))
        date_axis(ax)
        handles = [Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                          markeredgecolor=colors[c], markersize=8,
                          label=h.get_label()) for c, h in enumerate(ax.collections[:best])]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                  ncol=min(best, 3), borderaxespad=0, columnspacing=1.5)
        sas_frame(fig)
        save(fig, "08_kmeans_loket")


def dbscan(dev, local):
    section(f"DBSCAN na odchylce (min_samples = {K})")
    X = dev.to_numpy().reshape(-1, 1)
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
    # Smallest eps at which the tail fragments merge into one cluster.
    eps_stable = next((e for e, (c, _) in zip(grid, sensitivity) if c == 1), eps_knee)

    results = {}
    for e in dict.fromkeys([eps_knee, eps_stable]):
        labels = run_dbscan(X, e)
        noise = labels == -1
        noise_days = sorted(set(local[noise].dt.strftime("%d. %m.")))
        print(f"eps = {e:.1f}: šum ({noise.sum()}), dny: {', '.join(noise_days) or '-'}")
        for i in np.flatnonzero(noise)[:20]:
            print(f"    {stamp(local[i])}: {X[i, 0]:+.2f} °C")
        for c in sorted(set(labels) - {-1}):
            v = X[labels == c, 0]
            print(f"  shluk {c}: n = {v.size}, rozsah = [{v.min():.2f}; {v.max():.2f}] °C")
        results[e] = labels

    plot_dbscan(X, local, kth_sorted, eps_knee, eps_stable, grid, sensitivity, results)
    return eps_stable


def plot_dbscan(X, local, kth_sorted, eps_knee, eps_stable, grid, sensitivity, results):
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
        for e, (c, n) in zip(grid, sensitivity):
            ax.annotate(str(c), (e, n), xytext=(4, 4), textcoords="offset points",
                        fontsize=9)
        ax.set(title="Citlivost na eps (popisek = počet shluků)", xlabel="eps [°C]",
               ylabel="Počet bodů šumu", xticks=grid)
        ax.set_ylim(0, max(max(noise_counts) * 1.12, 1))

        open_circle = dict(marker="o", facecolors="none", linewidths=0.5)
        for row, e in enumerate([eps_knee, eps_stable], start=1):
            labels = results[e]
            noise = labels == -1
            main = labels == np.bincount(labels[~noise]).argmax()
            minor = ~noise & ~main
            n_clusters = len(set(labels) - {-1})
            ax = fig.add_subplot(gs[row, :])
            ax.scatter(local[main], X[main, 0], s=8, edgecolors=SAS_BLUE, **open_circle)
            ax.scatter(local[minor], X[minor, 0], s=8, edgecolors=SAS_TEAL, **open_circle)
            ax.scatter(local[noise], X[noise, 0], s=24, color=SAS_RED, zorder=3)
            ax.set(title=f"DBSCAN, eps = {num(e)} °C (shluky: {n_clusters}, šum: {noise.sum()})",
                   xlabel="Datum (místní čas)", ylabel="Odchylka [°C]",
                   xlim=(local.iloc[0], local.iloc[-1]))
            date_axis(ax)
            handles = [Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                              markeredgecolor=SAS_BLUE, label="Hlavní shluk")]
            if minor.any():
                handles.append(Line2D([], [], linestyle="none", marker="o",
                                      markerfacecolor="none", markeredgecolor=SAS_TEAL,
                                      label=f"Malé shluky ({n_clusters - 1})"))
            handles.append(Line2D([], [], linestyle="none", marker="o", color=SAS_RED,
                                  label="Šum"))
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.3),
                      ncol=len(handles), borderaxespad=0, columnspacing=2)
        sas_frame(fig)
        save(fig, "09_dbscan")


def iterative_removal(dev, local, eps):
    """Repeat each rule on the deviations left after removing its previous flags."""
    section("Iterativní odstraňování (odchylka)")

    def sigma(x):
        return np.abs(x - x.mean()) > 3 * x.std(ddof=1)

    def mad(x):
        return np.abs(0.6745 * (x - np.median(x)) / stats.median_abs_deviation(x)) > MAD_THRESHOLD

    def density(x):
        return run_dbscan(x.reshape(-1, 1), eps) == -1

    history = {}
    for name, rule in [("3σ", sigma), ("MAD", mad), ("DBSCAN", density)]:
        x = dev.to_numpy()
        keep = np.arange(dev.size)
        removed = []
        for i in range(MAX_ITER):
            mask = rule(x)
            if not mask.any():
                break
            removed.append(int(mask.sum()))
            print(f"{name} iterace {i + 1}: odstraněno {mask.sum():3d}, "
                  f"rozsah zbytku = [{x[~mask].min():.2f}; {x[~mask].max():.2f}] °C, "
                  f"celkem {sum(removed)} ({sum(removed) / dev.size * 100:.2f} %)")
            x, keep = x[~mask], keep[~mask]
        if len(removed) == MAX_ITER:
            print(f"{name}: po {MAX_ITER} iteracích stále nekonverguje")
        if removed:
            gone = np.setdiff1d(np.arange(dev.size), keep)
            print(f"{name}: odstraněné hodnoty pocházejí z {local[gone].dt.date.nunique()} dní")
        history[name] = np.cumsum(removed)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (name, cum), color in zip(history.items(), [C_FLAG, C_DATA, C_REF]):
        if not cum.size:
            continue
        it = np.arange(1, cum.size + 1)
        ax.plot(it, cum, marker="o", ms=5, color=color, label=name)
        ax.annotate(name, (it[-1], cum[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set(title="Kumulativní počet odstraněných hodnot (odchylka)", xlabel="iterace",
           ylabel="odstraněno hodnot", xticks=range(1, MAX_ITER + 1))
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    save(fig, "10_iterative_removal")


def sequence_check(t, local):
    """10-minute changes, now with their real timestamps."""
    section("Změny mezi sousedními měřeními")
    d = t.diff()
    top = d.abs().sort_values(ascending=False).head(6)
    print("největší změny:",
          ", ".join(f"{stamp(local[i])}: {t[i - 1]:.1f} -> {t[i]:.1f} ({d[i]:+.1f})"
                    for i in top.index))
    m, s = d.mean(), d.std()
    print(f"změna za 10 min: průměr = {m:.3f}, směr. odchylka = {s:.2f} °C, "
          f"|změna| > 3 °C: {(d.abs() > 3).sum()}, mimo 3σ změn: {((d - m).abs() > 3 * s).sum()}")
    for i in top.index[:2]:
        after = t[i:i + 7]
        print(f"po poklesu {stamp(local[i])}: následující hodina {after.min():.1f} – "
              f"{after.max():.1f} °C (pokles přetrvává, nejde o ojedinělý bod)")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[1.2, 1])
    jumps = top.index[:2]
    ax = axes[0]
    ax.plot(local, t, color=C_DATA, linewidth=0.7)
    ax.scatter(local[jumps], t[jumps], s=40, color=C_FLAG, edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="největší skokové poklesy")
    # The two drops are a day apart: label the first to the left, the second to the right.
    for i, (dx, ha) in zip(jumps, [(-10, "right"), (10, "left")]):
        ax.annotate(f"{stamp(local[i])}\n{num(d[i])} °C", (local[i], t[i]), xytext=(dx, -26),
                    textcoords="offset points", fontsize=8, color=INK_2, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    ax.set(title="Teplota v čase", ylabel="teplota [°C]")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(local, d, color=INK_2, linewidth=0.6)
    for b in [m - 3 * s, m + 3 * s]:
        ax.axhline(b, color=C_FLAG, linewidth=1, linestyle="--")
    ax.plot([], [], color=C_FLAG, linestyle="--", label=f"±3σ změn (±{num(3 * s, 2)} °C)")
    ax.set(title="Změna teploty za 10 min", ylabel="Δ teplota [°C]",
           xlabel="datum (místní čas)", xlim=(local.iloc[0], local.iloc[-1]))
    ax.legend(loc="lower right")
    date_axis(ax)
    fig.tight_layout()
    save(fig, "11_sequence")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_series()
    df.assign(time=df["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(
        OUT_DIR / "temperature_time_series.csv", index=False)
    t = df["temperature"]
    # Local wall-clock time without tz info, so matplotlib plots it as is.
    local = df["time"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    describe(t, local)
    dev = hourly_deviation(t, local)
    autocorrelation(t, dev)
    normality(t, dev)
    mad_mask, mad_bounds, iqr_bounds = rules(t, dev, local)
    plot_histogram(dev, mad_bounds)
    plot_boxplot(t, dev, local, iqr_bounds)
    ecdf(dev)
    kmeans_elbow(dev, local, mad_mask)
    eps = dbscan(dev, local)
    iterative_removal(dev, local, eps)
    sequence_check(t, local)
    print(f"\nGrafy uloženy do {FIG_DIR}, data do {OUT_DIR / 'temperature_time_series.csv'}")


if __name__ == "__main__":
    main()
