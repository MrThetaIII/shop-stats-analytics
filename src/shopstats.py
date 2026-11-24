"""Shared helpers for the Shop Stats notebooks: paths, loading, statistics, plot style."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2024-11-01")
END = pd.Timestamp("2026-10-31")
DAYS = pd.date_range(START, END, freq="D")
D = len(DAYS)
FLAGS = ["viewed_product", "added_to_cart", "began_checkout", "entered_payment", "purchased"]
STAGES = ["sessions"] + FLAGS
STAGE_LABELS = ["Visit", "Product view", "Add to cart", "Checkout", "Payment", "Purchase"]
STEP_NAMES = ["visit→view", "view→cart", "cart→checkout", "checkout→payment", "payment→purchase"]

# ---- plot style (one accent + grey; fixed categorical order) ----------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GREY, INK, INK2 = "#9a9993", "#0b0b0b", "#52514e"
GOOD, BAD = "#0ca30c", "#d03b3b"


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c8c2",
        "axes.grid": True, "grid.color": "#e9e8e3", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.8, "font.family": "DejaVu Sans",
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def day_index(date):
    return int((pd.Timestamp(date) - START).days)


def window(start, end):
    m = np.zeros(D, bool)
    m[max(day_index(start), 0):min(day_index(end), D - 1) + 1] = True
    return m


# ---- loading -----------------------------------------------------------------
def load_sessions(columns=None):
    s = pd.read_parquet(DATA / "sessions.parquet", columns=columns)
    if "session_start" in s.columns:
        s["day"] = (s["session_start"].values.astype("datetime64[D]") - np.datetime64(START, "D")).astype("int16")
    return s


def bot_mask(s):
    """Suspected bot sessions (identified in notebook 01): Direct / desktop, one page, <= 2 seconds,
    during the 3-6 Nov 2025 wave. Needs columns day, channel, device, pageviews, duration_sec."""
    return ((s["pageviews"] == 1) & (s["duration_sec"] <= 2) & s["channel"].eq("Direct") & s["device"].eq("desktop")
            & s["day"].between(day_index("2025-11-03"), day_index("2025-11-06"))).values


def load_orders():
    o = pd.read_parquet(DATA / "orders.parquet")
    o["day"] = (o["order_ts"].values.astype("datetime64[D]") - np.datetime64(START, "D")).astype("int16")
    o["day"] = o["day"].clip(upper=D - 1)      # a few orders complete minutes after midnight on the last day
    o["profit"] = o["revenue"] - o["cogs"]
    return o


def load_events():
    ev = pd.read_csv(DATA / "events_calendar.csv", parse_dates=["start_date", "end_date"])
    pr = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    return ev, pr


# ---- statistics --------------------------------------------------------------
def wilson(k, n, alpha=0.05):
    """Wilson score interval for a proportion (vectorised). Returns (p, lo, hi)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    z = stats.norm.ppf(1 - alpha / 2)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, mid - half, mid + half


def two_prop(k1, n1, k2, n2, alpha=0.05):
    """Two-proportion z-test, group 2 vs group 1 (baseline).
    Returns dict with rates, absolute diff + CI, relative lift + CI (log method), z, p, Cohen's h."""
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se0 = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    zstat = (p2 - p1) / se0 if se0 > 0 else np.nan
    pval = 2 * stats.norm.sf(abs(zstat))
    zc = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    se_log = np.sqrt((1 - p1) / k1 + (1 - p2) / k2) if min(k1, k2) > 0 else np.nan
    rr = p2 / p1 if p1 > 0 else np.nan
    return dict(p1=p1, p2=p2, diff=p2 - p1, diff_lo=p2 - p1 - zc * se, diff_hi=p2 - p1 + zc * se,
                lift=rr - 1, lift_lo=rr * np.exp(-zc * se_log) - 1, lift_hi=rr * np.exp(zc * se_log) - 1,
                z=zstat, p=pval, h=2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(q)
    adj[order] = np.minimum(ranked, 1)
    out[ok] = adj
    return out


def cramers_v(table):
    """Chi-square test of homogeneity on a contingency table + Cramér's V effect size."""
    table = np.asarray(table, float)
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return chi2, p, dof, v


def ratio_delta(num_c, den_c, num_t, den_t, alpha=0.05):
    """Ratio metric (sum num / sum den) compared between two arms when the randomisation unit
    (user) differs from the analysis unit (session). Inputs are per-user arrays.
    Variance by the delta method, which accounts for within-user correlation."""
    def arm(num, den):
        n = len(num)
        mn, md = num.mean(), den.mean()
        r = mn / md
        cov = np.cov(num, den)
        var = (cov[0, 0] / md ** 2 - 2 * mn * cov[0, 1] / md ** 3 + mn ** 2 * cov[1, 1] / md ** 4) / n
        return r, var
    rc, vc = arm(np.asarray(num_c, float), np.asarray(den_c, float))
    rt, vt = arm(np.asarray(num_t, float), np.asarray(den_t, float))
    se = np.sqrt(vc + vt)
    zstat = (rt - rc) / se
    zc = stats.norm.ppf(1 - alpha / 2)
    se_rel = np.sqrt(vt / rc ** 2 + vc * rt ** 2 / rc ** 4)
    return dict(p1=rc, p2=rt, diff=rt - rc, diff_lo=rt - rc - zc * se, diff_hi=rt - rc + zc * se,
                lift=rt / rc - 1, lift_lo=rt / rc - 1 - zc * se_rel, lift_hi=rt / rc - 1 + zc * se_rel,
                z=zstat, p=2 * stats.norm.sf(abs(zstat)), se=se)


def funnel_rates(df):
    """Add step-to-step and overall conversion columns to a frame holding stage counts."""
    out = df.copy()
    for prev, cur, name in zip(STAGES[:-1], STAGES[1:], STEP_NAMES):
        out[name] = out[cur] / out[prev]
    out["overall"] = out["purchased"] / out["sessions"]
    return out


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def fmt_p(p):
    return "<1e-300" if p == 0 else f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
