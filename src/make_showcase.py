"""Ten presentation images summarising the Shop Stats project and its findings.

Reads the aggregates the notebooks exported to outputs/ (plus orders / order_items for two
drill-downs) and writes 2400x1350 PNGs to showcase/.

    python src/make_showcase.py          # all ten
    python src/make_showcase.py 3 7      # only images 3 and 7
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shopstats import (BAD, BLUE, AQUA, DATA, GOOD, GREY, INK, INK2, ORANGE, OUT, ROOT, START, VIOLET, YELLOW,
                       MAGENTA, GREEN, RED, set_style)

SHOW = ROOT / "showcase"
SHOW.mkdir(exist_ok=True)
BG, CARD, LINE = "#fbfaf7", "#ffffff", "#e3e1da"
PALE_BLUE, PALE_ORANGE = "#dbe8f8", "#fbe3d8"
FOOT = "Shop Stats  ·  synthetic e-commerce dataset  ·  1 Nov 2024 – 31 Oct 2026"

set_style()
mpl.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "axes.titlesize": 15, "axes.titlepad": 12, "axes.labelsize": 12,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5, "lines.linewidth": 2.2,
})


# ---------------------------------------------------------------- helpers ----
def r(name):
    return pd.read_csv(OUT / name)


def dates(day):
    return START + pd.to_timedelta(np.asarray(day, dtype="int64"), unit="D")


def frame(n, title, subtitle):
    fig = plt.figure(figsize=(16, 9))
    fig.add_artist(Rectangle((0.04, 0.955), 0.045, 0.007, color=BLUE, transform=fig.transFigure))
    fig.text(0.04, 0.905, title, fontsize=29, weight="bold", color=INK, va="center")
    fig.text(0.04, 0.855, subtitle, fontsize=15.5, color=INK2, va="center")
    fig.add_artist(Rectangle((0.04, 0.055), 0.92, 0.0012, color=LINE, transform=fig.transFigure))
    fig.text(0.04, 0.032, FOOT, fontsize=10.5, color=GREY, va="center")
    fig.text(0.96, 0.032, f"{n:02d} / 10", fontsize=10.5, color=GREY, va="center", ha="right")
    return fig


def finish(fig, n, name):
    path = SHOW / f"{n:02d}_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path.name)


def canvas(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    return ax


def card(ax, x, y, w, h, fc=CARD, ec=LINE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=1.2,
                                transform=ax.transAxes, clip_on=False, mutation_aspect=ax.bbox.width / ax.bbox.height))


def tile(ax, x, y, w, h, value, label, color=INK, vsize=30, lsize=12.5):
    card(ax, x, y, w, h)
    ax.text(x + w / 2, y + h * 0.60, value, fontsize=vsize, weight="bold", color=color, ha="center", va="center")
    ax.text(x + w / 2, y + h * 0.22, label, fontsize=lsize, color=INK2, ha="center", va="center")


def big(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f}M" if n >= 1e7 else f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def pct(x, d=1, sign=True):
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def pct_axis(ax, axis="x", d=0):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{d}f}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def note(ax, text, x=0.0, y=1.02, **kw):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11.5, color=INK2, va="bottom", **kw)


def month_axis(ax, interval=1, fmt="%b %y"):
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(fmt))


# ------------------------------------------------------------ 01 overview ----
def img01():
    fig = frame(1, "Two years of an online store, end to end",
                "A seeded synthetic dataset with known causes planted in it, then analysed as if the causes were unknown.")
    d = r("daily_metrics.csv")
    fo = r("funnel_overall.csv")
    o = pd.read_parquet(DATA / "orders.parquet", columns=["status", "refund_amount", "review_rating"])
    ax = canvas(fig, [0.04, 0.09, 0.92, 0.72])

    kp = [("23.9M", "sessions"), ("3.32M", "users"), ("1.18M", "orders"), ("1.83M", "order lines"),
          ("3,000", "products"), ("$134.6M", "revenue")]
    w, gap = 0.155, 0.014
    for i, (v, l) in enumerate(kp):
        tile(ax, i * (w + gap), 0.80, w, 0.20, v, l, color=BLUE if i in (0, 5) else INK)

    # life-cycle strip
    ax.text(0, 0.735, "THE PURCHASE LIFE CYCLE", fontsize=11.5, weight="bold", color=GREY)
    delivered = int((o["status"] != "cancelled").sum())
    stages = list(zip(fo["stage"], fo["sessions"])) + [
        ("Delivered", delivered), ("Returned", int((o["refund_amount"] > 0).sum())),
        ("Reviewed", int(o["review_rating"].notna().sum()))]
    n = len(stages)
    sw, sg = 0.094, (1 - 9 * 0.094) / 8
    for i, (s, c) in enumerate(stages):
        x = i * (sw + sg)
        post = i >= 6
        card(ax, x, 0.54, sw, 0.17, fc=PALE_ORANGE if post else PALE_BLUE, ec="none")
        ax.text(x + sw / 2, 0.655, s, fontsize=12.5, weight="bold", color=INK, ha="center", va="center")
        ax.text(x + sw / 2, 0.59, big(c), fontsize=17, color=INK, ha="center", va="center")
        if i < n - 1:
            ax.annotate("", (x + sw + sg * 0.92, 0.625), (x + sw + sg * 0.08, 0.625),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))
    ax.text(0, 0.50, "session funnel", fontsize=11, color=BLUE, va="top")
    ax.text(6 * (sw + sg), 0.50, "post-purchase", fontsize=11, color=ORANGE, va="top")

    # three text cards
    cols = [
        ("PLANTED IN THE DATA", [
            "12 promotions of 5 types", "11 internal events: outage, bugs,\nprice rise, launches, budget cut",
            "13 external events: competitor,\nviral spike, bots, strike, heatwave",
            "2 user-randomised A/B tests,\none real effect and one true null"]),
        ("ANALYSES RUN", [
            "Trend regression with event terms", "Funnels under 9 conditions, with controls",
            "Segment statistics with effect sizes", "Promotion lift, profit and pull-forward",
            "A/B tests: SRM, CUPED, heterogeneity", "Anomaly detection, blind to the calendar"]),
        ("DELIVERABLES", [
            "4 executed Jupyter notebooks", "Excel workbook: 14 sheets,\n20.7k live formulas, 0 errors",
            "Written report with caveats", "Reproducible generator (one seed)"]),
    ]
    cw, cg = 0.322, 0.017
    for i, (head, lines) in enumerate(cols):
        x = i * (cw + cg)
        card(ax, x, 0.0, cw, 0.44)
        ax.text(x + 0.018, 0.40, head, fontsize=11.5, weight="bold", color=GREY, va="center")
        yy = 0.345
        for ln in lines:
            ax.add_patch(Rectangle((x + 0.018, yy - 0.006), 0.006, 0.012, color=BLUE, transform=ax.transAxes))
            ax.text(x + 0.034, yy + 0.012, ln, fontsize=12.8, color=INK, va="top", linespacing=1.25)
            yy -= 0.052 * (1 + ln.count("\n") * 0.82)
    finish(fig, 1, "project_at_a_glance")


# -------------------------------------------------------------- 02 trends ----
def img02():
    d = r("daily_metrics.csv")
    d["date"] = pd.to_datetime(d["date"])
    y1, y2 = d.iloc[:365], d.iloc[-365:]
    yoy = y2["revenue"].sum() / y1["revenue"].sum() - 1
    fig = frame(2, f"Revenue grew {yoy * 100:.0f}% year on year, and every bump has a name",
                "Daily revenue (7-day average). Shaded bands are promotions; dashed lines are lasting changes.")
    ax = fig.add_axes([0.065, 0.42, 0.895, 0.335])
    roll = d["revenue"].rolling(7, center=True, min_periods=4).mean() / 1e3
    ax.fill_between(d["date"], roll, color=PALE_BLUE, lw=0)
    ax.plot(d["date"], roll, color=BLUE, lw=2)
    promos = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    for _, p in promos.iterrows():
        ax.axvspan(p.start_date, p.end_date + pd.Timedelta(days=1), color=YELLOW, alpha=0.28, lw=0)
    top = roll.max() * 1.08
    ax.set_ylim(0, top)
    peaks = [("2024-11-30", "Black Friday 2024"), ("2025-11-29", "Black Friday 2025"),
             ("2025-03-23", "Spring\nSale"), ("2026-03-22", "Spring\nSale")]
    for dt, lab in peaks:
        t = pd.Timestamp(dt)
        v = roll[d["date"].between(t - pd.Timedelta(days=4), t + pd.Timedelta(days=4))].max()
        left = lab.startswith("Spring") or lab.endswith("2025")
        ax.text(t + pd.Timedelta(days=-12 if left else 9), v, lab, fontsize=11.5, color=INK, va="top", weight="bold",
                ha="right" if left else "left")
    marks = [("2025-04-01", "Loyalty\nlaunch"), ("2025-07-01", "Pet Supplies\nlaunch"),
             ("2025-09-01", "Electronics\n+8% price"), ("2026-01-15", "Paid search cut,\nmacro slowdown"),
             ("2026-05-04", "One-page checkout\nat 100%")]
    for dt, lab in marks:
        t = pd.Timestamp(dt)
        ax.axvline(t, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(t, top * 1.02, lab, fontsize=11, color=INK2, va="bottom", ha="left", linespacing=1.15)
    ax.set_xlim(d["date"].min(), d["date"].max())
    month_axis(ax, 2)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:.0f}k"))
    ax.set_ylabel("revenue per day")
    ax.grid(axis="x", visible=False)

    # monthly by category
    m = r("monthly_revenue_by_category.csv").fillna(0)
    ax2 = fig.add_axes([0.065, 0.105, 0.56, 0.235])
    cats = list(m.columns[1:])
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    x = np.arange(len(m))
    bottom = np.zeros(len(m))
    for c, col in zip(cats, cols):
        ax2.bar(x, m[c] / 1e6, bottom=bottom, color=col, width=0.78, label=c, edgecolor=BG, lw=0.8)
        bottom += m[c].values / 1e6
    ax2.set_xticks(x[::3])
    ax2.set_xticklabels(pd.to_datetime(m["month"]).dt.strftime("%b %y")[::3])
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
    ax2.set_title("Monthly revenue by category")
    ax2.grid(axis="x", visible=False)
    ax2.legend(ncol=1, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=10.5, handlelength=1, labelspacing=0.3)

    # year-on-year
    ax3 = fig.add_axes([0.80, 0.105, 0.16, 0.235])
    items = [("Sessions", y2["sessions"].sum() / y1["sessions"].sum() - 1),
             ("Orders", y2["orders"].sum() / y1["orders"].sum() - 1),
             ("Revenue", yoy),
             ("Order value", (y2["revenue"].sum() / y2["orders"].sum()) / (y1["revenue"].sum() / y1["orders"].sum()) - 1),
             ("Conversion", (y2["purchased"].sum() / y2["sessions"].sum()) / (y1["purchased"].sum() / y1["sessions"].sum()) - 1)]
    yy = np.arange(len(items))[::-1]
    vals = [v for _, v in items]
    ax3.barh(yy, vals, color=[BLUE if v >= 0 else ORANGE for v in vals], height=0.6)
    for y, v in zip(yy, vals):
        ax3.text(v + (0.006 if v >= 0 else -0.006), y, pct(v, 0), va="center", ha="left" if v >= 0 else "right",
                 fontsize=12, color=INK, weight="bold")
    ax3.set_yticks(yy)
    ax3.set_yticklabels([k for k, _ in items])
    ax3.axvline(0, color=INK2, lw=1)
    ax3.set_xlim(min(vals) - 0.07, max(vals) + 0.08)
    ax3.set_xticks([])
    ax3.grid(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.set_title("Year 2 vs year 1")
    finish(fig, 2, "revenue_trend_and_drivers")


# -------------------------------------------------------------- 03 funnel ----
def img03():
    fo = r("funnel_overall.csv")
    dev = r("funnel_by_device.csv").set_index("segment")
    fig = frame(3, "Four in five product viewers never add to cart, and mobile leaks at every step",
                "23.8M human sessions. Mobile brings 55% of visits but converts at 3.5%, against 7.2% on desktop.")
    ax = fig.add_axes([0.125, 0.11, 0.39, 0.66])
    n = len(fo)
    y = np.arange(n)[::-1]
    share = fo["share of visits"].values
    worst = int(np.nanargmax(fo["drop-off at step"].values))
    ax.barh(y, share, color=[ORANGE if i == worst else BLUE for i in range(n)], height=0.62)
    for i in range(n):
        ax.text(share[i] + 0.012, y[i], f"{big(fo['sessions'][i])}   {share[i] * 100:.1f}% of visits",
                va="center", fontsize=12.5, color=INK)
        if i > 0:
            ax.text(0.004, y[i] + 0.5, f"\u2193 {fo['step conversion'][i] * 100:.0f}% continue", ha="left", va="center",
                    fontsize=11.5, color=ORANGE if i == worst else INK2, weight="bold" if i == worst else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(fo["stage"], fontsize=13, weight="bold")
    ax.tick_params(axis="y", pad=8, length=0)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False), ax.spines["left"].set_visible(False)
    ax.set_title("Overall funnel", x=-0.21)

    steps = ["visit→view", "view→cart", "cart→checkout", "checkout→payment", "payment→purchase"]
    ax2 = fig.add_axes([0.58, 0.50, 0.38, 0.27])
    x = np.arange(len(steps))
    for k, (dv, col) in enumerate(zip(["desktop", "tablet", "mobile"], [BLUE, AQUA, ORANGE])):
        v = dev.loc[dv, steps].values.astype(float)
        ax2.bar(x + (k - 1) * 0.27, v, width=0.25, color=col,
                label=f"{dv}  ({dev.loc[dv, 'overall'] * 100:.1f}% overall)")
        for xi, vi in zip(x, v):
            ax2.text(xi + (k - 1) * 0.27, vi + 0.012, f"{vi * 100:.0f}", ha="center", fontsize=10, color=INK2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("→", " →\n") for s in steps], fontsize=11)
    ax2.set_ylim(0, 1.18)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    pct_axis(ax2, "y")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Step conversion by device")
    ax2.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=10.5, handlelength=1, columnspacing=1.2)

    g = pd.read_csv(OUT / "funnel_glm_odds_ratios.csv", index_col=0)
    names = {"device: mobile": "Mobile  (vs desktop)", "device: tablet": "Tablet  (vs desktop)",
             "channel: Social": "Social  (vs organic search)", "new user": "New user  (vs returning)",
             "channel: Direct": "Direct  (vs organic search)", "channel: Email": "Email  (vs organic search)",
             "loyalty member": "Loyalty member  (vs non-member)"}
    g = g.loc[list(names)].sort_values("odds ratio")
    ax3 = fig.add_axes([0.72, 0.125, 0.24, 0.245])
    yy = np.arange(len(g))
    ax3.hlines(yy, g["lo"], g["hi"], color=INK2, lw=1.5)
    ax3.scatter(g["odds ratio"], yy, s=90, color=[ORANGE if v < 1 else BLUE for v in g["odds ratio"]], zorder=3)
    for yi, v in zip(yy, g["odds ratio"]):
        ax3.text(v, yi + 0.33, f"{v:.2f}", ha="center", fontsize=10.5, color=INK)
    ax3.axvline(1, color=INK2, lw=1)
    ax3.set_yticks(yy)
    ax3.set_yticklabels([names[i] for i in g.index])
    ax3.set_xlim(0.35, 1.75)
    ax3.set_ylim(-0.6, len(g) - 0.3)
    ax3.grid(axis="y", visible=False)
    ax3.set_title("Odds of purchase, all else equal", x=-0.72)
    ax3.set_xlabel("odds ratio (logistic model)")
    finish(fig, 3, "funnel_overall_and_by_device")


# --------------------------------------------------- 04 funnel conditions ----
def img04():
    c = r("funnel_conditions.csv")
    fig = frame(4, "Each shock hits one funnel step in one group, and the control group stays flat",
                "Relative change in step conversion against the weeks before, with 95% intervals. Grey is the unaffected control.")
    rows = [  # (label, condition prefix, affected group, step, control group)
        ("Black Friday 2025\nview → cart, all shoppers", "Black Friday 2025", "all", "view→cart", None),
        ("Heatwave, Jun 2026\nview → cart, Home & Garden", "Heatwave", "Home & Garden", "view→cart", "other categories (control)"),
        ("One-page checkout rollout\ncheckout → payment, mobile", "One-page checkout", "mobile", "checkout→payment", "desktop (control)"),
        ("Weekend vs weekday\nview → cart, all shoppers", "Weekend", "all", "view→cart", None),
        ("Electronics +8% price\nview → cart, Electronics", "Electronics +8%", "Electronics", "view→cart", "other categories (control)"),
        ("Competitor campaign, May 2025\nview → cart, Electronics", "Competitor", "Electronics", "view→cart", "other categories (control)"),
        ("Mobile checkout bug, Jun 2025\npayment → purchase, mobile", "Mobile checkout bug", "mobile", "payment→purchase", "desktop (control)"),
        ("Viral social spike, Aug 2025\nview → cart, Social visitors", "Viral social", "Social", "view→cart", None),
    ]
    ax = fig.add_axes([0.235, 0.12, 0.40, 0.66])
    n = len(rows)
    for i, (lab, cond, grp, step, ctl) in enumerate(rows):
        y = n - 1 - i
        sub = c[c["condition"].str.startswith(cond) & (c["step"] == step)]
        a = sub[sub["group"] == grp].iloc[0]
        col = BLUE if a.rel_change > 0 else ORANGE
        ax.barh(y + 0.13, a.rel_change, height=0.34, color=col)
        ax.hlines(y + 0.13, a.lo, a.hi, color=INK, lw=1.6)
        off = 0.012 if a.rel_change > 0 else -0.012
        ax.text((a.hi if a.rel_change > 0 else a.lo) + off, y + 0.13, pct(a.rel_change), va="center",
                ha="left" if a.rel_change > 0 else "right", fontsize=13, weight="bold", color=INK)
        if ctl:
            b = sub[sub["group"] == ctl].iloc[0]
            ax.hlines(y - 0.22, b.lo, b.hi, color=GREY, lw=1.6)
            ax.scatter(b.rel_change, y - 0.22, s=70, color=GREY, zorder=3)
            side = -1 if a.rel_change > 0 else 1
            ax.text(b.rel_change + side * 0.022, y - 0.22, f"control {pct(b.rel_change)}", va="center",
                    ha="right" if side < 0 else "left", fontsize=11, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([x[0] for x in rows], fontsize=12, linespacing=1.3)
    ax.set_xlim(-0.68, 0.48)
    ax.set_ylim(-0.6, n - 0.4)
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("relative change in step conversion")
    ax.text(0.03, 4.13, "significant (p < 0.001), but only −0.5%", fontsize=11, color=INK2,
            ha="left", va="center", style="italic")

    t = r("segments_price_tier_macro.csv")
    ax2 = fig.add_axes([0.745, 0.47, 0.215, 0.285])
    yy = np.arange(len(t))[::-1]
    ax2.barh(yy, t["rel_change"], color=[BLUE if v > 0 else ORANGE for v in t["rel_change"]], height=0.6)
    ax2.hlines(yy, t["lo"], t["hi"], color=INK, lw=1.5)
    for y, v in zip(yy, t["rel_change"]):
        ax2.text(v + (0.012 if v > 0 else -0.012), y, pct(v), va="center", ha="left" if v > 0 else "right",
                 fontsize=12, weight="bold")
    ax2.axvline(0, color=INK2, lw=1)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(["Budget\nquartile", "Q2", "Q3", "Premium\nquartile"])
    ax2.set_xlim(-0.21, 0.17)
    ax2.set_xticks([])
    ax2.grid(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.set_title("Macro slowdown: shoppers trade down", x=-0.22, pad=28)
    note(ax2, "add-to-cart rate by price tier, Jan–Apr 2026 vs 2025", x=-0.22, y=1.0)

    m = r("mix_shift_paid_cut.csv").set_index("component")["value"]
    ax3 = fig.add_axes([0.745, 0.12, 0.215, 0.20])
    b0, mix, rate, b1 = m["blended before"], m["mix effect"], m["rate effect"], m["blended during"]
    ax3.bar([0, 3], [b0, b1], color=GREY, width=0.62)
    ax3.bar(1, mix, bottom=b0, color=BLUE, width=0.62)
    ax3.bar(2, rate, bottom=b0 + mix, color=ORANGE, width=0.62)
    for x, v, lab in [(0, b0, f"{b0 * 100:.2f}%"), (1, b0 + mix, f"{mix * 1e4:+.0f} bp"),
                      (2, b0 + mix, f"{rate * 1e4:+.0f} bp"), (3, b1, f"{b1 * 100:.2f}%")]:
        ax3.text(x, v + 0.00012, lab, ha="center", fontsize=11.5, weight="bold")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(["before", "traffic\nmix", "rates", "during"], fontsize=11)
    ax3.set_ylim(0.046, 0.0508)
    ax3.set_yticks([])
    ax3.grid(False)
    ax3.spines["left"].set_visible(False)
    ax3.set_title("Paid search cut: mix vs rate", x=-0.22, pad=28)
    note(ax3, "blended conversion, decomposed (broken axis)", x=-0.22, y=1.0)
    finish(fig, 4, "funnel_under_conditions")


# ------------------------------------------------------------ 05 segments ----
def img05():
    s = r("segments_users.csv")
    t = r("segments_users_tests.csv").set_index("dimension")
    cat = r("segments_category.csv")
    fig = frame(5, "Every segment split is “significant”; only two are big enough to matter",
                "With 3.3M users every p-value is tiny, so segments are ranked by effect size instead.")
    nice = {"preferred_device": "Preferred device", "member": "Loyalty", "age_group": "Age group",
            "acquisition_channel": "Acquisition channel", "country": "Country", "gender": "Gender"}
    order = t["cramers_v"].sort_values(ascending=False).index.tolist()
    overall = (s[s.dimension == "gender"]["buyer_rate"] * s[s.dimension == "gender"]["users"]).sum() / \
        s[s.dimension == "gender"]["users"].sum()

    ax = fig.add_axes([0.135, 0.10, 0.27, 0.68])
    labels, ypos, y = [], [], 0
    heads = []
    for dim in order:
        sub = s[s.dimension == dim]
        strong = t.loc[dim, "cramers_v"] >= 0.1
        heads.append((y, nice[dim]))
        y -= 1
        for _, row in sub.iterrows():
            col = BLUE if strong else GREY
            ax.hlines(y, overall, row.buyer_rate, color=col, lw=2.2)
            ax.scatter(row.buyer_rate, y, s=95 if strong else 55, color=col, zorder=3)
            if strong:
                ax.text(row.buyer_rate + (0.006 if row.buyer_rate > overall else -0.006), y, pct(row.buyer_rate, 1, False),
                        va="center", ha="left" if row.buyer_rate > overall else "right", fontsize=11.5, weight="bold")
            labels.append(row.segment.replace("Loyalty member", "Member")), ypos.append(y)
            y -= 1
        y -= 0.35
    for hy, h in heads:
        ax.text(-0.03, hy, h, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12, weight="bold",
                color=INK)
    ax.axvline(overall, color=INK2, lw=1)
    ax.text(overall + 0.004, 0.75, f"all users {overall * 100:.1f}%", ha="left", fontsize=11, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_ylim(y + 0.6, 1.6)
    ax.set_xlim(0.145, 0.37)
    ax.set_xticks([0.15, 0.20, 0.25, 0.30, 0.35])
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title("Share of users who bought", x=-0.42)

    ax2 = fig.add_axes([0.55, 0.50, 0.41, 0.28])
    v = t.loc[order, "cramers_v"][::-1]
    yy = np.arange(len(v))
    ax2.barh(yy, v, color=[BLUE if x >= 0.1 else GREY for x in v], height=0.62)
    for yi, x in zip(yy, v):
        ax2.text(x + 0.002, yi, f"{x:.3f}", va="center", fontsize=12, weight="bold" if x >= 0.1 else "normal")
    ax2.axvline(0.1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax2.text(0.101, len(v) - 0.42, "conventional “small effect” line", fontsize=10.5, color=INK2, va="bottom")
    ax2.set_yticks(yy)
    ax2.set_yticklabels([nice[i] for i in v.index])
    ax2.set_xlim(0, 0.15)
    ax2.set_ylim(-0.6, len(v) + 0.1)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Effect size on buying (Cramér’s V)", x=-0.27)
    note(ax2, "all six dimensions have p < 1e-40", x=0.52, y=1.045)

    ax3 = fig.add_axes([0.55, 0.135, 0.41, 0.255])
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    ax3.scatter(cat["margin_pct"], cat["return_rate"], s=cat["revenue"] / 2.2e4, color=cols, alpha=0.9,
                edgecolor=BG, lw=2, zorder=3)
    off = {"Electronics": (0.028, 0.012), "Fashion": (0.018, 0.0), "Home & Garden": (0.016, 0.014),
           "Beauty": (0.010, 0.008), "Sports & Outdoors": (0.0, 0.021), "Toys & Games": (0.010, 0.004),
           "Books & Media": (-0.008, 0.0), "Pet Supplies": (0.007, -0.004)}
    for _, row in cat.iterrows():
        dx, dy = off[row.category]
        ax3.text(row.margin_pct + dx, row.return_rate + dy,
                 f"{row.category}", fontsize=11, va="center", ha="right" if dx < 0 else "left", color=INK)
    ax3.set_xlim(0.2, 0.66)
    ax3.set_ylim(0, 0.235)
    pct_axis(ax3), pct_axis(ax3, "y")
    ax3.set_xlabel("gross margin")
    ax3.set_ylabel("return rate")
    ax3.set_title("Products: margin vs returns (bubble = revenue)", x=-0.105)
    finish(fig, 5, "segments_effect_sizes")


# ---------------------------------------------------------- 06 promotions ----
def img06():
    p = r("promo_results.csv")
    bf = r("promo_black_friday_net.csv")
    fig = frame(6, "Black Friday triples revenue, but Spring Sale earns more profit",
                "Lift against a regression baseline with 95% intervals. Right: Black Friday profit net of orders pulled from the weeks around it.")
    short = p["name"].str.replace(" / Cyber Week", "").str.replace("Free Shipping Weekend", "Free Shipping") \
        .str.replace("LOYAL20 Member Flash Sale", "Member Flash Sale 2025").str.replace("Beauty BOGO Week", "Beauty BOGO 2026") \
        .str.replace(r" \((\w+) (\d+)\)", r" \1 \2", regex=True)
    n = len(p)
    y = np.arange(n)[::-1]
    ax = fig.add_axes([0.175, 0.11, 0.20, 0.67])
    ax.barh(y, p["revenue_lift"], color=[BLUE if lo > 0 else (ORANGE if hi < 0 else GREY)
                                         for lo, hi in zip(p.revenue_lo, p.revenue_hi)], height=0.6)
    ax.hlines(y, p["revenue_lo"], p["revenue_hi"], color=INK, lw=1.4)
    for yi, v, hi, lo in zip(y, p.revenue_lift, p.revenue_hi, p.revenue_lo):
        ax.text(max(hi, 0) + 0.04, yi, pct(v, 0), va="center", fontsize=12, weight="bold")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}\n{t}, {int(d * 100)}% off" if d else f"{s}\n{t}" for s, t, d in zip(short, p.type, p.discount)],
                       fontsize=10.8, linespacing=1.25)
    ax.set_xlim(-0.3, 2.45)
    ax.set_xticks([0, 1, 2])
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title("Revenue in promoted scope")

    ax2 = fig.add_axes([0.405, 0.11, 0.20, 0.67])
    cols = [GOOD if lo > 0 else (BAD if hi < 0 else GREY) for lo, hi in zip(p.store_profit_lo, p.store_profit_hi)]
    ax2.barh(y, p["store_profit_lift"], color=cols, height=0.6)
    ax2.hlines(y, p["store_profit_lo"], p["store_profit_hi"], color=INK, lw=1.4)
    for yi, v, hi, lo in zip(y, p.store_profit_lift, p.store_profit_hi, p.store_profit_lo):
        ax2.text(hi + 0.012 if v >= 0 else lo - 0.012, yi, pct(v, 0), va="center", ha="left" if v >= 0 else "right",
                 fontsize=12, weight="bold")
    ax2.axvline(0, color=INK2, lw=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(-0.17, 0.48)
    pct_axis(ax2)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Store-wide gross profit")
    for lab, col, yy in [("gain", GOOD, 0.115), ("loss", BAD, 0.075), ("within noise", GREY, 0.035)]:
        ax2.add_patch(Rectangle((0.66, yy - 0.009), 0.035, 0.018, color=col, transform=ax2.transAxes))
        ax2.text(0.715, yy, lab, transform=ax2.transAxes, fontsize=10.5, va="center", color=INK2)

    for k, yr in enumerate([2024, 2025]):
        b = bf[(bf.metric == "gross profit") & (bf.year == yr)].iloc[0]
        ax3 = fig.add_axes([0.70, 0.50 - k * 0.39, 0.26, 0.255])
        vals = [b.during, b.pre_week, b.post_week]
        cum = np.cumsum([0] + vals)
        for i, v in enumerate(vals):
            ax3.bar(i, v / 1e3, bottom=cum[i] / 1e3, color=GOOD if v > 0 else BAD, width=0.6)
            ax3.text(i, (max(cum[i], cum[i + 1])) / 1e3 + 3, f"{v / 1e3:+.0f}k", ha="center", fontsize=12, weight="bold")
            ax3.hlines(cum[i + 1] / 1e3, i + 0.3, i + 0.7, color=INK2, lw=1, ls=(0, (2, 2)))
        ax3.bar(3, b.net / 1e3, color=GOOD if b.net > 0 else BAD, width=0.6, hatch="///", edgecolor=BG, lw=0)
        ax3.text(3, max(b.net, 0) / 1e3 + 3, f"{b.net / 1e3:+.0f}k", ha="center", fontsize=13, weight="bold")
        ax3.axhline(0, color=INK2, lw=1)
        ax3.set_xticks(range(4))
        ax3.set_xticklabels(["4 promo\ndays", "week\nbefore", "week\nafter", "net"], fontsize=11)
        ax3.set_ylim(min(b.net / 1e3, 0) - 12, b.during / 1e3 + 22)
        ax3.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"-${-v:.0f}k" if v < 0 else f"${v:.0f}k"))
        ax3.grid(axis="x", visible=False)
        ax3.set_title(f"Black Friday {yr}: profit after pull-forward")
    finish(fig, 6, "promotion_scorecard")


# ----------------------------------------------------------------- 07 AB1 ----
def img07():
    res = r("ab1_results.csv")
    seg = r("ab1_segments.csv")
    cum = r("ab1_daily_cumulative.csv")
    het = r("ab1_heterogeneity.csv").set_index("dimension")
    u = r("ab1_users.csv")
    pr = res.iloc[0]
    fig = frame(7, "A/B test 1: one-page checkout wins, entirely on small screens",
                f"{u['users'].sum():,} users randomised 50/50 for six weeks. Primary metric: checkout → purchase.")
    ax0 = canvas(fig, [0.04, 0.63, 0.92, 0.16])
    tiles = [(f"{pr.control * 100:.1f}%", "control"), (f"{pr.treatment * 100:.1f}%", "one-page checkout"),
             (pct(pr.rel_lift), f"relative lift (95% CI {pct(pr.lift_lo)} to {pct(pr.lift_hi)})"),
             (f"z = {pr.z:.2f}", "p = 0.000002, user-clustered"),
             ("+13,000", "extra orders a year at full rollout")]
    ws = [0.13, 0.16, 0.27, 0.20, 0.20]
    x = 0
    for (v, l), w in zip(tiles, ws):
        tile(ax0, x, 0, w, 1, v, l, color=BLUE if "lift" in l else INK, vsize=27, lsize=11.5)
        x += w + 0.01

    rows = [("All users", pr.rel_lift, pr.lift_lo, pr.lift_hi, "***", True)]
    for dim, lab in [("device", "Device"), ("user_type", "User type"), ("channel", "Channel")]:
        rows.append((lab, None, None, None, None, False))
        for _, s in seg[seg.dimension == dim].iterrows():
            rows.append(("   " + s.segment, s.rel_lift, s.lift_lo, s.lift_hi, s.sig_adj, False))
    ax = fig.add_axes([0.135, 0.105, 0.37, 0.45])
    n = len(rows)
    for i, (lab, v, lo, hi, sig, top) in enumerate(rows):
        y = n - 1 - i
        if v is None:
            continue
        col = BLUE if sig != "ns" else GREY
        ax.hlines(y, lo, hi, color=col, lw=3 if top else 2)
        ax.scatter(v, y, s=130 if top else 80, color=col, zorder=3, marker="D" if top else "o")
        ax.text(0.109, y, pct(v), va="center", ha="right", fontsize=11.5, weight="bold" if sig != "ns" else "normal",
                color=INK if sig != "ns" else INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(pr.rel_lift, color=BLUE, lw=1, ls=(0, (3, 3)), alpha=0.7)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([x[0] for x in rows], fontsize=11.5)
    for lbl, row in zip(ax.get_yticklabels(), rows):
        if row[1] is None or row[5]:
            lbl.set_weight("bold")
    ax.set_xlim(-0.03, 0.111)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
    ax.set_ylim(-0.7, n - 0.3)
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title("Lift by segment (blue = significant after correction)", x=-0.27)
    ax.text(0.035, n - 3.0, f"device differs: Q = {het.loc['device', 'Q']:.0f}, p < 1e-8\nno other split does", fontsize=10.5,
            color=INK2, va="center", style="italic")

    ax2 = fig.add_axes([0.60, 0.13, 0.36, 0.425])
    ax2.plot(cum["day"], cum["z"], color=BLUE, lw=2.4, label="test statistic so far")
    ob = cum["obf_bound"].clip(upper=7)
    ax2.plot(cum["day"], ob, color=INK, lw=1.6, ls=(0, (5, 3)), label="boundary that allows daily peeking")
    ax2.axhline(1.96, color=GREY, lw=1.4, ls=(0, (2, 2)))
    ax2.text(42, 1.82, "naive 1.96 line", ha="right", va="top", fontsize=10.5, color=INK2)
    early = cum[(cum.z > 1.96) & (cum.z < cum.obf_bound)]
    ax2.scatter(early["day"], early["z"], s=42, color=ORANGE, zorder=4, label="looks “significant” too early")
    cross = cum[cum.z >= cum.obf_bound]["day"].min()
    ax2.axvline(cross, color=GOOD, lw=1.2)
    ax2.text(cross + 0.5, 0.25, f"safe to call\nfrom day {int(cross)}", fontsize=11, color=GOOD, va="bottom")
    ax2.set_ylim(0, 7)
    ax2.set_xlim(1, 42)
    ax2.set_xlabel("day of test")
    ax2.set_ylabel("z")
    ax2.set_title("Why the test was not stopped early")
    ax2.legend(loc="upper right", fontsize=10.5)
    finish(fig, 7, "ab_test_one_page_checkout")


# ------------------------------------------------------- 08 AB2 + loyalty ----
def img08():
    res = r("ab2_results.csv")
    seg = r("ab2_segments.csv")
    lo = r("loyalty_did.csv").set_index("measure")["value"]
    fig = frame(8, "Knowing when the answer is “nothing”, and when it is “less than it looks”",
                "A well-powered null result for the social-proof badge, and a loyalty effect cut by two thirds once selection is removed.")
    ax = fig.add_axes([0.17, 0.11, 0.37, 0.63])
    rows = [("Primary: view → add to cart", res.iloc[0].rel_lift, res.iloc[0].lift_lo, res.iloc[0].lift_hi, True)]
    rows += [("Session conversion", res.iloc[1].rel_lift, res.iloc[1].lift_lo, res.iloc[1].lift_hi, True)]
    for _, s in seg.iterrows():
        rows.append((f"{s.segment}", s.rel_lift, s.lift_lo, s.lift_hi, False))
    n = len(rows)
    ax.axvspan(-0.014, 0.014, color=PALE_BLUE, lw=0)
    for i, (lab, v, l, h, main) in enumerate(rows):
        y = n - 1 - i - (0.6 if not main else 0)
        ax.hlines(y, l, h, color=BLUE if main else GREY, lw=3 if main else 1.8)
        ax.scatter(v, y, s=120 if main else 45, color=BLUE if main else GREY, zorder=3, marker="D" if main else "o")
        rows[i] = (lab, y, main)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([x[1] for x in rows])
    ax.set_yticklabels([x[0] for x in rows], fontsize=10.5)
    for lbl, row in zip(ax.get_yticklabels(), rows):
        if row[2]:
            lbl.set_weight("bold"), lbl.set_fontsize(12)
    ax.set_xlim(-0.062, 0.062)
    ax.set_ylim(-2.3, n - 0.3)
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("relative lift, treatment vs control (95% CI)")
    ax.set_title("A/B test 2: social-proof badge", x=-0.40, pad=30)
    note(ax, f"primary lift {pct(res.iloc[0].rel_lift, 2)}; 0 of {len(seg)} segments significant after correction", x=-0.40, y=1.0)
    ax.text(0.0, -1.75, "shaded: lifts too small for this test to detect (±1.4%)", fontsize=10.5, color=BLUE, ha="center")

    ax2 = fig.add_axes([0.64, 0.125, 0.32, 0.615])
    vals = [lo["naive member vs non-member"], lo["pre-existing gap (selection)"], lo["DiD programme effect"]]
    labs = ["Naive: members\nvs non-members", "Gap that existed\nbefore joining", "Programme effect\n(diff-in-diff)"]
    cols = [GREY, PALE_ORANGE, BLUE]
    ax2.bar(range(3), vals, color=cols, width=0.62, edgecolor=[GREY, ORANGE, BLUE], lw=1.5)
    ax2.errorbar(2, vals[2], yerr=[[vals[2] - lo["DiD lo"]], [lo["DiD hi"] - vals[2]]], color=INK, lw=1.8, capsize=5)
    for i, v in enumerate(vals):
        ax2.text(i, v + (0.03 if i == 2 else 0.012), pct(v, 1 if i == 2 else 0), ha="center", fontsize=22 if i != 1 else 17,
                 weight="bold", color=INK if i != 1 else ORANGE)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labs, fontsize=11.5, linespacing=1.3)
    ax2.set_ylim(0, 0.63)
    pct_axis(ax2, "y")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Loyalty programme: conversion uplift", pad=30)
    note(ax2, "keen shoppers join, so most of the gap is who they are", y=1.0)
    finish(fig, 8, "null_result_and_selection_bias")


# ------------------------------------------------------- 09 anomaly map ----
def img09():
    f = r("anomalies_flagged.csv")
    f["date"] = pd.to_datetime(f["date"])
    rec = r("anomaly_calendar_recall.csv")
    det, miss = int((rec.status == "detected").sum()), int((rec.status == "missed").sum())
    nflag, nun = len(f), int((~f.explained).sum())
    fig = frame(9, f"A detector that never saw the calendar found {det} of {det + miss} planted events",
                f"45 daily metrics against a same-weekday robust baseline. Each dot is a flagged metric-day ({nflag:,} in all).")
    metrics = list(dict.fromkeys(f["metric"]))
    grp = f.drop_duplicates("metric").set_index("metric")["group"]
    ypos, y, prev, bounds = {}, 0, None, []
    for m in metrics:
        if grp[m] != prev:
            if prev is not None:
                y -= 1.2
            bounds.append([grp[m], y, y])
            prev = grp[m]
        ypos[m] = y
        bounds[-1][2] = y
        y -= 1
    f["y"] = f["metric"].map(ypos)
    ax = fig.add_axes([0.105, 0.105, 0.665, 0.585])
    for name, y0, y1 in bounds:
        ax.axhspan(y1 - 0.6, y0 + 0.6, color=CARD, lw=0, zorder=0)
        ax.text(-0.012, (y0 + y1) / 2, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12.5,
                weight="bold", color=INK)
    e = f[f.explained]
    size = np.clip(e["z"].abs(), 2, 12) * 5
    ax.scatter(e["date"], e["y"], s=size, c=np.where(e["z"] > 0, BLUE, ORANGE), alpha=0.75, lw=0, zorder=3)
    un = f[~f.explained]
    ax.scatter(un["date"], un["y"], s=34, facecolor="none", edgecolor=INK, lw=1.1, zorder=4)
    ev = [("2025-02-18", "Site outage"), ("2025-03-23", "Spring Sale"), ("2025-05-05", "Competitor"),
          ("2025-06-11", "Checkout bug"), ("2025-07-15", "Clearance"), ("2025-08-10", "Viral spike"),
          ("2025-09-19", "Member sale"), ("2025-10-08", "Tracking bug"), ("2025-11-04", "Bot wave"),
          ("2025-11-29", "Black Friday"), ("2025-12-17", "Carrier strike"), ("2026-01-15", "Paid search cut"),
          ("2026-03-22", "Spring Sale"), ("2026-05-16", "Free shipping"), ("2026-06-17", "Payment outage"),
          ("2026-07-01", "Heatwave"), ("2026-07-18", "Clearance")]
    ytop = 1.6
    for i, (dt, lab) in enumerate(ev):
        t = pd.Timestamp(dt)
        ax.axvline(t, color=LINE, lw=1, zorder=1)
        ax.text(t, ytop + 0.4, lab, rotation=60, ha="left", va="bottom", fontsize=10.5, color=INK2)
    ax.set_ylim(y + 0.2, ytop)
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-11-05"))
    ax.set_yticks([])
    month_axis(ax, 2)
    ax.grid(False)
    ax.spines["left"].set_visible(False)

    ax2 = canvas(fig, [0.795, 0.105, 0.165, 0.69])
    tile(ax2, 0, 0.78, 1, 0.20, f"{det} / {det + miss}", "abrupt events detected", color=BLUE, lsize=11.5)
    tile(ax2, 0, 0.555, 1, 0.20, f"{(nflag - nun) / nflag * 100:.0f}%", "of flags traced\nto a known cause", lsize=11.5)
    tile(ax2, 0, 0.33, 1, 0.20, f"{nun}", "unexplained flags,\nnone above |z| = 4.3", lsize=11.5)
    card(ax2, 0, 0.0, 1, 0.305)
    ax2.text(0.07, 0.265, "LEGEND", fontsize=11, weight="bold", color=GREY, va="center")
    for yy, col, lab, kw in [(0.205, BLUE, "above expected", {}), (0.150, ORANGE, "below expected", {}),
                             (0.095, "none", "no known cause", dict(edgecolor=INK, lw=1.2))]:
        ax2.scatter(0.11, yy, s=90, color=col, transform=ax2.transAxes, **kw)
        ax2.text(0.20, yy, lab, fontsize=11.5, va="center", color=INK)
    ax2.text(0.07, 0.04, "dot size = strength (|z|)", fontsize=10.5, color=INK2, va="center")
    finish(fig, 9, "anomaly_detection_map")


# ------------------------------------------------------ 10 drill-downs ----
def img10():
    fig = frame(10, "Six anomalies, each traced to its cause by slicing one level deeper",
                "The alert says something moved; the drill-down says who, where and why. Orange is the affected slice, grey the comparison.")
    cd = pd.read_parquet(OUT / "cube_day_channel_device.parquet")
    cb = pd.read_parquet(OUT / "cube_all_incl_bots.parquet")
    L, W, H = [0.055, 0.375, 0.695], 0.265, 0.235
    B = [0.475, 0.105]

    def panel(i, title, sub):
        ax = fig.add_axes([L[i % 3], B[i // 3], W, H])
        ax.set_title(title, fontsize=14, pad=24)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10.8, color=INK2, va="bottom")
        ax.grid(axis="x", visible=False)
        return ax

    def win(df, a, b, col="day"):
        lo, hi = (pd.Timestamp(a) - START).days, (pd.Timestamp(b) - START).days
        return df[df[col].between(lo, hi)]

    def dfmt(ax, fmt="%d %b"):
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(fmt))

    # a) mobile checkout bug
    ax = panel(0, "Mobile checkout bug, 10–13 Jun 2025", "payment → purchase rate by device: mobile collapses, desktop does not")
    g = win(cd, "2025-05-25", "2025-06-30").groupby(["day", "device"], observed=True)[["entered_payment", "purchased"]].sum()
    g = (g["purchased"] / g["entered_payment"]).unstack()
    ax.plot(dates(g.index), g["desktop"], color=GREY, label="desktop")
    ax.plot(dates(g.index), g["mobile"], color=ORANGE, label="mobile")
    ax.set_ylim(0.3, 1)
    pct_axis(ax, "y"), dfmt(ax)
    ax.legend(loc="lower left", ncol=2)

    # b) bot wave
    ax = panel(1, "Bot traffic wave, 3–6 Nov 2025", "Direct desktop sessions: the surge is all one-page, two-second visits")
    g = win(cb[(cb.channel == "Direct") & (cb.device == "desktop")], "2025-10-18", "2025-11-22").groupby("day")[["sessions", "bounce0"]].sum()
    x = dates(g.index)
    ax.fill_between(x, 0, (g.sessions - g.bounce0) / 1e3, color=GREY, lw=0, label="engaged")
    ax.fill_between(x, (g.sessions - g.bounce0) / 1e3, g.sessions / 1e3, color=ORANGE, lw=0, label="zero-engagement")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))
    dfmt(ax)
    ax.legend(loc="upper left")

    # c) viral spike
    ax = panel(2, "Viral social mention, 9–12 Aug 2025", "Social sessions up 5×, orders flat: conversion 5.0% → 3.6%")
    g = win(cb, "2025-07-22", "2025-08-28")
    s = g[g.channel == "Social"].groupby("day")["sessions"].sum()
    o = g.groupby("day")["purchased"].sum()
    ax.plot(dates(s.index), s / 1e3, color=ORANGE, label="Social sessions (k)")
    ax.plot(dates(o.index), o / 1e3, color=GREY, label="all orders (k)")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))
    dfmt(ax)
    ax.legend(loc="upper left")

    # d) carrier strike
    ax = panel(3, "Carrier strike in Germany, 4–19 Dec 2025", "average delivery days by order date: only DE orders are delayed")
    od = pd.read_parquet(DATA / "orders.parquet", columns=["order_ts", "country", "delivery_days"])
    od = od[od.order_ts.between("2025-11-10", "2026-01-10")].dropna(subset=["delivery_days"])
    od["d"] = od.order_ts.dt.floor("D")
    od["g"] = np.where(od.country == "DE", "DE", "other")
    g = od.groupby(["d", "g"])["delivery_days"].mean().astype(float).unstack()
    ax.plot(g.index, g["other"], color=GREY, label="other countries")
    ax.plot(g.index, g["DE"], color=ORANGE, label="Germany")
    ax.set_ylim(0, None)
    dfmt(ax)
    ax.legend(loc="upper left")

    # e) defective SKU
    ax = panel(4, "Defective batch of one product, Feb 2026", "monthly return rate: one product at 57%, its category at 7%")
    it = pd.read_parquet(DATA / "order_items.parquet", columns=["order_id", "product_id", "category", "quantity", "returned"])
    it = it[it.category == "Electronics"]
    ts = pd.read_parquet(DATA / "orders.parquet", columns=["order_id", "order_ts"]).set_index("order_id")["order_ts"]
    it["m"] = it["order_id"].map(ts).dt.to_period("M")
    it = it[(it.m >= "2025-10") & (it.m <= "2026-06")]
    sku = it[it.product_id == 385].groupby("m")["returned"].mean()
    catr = it.groupby("m")["returned"].mean()
    xx = np.arange(len(sku))
    ax.bar(xx, sku.values, color=[ORANGE if v > 0.3 else PALE_ORANGE for v in sku.values], width=0.66)
    ax.bar([0], [0], color=ORANGE, label="Electronics-0385")
    ax.plot(xx, catr.reindex(sku.index).values, color=INK2, lw=2, marker="o", ms=4, label="all Electronics")
    ax.set_xticks(xx)
    ax.set_xticklabels([p.strftime("%b") for p in sku.index])
    pct_axis(ax, "y")
    ax.legend(loc="upper left")

    # f) paid search cut
    ax = panel(5, "Paid search budget cut, 15 Jan – 28 Feb 2026", "Paid Search share of sessions: a level shift, dated to the day")
    g = win(cd, "2025-11-20", "2026-04-10").groupby(["day", "channel"], observed=True)["sessions"].sum().unstack()
    sh = g["Paid Search"] / g.sum(axis=1)
    ax.plot(dates(sh.index), sh, color=ORANGE)
    for dt in ["2026-01-15", "2026-03-01"]:
        ax.axvline(pd.Timestamp(dt), color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(pd.Timestamp(dt) + pd.Timedelta(days=2), 0.045, f"detected\n{pd.Timestamp(dt):%d %b}", fontsize=10.5, color=INK2, va="bottom")
    ax.set_ylim(0, 0.3)
    pct_axis(ax, "y"), dfmt(ax)
    finish(fig, 10, "anomaly_drilldowns")


if __name__ == "__main__":
    fns = {i: globals()[f"img{i:02d}"] for i in range(1, 11)}
    want = [int(a) for a in sys.argv[1:]] or list(fns)
    for i in want:
        fns[i]()
