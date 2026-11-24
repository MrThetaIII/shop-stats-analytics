# %% [markdown]
# # 01 · Data overview, quality checks and trend drivers
#
# **Shop Stats** is a synthetic online store: 24 months of traffic (Nov 2024 – Oct 2026), ~24M sessions,
# 3.3M users, 1.2M orders. The generator (`src/generate_data.py`) plants a growth trend, seasonality,
# 12 promotions, 18 internal / external events and two A/B tests. `data/events_calendar.csv` is the
# ground truth we use to *explain* what the analysis finds.
#
# This notebook: (1) describes the tables, (2) runs integrity checks, (3) builds the daily aggregates the
# other notebooks and the Excel workbook reuse, (4) decomposes the daily series into trend, seasonality
# and event effects with significance tests.

# %%
import sys, warnings
sys.path.insert(0, "../src"); sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import shopstats as ss
from shopstats import *
set_style()
pd.options.display.float_format = "{:,.4f}".format
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)
try:
    display
except NameError:
    display = print

# %% [markdown]
# ## 1. Tables

# %%
S = load_sessions()
O = load_orders()
I = pd.read_parquet(DATA / "order_items.parquet")
U = pd.read_parquet(DATA / "users.parquet")
P = pd.read_parquet(DATA / "products.parquet")
EV, PR = load_events()
tables = {"sessions": S, "orders": O, "order_items": I, "users": U, "products": P, "promotions": PR, "events_calendar": EV}
display(pd.DataFrame({"rows": {k: len(v) for k, v in tables.items()},
                      "columns": {k: v.shape[1] for k, v in tables.items()}}))

# %%
for name in ["sessions", "orders", "order_items", "users", "products"]:
    print(f"\n=== {name} ===")
    display(tables[name].head(3))

# %% [markdown]
# **Life-cycle stages.** A session carries five funnel flags (`viewed_product → added_to_cart →
# began_checkout → entered_payment → purchased`). A purchase creates an order, whose post-purchase stages
# are in `orders` (`status`, `delivery_days`, `refund_amount`, `review_rating`) and `order_items` (`returned`).

# %% [markdown]
# ## 2. Integrity checks

# %%
checks = {
    "session_id unique": S["session_id"].is_unique,
    "order_id unique": O["order_id"].is_unique,
    "orders.session_id ⊂ sessions": bool(O["session_id"].between(1, len(S)).all()),
    "order_items.order_id ⊂ orders": bool(I["order_id"].between(1, len(O)).all()),
    "order_items.product_id ⊂ products": bool(I["product_id"].between(1, len(P)).all()),
    "every purchase session has an order": int(S["purchased"].sum()) == len(O),
    "no negative revenue": bool((O["revenue"] >= 0).all()),
    "revenue = gross − discount": bool(np.allclose(O["revenue"], O["gross_amount"] - O["discount_amount"], atol=0.02)),
}
display(pd.Series(checks, name="passed").to_frame())

# %% [markdown]
# ### 2.1 Funnel monotonicity
# Every stage should imply the previous one. A session that purchased without a recorded add-to-cart is
# logically impossible, so any such row is a **tracking problem**, not behaviour.

# %%
viol = S["began_checkout"] & ~S["added_to_cart"]
print(f"sessions with checkout but no recorded add-to-cart: {viol.sum():,} ({viol.mean():.4%})")
v = S.loc[viol].groupby(["day", "device"], observed=True).size().unstack(fill_value=0)
v.index = DAYS[v.index]
display(v)
base = S.loc[S["device"].eq("desktop") & S["day"].between(day_index("2025-09-21"), day_index("2025-10-04"))]
bug = S.loc[S["device"].eq("desktop") & S["day"].between(day_index("2025-10-05"), day_index("2025-10-07"))]
r = two_prop(base["added_to_cart"].sum(), base["viewed_product"].sum(), bug["added_to_cart"].sum(), bug["viewed_product"].sum())
print(f"desktop view→cart rate: prior 2 weeks {r['p1']:.3%} vs 5–7 Oct 2025 {r['p2']:.3%} "
      f"(relative {r['lift']:+.1%}, z={r['z']:.1f}, p={fmt_p(r['p'])})")
r2 = two_prop(base["purchased"].sum(), len(base), bug["purchased"].sum(), len(bug))
print(f"desktop purchase rate:  prior 2 weeks {r2['p1']:.3%} vs 5–7 Oct 2025 {r2['p2']:.3%} (p={r2['p']:.3f})")

# %% [markdown]
# All violations sit on **desktop, 5–7 Oct 2025**: the recorded add-to-cart rate drops by ~70% while the
# purchase rate does not move. Customers behaved normally; the add-to-cart *event* was not logged
# (calendar event E08). Any cart-stage metric for those three days must be treated as missing data.

# %% [markdown]
# ### 2.2 Zero-engagement sessions (bots)

# %%
S["bounce0"] = (S["pageviews"] == 1) & (S["duration_sec"] <= 2)
b = S.groupby("day")["bounce0"].agg(["sum", "mean"])
b.index = DAYS[b.index]
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.plot(b.index, b["mean"], color=GREY)
top = b["mean"].nlargest(4).sort_index()
ax.plot(top.index, top.values, "o", color=BLUE)
ax.annotate("3–6 Nov 2025", (top.index[0], top.max()), xytext=(-78, -4), textcoords="offset points", color=INK)
ax.set_title("Share of single-page sessions lasting ≤ 2 seconds spikes on four days")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
save(fig, "01_bot_share")
bots = S[S["bounce0"] & S["day"].between(day_index("2025-11-03"), day_index("2025-11-06"))]
print("profile of zero-engagement sessions on 3–6 Nov 2025:")
display(bots["channel"].value_counts(normalize=True).head(3).rename("share by channel").to_frame().T)
display(bots["device"].value_counts(normalize=True).rename("share by device").to_frame().T)
hourly = bots.groupby(bots["session_start"].dt.hour).size()
norm = S.loc[S["day"].between(day_index("2025-10-20"), day_index("2025-11-02"))].groupby(S["session_start"].dt.hour).size()
chi = stats.chisquare(hourly, f_exp=norm / norm.sum() * hourly.sum())
print(f"hour-of-day profile vs normal traffic: chi2={chi.statistic:,.0f}, p={fmt_p(chi.pvalue)} "
      f"(bots are flat across the night; humans are not)")

# %% [markdown]
# Roughly 49k extra sessions on 3–6 Nov 2025 are Direct / desktop, one page, ≤ 2 seconds, spread evenly over 24 hours —
# an automated **bot wave** (E09). We keep a `suspected_bot` flag (`pageviews == 1` and `duration ≤ 2s`
# inside that window) and exclude those rows from user-level statistics. Outside the window the same
# pattern is ~0.1% of traffic (fast human bounces), so the flag is only applied inside it.

# %%
S["suspected_bot"] = S["bounce0"] & S["day"].between(day_index("2025-11-03"), day_index("2025-11-06")) \
    & S["channel"].eq("Direct") & S["device"].eq("desktop")
print(f"suspected bot sessions: {S['suspected_bot'].sum():,}")

# %% [markdown]
# ## 3. Daily aggregates (reused by the other notebooks and by Excel)

# %%
keys = ["day", "channel", "device", "is_new_user", "is_loyalty_member"]
cube = (S[~S["suspected_bot"]].groupby(keys, observed=True)
        .agg(sessions=("purchased", "size"), **{f: (f, "sum") for f in FLAGS}).reset_index())
cube.to_parquet(OUT / "cube_day_channel_device.parquet", index=False)
cube_all = (S.groupby(["day", "channel", "device"], observed=True)
            .agg(sessions=("purchased", "size"), bounce0=("bounce0", "sum"), **{f: (f, "sum") for f in FLAGS}).reset_index())
cube_all.to_parquet(OUT / "cube_all_incl_bots.parquet", index=False)

daily = cube_all.groupby("day")[STAGES].sum()
od = O.groupby("day").agg(orders=("order_id", "size"), revenue=("revenue", "sum"), discount=("discount_amount", "sum"),
                          cogs=("cogs", "sum"), units=("units", "sum"), refunds=("refund_amount", "sum"),
                          cancelled=("status", lambda s: (s == "cancelled").sum()))
daily = daily.join(od)
daily["profit"] = daily["revenue"] - daily["cogs"]
daily["aov"] = daily["revenue"] / daily["orders"]
daily["conversion"] = daily["purchased"] / daily["sessions"]
ch = cube_all.pivot_table(index="day", columns="channel", values="sessions", aggfunc="sum", observed=True)
ch.columns = [f"sessions_{c.lower().replace(' ', '_')}" for c in ch.columns]
dv = cube_all.pivot_table(index="day", columns="device", values=["sessions", "purchased"], aggfunc="sum", observed=True)
dv.columns = [f"{a}_{b}" for a, b in dv.columns]
daily = daily.join(ch).join(dv)
daily.insert(0, "date", DAYS[daily.index])
daily.to_csv(OUT / "daily_metrics.csv", index=False)
display(daily.describe().T[["mean", "std", "min", "max"]].head(14))

# %% [markdown]
# ## 4. Trends and their drivers

# %%
wk = daily.set_index("date").resample("W-SUN").agg({"sessions": "sum", "revenue": "sum", "purchased": "sum"}).iloc[1:-1]
wk["conversion"] = wk["purchased"] / wk["sessions"]
marks = [("2024-11-29", "BF 2024"), ("2025-08-09", "viral\nspike"), ("2025-11-03", "bots"), ("2025-11-28", "BF 2025"),
         ("2026-01-15", "paid cut +\nmacro dip"), ("2026-05-04", "checkout\nrollout")]
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
for ax, col, title, f in zip(axes, ["sessions", "revenue", "conversion"],
                             ["Weekly sessions", "Weekly revenue ($)", "Weekly session → purchase conversion"],
                             [lambda x, _: f"{x/1e3:.0f}k", lambda x, _: f"${x/1e6:.1f}M", lambda x, _: f"{x:.1%}"]):
    ax.plot(wk.index, wk[col], color=BLUE)
    ax.set_title(title); ax.yaxis.set_major_formatter(plt.FuncFormatter(f))
    for d_, lab in marks:
        ax.axvline(pd.Timestamp(d_), color=GREY, lw=0.8, ls=":")
for d_, lab in marks:
    axes[0].annotate(lab, (pd.Timestamp(d_), axes[0].get_ylim()[1]), xytext=(3, -4), textcoords="offset points",
                     va="top", fontsize=8, color=INK2)
save(fig, "01_weekly_trends")

# %% [markdown]
# ### 4.1 Regression decomposition
# For each daily series we fit `y = trend + weekday + annual seasonality + event dummies`, with
# Newey–West (HAC, 7 lags) standard errors because daily residuals are autocorrelated. The response is
# `log(sessions)`, `log(revenue)` and `logit(conversion)`, so coefficients read as relative changes.
# P-values are adjusted for the number of terms tested (Benjamini–Hochberg).

# %%
def design():
    X = pd.DataFrame(index=np.arange(D))
    t = np.arange(D) / 365.25
    X["trend"] = t
    for k in range(1, 7):
        X[f"dow{k}"] = (DAYS.dayofweek == k).astype(float)
    for k in (1, 2, 3):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * DAYS.dayofyear / 365.25)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * DAYS.dayofyear / 365.25)
    for _, p in PR.iterrows():
        X[f"promo:{p.promo_id} {p['name']}"] = window(p.start_date, p.end_date).astype(float)
    bf_pre, bf_post = np.zeros(D), np.zeros(D)
    for _, p in PR[PR["type"].eq("sitewide % off")].iterrows():
        a, b = day_index(p.start_date), day_index(p.end_date)
        bf_pre[max(a - 7, 0):a] = 1; bf_post[b + 1:b + 8] = 1
    X["window:7 days before Black Friday"] = bf_pre
    X["window:7 days after Black Friday"] = bf_post
    for _, e in EV.iterrows():
        if e.type in ("experiment", "seasonality") or e.event_id in ("E02", "E05", "E13"):
            continue                      # experiments are balanced; loyalty / pet / defect are gradual or SKU-level;
                                          # category seasons do not move store totals (the harmonics carry total seasonality)
        X[f"event:{e.event_id} {e['name']}"] = window(e.start_date, e.end_date).astype(float)
    return sm.add_constant(X)

X = design()
y = {"sessions": np.log(daily["sessions"].values), "revenue": np.log(daily["revenue"].values),
     "conversion": np.log(daily["conversion"].values / (1 - daily["conversion"].values)),
     "aov": np.log(daily["aov"].values)}
rows = []
fits = {}
for name, yy in y.items():
    m = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    m0 = sm.OLS(yy, X).fit()
    fits[name] = m
    # a dummy covering only a few days fits those days exactly, which makes robust SEs far too small;
    # take the larger of the HAC and the classical standard error (conservative)
    se = np.maximum(m.bse, m0.bse)
    for term in X.columns:
        if ":" in term or term == "trend":
            b = m.params[term]
            rows.append(dict(metric=name, term=term, days=int(X[term].sum()) if ":" in term else D,
                             effect=np.exp(b) - 1, lo=np.exp(b - 1.96 * se[term]) - 1, hi=np.exp(b + 1.96 * se[term]) - 1,
                             p=2 * stats.norm.sf(abs(b / se[term]))))
    print(f"{name:11s} R² = {m.rsquared:.3f}   residual sd = {np.sqrt(m.scale):.3f}   Durbin–Watson = {sm.stats.durbin_watson(m.resid):.2f}")
eff = pd.DataFrame(rows)
eff["p_adj"] = bh(eff["p"])
eff["sig"] = eff["p_adj"].map(stars)
eff.to_csv(OUT / "event_effects_regression.csv", index=False)

tab = eff.pivot(index="term", columns="metric", values="effect")[["sessions", "conversion", "aov", "revenue"]]
sig = eff.pivot(index="term", columns="metric", values="sig")[["sessions", "conversion", "aov", "revenue"]]
show = tab.map(lambda v: f"{v:+.1%}") + " " + sig
order = [c for c in X.columns if c in show.index]
print("\nEstimated effect of each factor (relative change vs. the modelled baseline; trend = per year)")
print("significance after Benjamini–Hochberg adjustment:  *** p<.001  ** p<.01  * p<.05  ns = not significant")
pd.set_option("display.width", 200, "display.max_colwidth", 60)
display(show.loc[order])

# %% [markdown]
# **Reading the table**
#
# * **Trend**: sessions grow ~18% a year and revenue ~21%; growth is mostly traffic-led.
# * **Black Friday / Cyber Week** more than doubles sessions, lifts conversion and — despite the 25% discount — nearly
#   triples revenue. The 7 days **before and after** show depressed conversion and revenue: part of the lift is pulled
#   forward from neighbouring weeks (quantified in notebook 03).
# * **Traffic-only shocks** (viral mention, bot wave) raise sessions and *lower* the conversion rate while revenue barely moves:
#   the extra visitors do not buy. Conversion rate alone would read these as bad days; orders say otherwise.
# * **Incidents**: the site outage costs ~30% of a day's sessions; the mobile checkout bug and the card-payment outage hit
#   conversion, not traffic.
# * **Paid search budget cut** lowers sessions; its conversion coefficient is confounded with the macro slowdown that
#   overlaps it — separated below.
# * **Not significant** rows (e.g. Beauty BOGO on revenue) are as informative as the starred ones: no measurable sitewide effect.
# * **Caveat on permanent changes**: dummies that stay on until the end of the data (price increase, checkout rollout) are
#   poorly identified in a two-year regression — they are nearly collinear with the trend and seasonal terms, so their
#   coefficients here should not be read as effects. Those changes are measured properly with a control group
#   (difference-in-differences in notebook 02, the experiment in notebook 03).

# %%
fig, ax = plt.subplots(figsize=(11, 3.2))
res = pd.Series(fits["revenue"].resid, index=DAYS)
ax.plot(res.index, res, color=GREY, lw=0.9)
sd = res.std()
ax.axhline(0, color=INK2, lw=0.8)
for k in (-3, 3):
    ax.axhline(k * sd, color=BAD, lw=0.8, ls="--")
ax.set_title("Residual of the revenue model: what trend, seasonality and the known calendar do not explain")
ax.set_ylabel("log residual")
ax.annotate("±3 sd", (res.index[5], 3 * sd), xytext=(0, 4), textcoords="offset points", color=BAD, fontsize=8)
save(fig, "01_revenue_residual")
print(f"days beyond ±3 sd: {(res.abs() > 3 * sd).sum()} of {D} (expected by chance under normality: {D * 0.0027:.1f}); "
      f"residual sd = {sd:.1%} of daily revenue")
print("Once the calendar is in the model nothing unexplained is left. Notebook 04 does the reverse: it detects "
      "anomalies WITHOUT the calendar and then checks them against it.")

# %% [markdown]
# ### 4.2 Mix shift vs rate change: the paid search budget cut (Jan–Feb 2026)
# Paid Search converts below average, so halving it *raises* the blended conversion rate even if no channel
# converts better. At the same time the macro slowdown lowers every channel's rate. The Kitagawa
# decomposition splits the change in blended conversion into a **mix** component and a **rate** component.

# %%
def period(a, b):
    m = cube[cube["day"].between(day_index(a), day_index(b))]
    g = m.groupby("channel", observed=True)[["sessions", "purchased"]].sum()
    g["share"] = g["sessions"] / g["sessions"].sum()
    g["conv"] = g["purchased"] / g["sessions"]
    return g

pre, post = period("2025-10-06", "2025-11-02"), period("2026-01-26", "2026-02-22")   # 4 clean weeks each
blend_pre, blend_post = (pre["share"] * pre["conv"]).sum(), (post["share"] * post["conv"]).sum()
mix = ((post["share"] - pre["share"]) * (post["conv"] + pre["conv"]) / 2).sum()
rate = ((post["conv"] - pre["conv"]) * (post["share"] + pre["share"]) / 2).sum()
t = pd.DataFrame({"share before": pre["share"], "share during": post["share"], "conv before": pre["conv"], "conv during": post["conv"]})
t["conv change"] = t["conv during"] / t["conv before"] - 1
t["p (two-prop z)"] = [two_prop(pre.loc[c, "purchased"], pre.loc[c, "sessions"], post.loc[c, "purchased"], post.loc[c, "sessions"])["p"] for c in t.index]
display(t)
print(f"blended conversion: {blend_pre:.3%} → {blend_post:.3%}  (change {blend_post - blend_pre:+.3%} pts)")
print(f"   mix effect  (less Paid Search):        {mix:+.3%} pts")
print(f"   rate effect (within-channel change):   {rate:+.3%} pts")
pd.DataFrame({"component": ["blended before", "mix effect", "rate effect", "blended during"],
              "value": [blend_pre, mix, rate, blend_post]}).to_csv(OUT / "mix_shift_paid_cut.csv", index=False)

# %% [markdown]
# Every channel converts **worse** during the window (the macro slowdown: all six within-channel changes are negative,
# four of them significant on their own), yet the blended rate falls by less than the within-channel rates because the
# low-converting Paid Search share was cut almost in half. Reporting only the blended rate would hide a real, broad decline in purchase intent behind a
# favourable mix change. This is the practical form of Simpson's paradox; segment-level rates should be the default view.
