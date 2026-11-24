"""
Synthetic e-commerce store data generator ("Shop Stats").

Produces a relational dataset covering the full purchase life cycle:
    visit -> product view -> add to cart -> checkout -> payment -> purchase
          -> delivery -> return / refund / review

Planted structure (all recorded in data/events_calendar.csv as ground truth):
    * growth trend, weekly and annual seasonality
    * 12 promotions (some effective, one ineffective, pull-forward effects)
    * internal incidents (site outage, checkout bug, tracking bug, price increase,
      loyalty launch, new category, paid budget cut, checkout redesign rollout)
    * external shocks (competitor campaign, viral social spike, bot traffic,
      carrier strike, payment-provider outage, heatwave, macro slowdown)
    * two A/B tests (one real heterogeneous effect, one true null)

Usage:  python src/generate_data.py [scale]      (scale 1.0 ~= 12M sessions)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
SEED = 20261123
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

START = pd.Timestamp("2024-11-01")
END = pd.Timestamp("2026-10-31")
DAYS = pd.date_range(START, END, freq="D")
D = len(DAYS)
rng = np.random.default_rng(SEED)
t0 = time.time()


def di(date):
    """day index of a date string"""
    return int((pd.Timestamp(date) - START).days)


def win(start, end):
    """boolean mask over days for an inclusive date window"""
    m = np.zeros(D, bool)
    m[max(di(start), 0):min(di(end), D - 1) + 1] = True
    return m


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------
CHANNELS = ["Organic Search", "Paid Search", "Direct", "Social", "Email", "Affiliate"]
ORG, PAID, DIRECT, SOCIAL, EMAIL, AFF = range(6)
CH_SHARE = np.array([0.30, 0.22, 0.18, 0.12, 0.11, 0.07])
CH_GROWTH = np.array([1.35, 1.45, 1.40, 1.90, 1.60, 1.30])       # end / start
CH_NEW = np.array([0.12, 0.18, 0.05, 0.20, 0.00, 0.15])          # share of sessions from brand-new users

DEVICES = ["desktop", "mobile", "tablet"]
DESK, MOB, TAB = range(3)

COUNTRIES = ["US", "UK", "DE", "CA", "FR", "AU", "NL"]
CTRY_SHARE = np.array([0.45, 0.15, 0.12, 0.09, 0.08, 0.06, 0.05])
CTRY_CART = np.array([0.00, 0.03, 0.05, -0.02, -0.05, -0.04, 0.00])
CTRY_SHIP = np.array([3.0, 3.0, 3.0, 5.0, 4.0, 7.0, 3.5])
DE = COUNTRIES.index("DE")

AGES = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_SHARE = np.array([0.16, 0.30, 0.24, 0.16, 0.14])
AGE_CART = np.array([-0.15, 0.00, 0.08, 0.10, 0.05])
AGE_MOBILE = np.array([0.12, 0.05, 0.00, -0.07, -0.15])

GENDERS = ["F", "M", "Unspecified"]
GENDER_SHARE = np.array([0.52, 0.44, 0.04])

CATS = ["Electronics", "Fashion", "Home & Garden", "Beauty", "Sports & Outdoors",
        "Toys & Games", "Books & Media", "Pet Supplies"]
ELEC, FASH, HOME, BEAU, SPORT, TOYS, BOOKS, PET = range(8)
CAT_N = np.array([450, 700, 500, 350, 350, 250, 250, 150])
CAT_PRICE_MED = np.array([180, 45, 60, 28, 55, 30, 16, 24.0])
CAT_PRICE_SIG = np.array([0.80, 0.60, 0.70, 0.50, 0.60, 0.50, 0.40, 0.50])
CAT_MARGIN = np.array([0.22, 0.52, 0.42, 0.60, 0.40, 0.38, 0.30, 0.35])
CAT_RETURN = np.array([0.07, 0.19, 0.08, 0.04, 0.09, 0.05, 0.02, 0.03])
CAT_CART = np.array([-0.25, 0.00, -0.05, 0.15, -0.05, 0.05, 0.25, 0.20])
CAT_SHARE = np.array([0.20, 0.24, 0.15, 0.11, 0.10, 0.07, 0.08, 0.05])
GENDER_CAT = np.array([
    [0.80, 1.25, 1.05, 1.60, 0.85, 1.00, 1.05, 1.05],   # F
    [1.30, 0.80, 0.95, 0.50, 1.25, 1.00, 0.95, 0.95],   # M
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
])

PAY_METHODS = ["card", "paypal", "wallet", "bnpl"]
CARD, PAYPAL, WALLET, BNPL = range(4)
PAY_P = np.array([[0.68, 0.22, 0.04, 0.06],    # desktop
                  [0.50, 0.18, 0.26, 0.06],    # mobile
                  [0.60, 0.22, 0.12, 0.06]])   # tablet

# base funnel step probabilities by device (desktop, mobile, tablet)
B_VIEW = logit([0.70, 0.60, 0.63])
B_CART = logit([0.20, 0.16, 0.17])
B_CHK = logit([0.60, 0.50, 0.54])
B_PAY = logit([0.80, 0.70, 0.74])
B_PUR = logit([0.88, 0.82, 0.85])
CH_VIEW = np.array([0.00, 0.10, 0.15, -0.25, 0.30, 0.05])
CH_CARTL = np.array([0.00, -0.05, 0.25, -0.45, 0.45, 0.10])

# --------------------------------------------------------------------------
# Promotions (internal, planned)
# --------------------------------------------------------------------------
ALL = list(range(8))
PROMOS = [
    # id, name, start, end, categories, discount, kind, audience, traffic mult (all, email, paid)
    ("P01", "Black Friday / Cyber Week 2024", "2024-11-29", "2024-12-02", ALL, 0.25, "sitewide % off", "All", (1.9, 2.6, 2.3)),
    ("P02", "Spring Sale 2025", "2025-03-20", "2025-03-26", [FASH, HOME], 0.15, "category % off", "All", (1.22, 1.6, 1.25)),
    ("P03", "Summer Clearance 2025", "2025-07-10", "2025-07-20", [FASH], 0.30, "category % off", "All", (1.15, 1.4, 1.15)),
    ("P04", "Back to School 2025", "2025-08-18", "2025-08-31", [ELEC, BOOKS], 0.10, "category % off", "All", (1.08, 1.2, 1.1)),
    ("P05", "LOYAL20 Member Flash Sale", "2025-09-18", "2025-09-21", ALL, 0.20, "member code % off", "Loyalty members", (1.0, 2.0, 1.0)),
    ("P06", "Free Shipping Weekend (Oct 2025)", "2025-10-24", "2025-10-26", ALL, 0.0, "free shipping", "All", (1.06, 1.3, 1.0)),
    ("P07", "Black Friday / Cyber Week 2025", "2025-11-28", "2025-12-01", ALL, 0.25, "sitewide % off", "All", (1.9, 2.6, 2.3)),
    ("P08", "Beauty BOGO Week", "2026-02-09", "2026-02-15", [BEAU], 0.50, "buy one get one", "All", (1.02, 1.1, 1.0)),
    ("P09", "Spring Sale 2026", "2026-03-19", "2026-03-25", [FASH, HOME], 0.15, "category % off", "All", (1.22, 1.6, 1.25)),
    ("P10", "Free Shipping Weekend (May 2026)", "2026-05-15", "2026-05-17", ALL, 0.0, "free shipping", "All", (1.06, 1.3, 1.0)),
    ("P11", "Summer Clearance 2026", "2026-07-09", "2026-07-19", [FASH], 0.30, "category % off", "All", (1.15, 1.4, 1.15)),
    ("P12", "Back to School 2026", "2026-08-17", "2026-08-30", [ELEC, BOOKS], 0.10, "category % off", "All", (1.08, 1.2, 1.1)),
]

# other planted events -------------------------------------------------------
OUTAGE_DAY, OUTAGE_HOURS = di("2025-02-18"), range(10, 16)
LOYALTY_LAUNCH = di("2025-04-01")
COMPETITOR = win("2025-05-05", "2025-05-25")
CHECKOUT_BUG = win("2025-06-10", "2025-06-13")
PET_LAUNCH = di("2025-07-01")
VIRAL = {di("2025-08-09"): 5.5, di("2025-08-10"): 4.0, di("2025-08-11"): 2.2, di("2025-08-12"): 1.4}
PRICE_INC = di("2025-09-01")
PRICE_INC_PCT = 0.08
TRACK_BUG = win("2025-10-05", "2025-10-07")
BOTS = win("2025-11-03", "2025-11-06")
STRIKE = win("2025-12-04", "2025-12-19")
PAID_CUT = win("2026-01-15", "2026-02-28")
DEFECT = win("2026-02-01", "2026-02-28")
AB1 = win("2026-03-02", "2026-04-12")
ROLLOUT = di("2026-05-04")
PAY_OUT_DAY, PAY_OUT_HOURS = di("2026-06-17"), range(13, 17)
HEAT = win("2026-06-23", "2026-07-02")
AB2 = win("2026-09-07", "2026-10-04")
# macro slowdown: ramps in during January 2026, out during April 2026
MACRO = np.zeros(D)
a, b, c, e = di("2026-01-05"), di("2026-01-20"), di("2026-04-10"), di("2026-04-30")
MACRO[a:b] = np.linspace(0, 1, b - a)
MACRO[b:c] = 1.0
MACRO[c:e] = np.linspace(1, 0, e - c)

EVENTS = [
    ("E01", "Site outage (6 hours)", "2025-02-18", "2025-02-18", "internal", "incident",
     "Traffic lost 10:00-16:00; sessions down ~30% for the day"),
    ("E02", "Loyalty program launch", "2025-04-01", str(END.date()), "internal", "product change",
     "Members visit more often and convert better; gradual trend change (selection bias: high-intent users join)"),
    ("E03", "Competitor electronics campaign", "2025-05-05", "2025-05-25", "external", "competition",
     "Electronics add-to-cart rate down; paid and organic search traffic slightly down"),
    ("E04", "Mobile checkout bug", "2025-06-10", "2025-06-13", "internal", "incident",
     "Payment -> purchase conversion on mobile collapses; desktop unaffected"),
    ("E05", "Pet Supplies category launch", "2025-07-01", str(END.date()), "internal", "assortment",
     "New category ramps to ~5% of product interest over 60 days"),
    ("E06", "Viral social media mention", "2025-08-09", "2025-08-12", "external", "traffic shock",
     "Social traffic up to 5.5x, low intent: conversion rate falls while orders rise"),
    ("E07", "Electronics price increase (+8%)", "2025-09-01", str(END.date()), "internal", "pricing",
     "Electronics list prices +8%: AOV and margin up, add-to-cart rate down (level shift)"),
    ("E08", "Desktop add-to-cart tracking bug", "2025-10-05", "2025-10-07", "internal", "data quality",
     "70% of desktop add-to-cart events not recorded; purchases unaffected (impossible funnel states)"),
    ("E09", "Bot traffic wave", "2025-11-03", "2025-11-06", "external", "traffic shock",
     "Direct desktop traffic ~3x with single-page zero-duration sessions; conversion rate diluted, orders normal"),
    ("E10", "Carrier strike (Germany)", "2025-12-04", "2025-12-19", "external", "logistics",
     "DE delivery times +5 days, cancellations and returns up, ratings down"),
    ("E11", "Paid search budget cut", "2026-01-15", "2026-02-28", "internal", "marketing",
     "Paid search sessions -50%; blended conversion rate changes through channel mix only"),
    ("E12", "Macro slowdown / weak consumer confidence", "2026-01-05", "2026-04-30", "external", "macro",
     "Higher price sensitivity: add-to-cart rate and AOV drift down, premium items hit hardest"),
    ("E13", "Defective batch of one electronics SKU", "2026-02-01", "2026-02-28", "internal", "product quality",
     "Return rate of a single best-selling electronics product jumps to ~55%"),
    ("E14", "A/B test 1: one-page checkout", "2026-03-02", "2026-04-12", "internal", "experiment",
     "Treatment improves checkout completion on mobile (and half as much on tablet); no effect on desktop"),
    ("E15", "One-page checkout rolled out to 100%", "2026-05-04", str(END.date()), "internal", "product change",
     "Permanent level shift in mobile checkout -> purchase conversion"),
    ("E16", "Payment provider outage (cards, 4 hours)", "2026-06-17", "2026-06-17", "external", "incident",
     "Card payments fail 13:00-17:00; PayPal / wallet / BNPL unaffected"),
    ("E17", "Heatwave", "2026-06-23", "2026-07-02", "external", "weather",
     "Home & Garden demand (cooling) more than doubles"),
    ("E18", "A/B test 2: social-proof badge", "2026-09-07", "2026-10-04", "internal", "experiment",
     "True null effect (useful for false-positive / multiple-testing checks)"),
    # recurring seasonal effects (not incidents, but they move category and logistics metrics abruptly)
    ("E19", "Holiday toy season 2024", "2024-11-15", "2024-12-24", "external", "seasonality",
     "Toys & Games product interest x2.2 from mid-November until Christmas Eve"),
    ("E20", "Holiday toy season 2025", "2025-11-15", "2025-12-24", "external", "seasonality",
     "Toys & Games product interest x2.2 from mid-November until Christmas Eve"),
    ("E21", "Spring sports season 2025", "2025-04-01", "2025-06-30", "external", "seasonality",
     "Sports & Outdoors product interest x1.3 in April-June"),
    ("E22", "Spring sports season 2026", "2026-04-01", "2026-06-30", "external", "seasonality",
     "Sports & Outdoors product interest x1.3 in April-June"),
    ("E23", "Peak-season carrier congestion 2024", "2024-11-28", "2024-12-12", "external", "seasonality",
     "Delivery times +1 day in all countries during the holiday peak"),
    ("E24", "Peak-season carrier congestion 2025", "2025-11-28", "2025-12-12", "external", "seasonality",
     "Delivery times +1 day in all countries during the holiday peak"),
]

# --------------------------------------------------------------------------
# 1. Daily traffic intensity by channel
# --------------------------------------------------------------------------
tt = np.arange(D) / (D - 1)
doy = DAYS.dayofyear.values
dow = DAYS.dayofweek.values


def bump(center, sd):
    d = np.abs(doy - center)
    d = np.minimum(d, 365 - d)
    return np.exp(-0.5 * (d / sd) ** 2)


season = 1 + 0.22 * bump(339, 18) + 0.05 * bump(84, 14) - 0.07 * bump(201, 30)
DOW_ALL = np.array([1.00, 0.97, 0.96, 0.98, 1.02, 1.10, 1.12])
DOW_EMAIL = np.array([0.90, 1.25, 0.92, 1.22, 0.95, 0.88, 0.88])   # campaign sends Tue / Thu

BASE_DAILY = 12600 * SCALE
lam = np.zeros((D, 6))
day_noise = np.exp(rng.normal(0, 0.035, D))
for c in range(6):
    growth = CH_GROWTH[c] ** tt
    w = DOW_EMAIL[dow] if c == EMAIL else DOW_ALL[dow]
    lam[:, c] = BASE_DAILY * CH_SHARE[c] * growth * season * w * day_noise * np.exp(rng.normal(0, 0.04, D))

for pid, name, s, e, cats, pct, kind, aud, (m_all, m_email, m_paid) in PROMOS:
    m = win(s, e)
    lam[m] *= m_all
    lam[m, EMAIL] *= m_email / m_all
    lam[m, PAID] *= m_paid / m_all
lam[COMPETITOR, PAID] *= 0.88
lam[COMPETITOR, ORG] *= 0.95
lam[PAID_CUT, PAID] *= 0.50
lam[HEAT] *= 1.05
for d_, mult in VIRAL.items():
    lam[d_, SOCIAL] *= mult

HOUR_W = np.array([1, .6, .4, .3, .3, .5, 1, 1.8, 2.6, 3.2, 3.6, 3.9, 4.2, 4.1, 3.8, 3.7,
                   3.8, 4.1, 4.6, 5.2, 5.4, 4.8, 3.4, 2.0])
HOUR_P = HOUR_W / HOUR_W.sum()
hour_p_outage = HOUR_W.copy()
hour_p_outage[list(OUTAGE_HOURS)] *= 0.07
lam[OUTAGE_DAY] *= hour_p_outage.sum() / HOUR_W.sum()
hour_p_outage /= hour_p_outage.sum()

# --------------------------------------------------------------------------
# 2. Day loop: who visits (top-down counts, bottom-up user identity)
# --------------------------------------------------------------------------
N_SEED = int(200_000 * SCALE)
MAX_USERS = int(N_SEED + lam.sum() * 0.20 + 500_000 * SCALE)
u_signup = np.zeros(MAX_USERS, np.int32)
u_act = rng.lognormal(0, 1.0, MAX_USERS).astype(np.float64)        # latent visit frequency
u_z = rng.normal(0, 0.6, MAX_USERS).astype(np.float32)             # latent purchase propensity
u_acq = np.zeros(MAX_USERS, np.int8)
u_bot = np.zeros(MAX_USERS, bool)
u_join = np.full(MAX_USERS, 10 ** 6, np.int32)                     # loyalty join day (10**6 = never)
u_dec = np.zeros(MAX_USERS)                                        # recency decay state

u_signup[:N_SEED] = -rng.integers(1, 541, N_SEED)
u_acq[:N_SEED] = rng.choice(6, N_SEED, p=CH_SHARE)
u_dec[:N_SEED] = np.exp(u_signup[:N_SEED] / 75.0)


def assign_join(lo, hi):
    """loyalty enrolment: likelier for high-propensity users (selection effect)"""
    n_ = hi - lo
    joins = rng.random(n_) < sigmoid(-1.6 + 0.9 * u_z[lo:hi])
    jd = np.maximum(LOYALTY_LAUNCH, u_signup[lo:hi]) + rng.exponential(40, n_).astype(np.int32)
    u_join[lo:hi] = np.where(joins & (jd < D), jd, 10 ** 6)


assign_join(0, N_SEED)
n_users = N_SEED
DECAY = np.exp(-1 / 75.0)

s_day, s_ch, s_user, s_new, s_bot = [], [], [], [], []
for d in range(D):
    counts = rng.poisson(lam[d])
    new_frac = CH_NEW.copy()
    if d in VIRAL:
        new_frac[SOCIAL] = 0.70 if VIRAL[d] > 2 else 0.45
    n_new = rng.binomial(counts, new_frac)
    n_ret = counts - n_new
    n_bot = int(rng.poisson(lam[d, DIRECT] * 2.2)) if BOTS[d] else 0

    # returning visitors, sampled by activity x recency x loyalty boost
    w = u_act[:n_users] * (0.12 + 0.88 * u_dec[:n_users])
    w[u_join[:n_users] <= d] *= 1.5
    cw = np.cumsum(w)
    k = int(n_ret.sum())
    picks = np.searchsorted(cw, rng.random(k) * cw[-1]).astype(np.int32)
    ch_ret = np.repeat(np.arange(6, dtype=np.int8), n_ret)
    rng.shuffle(picks)

    # new visitors (first session on signup day)
    kn = int(n_new.sum())
    lo, hi = n_users, n_users + kn + n_bot
    ch_new = np.repeat(np.arange(6, dtype=np.int8), n_new)
    u_signup[lo:hi] = d
    u_acq[lo:lo + kn] = ch_new
    u_acq[lo + kn:hi] = DIRECT
    u_bot[lo + kn:hi] = True
    u_dec[lo:hi] = 1.0
    assign_join(lo, lo + kn)
    n_users = hi

    s_day.append(np.full(k + kn + n_bot, d, np.int16))
    s_ch.append(np.concatenate([ch_ret, ch_new, np.full(n_bot, DIRECT, np.int8)]))
    s_user.append(np.concatenate([picks, np.arange(lo, hi, dtype=np.int32)]))
    s_new.append(np.concatenate([np.zeros(k, bool), np.ones(kn + n_bot, bool)]))
    s_bot.append(np.concatenate([np.zeros(k + kn, bool), np.ones(n_bot, bool)]))
    u_dec[:n_users] *= DECAY

day = np.concatenate(s_day); ch = np.concatenate(s_ch); uid = np.concatenate(s_user)
is_new = np.concatenate(s_new); is_bot = np.concatenate(s_bot)
del s_day, s_ch, s_user, s_new, s_bot
N = len(day)
NU = n_users
log(f"sessions: {N:,}   users: {NU:,}")

# --------------------------------------------------------------------------
# 3. User attributes
# --------------------------------------------------------------------------
u_signup = u_signup[:NU]; u_z = u_z[:NU]; u_acq = u_acq[:NU]; u_bot = u_bot[:NU]; u_join = u_join[:NU]
u_ctry = rng.choice(7, NU, p=CTRY_SHARE).astype(np.int8)
u_ctry[u_bot] = 0
u_age = rng.choice(5, NU, p=AGE_SHARE).astype(np.int8)
u_gender = rng.choice(3, NU, p=GENDER_SHARE).astype(np.int8)
p_mob = 0.52 + 0.10 * np.clip(u_signup / (D - 1), -0.5, 1) + AGE_MOBILE[u_age]
r = rng.random(NU)
u_dev = np.where(r < p_mob, MOB, np.where(r < p_mob + 0.08, TAB, DESK)).astype(np.int8)
u_dev[u_bot] = DESK
u_ab1 = rng.integers(0, 2, NU).astype(np.int8)
u_ab2 = rng.integers(0, 2, NU).astype(np.int8)

# --------------------------------------------------------------------------
# 4. Session attributes
# --------------------------------------------------------------------------
r = rng.random(N)
dev = np.where(r < 0.80, u_dev[uid], rng.choice(3, N, p=[0.37, 0.55, 0.08])).astype(np.int8)
dev[is_bot] = DESK
hour = rng.choice(24, N, p=HOUR_P).astype(np.int8)
m = day == OUTAGE_DAY
hour[m] = rng.choice(24, m.sum(), p=hour_p_outage)
hour[is_bot] = rng.integers(0, 24, is_bot.sum())
sec = rng.integers(0, 3600, N).astype(np.int32)

ctry = u_ctry[uid]
age = u_age[uid]
z = u_z[uid]
member = u_join[uid] <= day

# --------------------------------------------------------------------------
# 5. Products
# --------------------------------------------------------------------------
NP_ = int(CAT_N.sum())
p_cat = np.repeat(np.arange(8), CAT_N).astype(np.int8)
p_price = np.round(np.exp(np.log(CAT_PRICE_MED[p_cat]) + rng.normal(0, 1, NP_) * CAT_PRICE_SIG[p_cat]), 0) - 0.01
p_price = np.maximum(p_price, 3.99)
p_margin = np.clip(CAT_MARGIN[p_cat] + rng.normal(0, 0.06, NP_), 0.08, 0.80)
p_cost = np.round(p_price * (1 - p_margin), 2)
p_quality = np.clip(rng.normal(4.3, 0.25, NP_), 3.2, 4.9)
p_pop = np.zeros(NP_)
cat_idx, cat_p, cat_logmean = [], [], np.zeros(8)
off = 0
for c in range(8):
    n_c = CAT_N[c]
    rank = rng.permutation(n_c) + 1
    pop = rank ** -0.9
    # cheaper items are somewhat more popular
    pop = pop * (p_price[off:off + n_c] / CAT_PRICE_MED[c]) ** -0.25
    p_pop[off:off + n_c] = pop / pop.sum()
    cat_idx.append(np.arange(off, off + n_c))
    cat_p.append(p_pop[off:off + n_c])
    cat_logmean[c] = np.sum(np.log(p_price[off:off + n_c]) * p_pop[off:off + n_c])
    off += n_c
# the defective SKU: 3rd most popular electronics product
DEFECT_PID = int(cat_idx[ELEC][np.argsort(-cat_p[ELEC])[2]])
p_launch = np.where(p_cat == PET, PET_LAUNCH, -rng.integers(30, 900, NP_))

# --------------------------------------------------------------------------
# 6. Funnel
# --------------------------------------------------------------------------
# product-category interest by day
T = np.tile(CAT_SHARE, (D, 1))
T[:, PET] = CAT_SHARE[PET] * np.clip((np.arange(D) - PET_LAUNCH) / 60.0, 0, 1)
toy_season = np.zeros(D, bool)
for yr in (2024, 2025):
    toy_season |= win(f"{yr}-11-15", f"{yr}-12-24")
T[toy_season, TOYS] *= 2.2
T[np.isin(DAYS.month, [4, 5, 6]), SPORT] *= 1.3
T[HEAT, HOME] *= 2.3

# promo matrices
disc = np.zeros((D, 8))            # public discount by day x category
promo_cart = np.zeros((D, 8))      # add-to-cart logit lift
pull = np.zeros((D, 8))            # pull-forward (negative) logit
free_ship = np.zeros(D, bool)
promo_code = np.full((D, 8), -1, np.int8)
bf_days = np.zeros(D, bool)
loyal_days = np.zeros(D, bool)
bogo_days = np.zeros(D, bool)
for i, (pid, name, s, e, cats, pct, kind, aud, mults) in enumerate(PROMOS):
    m = win(s, e)
    a_, b_ = di(s), di(e)
    if kind == "free shipping":
        free_ship |= m
        promo_code[m] = i
        continue
    if aud != "All":
        loyal_days |= m
        continue
    if kind == "buy one get one":
        bogo_days |= m
        promo_cart[np.ix_(m, cats)] += 0.04           # barely moves demand
        promo_code[np.ix_(m, cats)] = i
        continue
    disc[np.ix_(m, cats)] = pct
    promo_code[np.ix_(m, cats)] = i
    promo_cart[np.ix_(m, cats)] += 1.2 * pct
    if kind == "sitewide % off":
        bf_days |= m
        promo_cart[m] += 0.10
        pull[max(a_ - 7, 0):a_] -= 0.10
        pull[b_ + 1:b_ + 8] -= 0.18
    if "Clearance" in name:
        pull[b_ + 1:b_ + 6, FASH] -= 0.12
LOYAL_IDX = [p[0] for p in PROMOS].index("P05")

view_l = B_VIEW[dev] + CH_VIEW[ch]
viral_s = np.isin(day, list(VIRAL)) & (ch == SOCIAL)
view_l = view_l - 0.5 * viral_s
view_l[is_bot] = -6.0
viewed = rng.random(N) < sigmoid(view_l)
del view_l

# category + main product for sessions with a product view
cat = np.full(N, -1, np.int8)
prod = np.full(N, -1, np.int32)
vi = np.flatnonzero(viewed)
g_v = u_gender[uid[vi]]
CH = 1_000_000
for s0 in range(0, len(vi), CH):
    sl = slice(s0, s0 + CH)
    P = T[day[vi[sl]]] * GENDER_CAT[g_v[sl]]
    cum = np.cumsum(P, axis=1)
    rr = rng.random(len(cum)) * cum[:, -1]
    cat[vi[sl]] = (rr[:, None] > cum).sum(1)
for c in range(8):
    m = np.flatnonzero(cat == c)
    prod[m] = cat_idx[c][rng.choice(CAT_N[c], len(m), p=cat_p[c])]
del g_v

cv, dv = cat[vi], day[vi]
price_eff = p_price[prod[vi]] * np.where((cv == ELEC) & (dv >= PRICE_INC), 1 + PRICE_INC_PCT, 1.0)
pricez = np.log(p_price[prod[vi]]) - cat_logmean[cv]
beta = 0.20 + 0.20 * MACRO
cart_l = (B_CART[dev[vi]] + CH_CARTL[ch[vi]] + CAT_CART[cv] + AGE_CART[age[vi]] + CTRY_CART[ctry[vi]]
          + z[vi] - 0.35 * is_new[vi] + 0.20 * member[vi]
          + promo_cart[dv, cv] + pull[dv, cv]
          + 0.24 * (member[vi] & loyal_days[dv])
          - beta[dv] * pricez
          - 0.10 * ((cv == ELEC) & (dv >= PRICE_INC))
          - 0.30 * ((cv == ELEC) & COMPETITOR[dv])
          - 0.60 * viral_s[vi]
          - 0.06 * MACRO[dv]
          + 0.25 * ((cv == HOME) & HEAT[dv]))
carted = np.zeros(N, bool)
carted[vi] = rng.random(len(vi)) < sigmoid(cart_l)
del cart_l, pricez

ci = np.flatnonzero(carted)
dc = day[ci]
chk_l = (B_CHK[dev[ci]] + 0.3 * z[ci] + 0.10 * member[ci] + 0.20 * free_ship[dc]
         - 0.10 * ((ctry[ci] == DE) & STRIKE[dc]) + 0.15 * bf_days[dc])
checkout = np.zeros(N, bool)
checkout[ci] = rng.random(len(ci)) < sigmoid(chk_l)

# A/B test 1 (one-page checkout): user-level 50/50, then full rollout
ki = np.flatnonzero(checkout)
dk = day[ki]
new_checkout = (AB1[dk] & (u_ab1[uid[ki]] == 1)) | (dk >= ROLLOUT)
dev_eff = np.array([0.0, 1.0, 0.5])[dev[ki]]
pay_l = B_PAY[dev[ki]] + 0.14 * dev_eff * new_checkout
payment = np.zeros(N, bool)
payment[ki] = rng.random(len(ki)) < sigmoid(pay_l)

pi_ = np.flatnonzero(payment)
dp = day[pi_]
pm = np.zeros(len(pi_), np.int8)
rr = rng.random(len(pi_))
cumP = np.cumsum(PAY_P, axis=1)[dev[pi_]]
pm = (rr[:, None] > cumP).sum(1).astype(np.int8)
new_checkout_p = (AB1[dp] & (u_ab1[uid[pi_]] == 1)) | (dp >= ROLLOUT)
pur_l = (B_PUR[dev[pi_]] + 0.08 * np.array([0.0, 1.0, 0.5])[dev[pi_]] * new_checkout_p
         - 1.6 * (CHECKOUT_BUG[dp] & (dev[pi_] == MOB))
         - 3.5 * ((dp == PAY_OUT_DAY) & np.isin(hour[pi_], list(PAY_OUT_HOURS)) & (pm == CARD))
         - 0.10 * (pm == BNPL))
purchased = np.zeros(N, bool)
purchased[pi_] = rng.random(len(pi_)) < sigmoid(pur_l)
pay_method = np.full(N, -1, np.int8)
pay_method[pi_] = pm
log(f"funnel: view {viewed.mean():.3f} cart {carted.mean():.3f} checkout {checkout.mean():.3f} "
    f"payment {payment.mean():.3f} purchase {purchased.mean():.4f}")

# recorded add-to-cart flag (tracking bug loses desktop events for 3 days)
cart_rec = carted.copy()
m = carted & TRACK_BUG[day] & (dev == DESK) & (rng.random(N) < 0.70)
cart_rec[m] = False

# engagement
pv = (1 + rng.poisson(0.6, N) + viewed * rng.poisson(2.6, N) + carted * rng.poisson(2.0, N)
      + checkout * 2 + payment * 1 + purchased * 1).astype(np.int16)
dur = (pv * rng.gamma(2.0, 24.0, N)).astype(np.int32)
pv[is_bot] = 1
dur[is_bot] = rng.integers(0, 3, is_bot.sum())

# --------------------------------------------------------------------------
# 7. Sort by time, assign session ids
# --------------------------------------------------------------------------
ts = day.astype(np.int64) * 86400 + hour.astype(np.int64) * 3600 + sec
order = np.argsort(ts, kind="stable")
inv = np.empty(N, np.int64)
inv[order] = np.arange(N)          # old index -> new session row
session_id = np.arange(1, N + 1, dtype=np.int64)
start_ts = (START.to_datetime64().astype("datetime64[s]") + ts[order].astype("timedelta64[s]"))

ab_test = np.full(N, -1, np.int8)
ab_var = np.full(N, -1, np.int8)
ab_test[AB1[day]] = 0; ab_var[AB1[day]] = u_ab1[uid[AB1[day]]]
ab_test[AB2[day]] = 1; ab_var[AB2[day]] = u_ab2[uid[AB2[day]]]
ab_test[is_bot] = -1; ab_var[is_bot] = -1


def cat_col(codes, labels):
    return pd.Categorical.from_codes(codes, categories=labels)


# --------------------------------------------------------------------------
# 8. Orders and order items
# --------------------------------------------------------------------------
oi = np.flatnonzero(purchased)
oi = oi[np.argsort(ts[oi], kind="stable")]
NO = len(oi)
o_day = day[oi]; o_user = uid[oi]; o_ctry = ctry[oi]; o_member = member[oi]; o_cat = cat[oi]
order_id = np.arange(1, NO + 1, dtype=np.int64)

n_extra = rng.poisson(0.55 + 0.30 * bf_days[o_day])
it_order = np.concatenate([np.arange(NO), np.repeat(np.arange(NO), n_extra)])
it_first = np.concatenate([np.ones(NO, bool), np.zeros(n_extra.sum(), bool)])
it_prod = np.concatenate([prod[oi], np.full(n_extra.sum(), -1, np.int32)])
xi = np.flatnonzero(~it_first)
x_day = o_day[it_order[xi]]
same = rng.random(len(xi)) < 0.60
x_cat = np.where(same, o_cat[it_order[xi]], -1).astype(np.int8)
need = np.flatnonzero(~same)
cum = np.cumsum(T[x_day[need]], axis=1)
rr = rng.random(len(need)) * cum[:, -1]
x_cat[need] = (rr[:, None] > cum).sum(1)
for c in range(8):
    m = np.flatnonzero(x_cat == c)
    it_prod[xi[m]] = cat_idx[c][rng.choice(CAT_N[c], len(m), p=cat_p[c])]
items = pd.DataFrame({"o": it_order, "product": it_prod, "first": it_first})
items = items.sort_values(["o", "first"], ascending=[True, False]).drop_duplicates(["o", "product"])
it_o = items["o"].values; it_p = items["product"].values
NI = len(items)
it_day = o_day[it_o]; it_cat = p_cat[it_p]
it_qty = rng.choice([1, 2, 3], NI, p=[0.88, 0.09, 0.03]).astype(np.int16)
bogo = bogo_days[it_day] & (it_cat == BEAU)
it_qty[bogo] = 2
it_list = np.round(p_price[it_p] * np.where((it_cat == ELEC) & (it_day >= PRICE_INC), 1 + PRICE_INC_PCT, 1.0), 2)
it_disc = disc[it_day, it_cat]
it_disc = np.maximum(it_disc, 0.20 * (loyal_days[it_day] & o_member[it_o]))
it_disc[bogo] = 0.50
it_paid = np.round(it_list * (1 - it_disc), 2)
it_promo = promo_code[it_day, it_cat].astype(np.int16)
it_promo[(loyal_days[it_day] & o_member[it_o]) & (it_disc == 0.20)] = LOYAL_IDX

gross = np.bincount(it_o, it_list * it_qty, NO)
revenue = np.bincount(it_o, it_paid * it_qty, NO)
cogs = np.bincount(it_o, p_cost[it_p] * it_qty, NO)
units = np.bincount(it_o, it_qty, NO).astype(np.int32)
n_lines = np.bincount(it_o, minlength=NO).astype(np.int16)
ship_fee = np.where((revenue >= 60) | free_ship[o_day] | o_member, 0.0, 5.99)
o_promo = np.full(NO, -1, np.int16)
has = it_promo >= 0
o_promo[it_o[has]] = it_promo[has]
o_promo[free_ship[o_day] & (o_promo < 0)] = promo_code[o_day[free_ship[o_day] & (o_promo < 0)], 0]

o_pm = pay_method[oi]
strike_o = STRIKE[o_day] & (o_ctry == DE)
p_cancel = 0.02 + 0.07 * strike_o + 0.015 * (o_pm == BNPL)
cancelled = rng.random(NO) < p_cancel
peak = np.zeros(D, bool)
for yr in (2024, 2025):
    peak |= win(f"{yr}-11-28", f"{yr}-12-12")
deliv = (CTRY_SHIP[o_ctry] + rng.gamma(2.0, 0.6, NO) + 1.0 * peak[o_day]
         + strike_o * np.clip(rng.normal(5.0, 1.5, NO), 1, None))
deliv = np.maximum(np.round(deliv), 1).astype(np.int16)

clearance_item = (it_disc == 0.30)
p_ret = CAT_RETURN[it_cat] * np.where(clearance_item, 1.4, 1.0) * np.where(deliv[it_o] > 8, 1.35, 1.0)
p_ret = np.where((it_p == DEFECT_PID) & DEFECT[it_day], 0.55, p_ret)
it_ret = (rng.random(NI) < p_ret) & ~cancelled[it_o]
n_ret_lines = np.bincount(it_o, it_ret, NO)
refund = np.bincount(it_o, it_ret * it_paid * it_qty, NO)
refund = np.where(cancelled, revenue + ship_fee, refund)
status = np.where(cancelled, 0, np.where(n_ret_lines == 0, 1, np.where(n_ret_lines == n_lines, 3, 2))).astype(np.int8)
STATUS = ["cancelled", "delivered", "partially_returned", "returned"]

qual = np.bincount(it_o, p_quality[it_p], NO) / n_lines
r_mean = qual - 0.12 * np.clip(deliv - CTRY_SHIP[o_ctry] - 3, 0, None) - 1.4 * (n_ret_lines > 0)
rating = np.clip(np.round(r_mean + rng.normal(0, 0.9, NO)), 1, 5)
has_rating = (rng.random(NO) < 0.30) & ~cancelled
rating = np.where(has_rating, rating, np.nan)

o_ts_sec = ts[oi] + dur[oi]
o_ts = START.to_datetime64().astype("datetime64[s]") + o_ts_sec.astype("timedelta64[s]")
deliv_date = (o_ts.astype("datetime64[D]") + deliv.astype("timedelta64[D]")).astype("datetime64[s]")

orders = pd.DataFrame({
    "order_id": order_id,
    "session_id": session_id[inv[oi]],
    "user_id": o_user.astype(np.int64) + 1,
    "order_ts": o_ts,
    "country": cat_col(o_ctry, COUNTRIES),
    "device": cat_col(dev[oi], DEVICES),
    "channel": cat_col(ch[oi], CHANNELS),
    "payment_method": cat_col(o_pm, PAY_METHODS),
    "promo_id": pd.Categorical.from_codes(o_promo, categories=[p[0] for p in PROMOS]),
    "n_lines": n_lines,
    "units": units,
    "gross_amount": np.round(gross, 2),
    "discount_amount": np.round(gross - revenue, 2),
    "revenue": np.round(revenue, 2),
    "shipping_fee": ship_fee,
    "cogs": np.round(cogs, 2),
    "status": cat_col(status, STATUS),
    "delivery_days": pd.array(np.where(cancelled, -1, deliv), dtype="Int16"),
    "delivered_date": pd.Series(deliv_date).where(~cancelled),
    "refund_amount": np.round(refund, 2),
    "review_rating": pd.array(rating, dtype="Float32"),
})
orders.loc[cancelled, "delivery_days"] = pd.NA

order_items = pd.DataFrame({
    "order_id": order_id[it_o],
    "product_id": it_p.astype(np.int32) + 1,
    "category": cat_col(it_cat, CATS),
    "quantity": it_qty,
    "list_price": it_list,
    "discount_pct": np.round(it_disc, 2).astype(np.float32),
    "unit_price_paid": it_paid,
    "unit_cost": p_cost[it_p],
    "returned": it_ret,
})
log(f"orders: {NO:,}   order items: {NI:,}")

# --------------------------------------------------------------------------
# 9. Assemble and write
# --------------------------------------------------------------------------
o_id_for_session = np.zeros(N, np.int64)
o_id_for_session[oi] = order_id


def r_(x):
    return x[order]


sessions = pd.DataFrame({
    "session_id": session_id,
    "user_id": r_(uid).astype(np.int64) + 1,
    "session_start": start_ts,
    "channel": cat_col(r_(ch), CHANNELS),
    "device": cat_col(r_(dev), DEVICES),
    "country": cat_col(r_(ctry), COUNTRIES),
    "is_new_user": r_(is_new),
    "is_loyalty_member": r_(member),
    "pageviews": r_(pv),
    "duration_sec": r_(dur),
    "viewed_category": cat_col(r_(cat), CATS),
    "viewed_product_id": pd.array(np.where(r_(prod) >= 0, r_(prod) + 1, 0), dtype="Int32"),
    "viewed_product": r_(viewed),
    "added_to_cart": r_(cart_rec),
    "began_checkout": r_(checkout),
    "entered_payment": r_(payment),
    "purchased": r_(purchased),
    "payment_method": cat_col(r_(pay_method), PAY_METHODS),
    "ab_test": cat_col(r_(ab_test), ["AB1_one_page_checkout", "AB2_social_proof_badge"]),
    "ab_variant": cat_col(r_(ab_var), ["control", "treatment"]),
    "order_id": pd.array(r_(o_id_for_session), dtype="Int64"),
})
sessions.loc[sessions["viewed_product_id"] == 0, "viewed_product_id"] = pd.NA
sessions.loc[sessions["order_id"] == 0, "order_id"] = pd.NA

signup_dt = START.to_datetime64().astype("datetime64[D]") + u_signup.astype("timedelta64[D]")
join_dt = np.where(u_join < D, START.to_datetime64().astype("datetime64[D]")
                   + np.minimum(u_join, D).astype("timedelta64[D]"), np.datetime64("NaT"))
users = pd.DataFrame({
    "user_id": np.arange(1, NU + 1, dtype=np.int64),
    "signup_date": signup_dt.astype("datetime64[s]"),
    "country": cat_col(u_ctry, COUNTRIES),
    "age_group": cat_col(u_age, AGES),
    "gender": cat_col(u_gender, GENDERS),
    "acquisition_channel": cat_col(u_acq, CHANNELS),
    "preferred_device": cat_col(u_dev, DEVICES),
    "loyalty_join_date": join_dt.astype("datetime64[s]"),
})

launch_dt = START.to_datetime64().astype("datetime64[D]") + p_launch.astype("timedelta64[D]")
products = pd.DataFrame({
    "product_id": np.arange(1, NP_ + 1, dtype=np.int32),
    "product_name": [f"{CATS[c].split(' ')[0]}-{i + 1:04d}" for i, c in enumerate(p_cat)],
    "category": cat_col(p_cat, CATS),
    "base_price": p_price,
    "unit_cost": p_cost,
    "launch_date": launch_dt.astype("datetime64[s]"),
})
price_history = pd.DataFrame({
    "product_id": np.flatnonzero(p_cat == ELEC) + 1,
    "effective_from": pd.Timestamp("2025-09-01"),
    "new_price": np.round(p_price[p_cat == ELEC] * (1 + PRICE_INC_PCT), 2),
})
promotions = pd.DataFrame([{
    "promo_id": p[0], "name": p[1], "start_date": p[2], "end_date": p[3],
    "categories": "All" if len(p[4]) == 8 else "; ".join(CATS[c] for c in p[4]),
    "discount_pct": p[5], "type": p[6], "audience": p[7]} for p in PROMOS])
events = pd.DataFrame(EVENTS, columns=["event_id", "name", "start_date", "end_date", "origin", "type", "expected_signature"])
ab_tests = pd.DataFrame([
    {"ab_test": "AB1_one_page_checkout", "start_date": "2026-03-02", "end_date": "2026-04-12",
     "unit": "user", "allocation": "50/50", "primary_metric": "checkout -> purchase conversion",
     "hypothesis": "A single-page checkout reduces abandonment between checkout start and purchase"},
    {"ab_test": "AB2_social_proof_badge", "start_date": "2026-09-07", "end_date": "2026-10-04",
     "unit": "user", "allocation": "50/50", "primary_metric": "product view -> add to cart conversion",
     "hypothesis": "A 'popular right now' badge on product pages increases add-to-cart rate"},
])

sessions.to_parquet(OUT / "sessions.parquet", index=False)
orders.to_parquet(OUT / "orders.parquet", index=False)
order_items.to_parquet(OUT / "order_items.parquet", index=False)
users.to_parquet(OUT / "users.parquet", index=False)
products.to_parquet(OUT / "products.parquet", index=False)
products.to_csv(OUT / "products.csv", index=False)
price_history.to_csv(OUT / "price_history.csv", index=False)
promotions.to_csv(OUT / "promotions.csv", index=False)
events.to_csv(OUT / "events_calendar.csv", index=False)
ab_tests.to_csv(OUT / "ab_tests.csv", index=False)
(OUT / "ground_truth_notes.txt").write_text(
    f"seed={SEED} scale={SCALE}\ndefective_product_id={DEFECT_PID + 1}\n"
    f"sessions={N} users={NU} orders={NO} order_items={NI}\n")

log("written:")
for f in sorted(OUT.iterdir()):
    print(f"   {f.name:28s} {f.stat().st_size / 1e6:9.1f} MB")
print(f"rows -> sessions {N:,} | users {NU:,} | orders {NO:,} | order_items {NI:,} | products {NP_:,}")
print(f"revenue ${orders.revenue.sum():,.0f}  AOV ${orders.revenue.mean():.2f}  conv {purchased.mean():.4f}")
