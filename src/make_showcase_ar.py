"""Arabic edition of the ten Shop Stats showcase images.

Same data and layout as src/make_showcase.py, with every visible string translated into Arabic and the
page furniture (titles, subtitles, notes, cards) laid out right to left. Writes 2400x1350 PNGs to
showcase_ar/ under the same file names; the English images in showcase/ are not touched.

Arabic text is shaped with arabic_reshaper and reordered with python-bidi before matplotlib draws it.

    python src/make_showcase_ar.py          # all ten
    python src/make_showcase_ar.py 3 7      # only images 3 and 7
"""
import re
import sys
from pathlib import Path

import arabic_reshaper
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi.algorithm import get_display
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shopstats import (BAD, BLUE, AQUA, DATA, GOOD, GREY, INK, INK2, ORANGE, OUT, ROOT, START, VIOLET, YELLOW,
                       MAGENTA, GREEN, RED, set_style)

SHOW = ROOT / "showcase_ar"
SHOW.mkdir(exist_ok=True)
BG, CARD, LINE = "#fbfaf7", "#ffffff", "#e3e1da"
PALE_BLUE, PALE_ORANGE = "#dbe8f8", "#fbe3d8"

set_style()
mpl.rcParams.update({
    "font.family": ["Segoe UI", "Tahoma", "DejaVu Sans"], "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "axes.titlesize": 15, "axes.titlepad": 12, "axes.labelsize": 12,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5, "lines.linewidth": 2.2,
    "axes.titlelocation": "right",
})

# ------------------------------------------------------------- Arabic text ----
_ARABIC = re.compile(r"[؀-ۿ]")
# Left-to-right runs inside Arabic text (numbers with sign, %, $, k; Latin words; formulas such as p < 1e-40).
# Each run is kept as one left-to-right unit so the bidi algorithm keeps "+2.1%" or "-$19k" in reading order.
_LTR = re.compile(r"[+\-−±$|↓×]*[A-Za-z0-9À-ɏ]"
                  r"[A-Za-z0-9À-ɏ .,:%$+\-−±=<>|/&'’_×]*")
# (the run is put in an LRI ... PDI isolate on matplotlib >= 3.11, or LRE ... PDF for python-bidi, see A())


# Matplotlib >= 3.11 lays text out with libraqm, which already shapes Arabic and applies the bidi algorithm.
# Running get_display() on top of that would reverse every line a second time, so on those versions the
# text is passed in logical order: each line gets a right-to-left mark (RTL paragraph direction) and every
# left-to-right run is put in a directional isolate (LRI ... PDI). Older matplotlib has no shaping, so
# there the text is reshaped and reordered with arabic_reshaper + python-bidi (LRE ... PDF around runs).
NATIVE_BIDI = tuple(int(p) for p in mpl.__version__.split(".")[:2]) >= (3, 11)


def _wrapper(open_, close):
    def wrap(m):
        s = m.group(0)
        core = s.rstrip(" .,:")
        return open_ + core + close + s[len(core):]
    return wrap


_WRAP_ISO, _WRAP_EMB = _wrapper("⁦", "⁩"), _wrapper("‪", "‬")


def A(s):
    """Prepare an Arabic string for matplotlib (line by line). Pure Latin text is returned as is."""
    if not _ARABIC.search(s):
        return s
    out = []
    for line in s.split("\n"):
        if _ARABIC.search(line):
            if NATIVE_BIDI:
                line = "‏" + _LTR.sub(_WRAP_ISO, line)
            else:
                line = get_display(arabic_reshaper.reshape(_LTR.sub(_WRAP_EMB, line)), base_dir="R")
        out.append(line)
    return "\n".join(out)


FOOT = A("Shop Stats  ·  مجموعة بيانات اصطناعية لتجارة إلكترونية  ·  1 نوفمبر 2024 – 31 أكتوبر 2026")

MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر",
          "ديسمبر"]

STAGE = {"Visit": "زيارة", "Product view": "مشاهدة منتج", "Add to cart": "إضافة للسلة", "Checkout": "إتمام الطلب",
         "Payment": "الدفع", "Purchase": "الشراء", "Delivered": "التسليم", "Returned": "الإرجاع",
         "Reviewed": "التقييم"}
DEVICE = {"desktop": "حاسوب مكتبي", "mobile": "جوال", "tablet": "جهاز لوحي"}
CHANNEL = {"Organic Search": "البحث العضوي", "Paid Search": "البحث المدفوع", "Direct": "مباشر",
           "Social": "التواصل الاجتماعي", "Email": "البريد الإلكتروني", "Affiliate": "التسويق بالعمولة"}
COUNTRY = {"US": "الولايات المتحدة", "UK": "المملكة المتحدة", "DE": "ألمانيا", "CA": "كندا", "FR": "فرنسا",
           "AU": "أستراليا", "NL": "هولندا"}
CATEGORY = {"Electronics": "الإلكترونيات", "Fashion": "الأزياء", "Home & Garden": "المنزل والحديقة",
            "Beauty": "التجميل", "Sports & Outdoors": "الرياضة والهواء الطلق", "Toys & Games": "الألعاب",
            "Books & Media": "الكتب والوسائط", "Pet Supplies": "مستلزمات الحيوانات الأليفة"}
SEGMENT = {**DEVICE, **CHANNEL, **COUNTRY, "F": "أنثى", "M": "ذكر", "Unspecified": "غير محدد",
           "Loyalty member": "عضو", "Non-member": "غير عضو", "member": "عضو", "non-member": "غير عضو",
           "new": "جديد", "returning": "عائد"}
GROUP = {"traffic": "الزيارات", "funnel": "القمع", "category": "الفئات", "sales": "المبيعات", "payment": "الدفع",
         "post-purchase": "ما بعد الشراء"}


# ---------------------------------------------------------------- helpers ----
def r(name):
    return pd.read_csv(OUT / name)


def dates(day):
    return START + pd.to_timedelta(np.asarray(day, dtype="int64"), unit="D")


def frame(n, title, subtitle):
    fig = plt.figure(figsize=(16, 9))
    fig.add_artist(Rectangle((0.915, 0.955), 0.045, 0.007, color=BLUE, transform=fig.transFigure))
    fig.text(0.96, 0.905, A(title), fontsize=29, weight="bold", color=INK, va="center", ha="right")
    fig.text(0.96, 0.855, A(subtitle), fontsize=15.5, color=INK2, va="center", ha="right")
    fig.add_artist(Rectangle((0.04, 0.055), 0.92, 0.0012, color=LINE, transform=fig.transFigure))
    fig.text(0.96, 0.032, FOOT, fontsize=10.5, color=GREY, va="center", ha="right")
    fig.text(0.04, 0.032, f"{n:02d} / 10", fontsize=10.5, color=GREY, va="center", ha="left")
    return fig


def finish(fig, n, name):
    path = SHOW / f"{n:02d}_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def canvas(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    return ax


def card(ax, x, y, w, h, fc=CARD, ec=LINE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=1.2,
                                transform=ax.transAxes, clip_on=False, mutation_aspect=ax.bbox.width / ax.bbox.height))


def tile(ax, x, y, w, h, value, label, color=INK, vsize=30, lsize=12.5):
    card(ax, x, y, w, h)
    ax.text(x + w / 2, y + h * 0.60, A(value), fontsize=vsize, weight="bold", color=color, ha="center", va="center")
    ax.text(x + w / 2, y + h * 0.22, A(label), fontsize=lsize, color=INK2, ha="center", va="center")


def big(n):
    """Arabic form of big(): 15.5 مليون / 318 ألف."""
    if n >= 1e6:
        return (f"{n / 1e6:.1f}" if n >= 1e7 else f"{n / 1e6:.2f}") + " مليون"
    if n >= 1e3:
        return f"{n / 1e3:.0f} ألف"
    return f"{n:.0f}"


def pct(x, d=1, sign=True):
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def pct_axis(ax, axis="x", d=0):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{d}f}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def note(ax, text, x=1.0, y=1.02, **kw):
    kw.setdefault("ha", "right")
    kw.setdefault("fontsize", 11.5)
    ax.text(x, y, A(text), transform=ax.transAxes, color=INK2, va="bottom", **kw)


def month_axis(ax, interval=1):
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: A(f"{MONTHS[mpl.dates.num2date(v).month - 1]} {mpl.dates.num2date(v).year % 100:02d}")))


def money_k(v):
    return A(f"-${-v:.0f} ألف" if v < 0 else f"${v:.0f} ألف")


# ------------------------------------------------------------ 01 overview ----
def img01():
    fig = frame(1, "عامان من عمر متجر إلكتروني، من البداية إلى النهاية",
                "مجموعة بيانات اصطناعية مولّدة ببذرة ثابتة، زُرعت فيها أسباب معروفة، ثم حُلّلت كأن تلك الأسباب مجهولة.")
    fo = r("funnel_overall.csv")
    o = pd.read_parquet(DATA / "orders.parquet", columns=["status", "refund_amount", "review_rating"])
    ax = canvas(fig, [0.04, 0.09, 0.92, 0.72])

    kp = [("23.9", "مليون جلسة"), ("3.32", "مليون مستخدم"), ("1.18", "مليون طلب"), ("1.83", "مليون سطر طلب"),
          ("3,000", "منتج"), ("$134.6", "مليون دولار إيرادات")]
    w, gap = 0.155, 0.014
    for i, (v, l) in enumerate(kp):
        tile(ax, 1 - w - i * (w + gap), 0.80, w, 0.20, v, l, color=BLUE if i in (0, 5) else INK)

    # life-cycle strip, flowing right to left
    ax.text(1, 0.735, A("دورة حياة الشراء"), fontsize=12.5, weight="bold", color=GREY, ha="right")
    delivered = int((o["status"] != "cancelled").sum())
    stages = list(zip(fo["stage"], fo["sessions"])) + [
        ("Delivered", delivered), ("Returned", int((o["refund_amount"] > 0).sum())),
        ("Reviewed", int(o["review_rating"].notna().sum()))]
    n = len(stages)
    sw, sg = 0.094, (1 - 9 * 0.094) / 8
    for i, (s, c) in enumerate(stages):
        x = 1 - sw - i * (sw + sg)
        post = i >= 6
        card(ax, x, 0.54, sw, 0.17, fc=PALE_ORANGE if post else PALE_BLUE, ec="none")
        ax.text(x + sw / 2, 0.655, A(STAGE[s]), fontsize=12.5, weight="bold", color=INK, ha="center", va="center")
        ax.text(x + sw / 2, 0.59, A(big(c)), fontsize=16, color=INK, ha="center", va="center")
        if i < n - 1:
            ax.annotate("", (x - sg * 0.92, 0.625), (x - sg * 0.08, 0.625),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))
    ax.text(1, 0.50, A("قمع الجلسة"), fontsize=12, color=BLUE, va="top", ha="right")
    ax.text(1 - 6 * (sw + sg), 0.50, A("ما بعد الشراء"), fontsize=12, color=ORANGE, va="top", ha="right")

    # three text cards, first one on the right
    cols = [
        ("مزروع في البيانات", [
            "12 عرضًا ترويجيًا من 5 أنواع",
            "11 حدثًا داخليًا: انقطاع، أعطال،\nرفع أسعار، إطلاقات، خفض ميزانية",
            "13 حدثًا خارجيًا: منافس، طفرة انتشار،\nروبوتات، إضراب، موجة حر",
            "اختباران A/B بتوزيع عشوائي للمستخدمين،\nأحدهما بأثر حقيقي والآخر بلا أثر فعلًا"]),
        ("التحليلات المنفذة", [
            "انحدار الاتجاه مع متغيرات للأحداث", "القمع تحت 9 ظروف، مع مجموعات ضابطة",
            "إحصاءات الشرائح مع أحجام الأثر", "رفع العروض والربح وسحب الطلب المسبق",
            "اختبارات A/B: SRM و CUPED وعدم التجانس", "كشف الشذوذ دون الاطلاع على التقويم"]),
        ("المخرجات", [
            "4 دفاتر Jupyter منفّذة", "مصنف Excel: 14 ورقة،\n20.7 ألف صيغة حية، 0 أخطاء",
            "تقرير مكتوب مع التحفظات", "مولّد قابل لإعادة الإنتاج (بذرة واحدة)"]),
    ]
    cw, cg = 0.322, 0.017
    for i, (head, lines) in enumerate(cols):
        x = 1 - cw - i * (cw + cg)
        card(ax, x, 0.0, cw, 0.44)
        ax.text(x + cw - 0.018, 0.40, A(head), fontsize=12.5, weight="bold", color=GREY, va="center", ha="right")
        yy = 0.345
        for ln in lines:
            ax.add_patch(Rectangle((x + cw - 0.024, yy - 0.006), 0.006, 0.012, color=BLUE, transform=ax.transAxes))
            ax.text(x + cw - 0.034, yy + 0.014, A(ln), fontsize=12.8, color=INK, va="top", ha="right",
                    linespacing=1.3)
            yy -= 0.052 * (1 + ln.count("\n") * 0.86)
    finish(fig, 1, "project_at_a_glance")


# -------------------------------------------------------------- 02 trends ----
def img02():
    d = r("daily_metrics.csv")
    d["date"] = pd.to_datetime(d["date"])
    y1, y2 = d.iloc[:365], d.iloc[-365:]
    yoy = y2["revenue"].sum() / y1["revenue"].sum() - 1
    fig = frame(2, f"نمت الإيرادات {yoy * 100:.0f}% على أساس سنوي، ولكل قفزة اسم",
                "الإيرادات اليومية (متوسط 7 أيام). الأشرطة المظللة عروض ترويجية؛ والخطوط المتقطعة تغييرات دائمة.")
    ax = fig.add_axes([0.075, 0.42, 0.885, 0.335])
    roll = d["revenue"].rolling(7, center=True, min_periods=4).mean() / 1e3
    ax.fill_between(d["date"], roll, color=PALE_BLUE, lw=0)
    ax.plot(d["date"], roll, color=BLUE, lw=2)
    promos = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    for _, p in promos.iterrows():
        ax.axvspan(p.start_date, p.end_date + pd.Timedelta(days=1), color=YELLOW, alpha=0.28, lw=0)
    top = roll.max() * 1.08
    ax.set_ylim(0, top)
    peaks = [("2024-11-30", "الجمعة السوداء 2024", False), ("2025-11-29", "الجمعة السوداء 2025", True),
             ("2025-03-23", "تخفيضات\nالربيع", True), ("2026-03-22", "تخفيضات\nالربيع", True)]
    for dt, lab, left in peaks:
        t = pd.Timestamp(dt)
        v = roll[d["date"].between(t - pd.Timedelta(days=4), t + pd.Timedelta(days=4))].max()
        ax.text(t + pd.Timedelta(days=-12 if left else 9), v, A(lab), fontsize=11.5, color=INK, va="top",
                weight="bold", ha="right" if left else "left")
    marks = [("2025-04-01", "إطلاق برنامج\nالولاء"), ("2025-07-01", "إطلاق مستلزمات\nالحيوانات الأليفة"),
             ("2025-09-01", "الإلكترونيات\n+8% في السعر"), ("2026-01-15", "خفض البحث المدفوع،\nتباطؤ اقتصادي"),
             ("2026-05-04", "الدفع بصفحة واحدة\nبنسبة 100%")]
    for dt, lab in marks:
        t = pd.Timestamp(dt)
        ax.axvline(t, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(t, top * 1.02, A(lab), fontsize=11, color=INK2, va="bottom", ha="left", linespacing=1.15)
    ax.set_xlim(d["date"].min(), d["date"].max())
    month_axis(ax, 2)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: money_k(v)))
    ax.set_ylabel(A("الإيرادات اليومية"))
    ax.grid(axis="x", visible=False)

    # monthly by category
    m = r("monthly_revenue_by_category.csv").fillna(0)
    ax2 = fig.add_axes([0.075, 0.105, 0.52, 0.235])
    cats = list(m.columns[1:])
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    x = np.arange(len(m))
    bottom = np.zeros(len(m))
    for c, col in zip(cats, cols):
        ax2.bar(x, m[c] / 1e6, bottom=bottom, color=col, width=0.78, label=A(CATEGORY[c]), edgecolor=BG, lw=0.8)
        bottom += m[c].values / 1e6
    ax2.set_xticks(x[::3])
    ax2.set_xticklabels([A(f"{MONTHS[t.month - 1]} {t.year % 100:02d}") for t in pd.to_datetime(m["month"])[::3]])
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: A(f"${v:.0f} مليون")))
    ax2.set_title(A("الإيرادات الشهرية حسب الفئة"))
    ax2.grid(axis="x", visible=False)
    ax2.legend(ncol=1, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=10.5, handlelength=1,
               labelspacing=0.3)

    # year-on-year
    ax3 = fig.add_axes([0.81, 0.105, 0.15, 0.235])
    items = [("الجلسات", y2["sessions"].sum() / y1["sessions"].sum() - 1),
             ("الطلبات", y2["orders"].sum() / y1["orders"].sum() - 1),
             ("الإيرادات", yoy),
             ("قيمة الطلب", (y2["revenue"].sum() / y2["orders"].sum()) / (y1["revenue"].sum() / y1["orders"].sum()) - 1),
             ("التحويل", (y2["purchased"].sum() / y2["sessions"].sum()) / (y1["purchased"].sum() / y1["sessions"].sum()) - 1)]
    yy = np.arange(len(items))[::-1]
    vals = [v for _, v in items]
    ax3.barh(yy, vals, color=[BLUE if v >= 0 else ORANGE for v in vals], height=0.6)
    for y, v in zip(yy, vals):
        ax3.text(v + (0.006 if v >= 0 else -0.006), y, pct(v, 0), va="center", ha="left" if v >= 0 else "right",
                 fontsize=12, color=INK, weight="bold")
    ax3.set_yticks(yy)
    ax3.set_yticklabels([A(k) for k, _ in items])
    ax3.axvline(0, color=INK2, lw=1)
    ax3.set_xlim(min(vals) - 0.07, max(vals) + 0.08)
    ax3.set_xticks([])
    ax3.grid(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.set_title(A("السنة 2 مقابل السنة 1"))
    finish(fig, 2, "revenue_trend_and_drivers")


# -------------------------------------------------------------- 03 funnel ----
def img03():
    fo = r("funnel_overall.csv")
    dev = r("funnel_by_device.csv").set_index("segment")
    fig = frame(3, "أربعة من كل خمسة مشاهدين للمنتج لا يضيفون إلى السلة، والجوال يتسرّب في كل خطوة",
                "23.8 مليون جلسة بشرية. يجلب الجوال 55% من الزيارات لكنه يحوّل بنسبة 3.5%، مقابل 7.2% على الحاسوب المكتبي.")
    ax = fig.add_axes([0.125, 0.11, 0.39, 0.66])
    n = len(fo)
    y = np.arange(n)[::-1]
    share = fo["share of visits"].values
    worst = int(np.nanargmax(fo["drop-off at step"].values))
    ax.barh(y, share, color=[ORANGE if i == worst else BLUE for i in range(n)], height=0.62)
    for i in range(n):
        ax.text(share[i] + 0.012, y[i], A(f"{big(fo['sessions'][i])}   {share[i] * 100:.1f}% من الزيارات"),
                va="center", fontsize=12.5, color=INK)
        if i > 0:
            ax.text(0.004, y[i] + 0.5, A(f"↓ يستمر {fo['step conversion'][i] * 100:.0f}%"), ha="left",
                    va="center", fontsize=11.5, color=ORANGE if i == worst else INK2,
                    weight="bold" if i == worst else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([A(STAGE[s]) for s in fo["stage"]], fontsize=13, weight="bold")
    ax.tick_params(axis="y", pad=8, length=0)
    ax.set_xlim(0, 1.62)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False), ax.spines["left"].set_visible(False)
    ax.set_title(A("القمع الإجمالي"))

    steps = ["visit→view", "view→cart", "cart→checkout", "checkout→payment", "payment→purchase"]
    step_ar = ["زيارة ←\nمشاهدة", "مشاهدة ←\nسلة", "سلة ←\nإتمام الطلب", "إتمام الطلب ←\nدفع", "دفع ←\nشراء"]
    ax2 = fig.add_axes([0.58, 0.50, 0.38, 0.27])
    x = np.arange(len(steps))
    for k, (dv, col) in enumerate(zip(["desktop", "tablet", "mobile"], [BLUE, AQUA, ORANGE])):
        v = dev.loc[dv, steps].values.astype(float)
        ax2.bar(x + (k - 1) * 0.27, v, width=0.25, color=col,
                label=A(f"{DEVICE[dv]}  ({dev.loc[dv, 'overall'] * 100:.1f}% إجمالًا)"))
        for xi, vi in zip(x, v):
            ax2.text(xi + (k - 1) * 0.27, vi + 0.012, f"{vi * 100:.0f}", ha="center", fontsize=10, color=INK2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([A(s) for s in step_ar], fontsize=11)
    ax2.set_ylim(0, 1.18)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    pct_axis(ax2, "y")
    ax2.grid(axis="x", visible=False)
    ax2.set_title(A("تحويل الخطوات حسب الجهاز"))
    ax2.legend(ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=10.5, handlelength=1,
               columnspacing=1.2)

    g = pd.read_csv(OUT / "funnel_glm_odds_ratios.csv", index_col=0)
    names = {"device: mobile": "جوال  (مقابل المكتبي)", "device: tablet": "جهاز لوحي  (مقابل المكتبي)",
             "channel: Social": "التواصل الاجتماعي  (مقابل البحث العضوي)", "new user": "مستخدم جديد  (مقابل العائد)",
             "channel: Direct": "مباشر  (مقابل البحث العضوي)", "channel: Email": "البريد الإلكتروني  (مقابل البحث العضوي)",
             "loyalty member": "عضو ولاء  (مقابل غير العضو)"}
    g = g.loc[list(names)].sort_values("odds ratio")
    ax3 = fig.add_axes([0.765, 0.125, 0.195, 0.245])
    yy = np.arange(len(g))
    ax3.hlines(yy, g["lo"], g["hi"], color=INK2, lw=1.5)
    ax3.scatter(g["odds ratio"], yy, s=90, color=[ORANGE if v < 1 else BLUE for v in g["odds ratio"]], zorder=3)
    for yi, v in zip(yy, g["odds ratio"]):
        ax3.text(v, yi + 0.33, f"{v:.2f}", ha="center", fontsize=10.5, color=INK)
    ax3.axvline(1, color=INK2, lw=1)
    ax3.set_yticks(yy)
    ax3.set_yticklabels([A(names[i]) for i in g.index])
    ax3.set_xlim(0.35, 1.75)
    ax3.set_ylim(-0.6, len(g) - 0.3)
    ax3.grid(axis="y", visible=False)
    ax3.set_title(A("أرجحية الشراء، مع ثبات العوامل الأخرى"))
    ax3.set_xlabel(A("نسبة الأرجحية (نموذج لوجستي)"))
    finish(fig, 3, "funnel_overall_and_by_device")


# --------------------------------------------------- 04 funnel conditions ----
def img04():
    c = r("funnel_conditions.csv")
    fig = frame(4, "كل صدمة تصيب خطوة واحدة من القمع في مجموعة واحدة، وتبقى المجموعة الضابطة ثابتة",
                "التغير النسبي في تحويل الخطوة مقارنة بالأسابيع السابقة، مع فترات ثقة 95%. الرمادي هو المجموعة الضابطة غير المتأثرة.")
    rows = [  # (label, condition prefix, affected group, step, control group)
        ("الجمعة السوداء 2025\nمشاهدة ← سلة، كل المتسوقين", "Black Friday 2025", "all", "view→cart", None),
        ("موجة الحر، يونيو 2026\nمشاهدة ← سلة، المنزل والحديقة", "Heatwave", "Home & Garden", "view→cart", "other categories (control)"),
        ("طرح الدفع بصفحة واحدة\nإتمام الطلب ← دفع، الجوال", "One-page checkout", "mobile", "checkout→payment", "desktop (control)"),
        ("نهاية الأسبوع مقابل أيام العمل\nمشاهدة ← سلة، كل المتسوقين", "Weekend", "all", "view→cart", None),
        ("الإلكترونيات +8% في السعر\nمشاهدة ← سلة، الإلكترونيات", "Electronics +8%", "Electronics", "view→cart", "other categories (control)"),
        ("حملة منافس، مايو 2025\nمشاهدة ← سلة، الإلكترونيات", "Competitor", "Electronics", "view→cart", "other categories (control)"),
        ("عطل إتمام الطلب على الجوال، يونيو 2025\nدفع ← شراء، الجوال", "Mobile checkout bug", "mobile", "payment→purchase", "desktop (control)"),
        ("طفرة انتشار اجتماعي، أغسطس 2025\nمشاهدة ← سلة، زوار التواصل الاجتماعي", "Viral social", "Social", "view→cart", None),
    ]
    ax = fig.add_axes([0.245, 0.12, 0.39, 0.66])
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
            ax.text(b.rel_change + side * 0.022, y - 0.22, A(f"الضابطة {pct(b.rel_change)}"), va="center",
                    ha="right" if side < 0 else "left", fontsize=11, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([A(x[0]) for x in rows], fontsize=12, linespacing=1.3)
    ax.set_xlim(-0.68, 0.48)
    ax.set_ylim(-0.6, n - 0.4)
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(A("التغير النسبي في تحويل الخطوة"))
    ax.text(0.03, 4.13, A("دال (p < 0.001)، لكنه −0.5% فقط"), fontsize=11, color=INK2,
            ha="left", va="center", style="italic")

    t = r("segments_price_tier_macro.csv")
    ax2 = fig.add_axes([0.765, 0.47, 0.195, 0.285])
    yy = np.arange(len(t))[::-1]
    ax2.barh(yy, t["rel_change"], color=[BLUE if v > 0 else ORANGE for v in t["rel_change"]], height=0.6)
    ax2.hlines(yy, t["lo"], t["hi"], color=INK, lw=1.5)
    for y, v in zip(yy, t["rel_change"]):
        ax2.text(v + (0.012 if v > 0 else -0.012), y, pct(v), va="center", ha="left" if v > 0 else "right",
                 fontsize=12, weight="bold")
    ax2.axvline(0, color=INK2, lw=1)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([A(s) for s in ["الربع الأرخص\n(اقتصادي)", "الربع 2", "الربع 3", "الربع الأغلى\n(ممتاز)"]])
    ax2.set_xlim(-0.21, 0.17)
    ax2.set_xticks([])
    ax2.grid(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.set_title(A("التباطؤ الاقتصادي: المتسوقون يتجهون إلى الأرخص"), pad=28)
    note(ax2, "معدل الإضافة إلى السلة حسب شريحة السعر، يناير–أبريل 2026 مقابل 2025", y=1.0)

    m = r("mix_shift_paid_cut.csv").set_index("component")["value"]
    ax3 = fig.add_axes([0.765, 0.12, 0.195, 0.20])
    b0, mix, rate, b1 = m["blended before"], m["mix effect"], m["rate effect"], m["blended during"]
    ax3.bar([0, 3], [b0, b1], color=GREY, width=0.62)
    ax3.bar(1, mix, bottom=b0, color=BLUE, width=0.62)
    ax3.bar(2, rate, bottom=b0 + mix, color=ORANGE, width=0.62)
    for x, v, lab in [(0, b0, f"{b0 * 100:.2f}%"), (1, b0 + mix, A(f"{mix * 1e4:+.0f} ن.أ")),
                      (2, b0 + mix, A(f"{rate * 1e4:+.0f} ن.أ")), (3, b1, f"{b1 * 100:.2f}%")]:
        ax3.text(x, v + 0.00012, lab, ha="center", fontsize=11.5, weight="bold")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels([A(s) for s in ["قبل", "مزيج\nالزيارات", "المعدلات", "أثناء"]], fontsize=11)
    ax3.set_ylim(0.046, 0.0508)
    ax3.set_yticks([])
    ax3.grid(False)
    ax3.spines["left"].set_visible(False)
    ax3.set_title(A("خفض البحث المدفوع: المزيج مقابل المعدل"), pad=28)
    note(ax3, "التحويل المجمّع مفكَّكًا (محور مقطوع؛ ن.أ = نقطة أساس)", y=1.0, fontsize=10.5)
    finish(fig, 4, "funnel_under_conditions")


# ------------------------------------------------------------ 05 segments ----
def img05():
    s = r("segments_users.csv")
    t = r("segments_users_tests.csv").set_index("dimension")
    cat = r("segments_category.csv")
    fig = frame(5, "كل تقسيم للشرائح «دال إحصائيًا»؛ واثنان فقط كبيران بما يكفي ليُعتدّ بهما",
                "مع 3.3 مليون مستخدم تكون كل قيم p ضئيلة، لذا رُتّبت الشرائح حسب حجم الأثر بدلًا منها.")
    nice = {"preferred_device": "الجهاز المفضل", "member": "الولاء", "age_group": "الفئة العمرية",
            "acquisition_channel": "قناة الاستقطاب", "country": "الدولة", "gender": "الجنس"}
    order = t["cramers_v"].sort_values(ascending=False).index.tolist()
    overall = (s[s.dimension == "gender"]["buyer_rate"] * s[s.dimension == "gender"]["users"]).sum() / \
        s[s.dimension == "gender"]["users"].sum()

    ax = fig.add_axes([0.155, 0.10, 0.27, 0.68])
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
            labels.append(A(SEGMENT.get(row.segment, row.segment))), ypos.append(y)
            y -= 1
        y -= 0.35
    for hy, h in heads:
        ax.text(-0.03, hy, A(h), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12,
                weight="bold", color=INK)
    ax.axvline(overall, color=INK2, lw=1)
    ax.text(overall + 0.004, 0.75, A(f"كل المستخدمين {overall * 100:.1f}%"), ha="left", fontsize=11, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_ylim(y + 0.6, 1.6)
    ax.set_xlim(0.145, 0.37)
    ax.set_xticks([0.15, 0.20, 0.25, 0.30, 0.35])
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title(A("نسبة المستخدمين الذين اشتروا"))

    ax2 = fig.add_axes([0.58, 0.50, 0.38, 0.28])
    v = t.loc[order, "cramers_v"][::-1]
    yy = np.arange(len(v))
    ax2.barh(yy, v, color=[BLUE if x >= 0.1 else GREY for x in v], height=0.62)
    for yi, x in zip(yy, v):
        ax2.text(x + 0.002, yi, f"{x:.3f}", va="center", fontsize=12, weight="bold" if x >= 0.1 else "normal")
    ax2.axvline(0.1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax2.text(0.101, len(v) - 0.42, A("خط «الأثر الصغير» المتعارف عليه"), fontsize=10.5, color=INK2, va="bottom")
    ax2.set_yticks(yy)
    ax2.set_yticklabels([A(nice[i]) for i in v.index])
    ax2.set_xlim(0, 0.15)
    ax2.set_ylim(-0.6, len(v) + 0.1)
    ax2.grid(axis="y", visible=False)
    ax2.set_title(A("حجم الأثر على الشراء (Cramér’s V)"))
    note(ax2, "الأبعاد الستة كلها p < 1e-40", x=0.0, y=1.045, ha="left")

    ax3 = fig.add_axes([0.58, 0.135, 0.38, 0.255])
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    ax3.scatter(cat["margin_pct"], cat["return_rate"], s=cat["revenue"] / 2.2e4, color=cols, alpha=0.9,
                edgecolor=BG, lw=2, zorder=3)
    off = {"Electronics": (0.028, 0.012), "Fashion": (0.018, 0.0), "Home & Garden": (0.018, 0.0),
           "Beauty": (0.010, 0.008), "Sports & Outdoors": (-0.012, 0.013), "Toys & Games": (0.010, 0.004),
           "Books & Media": (-0.008, 0.0), "Pet Supplies": (0.007, -0.006)}
    for _, row in cat.iterrows():
        dx, dy = off[row.category]
        ax3.text(row.margin_pct + dx, row.return_rate + dy, A(CATEGORY[row.category]), fontsize=11, va="center",
                 ha="right" if dx < 0 else "left", color=INK)
    ax3.set_xlim(0.2, 0.68)
    ax3.set_ylim(0, 0.235)
    pct_axis(ax3), pct_axis(ax3, "y")
    ax3.set_xlabel(A("الهامش الإجمالي"))
    ax3.set_ylabel(A("معدل الإرجاع"))
    ax3.set_title(A("المنتجات: الهامش مقابل المرتجعات (الفقاعة = الإيرادات)"))
    finish(fig, 5, "segments_effect_sizes")


# ---------------------------------------------------------- 06 promotions ----
def img06():
    p = r("promo_results.csv")
    bf = r("promo_black_friday_net.csv")
    fig = frame(6, "الجمعة السوداء تضاعف الإيرادات ثلاث مرات، لكن تخفيضات الربيع تحقق ربحًا أكبر",
                "الرفع مقارنة بخط أساس انحداري مع فترات ثقة 95%. يمينًا: ربح الجمعة السوداء بعد خصم الطلبات المسحوبة من الأسابيع المحيطة.")
    short = {"P01": "الجمعة السوداء 2024", "P02": "تخفيضات الربيع 2025", "P03": "تصفية الصيف 2025",
             "P04": "العودة إلى المدارس 2025", "P05": "تخفيضات الأعضاء الخاطفة 2025", "P06": "شحن مجاني أكتوبر 2025",
             "P07": "الجمعة السوداء 2025", "P08": "BOGO التجميل 2026", "P09": "تخفيضات الربيع 2026",
             "P10": "شحن مجاني مايو 2026", "P11": "تصفية الصيف 2026", "P12": "العودة إلى المدارس 2026"}
    ptype = {"sitewide % off": "خصم على كامل المتجر", "category % off": "خصم على فئات",
             "member code % off": "خصم برمز الأعضاء", "free shipping": "شحن مجاني",
             "buy one get one": "اشترِ واحدًا واحصل على آخر"}
    n = len(p)
    y = np.arange(n)[::-1]
    ax = fig.add_axes([0.185, 0.11, 0.19, 0.67])
    ax.barh(y, p["revenue_lift"], color=[BLUE if lo > 0 else (ORANGE if hi < 0 else GREY)
                                         for lo, hi in zip(p.revenue_lo, p.revenue_hi)], height=0.6)
    ax.hlines(y, p["revenue_lo"], p["revenue_hi"], color=INK, lw=1.4)
    for yi, v, hi, lo in zip(y, p.revenue_lift, p.revenue_hi, p.revenue_lo):
        ax.text(max(hi, 0) + 0.04, yi, pct(v, 0), va="center", fontsize=12, weight="bold")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([A(f"{short[i]}\n{ptype[t]}، {int(d * 100)}%" if d else f"{short[i]}\n{ptype[t]}")
                        for i, t, d in zip(p.promo_id, p.type, p.discount)], fontsize=10.8, linespacing=1.25)
    ax.set_xlim(-0.3, 2.45)
    ax.set_xticks([0, 1, 2])
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title(A("الإيرادات في النطاق المروَّج"))

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
    ax2.set_title(A("إجمالي الربح على مستوى المتجر"))
    for lab, col, yy in [("ربح", GOOD, 0.115), ("خسارة", BAD, 0.075), ("ضمن الضوضاء", GREY, 0.035)]:
        ax2.add_patch(Rectangle((0.945, yy - 0.009), 0.035, 0.018, color=col, transform=ax2.transAxes))
        ax2.text(0.925, yy, A(lab), transform=ax2.transAxes, fontsize=10.5, va="center", ha="right", color=INK2)

    for k, yr in enumerate([2024, 2025]):
        b = bf[(bf.metric == "gross profit") & (bf.year == yr)].iloc[0]
        ax3 = fig.add_axes([0.70, 0.50 - k * 0.39, 0.26, 0.255])
        vals = [b.during, b.pre_week, b.post_week]
        cum = np.cumsum([0] + vals)
        for i, v in enumerate(vals):
            ax3.bar(i, v / 1e3, bottom=cum[i] / 1e3, color=GOOD if v > 0 else BAD, width=0.6)
            ax3.text(i, (max(cum[i], cum[i + 1])) / 1e3 + 3, A(f"{v / 1e3:+.0f} ألف"), ha="center", fontsize=12,
                     weight="bold")
            ax3.hlines(cum[i + 1] / 1e3, i + 0.3, i + 0.7, color=INK2, lw=1, ls=(0, (2, 2)))
        ax3.bar(3, b.net / 1e3, color=GOOD if b.net > 0 else BAD, width=0.6, hatch="///", edgecolor=BG, lw=0)
        ax3.text(3, max(b.net, 0) / 1e3 + 3, A(f"{b.net / 1e3:+.0f} ألف"), ha="center", fontsize=13, weight="bold")
        ax3.axhline(0, color=INK2, lw=1)
        ax3.set_xticks(range(4))
        ax3.set_xticklabels([A(s) for s in ["4 أيام\nعرض", "الأسبوع\nالسابق", "الأسبوع\nاللاحق", "الصافي"]],
                            fontsize=11)
        ax3.set_ylim(min(b.net / 1e3, 0) - 12, b.during / 1e3 + 22)
        ax3.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: money_k(v)))
        ax3.grid(axis="x", visible=False)
        ax3.set_title(A(f"الجمعة السوداء {yr}: الربح بعد سحب الطلب المسبق"))
    finish(fig, 6, "promotion_scorecard")


# ----------------------------------------------------------------- 07 AB1 ----
def img07():
    res = r("ab1_results.csv")
    seg = r("ab1_segments.csv")
    cum = r("ab1_daily_cumulative.csv")
    het = r("ab1_heterogeneity.csv").set_index("dimension")
    u = r("ab1_users.csv")
    pr = res.iloc[0]
    fig = frame(7, "اختبار A/B الأول: الدفع بصفحة واحدة يفوز، وبالكامل على الشاشات الصغيرة",
                f"{u['users'].sum():,} مستخدم وُزّعوا عشوائيًا 50/50 لستة أسابيع. المقياس الأساسي: إتمام الطلب ← الشراء.")
    ax0 = canvas(fig, [0.04, 0.63, 0.92, 0.16])
    tiles = [(f"{pr.control * 100:.1f}%", "المجموعة الضابطة", False),
             (f"{pr.treatment * 100:.1f}%", "الدفع بصفحة واحدة", False),
             (pct(pr.rel_lift), f"رفع نسبي (فترة ثقة 95%: من {pct(pr.lift_lo)} إلى {pct(pr.lift_hi)})", True),
             (f"z = {pr.z:.2f}", "p = 0.000002، مُجمَّعة حسب المستخدم", False),
             ("+13,000", "طلب إضافي سنويًا عند الطرح الكامل", False)]
    ws = [0.13, 0.16, 0.27, 0.20, 0.20]
    x = 1
    for (v, l, hl), w in zip(tiles, ws):
        x -= w
        tile(ax0, x, 0, w, 1, v, l, color=BLUE if hl else INK, vsize=27, lsize=11.5)
        x -= 0.01

    rows = [("كل المستخدمين", pr.rel_lift, pr.lift_lo, pr.lift_hi, "***", True)]
    for dim, lab in [("device", "الجهاز"), ("user_type", "نوع المستخدم"), ("channel", "القناة")]:
        rows.append((lab, None, None, None, None, False))
        for _, s in seg[seg.dimension == dim].iterrows():
            rows.append(("   " + SEGMENT[s.segment], s.rel_lift, s.lift_lo, s.lift_hi, s.sig_adj, False))
    ax = fig.add_axes([0.155, 0.105, 0.35, 0.45])
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
    # indented segment names: the leading spaces sit at the reading start (right), pushing the name left
    ax.set_yticklabels([A((" " * 5 if x[0].startswith(" ") else "") + x[0].strip()) for x in rows],
                       fontsize=11.5)
    for lbl, row in zip(ax.get_yticklabels(), rows):
        if row[1] is None or row[5]:
            lbl.set_weight("bold")
    ax.set_xlim(-0.03, 0.111)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
    ax.set_ylim(-0.7, n - 0.3)
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title(A("الرفع حسب الشريحة (الأزرق = دال بعد التصحيح)"))
    ax.text(0.035, n - 2.8, A(f"الجهاز يُحدث فرقًا: Q = {het.loc['device', 'Q']:.0f}، p < 1e-8\nولا يُحدثه أي تقسيم آخر"),
            fontsize=10.5, color=INK2, va="center", style="italic")

    ax2 = fig.add_axes([0.60, 0.13, 0.36, 0.425])
    ax2.plot(cum["day"], cum["z"], color=BLUE, lw=2.4, label=A("إحصاءة الاختبار حتى تاريخه"))
    ob = cum["obf_bound"].clip(upper=7)
    ax2.plot(cum["day"], ob, color=INK, lw=1.6, ls=(0, (5, 3)), label=A("حدّ يسمح بالاطلاع اليومي"))
    ax2.axhline(1.96, color=GREY, lw=1.4, ls=(0, (2, 2)))
    ax2.text(42, 1.82, A("خط 1.96 الساذج"), ha="right", va="top", fontsize=10.5, color=INK2)
    early = cum[(cum.z > 1.96) & (cum.z < cum.obf_bound)]
    ax2.scatter(early["day"], early["z"], s=42, color=ORANGE, zorder=4, label=A("يبدو «دالًا» قبل الأوان"))
    cross = cum[cum.z >= cum.obf_bound]["day"].min()
    ax2.axvline(cross, color=GOOD, lw=1.2)
    ax2.text(cross + 0.5, 0.25, A(f"آمن للحسم\nمن اليوم {int(cross)}"), fontsize=11, color=GOOD, va="bottom")
    ax2.set_ylim(0, 7)
    ax2.set_xlim(1, 42)
    ax2.set_xlabel(A("يوم الاختبار"))
    ax2.set_ylabel("z")
    ax2.set_title(A("لماذا لم يُوقَف الاختبار مبكرًا"))
    ax2.legend(loc="upper right", fontsize=10.5)
    finish(fig, 7, "ab_test_one_page_checkout")


# ------------------------------------------------------- 08 AB2 + loyalty ----
def img08():
    res = r("ab2_results.csv")
    seg = r("ab2_segments.csv")
    lo = r("loyalty_did.csv").set_index("measure")["value"]
    fig = frame(8, "أن تعرف متى تكون الإجابة «لا شيء»، ومتى تكون «أقل مما تبدو»",
                "نتيجة صفرية باختبار عالي القوة لشارة الدليل الاجتماعي، وأثر ولاء يتقلص بمقدار الثلثين بعد استبعاد الانتقاء.")
    ax = fig.add_axes([0.17, 0.11, 0.37, 0.63])
    rows = [("الأساسي: مشاهدة ← إضافة للسلة", res.iloc[0].rel_lift, res.iloc[0].lift_lo, res.iloc[0].lift_hi, True)]
    rows += [("تحويل الجلسة", res.iloc[1].rel_lift, res.iloc[1].lift_lo, res.iloc[1].lift_hi, True)]
    for _, s in seg.iterrows():
        rows.append((SEGMENT[s.segment], s.rel_lift, s.lift_lo, s.lift_hi, False))
    n = len(rows)
    ax.axvspan(-0.014, 0.014, color=PALE_BLUE, lw=0)
    for i, (lab, v, l, h, main) in enumerate(rows):
        y = n - 1 - i - (0.6 if not main else 0)
        ax.hlines(y, l, h, color=BLUE if main else GREY, lw=3 if main else 1.8)
        ax.scatter(v, y, s=120 if main else 45, color=BLUE if main else GREY, zorder=3, marker="D" if main else "o")
        rows[i] = (lab, y, main)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([x[1] for x in rows])
    ax.set_yticklabels([A(x[0]) for x in rows], fontsize=10.5)
    for lbl, row in zip(ax.get_yticklabels(), rows):
        if row[2]:
            lbl.set_weight("bold"), lbl.set_fontsize(12)
    ax.set_xlim(-0.062, 0.062)
    ax.set_ylim(-2.3, n - 0.3)
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(A("الرفع النسبي، المعالجة مقابل الضابطة (فترة ثقة 95%)"))
    ax.set_title(A("اختبار A/B الثاني: شارة الدليل الاجتماعي"), pad=30)
    note(ax, f"الرفع الأساسي {pct(res.iloc[0].rel_lift, 2)}؛ 0 من {len(seg)} شريحة دالة بعد التصحيح", y=1.0)
    ax.text(0.0, -1.75, A("المظلّل: رفع أصغر من أن يكشفه هذا الاختبار (±1.4%)"), fontsize=10.5, color=BLUE,
            ha="center")

    ax2 = fig.add_axes([0.64, 0.125, 0.32, 0.615])
    vals = [lo["naive member vs non-member"], lo["pre-existing gap (selection)"], lo["DiD programme effect"]]
    labs = ["ساذج: الأعضاء\nمقابل غير الأعضاء", "فجوة قائمة\nقبل الانضمام", "أثر البرنامج\n(فرق الفروق)"]
    cols = [GREY, PALE_ORANGE, BLUE]
    ax2.bar(range(3), vals, color=cols, width=0.62, edgecolor=[GREY, ORANGE, BLUE], lw=1.5)
    ax2.errorbar(2, vals[2], yerr=[[vals[2] - lo["DiD lo"]], [lo["DiD hi"] - vals[2]]], color=INK, lw=1.8, capsize=5)
    for i, v in enumerate(vals):
        ax2.text(i, v + (0.03 if i == 2 else 0.012), pct(v, 1 if i == 2 else 0), ha="center", fontsize=22 if i != 1 else 17,
                 weight="bold", color=INK if i != 1 else ORANGE)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([A(s) for s in labs], fontsize=11.5, linespacing=1.3)
    ax2.set_ylim(0, 0.63)
    pct_axis(ax2, "y")
    ax2.grid(axis="x", visible=False)
    ax2.set_title(A("برنامج الولاء: رفع التحويل"), pad=30)
    note(ax2, "المتسوقون المتحمسون هم من ينضمون، فمعظم الفجوة يعود إلى هويتهم", y=1.0)
    finish(fig, 8, "null_result_and_selection_bias")


# ------------------------------------------------------- 09 anomaly map ----
def img09():
    f = r("anomalies_flagged.csv")
    f["date"] = pd.to_datetime(f["date"])
    rec = r("anomaly_calendar_recall.csv")
    det, miss = int((rec.status == "detected").sum()), int((rec.status == "missed").sum())
    nflag, nun = len(f), int((~f.explained).sum())
    fig = frame(9, f"كاشف لم يرَ التقويم قط وجد {det} من {det + miss} حدثًا مزروعًا",
                f"45 مقياسًا يوميًا مقارنة بخط أساس متين لنفس يوم الأسبوع. كل نقطة يوم-مقياس مُعلَّم ({nflag:,} إجمالًا).")
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
    ax = fig.add_axes([0.115, 0.105, 0.655, 0.585])
    for name, y0, y1 in bounds:
        ax.axhspan(y1 - 0.6, y0 + 0.6, color=CARD, lw=0, zorder=0)
        ax.text(-0.012, (y0 + y1) / 2, A(GROUP[name]), transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12.5, weight="bold", color=INK)
    e = f[f.explained]
    size = np.clip(e["z"].abs(), 2, 12) * 5
    ax.scatter(e["date"], e["y"], s=size, c=np.where(e["z"] > 0, BLUE, ORANGE), alpha=0.75, lw=0, zorder=3)
    un = f[~f.explained]
    ax.scatter(un["date"], un["y"], s=34, facecolor="none", edgecolor=INK, lw=1.1, zorder=4)
    ev = [("2025-02-18", "انقطاع الموقع"), ("2025-03-23", "تخفيضات الربيع"), ("2025-05-05", "منافس"),
          ("2025-06-11", "عطل إتمام الطلب"), ("2025-07-15", "التصفية"), ("2025-08-10", "طفرة انتشار"),
          ("2025-09-19", "تخفيضات الأعضاء"), ("2025-10-08", "عطل التتبع"), ("2025-11-04", "موجة روبوتات"),
          ("2025-11-29", "الجمعة السوداء"), ("2025-12-17", "إضراب الناقل"), ("2026-01-15", "خفض البحث المدفوع"),
          ("2026-03-22", "تخفيضات الربيع"), ("2026-05-16", "شحن مجاني"), ("2026-06-17", "انقطاع الدفع"),
          ("2026-07-01", "موجة حر"), ("2026-07-18", "التصفية")]
    ytop = 1.6
    nudge = {"إضراب الناقل": 3, "انقطاع الدفع": -3, "موجة حر": 3, "التصفية": 5}   # days, to keep labels apart
    for i, (dt, lab) in enumerate(ev):
        t = pd.Timestamp(dt)
        ax.axvline(t, color=LINE, lw=1, zorder=1)
        ax.text(t + pd.Timedelta(days=nudge.get(lab, 0) if dt > "2025-12" else 0), ytop + 0.4, A(lab),
                rotation=60, ha="left", va="bottom", fontsize=10.5, color=INK2, rotation_mode="anchor")
    ax.set_ylim(y + 0.2, ytop)
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-11-05"))
    ax.set_yticks([])
    month_axis(ax, 2)
    ax.grid(False)
    ax.spines["left"].set_visible(False)

    ax2 = canvas(fig, [0.795, 0.105, 0.165, 0.69])
    tile(ax2, 0, 0.78, 1, 0.20, f"{det} / {det + miss}", "حدثًا مفاجئًا اكتُشف", color=BLUE, lsize=11.5)
    tile(ax2, 0, 0.555, 1, 0.20, f"{(nflag - nun) / nflag * 100:.0f}%", "من الإشارات تُعزى\nإلى سبب معروف",
         lsize=11.5)
    tile(ax2, 0, 0.33, 1, 0.20, f"{nun}", "إشارة غير مفسَّرة،\nلا شيء منها فوق |z| = 4.3", lsize=11.5)
    card(ax2, 0, 0.0, 1, 0.305)
    ax2.text(0.93, 0.265, A("مفتاح الرموز"), fontsize=11.5, weight="bold", color=GREY, va="center", ha="right")
    for yy, col, lab, kw in [(0.205, BLUE, "أعلى من المتوقع", {}), (0.150, ORANGE, "أقل من المتوقع", {}),
                             (0.095, "none", "بلا سبب معروف", dict(edgecolor=INK, lw=1.2))]:
        ax2.scatter(0.89, yy, s=90, color=col, transform=ax2.transAxes, **kw)
        ax2.text(0.80, yy, A(lab), fontsize=11.5, va="center", ha="right", color=INK)
    ax2.text(0.93, 0.04, A("حجم النقطة = القوة (|z|)"), fontsize=10.5, color=INK2, va="center", ha="right")
    finish(fig, 9, "anomaly_detection_map")


# ------------------------------------------------------ 10 drill-downs ----
def img10():
    fig = frame(10, "ست حالات شذوذ، تُتبَّع كلٌّ منها إلى سببها بالتعمق مستوى واحدًا",
                "التنبيه يقول إن شيئًا ما تحرّك؛ والتعمق يقول من وأين ولماذا. البرتقالي هو الشريحة المتأثرة، والرمادي للمقارنة.")
    cd = pd.read_parquet(OUT / "cube_day_channel_device.parquet")
    cb = pd.read_parquet(OUT / "cube_all_incl_bots.parquet")
    R, W, H = [0.96, 0.64, 0.32], 0.265, 0.235   # right edges, so panel a) sits top right
    B = [0.475, 0.105]

    def panel(i, title, sub):
        ax = fig.add_axes([R[i % 3] - W, B[i // 3], W, H])
        ax.set_title(A(title), fontsize=14, pad=24)
        ax.text(1, 1.03, A(sub), transform=ax.transAxes, fontsize=10.8, color=INK2, va="bottom", ha="right")
        ax.grid(axis="x", visible=False)
        return ax

    def win(df, a, b, col="day"):
        lo, hi = (pd.Timestamp(a) - START).days, (pd.Timestamp(b) - START).days
        return df[df[col].between(lo, hi)]

    def dfmt(ax):
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: A(f"{mpl.dates.num2date(v).day} {MONTHS[mpl.dates.num2date(v).month - 1]}")))

    kfmt = mpl.ticker.FuncFormatter(lambda v, _: A(f"{v:.0f} ألف") if v else "0")

    # a) mobile checkout bug
    ax = panel(0, "عطل إتمام الطلب على الجوال، 10–13 يونيو 2025", "معدل الدفع ← الشراء حسب الجهاز: الجوال ينهار، والمكتبي لا")
    g = win(cd, "2025-05-25", "2025-06-30").groupby(["day", "device"], observed=True)[["entered_payment", "purchased"]].sum()
    g = (g["purchased"] / g["entered_payment"]).unstack()
    ax.plot(dates(g.index), g["desktop"], color=GREY, label=A("حاسوب مكتبي"))
    ax.plot(dates(g.index), g["mobile"], color=ORANGE, label=A("جوال"))
    ax.set_ylim(0.3, 1)
    pct_axis(ax, "y"), dfmt(ax)
    ax.legend(loc="lower left", ncol=2)

    # b) bot wave
    ax = panel(1, "موجة زيارات الروبوتات، 3–6 نوفمبر 2025", "الجلسات المباشرة على المكتبي: الطفرة كلها زيارات بصفحة واحدة وثانيتين")
    g = win(cb[(cb.channel == "Direct") & (cb.device == "desktop")], "2025-10-18", "2025-11-22").groupby("day")[["sessions", "bounce0"]].sum()
    x = dates(g.index)
    ax.fill_between(x, 0, (g.sessions - g.bounce0) / 1e3, color=GREY, lw=0, label=A("متفاعلة"))
    ax.fill_between(x, (g.sessions - g.bounce0) / 1e3, g.sessions / 1e3, color=ORANGE, lw=0, label=A("بلا أي تفاعل"))
    ax.yaxis.set_major_formatter(kfmt)
    dfmt(ax)
    ax.legend(loc="upper left")

    # c) viral spike
    ax = panel(2, "إشارة انتشرت على التواصل الاجتماعي، 9–12 أغسطس 2025", "جلسات التواصل تتضاعف 5 مرات والطلبات ثابتة: التحويل 5.0% ← 3.6%")
    g = win(cb, "2025-07-22", "2025-08-28")
    s = g[g.channel == "Social"].groupby("day")["sessions"].sum()
    o = g.groupby("day")["purchased"].sum()
    ax.plot(dates(s.index), s / 1e3, color=ORANGE, label=A("جلسات التواصل (ألف)"))
    ax.plot(dates(o.index), o / 1e3, color=GREY, label=A("كل الطلبات (ألف)"))
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(kfmt)
    dfmt(ax)
    ax.legend(loc="upper right", fontsize=10.5)

    # d) carrier strike
    ax = panel(3, "إضراب شركة الشحن في ألمانيا، 4–19 ديسمبر 2025", "متوسط أيام التسليم حسب تاريخ الطلب: طلبات ألمانيا وحدها تتأخر")
    od = pd.read_parquet(DATA / "orders.parquet", columns=["order_ts", "country", "delivery_days"])
    od = od[od.order_ts.between("2025-11-10", "2026-01-10")].dropna(subset=["delivery_days"])
    od["d"] = od.order_ts.dt.floor("D")
    od["g"] = np.where(od.country == "DE", "DE", "other")
    g = od.groupby(["d", "g"])["delivery_days"].mean().astype(float).unstack()
    ax.plot(g.index, g["other"], color=GREY, label=A("دول أخرى"))
    ax.plot(g.index, g["DE"], color=ORANGE, label=A("ألمانيا"))
    ax.set_ylim(0, None)
    dfmt(ax)
    ax.legend(loc="upper left")

    # e) defective SKU
    ax = panel(4, "دفعة معيبة من منتج واحد، فبراير 2026", "معدل الإرجاع الشهري: منتج واحد عند 57%، وفئته عند 7%")
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
    ax.plot(xx, catr.reindex(sku.index).values, color=INK2, lw=2, marker="o", ms=4, label=A("كل الإلكترونيات"))
    ax.set_xticks(xx)
    ax.set_xticklabels([A(MONTHS[p.month - 1]) for p in sku.index], fontsize=9.5)
    pct_axis(ax, "y")
    ax.legend(loc="upper left")

    # f) paid search cut
    ax = panel(5, "خفض ميزانية البحث المدفوع، 15 يناير – 28 فبراير 2026", "حصة البحث المدفوع من الجلسات: تحوّل في المستوى، محدَّد باليوم")
    g = win(cd, "2025-11-20", "2026-04-10").groupby(["day", "channel"], observed=True)["sessions"].sum().unstack()
    sh = g["Paid Search"] / g.sum(axis=1)
    ax.plot(dates(sh.index), sh, color=ORANGE)
    for dt in ["2026-01-15", "2026-03-01"]:
        t = pd.Timestamp(dt)
        ax.axvline(t, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(t + pd.Timedelta(days=2), 0.045, A(f"اكتُشف في\n{t.day} {MONTHS[t.month - 1]}"), fontsize=10.5,
                color=INK2, va="bottom")
    ax.set_ylim(0, 0.3)
    pct_axis(ax, "y"), dfmt(ax)
    finish(fig, 10, "anomaly_drilldowns")


if __name__ == "__main__":
    fns = {i: globals()[f"img{i:02d}"] for i in range(1, 11)}
    want = [int(a) for a in sys.argv[1:]] or list(fns)
    for i in want:
        fns[i]()
