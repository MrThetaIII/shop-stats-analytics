"""Build Shop_Stats_Analysis.xlsx from the aggregates exported by the notebooks.

The workbook holds aggregated counts as inputs (blue) and recomputes rates, confidence intervals and
significance tests with live Excel formulas (black). Model outputs that cannot be reproduced in a
spreadsheet (regressions) are imported as values and labelled as such.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L
from openpyxl.worksheet.formula import ArrayFormula

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA = ROOT / "outputs", ROOT / "data"
DEST = ROOT / "Shop_Stats_Analysis.xlsx"

F = "Arial"
f_norm = Font(name=F, size=10)
f_in = Font(name=F, size=10, color="0000FF")
f_bold = Font(name=F, size=10, bold=True)
f_head = Font(name=F, size=10, bold=True, color="FFFFFF")
f_title = Font(name=F, size=14, bold=True)
f_sub = Font(name=F, size=11, bold=True, color="1F4E78")
f_note = Font(name=F, size=9, italic=True, color="595959")
fill_head = PatternFill("solid", fgColor="1F4E78")
fill_key = PatternFill("solid", fgColor="FFFF00")
fill_band = PatternFill("solid", fgColor="F2F2F2")
thin = Side(style="thin", color="BFBFBF")
PCT, PCT2, INT, USD, USD2, NUM2, NUM3, DATE, PV = "0.0%", "0.00%", "#,##0", "$#,##0", "$#,##0.00", "0.00", "0.000", "yyyy-mm-dd", "0.0000"

wb = Workbook()


def sheet(name, title, note=None, widths=None):
    ws = wb.create_sheet(name)
    ws["A1"] = title; ws["A1"].font = f_title
    if note:
        ws["A2"] = note; ws["A2"].font = f_note
    for i, w in enumerate(widths or [], 1):
        ws.column_dimensions[L(i)].width = w
    ws.sheet_view.showGridLines = False
    return ws


def header(ws, r, labels, c0=1):
    for i, lab in enumerate(labels):
        c = ws.cell(r, c0 + i, lab)
        c.font, c.fill = f_head, fill_head
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[r].height = 30


def put(ws, r, c, v, fmt=None, font=None):
    cell = ws.cell(r, c, v)
    is_formula = isinstance(v, str) and v.startswith("=")
    cell.font = font or (f_norm if is_formula or isinstance(v, str) else f_in)
    if fmt:
        cell.number_format = fmt
    return cell


def sub(ws, r, text, c=1):
    ws.cell(r, c, text).font = f_sub


def note(ws, r, text, c=1):
    ws.cell(r, c, text).font = f_note


def style_chart(ch, title, ytitle=None, w=22, h=9):
    ch.title = title
    ch.roundedCorners = False
    ch.width, ch.height = w, h
    if ch.legend is not None:
        ch.legend.overlay = False
        ch.legend.position = "b"
    ch.title.overlay = False
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    if ytitle:
        ch.y_axis.title = ytitle
        ch.y_axis.title.overlay = False
    return ch


SIG = lambda p: f'=IF({p}<0.001,"***",IF({p}<0.01,"**",IF({p}<0.05,"*","ns")))'

# ============================================================================
# README
# ============================================================================
ws = wb.active
ws.title = "README"
ws.sheet_view.showGridLines = False
ws.column_dimensions["A"].width = 24
ws.column_dimensions["B"].width = 118
ws["A1"] = "Shop Stats – store analytics workbook"; ws["A1"].font = f_title
rows = [
    ("What this is", "Excel companion to the Jupyter notebooks. Synthetic online store, 1 Nov 2024 – 31 Oct 2026: 23.9M sessions, 3.3M users, 1.18M orders."),
    ("How it works", "The raw tables are too large for Excel (sessions = 23.9M rows), so the notebooks export aggregated counts to outputs/*.csv. "
                     "This workbook holds those counts and recomputes every rate, confidence interval and significance test with live formulas."),
    ("Colour legend", "Blue text = input data (aggregated counts from the notebooks).  Black text = formula.  Yellow fill = a setting you can change."),
    ("Significance", "*** p<0.001   ** p<0.01   * p<0.05   ns = not significant. With millions of sessions almost everything is 'significant', "
                     "so every test also shows an effect size (relative change, Cohen's h, Cramér's V). Judge importance by the effect size."),
    ("", ""),
    ("Sheet", "Contents"),
    ("Daily", "730 daily rows: traffic, funnel counts, orders, revenue; conversion / AOV / margin formulas. Source for Monthly, Promotions and Anomalies."),
    ("Monthly", "SUMIFS roll-up of Daily with month-over-month growth and charts."),
    ("Funnel", "Stage counts overall and by device, channel, user type, loyalty status. Step conversion, z-test vs a reference segment, Cohen's h."),
    ("Conditions", "The funnel under changing conditions (promotions, incidents, price change, rollout, external shocks): two-proportion z-tests with CIs."),
    ("Segments", "User segments (age, gender, country, acquisition channel, device, loyalty): buyer rate with Wilson CI, chi-square test, Cramér's V."),
    ("Categories", "Product categories: revenue share, margin, discount, return rate with Wilson CI and chi-square test."),
    ("Promotions", "12 promotions: model-based lift on revenue / units / profit (from notebook 03) next to a naive before/after lift computed here."),
    ("AB_Tests", "Two A/B tests recomputed from counts: sample-ratio check, lift, z, p, CI, device interaction, plus a sample-size calculator."),
    ("Anomalies", "Live anomaly detector on four daily metrics: same-weekday median baseline, MAD scale, robust z, flag, calendar look-up."),
    ("Incidents", "Output of the full 55-metric detector in notebook 04: incidents, their signature, the calendar entry that explains each, recall."),
    ("Calendar", "Promotions and internal / external events planted in the data (the ground truth used to explain findings)."),
    ("Data_Dictionary", "Tables and columns of the raw dataset in data/."),
    ("Orders_Sample", "Random sample of 5,000 orders for ad-hoc pivoting."),
]
for i, (a, b) in enumerate(rows, 3):
    ws.cell(i, 1, a).font = f_bold
    c = ws.cell(i, 2, b); c.font = f_norm; c.alignment = Alignment(wrap_text=True, vertical="top")
header(ws, 8, ["Sheet", "Contents"])

# ============================================================================
# Daily
# ============================================================================
d = pd.read_csv(OUT / "daily_metrics.csv", parse_dates=["date"])
ch_cols = [c for c in d.columns if c.startswith("sessions_") and c.split("_", 1)[1] not in ("desktop", "mobile", "tablet")]
dev_cols = ["sessions_desktop", "sessions_mobile", "sessions_tablet", "purchased_desktop", "purchased_mobile", "purchased_tablet"]
base_cols = ["sessions", "viewed_product", "added_to_cart", "began_checkout", "entered_payment", "orders", "revenue", "discount", "cogs", "units"]
ws = wb.create_sheet("Daily")
heads = ["date"] + base_cols + ["conversion", "aov", "gross_margin_pct"] + ch_cols + dev_cols
header(ws, 1, heads)
ND = len(d)
for i, row in enumerate(d.itertuples(index=False), 2):
    put(ws, i, 1, row.date.to_pydatetime(), DATE)
    for j, c in enumerate(base_cols, 2):
        put(ws, i, j, float(getattr(row, c)), USD if c in ("revenue", "discount", "cogs") else INT)
    put(ws, i, 12, f"=G{i}/B{i}", PCT2)
    put(ws, i, 13, f"=H{i}/G{i}", USD2)
    put(ws, i, 14, f"=(H{i}-J{i})/H{i}", PCT)
    for j, c in enumerate(ch_cols + dev_cols, 15):
        put(ws, i, j, float(getattr(row, c)), INT)
ws.freeze_panes = "B2"
for i in range(1, len(heads) + 1):
    ws.column_dimensions[L(i)].width = 13
DL = ND + 1          # last data row on Daily
rng = lambda col: f"Daily!${col}$2:${col}${DL}"

# ============================================================================
# Monthly
# ============================================================================
ws = sheet("Monthly", "Monthly roll-up (SUMIFS over the Daily sheet)", widths=[12, 13, 12, 15, 12, 12, 12, 12])
header(ws, 4, ["month", "sessions", "orders", "revenue", "conversion", "AOV", "revenue MoM", "sessions MoM"])
months = pd.date_range("2024-11-01", "2026-10-01", freq="MS")
for i, m in enumerate(months, 5):
    put(ws, i, 1, m.to_pydatetime(), "mmm yyyy")
    cond = f'{rng("A")},">="&$A{i},{rng("A")},"<"&EDATE($A{i},1)'
    put(ws, i, 2, f"=SUMIFS({rng('B')},{cond})", INT)
    put(ws, i, 3, f"=SUMIFS({rng('G')},{cond})", INT)
    put(ws, i, 4, f"=SUMIFS({rng('H')},{cond})", USD)
    put(ws, i, 5, f"=C{i}/B{i}", PCT2)
    put(ws, i, 6, f"=D{i}/C{i}", USD2)
    if i > 5:
        put(ws, i, 7, f"=D{i}/D{i-1}-1", PCT)
        put(ws, i, 8, f"=B{i}/B{i-1}-1", PCT)
last = 4 + len(months)
put(ws, last + 1, 1, "Total", font=f_bold)
for c in (2, 3, 4):
    put(ws, last + 1, c, f"=SUM({L(c)}5:{L(c)}{last})", USD if c == 4 else INT, f_bold)
put(ws, last + 1, 5, f"=C{last+1}/B{last+1}", PCT2, f_bold)
put(ws, last + 1, 6, f"=D{last+1}/C{last+1}", USD2, f_bold)
for col, title, yt, anchor in ((4, "Monthly revenue ($)", "revenue", "J4"), (5, "Monthly session → purchase conversion", "conversion", "J23")):
    ch = LineChart()
    ch.add_data(Reference(ws, min_col=col, min_row=4, max_row=last), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=last))
    ch.x_axis.number_format = "mmm yy"
    style_chart(ch, title, yt)
    ch.legend = None
    ws.add_chart(ch, anchor)

# ============================================================================
# Funnel
# ============================================================================
ws = sheet("Funnel", "Funnel analysis: stage counts and step-to-step conversion",
           "Counts are sessions reaching each stage (suspected bot sessions removed). Tests compare each segment's overall conversion with the reference segment (first row).",
           widths=[20, 13, 13, 13, 13, 13, 13, 11, 11, 11, 11, 11, 11, 11, 10, 10, 8])
ov = pd.read_csv(OUT / "funnel_overall.csv")
header(ws, 4, ["stage", "sessions", "share of visits", "step conversion", "drop-off at step"])
for i, r in enumerate(ov.itertuples(index=False), 5):
    put(ws, i, 1, r.stage)
    put(ws, i, 2, int(r.sessions), INT)
    put(ws, i, 3, f"=B{i}/$B$5", PCT)
    if i > 5:
        put(ws, i, 4, f"=B{i}/B{i-1}", PCT)
        put(ws, i, 5, f"=1-D{i}", PCT)
bc = BarChart(); bc.type = "bar"
bc.add_data(Reference(ws, min_col=2, min_row=4, max_row=10), titles_from_data=True)
bc.set_categories(Reference(ws, min_col=1, min_row=5, max_row=10))
bc.y_axis.scaling.orientation = "minMax"; bc.x_axis.scaling.orientation = "maxMin"
style_chart(bc, "Sessions reaching each funnel stage", w=18, h=8); bc.legend = None
ws.add_chart(bc, "G3")

stage_cols = ["sessions", "viewed_product", "added_to_cart", "began_checkout", "entered_payment", "purchased"]
steps = ["visit→view", "view→cart", "cart→checkout", "checkout→payment", "payment→purchase"]
r0 = 22
for dim, title, ref in (("device", "By device", "desktop"), ("channel", "By channel", "Organic Search"),
                        ("is_new_user", "New vs returning user (TRUE = first session)", "False"),
                        ("is_loyalty_member", "Loyalty member vs non-member", "False")):
    g = pd.read_csv(OUT / f"funnel_by_{dim}.csv")
    g["segment"] = g["segment"].astype(str)
    g = pd.concat([g[g["segment"] == ref], g[g["segment"] != ref]])
    sub(ws, r0, title)
    header(ws, r0 + 1, ["segment", "visits", "product view", "add to cart", "checkout", "payment", "purchase"] + steps
           + ["overall conv.", "lift vs ref.", "z", "p-value", "Cohen's h", "sig."])
    first = r0 + 2
    for i, r in enumerate(g.itertuples(index=False), first):
        put(ws, i, 1, r.segment + (" (ref.)" if i == first else ""))
        for j, c in enumerate(stage_cols, 2):
            put(ws, i, j, int(getattr(r, c)), INT)
        for j in range(5):
            put(ws, i, 8 + j, f"={L(3+j)}{i}/{L(2+j)}{i}", PCT)
        put(ws, i, 13, f"=G{i}/B{i}", PCT2)
        if i > first:
            put(ws, i, 14, f"=M{i}/M${first}-1", PCT)
            pooled = f"((G{i}+G${first})/(B{i}+B${first}))"
            put(ws, i, 15, f"=(M{i}-M${first})/SQRT({pooled}*(1-{pooled})*(1/B{i}+1/B${first}))", NUM2)
            put(ws, i, 16, f"=2*(1-NORMSDIST(ABS(O{i})))", PV)
            put(ws, i, 17, f"=2*ASIN(SQRT(M{i}))-2*ASIN(SQRT(M${first}))", NUM3)
            put(ws, i, 18, SIG(f"P{i}"))
    r0 = first + len(g) + 2
note(ws, r0, "Cohen's h: 0.2 = small, 0.5 = medium, 0.8 = large. A p-value of 0.0000 with |h| < 0.05 is statistically real but practically negligible.")

# ============================================================================
# Conditions
# ============================================================================
ws = sheet("Conditions", "The funnel under changing conditions",
           "Each row compares one funnel step between a baseline period and a test period (two-proportion z-test). Rates and tests are formulas on the counts.",
           widths=[52, 24, 18, 12, 12, 12, 12, 10, 10, 11, 10, 10, 9, 10, 11, 10, 7])
fc = pd.read_csv(OUT / "funnel_conditions.csv")
header(ws, 4, ["condition", "group", "funnel step", "base: entered step", "base: completed", "test: entered step", "test: completed",
               "base rate", "test rate", "relative change", "CI low", "CI high", "z", "p-value", "BH-adj. p (approx.)", "Cohen's h", "sig."])
n = len(fc)
for i, r in enumerate(fc.itertuples(index=False), 5):
    put(ws, i, 1, r.condition); put(ws, i, 2, r.group); put(ws, i, 3, r.step)
    put(ws, i, 4, int(r.n_base), INT); put(ws, i, 5, int(round(r.n_base * r.base)), INT)
    put(ws, i, 6, int(r.n_test), INT); put(ws, i, 7, int(round(r.n_test * r.test)), INT)
    put(ws, i, 8, f"=E{i}/D{i}", PCT2); put(ws, i, 9, f"=G{i}/F{i}", PCT2)
    put(ws, i, 10, f"=I{i}/H{i}-1", PCT)
    se = f"SQRT((1-H{i})/E{i}+(1-I{i})/G{i})"
    put(ws, i, 11, f"=(I{i}/H{i})*EXP(-1.96*{se})-1", PCT)
    put(ws, i, 12, f"=(I{i}/H{i})*EXP(1.96*{se})-1", PCT)
    pooled = f"((E{i}+G{i})/(D{i}+F{i}))"
    put(ws, i, 13, f"=(I{i}-H{i})/SQRT({pooled}*(1-{pooled})*(1/D{i}+1/F{i}))", NUM2)
    put(ws, i, 14, f"=2*(1-NORMSDIST(ABS(M{i})))", PV)
    put(ws, i, 15, f"=MIN(1,N{i}*COUNT($N$5:$N${4+n})/RANK(N{i},$N$5:$N${4+n},1))", PV)
    put(ws, i, 16, f"=2*ASIN(SQRT(I{i}))-2*ASIN(SQRT(H{i}))", NUM3)
    put(ws, i, 17, SIG(f"O{i}"))
ws.freeze_panes = "D5"
ws.conditional_formatting.add(f"J5:J{4+n}", CellIsRule(operator="lessThan", formula=["-0.05"], font=Font(name=F, size=10, color="C00000", bold=True)))
ws.conditional_formatting.add(f"J5:J{4+n}", CellIsRule(operator="greaterThan", formula=["0.05"], font=Font(name=F, size=10, color="007A33", bold=True)))
did = pd.read_csv(OUT / "funnel_did.csv")
r0 = 7 + n
sub(ws, r0, "Difference-in-differences (treated group's change relative to its control group's change) – values from notebook 02")
header(ws, r0 + 1, ["condition", "funnel step", "effect", "CI low", "CI high", "p-value"])
for i, r in enumerate(did.itertuples(index=False), r0 + 2):
    put(ws, i, 1, r.condition); put(ws, i, 2, r.step)
    put(ws, i, 3, float(r.did_effect), PCT); put(ws, i, 4, float(r.lo), PCT); put(ws, i, 5, float(r.hi), PCT); put(ws, i, 6, float(r.p), "0.00E+00")

# ============================================================================
# Segments
# ============================================================================
ws = sheet("Segments", "User segments: who becomes a buyer, and what are they worth?",
           "Per-user totals over 24 months (active, non-bot users; revenue net of refunds and cancellations). Chi-square tests whether buyer rate differs across the segments of a dimension.",
           widths=[22, 12, 12, 12, 15, 11, 10, 10, 11, 12, 13, 13])
su = pd.read_csv(OUT / "segments_users.csv")
titles = {"age_group": "Age group", "gender": "Gender", "country": "Country", "acquisition_channel": "Acquisition channel",
          "preferred_device": "Preferred device", "member": "Loyalty membership (selection bias – see report)"}
r0 = 4
for dim, g in su.groupby("dimension", sort=False):
    sub(ws, r0, titles[dim])
    header(ws, r0 + 1, ["segment", "users", "buyers", "orders", "net revenue", "buyer rate", "Wilson low", "Wilson high", "AOV", "revenue / user",
                        "expected buyers", "chi-sq. contribution"])
    a, b = r0 + 2, r0 + 1 + len(g)
    for i, r in enumerate(g.itertuples(index=False), a):
        put(ws, i, 1, str(r.segment)); put(ws, i, 2, int(r.users), INT); put(ws, i, 3, int(r.buyers), INT)
        put(ws, i, 4, int(r.orders), INT); put(ws, i, 5, float(r.revenue), USD)
        put(ws, i, 6, f"=C{i}/B{i}", PCT)
        put(ws, i, 7, f"=(F{i}+1.96^2/(2*B{i})-1.96*SQRT(F{i}*(1-F{i})/B{i}+1.96^2/(4*B{i}^2)))/(1+1.96^2/B{i})", PCT)
        put(ws, i, 8, f"=(F{i}+1.96^2/(2*B{i})+1.96*SQRT(F{i}*(1-F{i})/B{i}+1.96^2/(4*B{i}^2)))/(1+1.96^2/B{i})", PCT)
        put(ws, i, 9, f"=E{i}/D{i}", USD2); put(ws, i, 10, f"=E{i}/B{i}", USD2)
        put(ws, i, 11, f"=B{i}*SUM($C${a}:$C${b})/SUM($B${a}:$B${b})", INT)
        put(ws, i, 12, f"=(C{i}-K{i})^2/K{i}+((B{i}-C{i})-(B{i}-K{i}))^2/(B{i}-K{i})", NUM2)
    t = b + 1
    put(ws, t, 1, "chi-square", font=f_bold); put(ws, t, 2, f"=SUM(L{a}:L{b})", NUM2)
    put(ws, t, 3, "dof", font=f_bold); put(ws, t, 4, f"=ROWS(L{a}:L{b})-1", INT)
    put(ws, t, 5, "p-value", font=f_bold); put(ws, t, 6, f"=CHIDIST(B{t},D{t})", "0.00E+00")
    put(ws, t, 7, "Cramér's V", font=f_bold); put(ws, t, 9, f"=SQRT(B{t}/SUM(B{a}:B{b}))", NUM3)
    put(ws, t, 10, '=IF(I%d<0.03,"negligible",IF(I%d<0.1,"small","material"))' % (t, t))
    r0 = t + 2

# ============================================================================
# Categories
# ============================================================================
ws = sheet("Categories", "Product categories: revenue, margin and returns",
           "Order lines of non-cancelled orders. Return rate with Wilson 95% CI; chi-square tests whether return rates differ by category.",
           widths=[20, 12, 12, 15, 15, 15, 11, 11, 11, 11, 11, 11, 11, 13, 14])
cat = pd.read_csv(OUT / "segments_category.csv")
header(ws, 4, ["category", "order lines", "units", "revenue", "gross profit", "list value", "lines returned", "revenue share", "margin %",
               "avg discount", "return rate", "Wilson low", "Wilson high", "expected returns", "chi-sq. contribution"])
a, b = 5, 4 + len(cat)
for i, r in enumerate(cat.itertuples(index=False), a):
    put(ws, i, 1, r.category); put(ws, i, 2, int(r.lines), INT); put(ws, i, 3, int(r.units), INT)
    put(ws, i, 4, float(r.revenue), USD); put(ws, i, 5, float(r.profit), USD); put(ws, i, 6, float(r.list_value), USD); put(ws, i, 7, int(r.returned), INT)
    put(ws, i, 8, f"=D{i}/SUM($D${a}:$D${b})", PCT); put(ws, i, 9, f"=E{i}/D{i}", PCT); put(ws, i, 10, f"=1-D{i}/F{i}", PCT)
    put(ws, i, 11, f"=G{i}/B{i}", PCT)
    put(ws, i, 12, f"=(K{i}+1.96^2/(2*B{i})-1.96*SQRT(K{i}*(1-K{i})/B{i}+1.96^2/(4*B{i}^2)))/(1+1.96^2/B{i})", PCT)
    put(ws, i, 13, f"=(K{i}+1.96^2/(2*B{i})+1.96*SQRT(K{i}*(1-K{i})/B{i}+1.96^2/(4*B{i}^2)))/(1+1.96^2/B{i})", PCT)
    put(ws, i, 14, f"=B{i}*SUM($G${a}:$G${b})/SUM($B${a}:$B${b})", INT)
    put(ws, i, 15, f"=(G{i}-N{i})^2/N{i}+((B{i}-G{i})-(B{i}-N{i}))^2/(B{i}-N{i})", NUM2)
t = b + 1
put(ws, t, 1, "Total", font=f_bold)
for c in range(2, 8):
    put(ws, t, c, f"=SUM({L(c)}{a}:{L(c)}{b})", USD if c in (4, 5, 6) else INT, f_bold)
put(ws, t, 9, f"=E{t}/D{t}", PCT, f_bold); put(ws, t, 11, f"=G{t}/B{t}", PCT, f_bold)
put(ws, t + 2, 1, "chi-square", font=f_bold); put(ws, t + 2, 2, f"=SUM(O{a}:O{b})", NUM2)
put(ws, t + 2, 3, "dof", font=f_bold); put(ws, t + 2, 4, f"=ROWS(O{a}:O{b})-1", INT)
put(ws, t + 2, 5, "p-value", font=f_bold); put(ws, t + 2, 6, f"=CHIDIST(B{t+2},D{t+2})", "0.00E+00")
put(ws, t + 2, 7, "Cramér's V", font=f_bold); put(ws, t + 2, 9, f"=SQRT(B{t+2}/B{t})", NUM3)
for col, title, anchor in ((11, "Return rate by category", "A18"), (9, "Gross margin by category", "H18")):
    bc = BarChart(); bc.type = "bar"
    bc.add_data(Reference(ws, min_col=col, min_row=4, max_row=b), titles_from_data=True)
    bc.set_categories(Reference(ws, min_col=1, min_row=a, max_row=b))
    bc.x_axis.scaling.orientation = "maxMin"; bc.y_axis.number_format = "0%"
    style_chart(bc, title, w=16, h=8); bc.legend = None
    ws.add_chart(bc, anchor)

# ============================================================================
# Promotions
# ============================================================================
ws = sheet("Promotions", "Promotion effects",
           "Model columns come from notebook 03 (regression counterfactual on the promoted categories, HAC errors, BH-adjusted). "
           "The naive columns are computed here from the Daily sheet (store-wide revenue vs the 28 days before) to show why a model is needed.",
           widths=[6, 34, 11, 11, 26, 9, 6, 10, 9, 9, 10, 10, 10, 13, 13, 44, 14, 14, 10, 12, 12, 13])
pr = pd.read_csv(OUT / "promo_results.csv")
cal = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"]).set_index("promo_id")
header(ws, 4, ["id", "promotion", "start", "end", "promoted categories", "discount", "days", "MODEL revenue lift", "CI low", "CI high", "adj. p",
               "MODEL units lift", "MODEL profit lift", "incremental revenue", "incremental profit", "verdict",
               "NAIVE avg daily revenue: during", "NAIVE: prior 28 days", "NAIVE store-wide lift",
               "MODEL rest-of-store revenue lift", "MODEL store-wide profit lift", "store-wide incremental profit"])
for i, r in enumerate(pr.itertuples(index=False), 5):
    put(ws, i, 1, r.promo_id); put(ws, i, 2, r.name)
    put(ws, i, 3, cal.loc[r.promo_id, "start_date"].to_pydatetime(), DATE); put(ws, i, 4, cal.loc[r.promo_id, "end_date"].to_pydatetime(), DATE)
    put(ws, i, 5, r.scope); put(ws, i, 6, float(r.discount), "0%"); put(ws, i, 7, f"=D{i}-C{i}+1", INT)
    put(ws, i, 8, float(r.revenue_lift), PCT); put(ws, i, 9, float(r.revenue_lo), PCT); put(ws, i, 10, float(r.revenue_hi), PCT)
    put(ws, i, 11, float(r.revenue_p_adj), PV); put(ws, i, 12, float(r.units_lift), PCT); put(ws, i, 13, float(r.profit_lift), PCT)
    put(ws, i, 14, float(r.revenue_inc), USD); put(ws, i, 15, float(r.profit_inc), USD); put(ws, i, 16, r.verdict)
    put(ws, i, 17, f'=AVERAGEIFS({rng("H")},{rng("A")},">="&C{i},{rng("A")},"<="&D{i})', USD)
    put(ws, i, 18, f'=AVERAGEIFS({rng("H")},{rng("A")},">="&C{i}-28,{rng("A")},"<"&C{i})', USD)
    put(ws, i, 19, f"=IFERROR(Q{i}/R{i}-1,\"n/a\")", PCT)
    if not pd.isna(r.rest_lift):
        put(ws, i, 20, float(r.rest_lift), PCT)
    put(ws, i, 21, float(r.store_profit_lift), PCT); put(ws, i, 22, float(r.store_profit_inc), USD)
e = 4 + len(pr)
ws.conditional_formatting.add(f"U5:U{e}", CellIsRule(operator="lessThan", formula=["0"], font=Font(name=F, size=10, color="C00000", bold=True)))
ws.conditional_formatting.add(f"M5:M{e}", CellIsRule(operator="lessThan", formula=["0"], font=Font(name=F, size=10, color="C00000", bold=True)))
note(ws, e + 2, "Why naive ≠ model: the naive lift is store-wide (a Fashion-only sale is diluted by the other categories), ignores weekday mix and season, "
                "and for the first Black Friday has no 28 clean days of history. It still gets the sign and rough size right for store-wide promotions.")
bc = BarChart(); bc.type = "bar"
bc.add_data(Reference(ws, min_col=8, min_row=4, max_row=e), titles_from_data=True)
bc.add_data(Reference(ws, min_col=13, min_row=4, max_row=e), titles_from_data=True)
bc.set_categories(Reference(ws, min_col=2, min_row=5, max_row=e))
bc.x_axis.scaling.orientation = "maxMin"; bc.y_axis.number_format = "0%"
style_chart(bc, "Revenue lift vs gross-profit lift on the promoted categories", w=24, h=12)
ws.add_chart(bc, f"B{e+5}")
bf = pd.read_csv(OUT / "promo_black_friday_net.csv")
r0 = e + 31
sub(ws, r0, "Black Friday net of pull-forward (model, notebook 03): incremental $ during the event, and what was borrowed from the week before / after")
header(ws, r0 + 1, ["", "metric", "year", "during", "week before", "week after", "net effect", "borrowed share"])
for i, r in enumerate(bf.itertuples(index=False), r0 + 2):
    put(ws, i, 2, r.metric); put(ws, i, 3, str(r.year)); put(ws, i, 4, float(r.during), USD); put(ws, i, 5, float(r.pre_week), USD)
    put(ws, i, 6, float(r.post_week), USD); put(ws, i, 7, f"=D{i}+E{i}+F{i}", USD); put(ws, i, 8, f"=-(E{i}+F{i})/D{i}", PCT)

# ============================================================================
# AB_Tests
# ============================================================================
ws = sheet("AB_Tests", "A/B tests recomputed from counts",
           "Session-level two-proportion tests on the counts below. Notebook 03 repeats them with user-clustered (delta-method) errors; here the two agree to 3 decimals.",
           widths=[30, 14, 14, 14, 14, 11, 11, 11, 11, 11, 11, 9, 10, 7])
c1 = pd.read_csv(OUT / "ab1_counts.csv"); u1 = pd.read_csv(OUT / "ab1_users.csv"); c2 = pd.read_csv(OUT / "ab2_counts.csv")

def ab_block(r0, title, cnt, den, num, den_lab, num_lab):
    sub(ws, r0, title)
    header(ws, r0 + 1, ["device", f"control: {den_lab}", f"control: {num_lab}", f"treatment: {den_lab}", f"treatment: {num_lab}",
                        "control rate", "treatment rate", "abs. diff", "relative lift", "CI low", "CI high", "z", "p-value", "sig."])
    devs = ["desktop", "mobile", "tablet"]
    a = r0 + 2
    for i, dv in enumerate(devs + ["ALL DEVICES"], a):
        put(ws, i, 1, dv, font=f_bold if dv == "ALL DEVICES" else None)
        if dv == "ALL DEVICES":
            for c in range(2, 6):
                put(ws, i, c, f"=SUM({L(c)}{a}:{L(c)}{a+2})", INT, f_bold)
        else:
            cc = cnt[(cnt["device"] == dv) & (cnt["ab_variant"] == "control")].iloc[0]
            tt = cnt[(cnt["device"] == dv) & (cnt["ab_variant"] == "treatment")].iloc[0]
            put(ws, i, 2, int(cc[den]), INT); put(ws, i, 3, int(cc[num]), INT); put(ws, i, 4, int(tt[den]), INT); put(ws, i, 5, int(tt[num]), INT)
        put(ws, i, 6, f"=C{i}/B{i}", PCT2); put(ws, i, 7, f"=E{i}/D{i}", PCT2); put(ws, i, 8, f"=G{i}-F{i}", "0.00%;-0.00%")
        put(ws, i, 9, f"=G{i}/F{i}-1", "+0.00%;-0.00%")
        se = f"SQRT((1-F{i})/C{i}+(1-G{i})/E{i})"
        put(ws, i, 10, f"=(G{i}/F{i})*EXP(-1.96*{se})-1", "+0.00%;-0.00%"); put(ws, i, 11, f"=(G{i}/F{i})*EXP(1.96*{se})-1", "+0.00%;-0.00%")
        pooled = f"((C{i}+E{i})/(B{i}+D{i}))"
        put(ws, i, 12, f"=(G{i}-F{i})/SQRT({pooled}*(1-{pooled})*(1/B{i}+1/D{i}))", NUM2)
        put(ws, i, 13, f"=2*(1-NORMSDIST(ABS(L{i})))", PV); put(ws, i, 14, SIG(f"M{i}"))
    return a

a1 = ab_block(4, "AB1 · one-page checkout (2 Mar – 12 Apr 2026) · PRIMARY metric: checkout → purchase", c1, "began_checkout", "purchased", "checkouts", "purchases")
r = a1 + 5
put(ws, r, 1, "Interaction: mobile lift − desktop lift", font=f_bold)
put(ws, r, 2, f"=I{a1+1}-I{a1}", "+0.00%;-0.00%")
put(ws, r, 3, "z", font=f_bold)
put(ws, r, 4, f"=(LN(G{a1+1}/F{a1+1})-LN(G{a1}/F{a1}))/SQRT((1-F{a1+1})/C{a1+1}+(1-G{a1+1})/E{a1+1}+(1-F{a1})/C{a1}+(1-G{a1})/E{a1})", NUM2)
put(ws, r, 5, "p-value", font=f_bold); put(ws, r, 6, f"=2*(1-NORMSDIST(ABS(D{r})))", "0.00E+00"); put(ws, r, 7, SIG(f"F{r}"))
note(ws, r + 1, "A significant interaction means the effect genuinely differs by device – the lift is a mobile / tablet effect; desktop is unaffected.")

r0 = r + 3
sub(ws, r0, "AB1 · validity and user-level metrics")
header(ws, r0 + 1, ["", "control", "treatment", "difference", "relative", "statistic", "p-value", "verdict"])
uc, ut = u1[u1["ab_variant"] == "control"].iloc[0], u1[u1["ab_variant"] == "treatment"].iloc[0]
x = r0 + 2
put(ws, x, 1, "users (randomisation unit)"); put(ws, x, 2, int(uc["users"]), INT); put(ws, x, 3, int(ut["users"]), INT)
put(ws, x, 4, f"=C{x}-B{x}", INT); put(ws, x, 5, f"=C{x}/(B{x}+C{x})", PCT2)
put(ws, x, 6, f"=(B{x}-(B{x}+C{x})/2)^2/((B{x}+C{x})/2)+(C{x}-(B{x}+C{x})/2)^2/((B{x}+C{x})/2)", NUM2)
put(ws, x, 7, f"=CHIDIST(F{x},1)", PV); put(ws, x, 8, f'=IF(G{x}<0.001,"SAMPLE-RATIO MISMATCH","no sample-ratio mismatch")')
put(ws, x + 1, 1, "buyers during the test"); put(ws, x + 1, 2, int(uc["buyers"]), INT); put(ws, x + 1, 3, int(ut["buyers"]), INT)
put(ws, x + 2, 1, "user conversion"); put(ws, x + 2, 2, f"=B{x+1}/B{x}", PCT2); put(ws, x + 2, 3, f"=C{x+1}/C{x}", PCT2)
put(ws, x + 2, 4, f"=C{x+2}-B{x+2}", "0.00%;-0.00%"); put(ws, x + 2, 5, f"=C{x+2}/B{x+2}-1", "+0.00%;-0.00%")
pooled = f"((B{x+1}+C{x+1})/(B{x}+C{x}))"
put(ws, x + 2, 6, f"=(C{x+2}-B{x+2})/SQRT({pooled}*(1-{pooled})*(1/B{x}+1/C{x}))", NUM2)
put(ws, x + 2, 7, f"=2*(1-NORMSDIST(ABS(F{x+2})))", PV); put(ws, x + 2, 8, SIG(f"G{x+2}"))
put(ws, x + 3, 1, "revenue (total $)"); put(ws, x + 3, 2, float(uc["revenue"]), USD); put(ws, x + 3, 3, float(ut["revenue"]), USD)
put(ws, x + 4, 1, "revenue sd per user"); put(ws, x + 4, 2, float(uc["revenue_sd"]), USD2); put(ws, x + 4, 3, float(ut["revenue_sd"]), USD2)
put(ws, x + 5, 1, "revenue per user (Welch test)"); put(ws, x + 5, 2, f"=B{x+3}/B{x}", USD2); put(ws, x + 5, 3, f"=C{x+3}/C{x}", USD2)
put(ws, x + 5, 4, f"=C{x+5}-B{x+5}", "$0.000"); put(ws, x + 5, 5, f"=C{x+5}/B{x+5}-1", "+0.00%;-0.00%")
put(ws, x + 5, 6, f"=D{x+5}/SQRT(B{x+4}^2/B{x}+C{x+4}^2/C{x})", NUM2)
put(ws, x + 5, 7, f"=2*(1-NORMSDIST(ABS(F{x+5})))", PV); put(ws, x + 5, 8, SIG(f"G{x+5}"))
note(ws, x + 6, "Revenue per user is far noisier than conversion: the same real effect is clearly significant on the primary metric and not significant on revenue.")

a2 = ab_block(x + 9, "AB2 · social-proof badge (7 Sep – 4 Oct 2026) · PRIMARY metric: product view → add to cart", c2, "viewed_product", "added_to_cart", "product views", "add-to-carts")
note(ws, a2 + 4, "AB2 is a true null: no device shows an effect. Do not ship on this evidence.")

r0 = a2 + 7
sub(ws, r0, "Sample-size calculator (two-sided test of two proportions) – change the yellow cells")
labels = [("baseline conversion rate", 0.645, PCT), ("minimum detectable effect (relative)", 0.02, PCT), ("significance level α", 0.05, "0.00"), ("power", 0.80, "0%")]
for i, (lab, val, fmt) in enumerate(labels, r0 + 1):
    put(ws, i, 1, lab); c = put(ws, i, 2, val, fmt); c.fill = fill_key
b0 = r0 + 1
put(ws, b0 + 4, 1, "treatment rate implied"); put(ws, b0 + 4, 2, f"=B{b0}*(1+B{b0+1})", PCT2)
put(ws, b0 + 5, 1, "required sample per arm", font=f_bold)
put(ws, b0 + 5, 2, f"=ROUNDUP((NORMSINV(1-B{b0+2}/2)+NORMSINV(B{b0+3}))^2*(B{b0}*(1-B{b0})+B{b0+4}*(1-B{b0+4}))/(B{b0+4}-B{b0})^2,0)", INT, f_bold)
put(ws, b0 + 6, 1, "AB1 actually had (checkouts per arm)"); put(ws, b0 + 6, 2, f"=B{a1+3}", INT)
note(ws, b0 + 7, "Units: sessions that started checkout. Default inputs reproduce AB1's primary metric (64.5% baseline, 2% relative lift).")

# ============================================================================
# Calendar
# ============================================================================
ws = sheet("Calendar", "Promotions and events planted in the data (ground truth)",
           "Sorted by duration so that the Anomalies look-up returns the most specific entry first. 'match until' = end date, capped at start + 14 days (onset window).",
           widths=[7, 44, 12, 12, 10, 16, 80, 12])
ev = pd.read_csv(DATA / "events_calendar.csv", parse_dates=["start_date", "end_date"])
pm = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
rows = [(r.promo_id, r["name"], r.start_date, r.end_date, "internal", "promotion", f"{r['type']}, {r.discount_pct:.0%} off, {r.categories}, audience: {r.audience}") for _, r in pm.iterrows()]
rows += [(r.event_id, r["name"], r.start_date, r.end_date, r.origin, r.type, r.expected_signature) for _, r in ev.iterrows() if r.type != "experiment"]
rows.sort(key=lambda t: (t[3] - t[2]).days)
rows += [(r.event_id, r["name"], r.start_date, r.end_date, r.origin, r.type, r.expected_signature) for _, r in ev.iterrows() if r.type == "experiment"]
header(ws, 4, ["id", "name", "start", "end", "origin", "type", "what it does", "match until"])
for i, r in enumerate(rows, 5):
    put(ws, i, 1, r[0]); put(ws, i, 2, r[1]); put(ws, i, 3, r[2].to_pydatetime(), DATE); put(ws, i, 4, r[3].to_pydatetime(), DATE)
    put(ws, i, 5, r[4]); put(ws, i, 6, r[5]); put(ws, i, 7, r[6])
    put(ws, i, 8, f'=IF(F{i}="experiment",C{i}-1,MIN(D{i},C{i}+13)+1)', DATE)
CAL_A, CAL_B = 5, 4 + len(rows)

# ============================================================================
# Anomalies (live detector)
# ============================================================================
ws = sheet("Anomalies", "Live anomaly detector on four daily metrics",
           "expected = median of the same weekday in the previous 6 weeks · deviation = LN(actual / expected) · scale = 1.4826 × MAD of the previous 56 deviations · z = deviation / scale",
           widths=[12, 6] + [11, 11, 9, 8, 7] * 4 + [8, 11, 22, 46])
put(ws, 3, 1, "flag when |z| ≥", font=f_bold); c = put(ws, 3, 3, 5, "0.0"); c.fill = fill_key
put(ws, 3, 5, "days flagged", font=f_bold); put(ws, 3, 7, '=COUNTIF(X7:X736,"ANOMALY")', INT)
put(ws, 3, 9, "of which matched to the calendar", font=f_bold); put(ws, 3, 13, '=COUNTIFS(X7:X736,"ANOMALY",Z7:Z736,"<>")', INT)
put(ws, 3, 15, "share explained", font=f_bold); put(ws, 3, 17, "=IF(G3=0,0,M3/G3)", PCT)
metrics = [("sessions", "Daily!B{r}"), ("conversion", "Daily!L{r}"), ("revenue", "Daily!H{r}"), ("mobile conversion", "Daily!Y{r}/Daily!V{r}")]
heads = ["date", "wkday"]
for m, _ in metrics:
    heads += [m, "expected", "deviation", "scale", "z"]
heads += ["max |z|", "flag", "metric", "calendar entry (first match)"]
header(ws, 6, heads)
R0 = 7
for k in range(ND):
    r, dr = R0 + k, 2 + k
    put(ws, r, 1, f"=Daily!A{dr}", DATE); put(ws, r, 2, f'=TEXT(A{r},"ddd")')
    zcells = []
    for mi, (m, src) in enumerate(metrics):
        c0 = 3 + 5 * mi
        v, ex, dv, sc, z = (L(c0 + j) for j in range(5))
        fmt = PCT2 if "conversion" in m else (USD if m == "revenue" else INT)
        put(ws, r, c0, "=" + src.format(r=dr), fmt)
        if k >= 42:
            put(ws, r, c0 + 1, "=MEDIAN(" + ",".join(f"{v}{r-7*j}" for j in range(1, 7)) + ")", fmt)
            put(ws, r, c0 + 2, f"=LN({v}{r}/{ex}{r})", NUM3)
        if k >= 98:
            rg = f"{dv}{r-56}:{dv}{r-1}"
            ws.cell(r, c0 + 3).value = ArrayFormula(f"{sc}{r}", f"=MAX(0.01,1.4826*MEDIAN(ABS({rg}-MEDIAN({rg}))))")
            ws.cell(r, c0 + 3).font = f_norm; ws.cell(r, c0 + 3).number_format = NUM3
            put(ws, r, c0 + 4, f"={dv}{r}/{sc}{r}", "0.0")
            zcells.append(f"{z}{r}")
    if zcells:
        put(ws, r, 23, "=MAX(" + ",".join(f"ABS({z})" for z in zcells) + ")", "0.0")
        put(ws, r, 24, f'=IF(W{r}>=$C$3,"ANOMALY","")')
        idx = "IF(ABS(G{r})=W{r},\"sessions\",IF(ABS(L{r})=W{r},\"conversion\",IF(ABS(Q{r})=W{r},\"revenue\",\"mobile conversion\")))".format(r=r)
        put(ws, r, 25, f'=IF(X{r}="","",{idx})')
        ws.cell(r, 26).value = ArrayFormula(
            f"Z{r}", f'=IF(X{r}="","",IFERROR(INDEX(Calendar!$B${CAL_A}:$B${CAL_B},MATCH(1,(Calendar!$C${CAL_A}:$C${CAL_B}<=A{r})*(Calendar!$H${CAL_A}:$H${CAL_B}>=A{r}),0)),""))')
        ws.cell(r, 26).font = f_norm
ws.freeze_panes = "C7"
RL = R0 + ND - 1
ws.conditional_formatting.add(f"A7:Z{RL}", FormulaRule(formula=['$X7="ANOMALY"'], fill=PatternFill("solid", bgColor="FDE9D9")))
ch = LineChart()
for col in (7, 12, 17, 22):
    ch.add_data(Reference(ws, min_col=col, min_row=6, max_row=RL), titles_from_data=True)
ch.set_categories(Reference(ws, min_col=1, min_row=R0, max_row=RL))
ch.x_axis.number_format = "mmm yy"; ch.x_axis.tickLblSkip = 60
style_chart(ch, "Robust z-score by day (sessions, conversion, revenue, mobile conversion)", "z", w=30, h=11)
for s, colr in zip(ch.series, ("2A78D6", "EB6834", "1BAF7A", "EDA100")):
    s.graphicalProperties.line.width = 9000
    s.graphicalProperties.line.solidFill = colr
    s.smooth = False
ws.add_chart(ch, "AB6")

# ============================================================================
# Incidents (from notebook 04)
# ============================================================================
ws = sheet("Incidents", "Anomaly detection results from notebook 04 (55 metrics across the purchase life cycle)",
           "Values imported from outputs/anomaly_incidents.csv and anomaly_calendar_recall.csv. The detector does not see the calendar; the match is made afterwards.",
           widths=[12, 12, 7, 8, 12, 9, 9, 70, 70])
inc = pd.read_csv(OUT / "anomaly_incidents.csv", parse_dates=["start", "end"])
sub(ws, 4, "Detected incidents")
header(ws, 5, ["start", "end", "days", "flags", "unexplained flags", "metrics", "peak z", "signature (largest deviations)", "explained by"])
for i, r in enumerate(inc.itertuples(index=False), 6):
    put(ws, i, 1, r.start.to_pydatetime(), DATE); put(ws, i, 2, r.end.to_pydatetime(), DATE); put(ws, i, 3, int(r.days), INT); put(ws, i, 4, int(r.flags), INT)
    put(ws, i, 5, int(r.unexplained_flags), INT); put(ws, i, 6, int(r.metrics), INT); put(ws, i, 7, float(r.peak_z), "0.0")
    put(ws, i, 8, r.signature); put(ws, i, 9, r.explained_by)
e = 5 + len(inc)
put(ws, e + 1, 1, "Total", font=f_bold); put(ws, e + 1, 4, f"=SUM(D6:D{e})", INT, f_bold); put(ws, e + 1, 5, f"=SUM(E6:E{e})", INT, f_bold)
put(ws, e + 1, 7, "explained share of flags", font=f_bold); put(ws, e + 1, 9, f"=1-E{e+1}/D{e+1}", PCT, f_bold)
rec = pd.read_csv(OUT / "anomaly_calendar_recall.csv")
r0 = e + 4
sub(ws, r0, "Calendar entries: was each one detected?")
header(ws, r0 + 1, ["id", "origin", "", "", "type", "lag (days)", "metrics", "name", "status / strongest signals"])
for i, r in enumerate(rec.itertuples(index=False), r0 + 2):
    put(ws, i, 1, r.id); put(ws, i, 2, r.origin); put(ws, i, 5, r.kind)
    if not pd.isna(r.first_flag_lag_days):
        put(ws, i, 6, int(r.first_flag_lag_days), INT)
    put(ws, i, 7, int(r.metrics_flagged), INT); put(ws, i, 8, r.name)
    put(ws, i, 9, r.status + (": " + r.strongest_signals if isinstance(r.strongest_signals, str) and r.strongest_signals else ""))
a, b = r0 + 2, r0 + 1 + len(rec)
put(ws, b + 2, 8, "detected", font=f_bold); put(ws, b + 2, 9, f'=COUNTIF(I{a}:I{b},"detected*")', INT)
put(ws, b + 3, 8, "missed", font=f_bold); put(ws, b + 3, 9, f'=COUNTIF(I{a}:I{b},"missed*")', INT)
put(ws, b + 4, 8, "recall on detectable events", font=f_bold); put(ws, b + 4, 9, f"=I{b+2}/(I{b+2}+I{b+3})", PCT, f_bold)

# ============================================================================
# Data dictionary
# ============================================================================
ws = sheet("Data_Dictionary", "Data dictionary (files in data/)", widths=[16, 24, 100])
dd = [
    ("sessions", "session_id, user_id", "One row per visit (23.9M). user_id links to users."),
    ("sessions", "session_start", "Timestamp of the visit (store local time)."),
    ("sessions", "channel, device, country", "Traffic source (6), device (3), visitor country (7)."),
    ("sessions", "is_new_user, is_loyalty_member", "First-ever session of the user; user was a loyalty member at the time of the visit."),
    ("sessions", "pageviews, duration_sec", "Engagement. One page and ≤ 2 seconds marks the bot wave of 3–6 Nov 2025."),
    ("sessions", "viewed_category, viewed_product_id", "Main product viewed (empty if no product view)."),
    ("sessions", "viewed_product … purchased", "Five funnel flags: viewed_product → added_to_cart → began_checkout → entered_payment → purchased."),
    ("sessions", "payment_method", "Method chosen at the payment step (card, paypal, wallet, bnpl); empty if payment was not reached."),
    ("sessions", "ab_test, ab_variant", "Experiment and arm for sessions inside an A/B test window (assignment is per user)."),
    ("sessions", "order_id", "Order created by the session, if any."),
    ("orders", "order_id, session_id, user_id, order_ts", "One row per order (1.18M)."),
    ("orders", "gross_amount, discount_amount, revenue", "List value, promotion discount, revenue = gross − discount (excludes shipping)."),
    ("orders", "shipping_fee, cogs, promo_id", "Shipping charged, cost of goods, promotion applied."),
    ("orders", "status, delivery_days, delivered_date", "cancelled / delivered / partially_returned / returned; delivery time in days."),
    ("orders", "refund_amount, review_rating", "Refunded value; 1–5 rating (about 30% of orders are rated)."),
    ("order_items", "order_id, product_id, category", "One row per order line (1.83M)."),
    ("order_items", "quantity, list_price, discount_pct", "Units, list price at the time of the order, discount applied."),
    ("order_items", "unit_price_paid, unit_cost, returned", "Price paid, unit cost, whether the line was returned."),
    ("users", "user_id, signup_date", "One row per visitor (3.32M); signup_date = first seen."),
    ("users", "country, age_group, gender", "Demographics."),
    ("users", "acquisition_channel, preferred_device", "Channel of the first visit; device used most."),
    ("users", "loyalty_join_date", "Date the user joined the loyalty programme (empty = never)."),
    ("products", "product_id, product_name, category", "3,000 products in 8 categories."),
    ("products", "base_price, unit_cost, launch_date", "List price before the Sept 2025 Electronics increase, unit cost, launch date."),
    ("price_history", "product_id, effective_from, new_price", "Electronics list prices after the +8% increase on 1 Sep 2025."),
    ("promotions", "promo_id … audience", "12 promotions: dates, categories, discount, type, audience."),
    ("events_calendar", "event_id … expected_signature", "24 internal / external events planted by the generator (ground truth)."),
    ("ab_tests", "ab_test … hypothesis", "Design of the two experiments (unit, allocation, pre-registered primary metric)."),
]
header(ws, 3, ["table", "columns", "description"])
for i, r in enumerate(dd, 4):
    for j, v in enumerate(r, 1):
        c = put(ws, i, j, v); c.alignment = Alignment(wrap_text=True, vertical="top")

# ============================================================================
# Orders sample
# ============================================================================
ws = wb.create_sheet("Orders_Sample")
o = pd.read_parquet(DATA / "orders.parquet").sample(5000, random_state=1).sort_values("order_ts")
cols = ["order_id", "order_ts", "country", "device", "channel", "payment_method", "promo_id", "units", "gross_amount", "discount_amount",
        "revenue", "shipping_fee", "cogs", "status", "delivery_days", "refund_amount", "review_rating"]
header(ws, 1, cols)
for i, r in enumerate(o[cols].itertuples(index=False), 2):
    for j, v in enumerate(r, 1):
        if pd.isna(v):
            continue
        if isinstance(v, pd.Timestamp):
            put(ws, i, j, v.to_pydatetime(), "yyyy-mm-dd hh:mm")
        elif isinstance(v, (int, float, np.integer, np.floating)):
            put(ws, i, j, float(v), USD2 if cols[j - 1] in ("gross_amount", "discount_amount", "revenue", "shipping_fee", "cogs", "refund_amount") else "0")
        else:
            put(ws, i, j, str(v), font=f_in)
ws.freeze_panes = "A2"
for i in range(1, len(cols) + 1):
    ws.column_dimensions[L(i)].width = 15
ws.auto_filter.ref = f"A1:{L(len(cols))}5001"

wb.save(DEST)
print("saved", DEST, f"{DEST.stat().st_size / 1e6:.1f} MB")
