#!/usr/bin/env python3
"""Part 1 slide figures: explaining the paper (forest MFD, our replication unless noted).

paper_1_construction.png   how MFD is built (diagram)
paper_2_example.png        the 100 forecasts for a low-MFD and a high-MFD stock, same month
paper_3_headline.png       VW return by MFD decile + cumulative low-minus-high MFD strategy, 1976-2025
paper_4_who.png            what high-MFD stocks look like, by MFD decile
paper_5_mechanism.png      the authors' borrowing-fee evidence (reported numbers, not replicated)

Stocks in paper_2 are anonymised (no tickers/names on slides: WRDS/JKP licence).
Style matches scripts/plot_results.py. Definitions: docs/definitions.md.
Usage (project root, mfd env): python scripts/plot_paper_overview.py
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
spec = importlib.util.spec_from_file_location("pr", str(Path(__file__).with_name("plot_results.py")))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)  # shared rcParams and colours
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
SURFACE, INK, INK2, GRID, BLUE = pr.SURFACE, pr.INK, pr.INK2, pr.GRID, pr.C_NO
EXAMPLE_MONTH = "2019-12"


def footnote(fig, text):
    fig.text(0.01, 0.005, text, color=INK2, fontsize=10)


# ----------------------------------------------------------------------------- 1. construction
def construction():
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 40)
    ax.axis("off")
    steps = [
        ("153 stock\ncharacteristics", "JKP dataset,\nranked to [-1, 1]\neach month"),
        ("100 \"investors\"", "each sees a random\n76 of the 153"),
        ("Each fits a\nrandom forest", "depth 3, 2,000 trees;\npast 120 months;\nrefit every 12 months"),
        ("100 forecasts", "of next-month return,\nper stock, per month"),
        ("MFD", "standard deviation\nof the 100 forecasts"),
        ("Portfolio", "sort stocks into MFD\ndeciles; long lowest,\nshort highest"),
    ]
    n = len(steps)
    w, gap = 13.6, (100 - 6 * 13.6) / 5
    for i, (head, body) in enumerate(steps):
        x0 = i * (w + gap)
        hl = head == "MFD"
        box = FancyBboxPatch((x0, 6), w, 26, boxstyle="round,pad=0.3,rounding_size=1.2",
                             fc="#e8f1fc" if hl else "#f0efec", ec=BLUE if hl else GRID, lw=2 if hl else 1.2)
        ax.add_patch(box)
        ax.text(x0 + w / 2, 25.5, head, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
        ax.text(x0 + w / 2, 15.5, body, ha="center", va="center", fontsize=10.5, color=INK2, linespacing=1.35)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.4, 19), (x0 + w + gap - 0.4, 19), arrowstyle="-|>",
                                         mutation_scale=16, color=INK2, lw=1.5))
    ax.text(0, 38, "How the paper builds Machine Forecast Disagreement (MFD)", fontsize=15, fontweight="bold",
            color=INK, va="center")
    ax.text(0, 0.5, "Different investors see different information, so their forecasts disagree. The paper argues "
            "high disagreement means overpricing\n(Miller, 1977): pessimists can't easily short, so prices reflect "
            "the optimists, and high-MFD stocks later underperform.", fontsize=10.5, color=INK2, linespacing=1.4)
    return fig


# ----------------------------------------------------------------------------- 2. example
def example(panel):
    tag = EXAMPLE_MONTH.replace("-", "")
    fc = pd.read_parquet(ROOT / f"data/mfd_full/month={tag}_forecasts.pq")
    d = panel[panel.month == pd.Period(EXAMPLE_MONTH, "M")][["permno", "std", "mean", "market_equity"]].copy()
    d["dec"] = pd.qcut(d["std"].rank(method="first"), 10, labels=False) + 1
    d["size_pct"] = d["market_equity"].rank(pct=True)
    # same consensus, different disagreement: in decile 1 and decile 10, the stock whose MFD is
    # near its decile's median AND whose mean forecast is closest to the month's median mean forecast
    target = d["mean"].median()
    picks = []
    for dec in (1, 10):
        g = d[d.dec == dec].copy()
        g = g[(g["std"] - g["std"].median()).abs() <= g["std"].std()]
        picks.append(g.loc[(g["mean"] - target).abs().idxmin()])
    inv = [c for c in fc.columns if c.startswith("inv_")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    allv = np.concatenate([fc.loc[fc.permno == p.permno, inv].to_numpy().ravel() for p in picks]) * 100
    bins = np.linspace(allv.min() - 0.05, allv.max() + 0.05, 45)
    for ax, p, name, lab in zip(axes, picks, ("Stock A", "Stock B"), ("lowest MFD decile", "highest MFD decile")):
        v = fc.loc[fc.permno == p.permno, inv].to_numpy().ravel() * 100
        ax.hist(v, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        m, s = v.mean(), v.std(ddof=1)
        ax.axvline(m, color=INK, lw=2, zorder=4)
        ax.axvspan(m - s, m + s, color=BLUE, alpha=0.10, zorder=1)
        n = int(round(p.size_pct * 100))
        suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        ax.set_title(f"{name}: {lab}", loc="left", fontsize=13)
        ax.text(0.99, 0.85, f"mean forecast {m:+.2f}%\nMFD (std) {s:.2f}%", transform=ax.transAxes, ha="right",
                va="top", fontsize=11.5, color=INK, linespacing=1.5)
        ax.set_ylabel("Investors")
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("Forecast of next-month excess return (% per month), one per investor")
    fig.suptitle(f"Same average forecast, different disagreement: 100 forecasts each, "
                 f"{pd.Period(EXAMPLE_MONTH, 'M').strftime('%b %Y')}", x=0.01, ha="left", fontsize=15,
                 fontweight="bold")
    footnote(fig, "Black line: mean forecast. Shaded band: mean ± 1 SD (= MFD). Typical stocks of MFD decile 1 "
                  "and 10, with a mean forecast near the month's median; anonymised.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


# ----------------------------------------------------------------------------- 3. headline
def headline(panel, factors):
    t3 = pd.read_csv(ROOT / "results/paper_tables/forest/table3_vw.csv", index_col=0).drop(index="H-L")
    d = panel.dropna(subset=["std"]).copy()
    pct = d.groupby("month")["std"].rank(method="dense", pct=True)
    pct = pct.where(pct != pct.groupby(d["month"]).transform("min"), 1e-8)
    d["dec"] = np.ceil(pct * 10).clip(lower=1).astype(int)
    d = d[d.dec.isin([1, 10])]
    g = d.assign(rw=d.ret_exc_lead1m * d.me)
    vw = (g.groupby(["month", "dec"]).rw.sum() / g.groupby(["month", "dec"]).me.sum()).unstack()
    ls = (vw[1] - vw[10])  # long lowest MFD, short highest (decimal, return over month t+1)
    ls.index = ls.index + 1  # plot at the month the return is earned
    growth = (1 + ls).cumprod()
    oos_start = tm.PAPER_END + 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(1, 11)
    a1.bar(x, t3["XRet"], color=BLUE, width=0.7, zorder=3)
    a1.axhline(0, color=INK2, lw=1)
    a1.set_xticks(x, ["Low"] + [str(i) for i in range(2, 10)] + ["High"])
    a1.set_xlabel("MFD decile")
    a1.set_ylabel("Next-month excess return (% per month)")
    a1.set_title("Value-weighted return by MFD decile, 1976–2022", loc="left", fontsize=13)
    a1.grid(axis="x", visible=False)
    for xi, yv in ((1, t3["XRet"].iloc[0]), (10, t3["XRet"].iloc[-1])):
        a1.annotate(f"{yv:+.2f}", (xi, max(yv, 0)), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=11, color=INK)
    a1.margins(y=0.12)
    ts = growth.index.to_timestamp()
    a2.plot(ts, growth.to_numpy(), color=BLUE, lw=2, zorder=3)
    a2.set_yscale("log")
    a2.axvspan(pd.Period(oos_start, "M").to_timestamp(), ts.max(), color=GRID, alpha=0.7, zorder=1)
    a2.annotate("after the paper's sample\n(Dec 2022 – Nov 2025)", xy=(pd.Period(oos_start, "M").to_timestamp(), 3),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=10, color=INK2)
    ins = ls[ls.index <= oos_start - 1] * 100
    oos = ls[ls.index >= oos_start] * 100
    a2.set_title("Growth of $1, long lowest / short highest MFD decile (VW)", loc="left", fontsize=13)
    a2.set_ylabel("Value of $1 (log scale)")
    a2.text(0.02, 0.95, f"1976–2022: {ins.mean():+.2f}% per month\n2023–2025: {oos.mean():+.2f}% per month",
            transform=a2.transAxes, va="top", fontsize=11.5, color=INK, linespacing=1.5)
    fig.suptitle("Headline: low-disagreement stocks beat high-disagreement stocks by about 1% a month",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    footnote(fig, "Our replication of the paper's forest MFD. Long-short excess returns, value-weighted by "
                  "market equity; no transaction costs.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


# ----------------------------------------------------------------------------- 4. who
def who(panel):
    px = pd.read_parquet(ROOT / "data/raw/usa.parquet", columns=["date", "permno", "prc"],
                         filters=[("date", "<=", 20221231)])
    px["month"] = pd.to_datetime(px.pop("date").astype(str)).dt.to_period("M")
    px["permno"] = px["permno"].astype(int)
    d = panel[panel.month <= tm.PAPER_END].merge(px, on=["month", "permno"], how="left").dropna(subset=["std"])
    d["dec"] = d.groupby("month")["std"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    d["ivol_ann"] = d["ivol_capm_252d"] * np.sqrt(252) * 100
    d["turn_pct"] = d["turnover_126d"] * 100
    d["mean_pct"] = d["mean"] * 100
    stats = [("market_equity", "Market cap, $ millions\n(median)", "median", "{:,.0f}"),
             ("prc", "Share price, $\n(median)", "median", "{:,.0f}"),
             ("ivol_ann", "Idiosyncratic volatility,\n% per year (median)", "median", "{:.0f}"),
             ("turn_pct", "Turnover,\n% of shares per day (median)", "median", "{:.2f}"),
             ("mean_pct", "Investors' mean forecast,\n% per month (average)", "mean", "{:+.2f}")]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.6))
    x = np.arange(1, 11)
    for ax, (col, title, how, fmt) in zip(axes, stats):
        per_month = d.groupby(["month", "dec"])[col].agg(how).unstack()
        vals = per_month.mean()
        ax.bar(x, vals.to_numpy(), color=BLUE, width=0.72, zorder=3)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(f"{title}\nLow {fmt.format(vals.iloc[0])} → High {fmt.format(vals.iloc[-1])}", loc="left",
                     fontsize=11.5)
        ax.set_xticks([1, 10], ["Low", "High"])
        ax.set_xlabel("MFD decile", fontsize=10.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Who are high-MFD stocks? Small, cheap, volatile, heavily traded, and forecast to do badly",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    footnote(fig, "Our replication, Aug 1976 – Nov 2022. Each month's cross-sectional statistic by MFD decile, "
                  "averaged over months.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return fig


# ----------------------------------------------------------------------------- 5. mechanism (reported)
def mechanism():
    fee = [0.67, 0.68, 0.88, 1.27, 3.82]  # package results/bivariate_IFEE.tex, Panel A
    alpha = {"Low": (-0.12, -0.85), "Middle": (-0.74, -2.46), "High": (-2.36, -6.67)}  # Panel B, FF6 alpha, t
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(1, 6)
    a1.bar(x, fee, color=BLUE, width=0.65, zorder=3)
    a1.set_xticks(x, ["Low", "2", "3", "4", "High"])
    a1.set_xlabel("MFD quintile")
    a1.set_ylabel("Average borrowing fee (% per year)")
    a1.set_title("High-MFD stocks are expensive to short", loc="left", fontsize=13)
    a1.grid(axis="x", visible=False)
    for xi, v in zip(x, fee):
        a1.annotate(f"{v:.2f}", (xi, v), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10.5)
    xs = np.arange(3)
    vals = [alpha[k][0] for k in alpha]
    a2.bar(xs, vals, color=BLUE, width=0.6, zorder=3)
    a2.axhline(0, color=INK2, lw=1)
    a2.set_xticks(xs, [f"{k} fee" for k in alpha])
    a2.set_xlabel("Borrowing-fee group")
    a2.set_ylabel("MFD high-minus-low FF6 alpha (% per month)")
    a2.set_title("…and the MFD effect lives where shorting is costly", loc="left", fontsize=13)
    a2.grid(axis="x", visible=False)
    for xi, (k, (v, t)) in zip(xs, alpha.items()):
        a2.annotate(f"{v:+.2f}\n(t {t:+.2f})", (xi, v), xytext=(0, -30), textcoords="offset points",
                    ha="center", fontsize=10.5)
    a2.set_ylim(min(vals) * 1.35, 0.3)
    fig.suptitle("The paper's mechanism evidence (Miller, 1977): reported by the authors",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    footnote(fig, "Source: authors' replication package (Feb 2026), Markit borrowing-fee data from 2004; "
                  "not replicated here (proprietary data). Right: bivariate sort, borrowing-fee group then MFD.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return fig


def investors():
    """Who sees what: 100 investors x 153 characteristics, plus the recipe inside one investor."""
    inv = pd.read_csv(ROOT / "data/mfd_full/investor_features.csv")
    fd = pd.read_excel(ROOT / "data/raw/Factor Details.xlsx").dropna(subset=["abr_jkp"])
    group = dict(zip(fd["abr_jkp"], fd["group"].fillna("Other").str.strip().str.capitalize()))
    chars = sorted(set(inv.to_numpy().ravel()) | set(group), key=lambda c: (group.get(c, "Other"), c))
    chars = [c for c in chars if c in group]
    col = {c: i for i, c in enumerate(chars)}
    M = np.zeros((inv.shape[1], len(chars)))
    for k in range(inv.shape[1]):
        for c in inv.iloc[:, k]:
            M[k, col[c]] = 1
    seen_by = M.sum(axis=0)
    shared = (M @ M.T)[np.triu_indices(M.shape[0], 1)]

    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax.imshow(M, aspect="auto", cmap=matplotlib.colors.ListedColormap([SURFACE, BLUE]), interpolation="nearest")
    ax.set_ylabel("Investor (1–100)")
    ax.set_yticks([0, 24, 49, 74, 99], ["1", "25", "50", "75", "100"])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    # theme brackets under the grid
    groups = [group[c] for c in chars]
    starts = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]] + [len(groups)]
    ax.set_xticks([])
    for a, b in zip(starts[:-1], starts[1:]):
        ax.annotate("", xy=(a - 0.5, 104), xytext=(b - 0.5, 104), xycoords="data", annotation_clip=False,
                    arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", color=INK2, lw=1))
        ax.text((a + b - 1) / 2, 109, f"{groups[a]}\n({b - a})", ha="center", va="top", fontsize=9.5,
                color=INK2, clip_on=False)
        if a > 0:
            ax.axvline(a - 0.5, color=GRID, lw=1)
    ax.set_title(f"Each row is an investor; blue = characteristic it sees (76 of {len(chars)}, drawn at random, "
                 "fixed for all years)", loc="left", fontsize=12.5)

    tx = fig.add_subplot(gs[1])
    tx.axis("off")
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    tx.text(0.04, 0.97, "Inside one investor", fontsize=14, fontweight="bold", va="top")
    steps = [
        ("Information", "a random 76 of the 153 characteristics,\nranked to [-1, 1] each month"),
        ("Training data", "past 120 months of stocks and their\nnext-month returns (~300k rows);\nrefit every 12 months"),
        ("Model", "random forest of 2,000 trees,\neach depth 3 (at most 8 leaves)"),
        ("Randomness", "each tree sees a random 5% of rows;\neach split tries a random 5%\nof the 76 characteristics (~4)"),
        ("Forecast", "average of the 2,000 trees:\none expected return per stock"),
    ]
    y = 0.89
    for head, body in steps:
        tx.add_patch(FancyBboxPatch((0.04, y - 0.158), 0.92, 0.163, boxstyle="round,pad=0.008,rounding_size=0.02",
                                    fc="#f0efec", ec=GRID, lw=1))
        tx.text(0.08, y - 0.012, head, fontsize=11.5, fontweight="bold", va="top")
        tx.text(0.08, y - 0.047, body, fontsize=10, color=INK2, va="top", linespacing=1.3)
        y -= 0.18
    fig.suptitle("How the 100 \"investors\" are made: same model and data, different information",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    footnote(fig, f"Our replication (paper's settings). Each characteristic is seen by {seen_by.mean():.0f} investors "
                  f"on average (range {seen_by.min():.0f}–{seen_by.max():.0f}); two investors share "
                  f"{shared.mean():.0f} of their 76 characteristics on average (range {shared.min():.0f}–"
                  f"{shared.max():.0f}). Themes from JKP.")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.2)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = tm.load_panel(ROOT / "data/mfd_full/mfd_quantiles.pq")
    factors = tm.ff_factors()
    figs = {"paper_1_construction": construction(), "paper_1b_investors": investors(),
            "paper_2_example": example(panel),
            "paper_3_headline": headline(panel, factors), "paper_4_who": who(panel),
            "paper_5_mechanism": mechanism()}
    for name, f in figs.items():
        f.savefig(OUT / f"{name}.png", dpi=200)
        plt.close(f)
    print(f"Wrote {len(figs)} figures to {OUT}")


if __name__ == "__main__":
    main()
