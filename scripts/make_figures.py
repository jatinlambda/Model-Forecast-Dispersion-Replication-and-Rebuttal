#!/usr/bin/env python3
"""Slide figures from finished results (no re-estimation; reads results/*).

  fig1_replication.png     MFD decile sorts: ours vs the authors' published MFD
  fig2_control_ladder.png  MFD slope under 8 control sets, alone / + mean / + mean, mean^2
  fig3_horizons.png        MFD slope for returns 1..12 months ahead, alone vs + mean
  fig4_ridge_vs_forest.png forest and ridge MFD, alone / + ridge mean / + forest mean
  fig6_short_interest.png  MFD slope by short-interest tercile, alone vs + mean
  fig7_who_disagrees.png   top-MFD share by mean-forecast bin; characteristics of high- vs low-MFD stocks

Coefficients: % per month per 1 SD (z-scored Fama-MacBeth), 95% intervals from
the Newey-West (6 lags) t-stats as coef +/- 1.96 * |coef / t|.

Usage (project root, mfd env): python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"

# reference palette (dataviz skill, validated categorical slots 1-3) + text tokens
C = ["#2a78d6", "#eb6834", "#1baf7a"]
MK = ["o", "s", "D"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
})


def ci(coef, t):
    return 1.96 * np.abs(coef / t)


def header(fig, title, subtitle, y=1.0):
    fig.text(0.01, y + 0.06, title, ha="left", va="bottom", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, y + 0.015, subtitle, ha="left", va="bottom", fontsize=10.5, color=INK2)


def footnote(fig, text):
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, color=MUTED)


def dot_whisker(ax, rows, series, get, label_t=True):
    """rows: list of row labels (top to bottom); series: list of names; get(row, s) -> (coef, t) or None."""
    n = len(series)
    off = np.linspace(-0.22, 0.22, n) if n > 1 else [0]
    ax.axvline(0, color=INK2, lw=1)
    for j, s in enumerate(series):
        for i, r in enumerate(rows):
            v = get(r, s)
            if v is None:
                continue
            b, t = v
            y = i + off[j]
            e = ci(b, t)
            ax.plot([b - e, b + e], [y, y], color=C[j], lw=2, solid_capstyle="round", zorder=2)
            ax.plot(b, y, MK[j], color=C[j], ms=8, mec=SURF, mew=1.5, zorder=3)
            if label_t:
                ax.text(b + e + 0.008, y, f"t {t:+.1f}", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], color=C[j], marker=MK[j], lw=2, ms=7, mec=SURF, label=s) for j, s in enumerate(series)]
    return handles


# ----------------------------------------------------------------------------- fig 1

def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True, gridspec_kw={"wspace": 0.08})
    x = np.arange(1, 11)
    alphas = {}
    for ax, w in zip(axes, ("vw", "ew")):
        mine = pd.read_csv(RES / f"miller/step0_ours_{w}.csv", index_col=0)
        auth = pd.read_csv(RES / f"miller/step0_authors_mfd.csv_{w}.csv", index_col=0)
        for j, (d, lab) in enumerate([(mine, "My replication"), (auth, "Authors' published MFD")]):
            ax.bar(x + (j - 0.5) * 0.38, d.loc[[str(i) for i in x], "ret"], width=0.36, color=C[j],
                   label=lab, edgecolor=SURF, linewidth=1)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x, [str(i) for i in x])
        ax.set_xlabel("MFD decile  (1 = least disagreement, 10 = most)")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(w == "vw")
        o, a = mine.loc["H-L"], auth.loc["H-L"]
        alphas[w] = (o.alpha_FF6, a.alpha_FF6)
        ax.set_title("Value-weighted" if w == "vw" else "Equal-weighted", loc="left", fontsize=13,
                     color=INK, fontweight="bold", pad=26)
        hl = f"Decile 10 − 1:  mine {o.ret:.2f}% (t {o.t:.1f}),  authors {a.ret:.2f}% (t {a.t:.1f})"
        ax.text(0, 1.02, hl.replace("-", "−"), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10.5, color=INK2)
    axes[0].set_ylabel("next-month excess return, % per month")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.005, 1.0),
               ncol=2, frameon=False, fontsize=11, handlelength=1.2)
    fig.text(0.01, 1.08, "My replication matches the paper: high-MFD stocks underperform by about 1% a month",
             ha="left", va="bottom", fontsize=15, fontweight="bold", color=INK)
    footnote(fig, f"Monthly decile portfolios, 1976–2022; Newey-West t, 6 lags. FF6 alphas of decile 10 − 1: "
                  f"value-weighted {alphas['vw'][0]:+.2f} (mine) vs {alphas['vw'][1]:+.2f} (authors), "
                  f"equal-weighted {alphas['ew'][0]:+.2f} vs {alphas['ew'][1]:+.2f}.")
    fig.savefig(OUT / "fig1_replication.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 2

def fig2():
    d = pd.read_csv(RES / "mfd_mean_controls/common_sample/fama_macbeth_by_controls.csv")
    d = d[d["var"] == "std"]
    names = {"none": "No controls",
             "FF4": "Size, B/M,\nmomentum,\nreversal",
             "paper12": "Paper's 12\ncontrols",
             "all153": "All 153\ncharacteristics",
             "all153+ind": "All 153\n+ industry\ndummies"}
    order = list(names)
    t_alone = [d[(d.controls == c) & (d.spec == "MFD")].t.iloc[0] for c in order]
    t_mean = [d[(d.controls == c) & (d.spec == "MFD+mean")].t.iloc[0] for c in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhspan(-1.96, 1.96, color="#ecebe7", zorder=0)
    ax.text(len(order) - 0.45, 1.5, "not significant (|t| < 1.96)", ha="right", va="center",
            fontsize=9.5, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    w = 0.3
    for j, (vals, lab) in enumerate([(t_alone, "MFD alone"), (t_mean, "MFD + mean forecast")]):
        xs = x + (j - 0.5) * (w + 0.04)
        ax.bar(xs, vals, width=w, color=C[j], label=lab, zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v - 0.15, f"{v:.1f}", ha="center", va="top", fontsize=9.5, color=INK)
    ax.set_xticks(x, [names[c] for c in order])
    ax.set_ylim(-7.6, 2.2)
    ax.set_ylabel("t-statistic on MFD")
    ax.set_xlabel("")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    ax.set_xlim(-0.6, len(order) - 0.4)
    header(fig, "MFD is significant under every control set,\nbut not once the mean forecast is added",
           "t-statistic on MFD, Fama-MacBeth, 1976 to 2022, same stocks in every pair.", y=0.97)
    footnote(fig, "Paper's 12: SUE, AG, MOM, ILLIQ, OP, IVOL, BETA, SIZE, BM, MAX, TURN, STR. Characteristics are the forests' "
                  "rank-normalized inputs;\nindustries are Fama-French 49. Newey-West t, 6 lags. Adding mean² as well gives t "
                  "between −2.2 and +0.2.")
    fig.savefig(OUT / "fig2_control_ladder.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 3

def fig3():
    h = pd.read_csv(RES / "miller/horizons.csv")
    h = h[h["var"] == "std"]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axhline(0, color=INK2, lw=1)
    for j, (spec, lab) in enumerate([("MFD + controls", "MFD alone"),
                                      ("MFD + mean + controls", "MFD + mean forecast")]):
        g = h[h.spec == spec].sort_values("k")
        e = ci(g.coef, g.t)
        ax.fill_between(g.k, g.coef - e, g.coef + e, color=C[j], alpha=0.14, lw=0)
        ax.plot(g.k, g.coef, color=C[j], lw=2, marker=MK[j], ms=7, mec=SURF, mew=1.5)
        last = g.iloc[-1]
        ax.text(12.25, last.coef, lab, va="center", ha="left", fontsize=10.5, color=INK)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.6, 13.9)
    ax.set_xlabel("return month after the signal (k = 1 is next month)")
    ax.set_ylabel("MFD slope, % per month per 1 SD")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    header(fig, "The mean absorbs MFD at every horizon, not just next month",
           "Fama-MacBeth slope on MFD for the return in month t+k, paper's 12 controls, 1976 to 2022. "
           "Shaded: 95% interval.", y=0.96)
    footnote(fig, "Newey-West t, 6 lags. Signals measured at t; horizon k uses the excess return in month t+k.")
    fig.savefig(OUT / "fig3_horizons.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 4

def fig4():
    d = pd.read_csv(RES / "ridge_mfd/fama_macbeth.csv")
    g = lambda spec, var: tuple(d[(d.spec == spec) & (d["var"] == var)][["coef", "t"]].iloc[0])
    lut = {
        ("Forest MFD", "alone"): g("forest MFD + 12 controls", "rf_std"),
        ("Forest MFD", "+ ridge investors' mean"): g("forest MFD + ridge investors' mean", "rf_std"),
        ("Forest MFD", "+ forests' mean"): g("forest MFD + forests' own mean", "rf_std"),
        ("Ridge MFD", "alone"): g("ridge MFD + 12 controls", "r_std"),
        ("Ridge MFD", "+ ridge investors' mean"): g("+ ridge investors' own mean", "r_std"),
        ("Ridge MFD", "+ forests' mean"): g("+ forests' mean forecast", "r_std"),
    }
    rows = ["Forest MFD", "Ridge MFD"]
    series = ["alone", "+ ridge investors' mean", "+ forests' mean"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    h = dot_whisker(ax, rows, series, lambda r, s: lut[(r, s)])
    ax.set_yticks(range(2), ["Forest MFD\n(paper's 100 random forests)", "Ridge MFD\n(100 ridge models,\nsame feature subsets)"])
    ax.set_xlabel("MFD slope, % per month per 1 SD (95% interval)")
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, title=None)
    header(fig, "Only the forests' own mean removes forest MFD; linear forecasts do not",
           "Fama-MacBeth, paper's 12 controls, 1976-08 to 2022-11, common sample (~1,879 stocks a month).", y=1.0)
    footnote(fig, "Ridge investors see the same random 76 of 153 characteristics as the matching forest investor; "
                  "Newey-West t, 6 lags.")
    fig.savefig(OUT / "fig4_ridge_vs_forest.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 6

def fig6():
    d = pd.read_csv(RES / "miller/step4_constraints.csv")
    d = d[(d.proxy == "SIR") & (d["var"] == "std")]
    terc = ["low SIR (easy)", "mid SIR", "high SIR (hard to short)"]
    rows = ["Low short interest\n(easy to short)", "Mid short interest", "High short interest\n(hard to short)"]
    series = ["MFD alone", "MFD + mean forecast"]
    key = {"MFD alone": "MFD + controls", "MFD + mean forecast": "MFD + mean + controls"}
    lut = {(rows[i], s): tuple(d[(d.tercile == t) & (d.spec == key[s])][["coef", "t"]].iloc[0])
           for i, t in enumerate(terc) for s in series}
    months = int(d.months.iloc[0])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    h = dot_whisker(ax, rows, series, lambda r, s: lut[(r, s)])
    ax.set_xlabel("MFD slope, % per month per 1 SD (95% interval)")
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    header(fig, "Opposite to Miller: what MFD effect there is shows up where shorting is easy",
           f"Fama-MacBeth within monthly short-interest terciles, paper's 12 controls, {months} months "
           "(2006 onward, when short-interest data start).", y=1.0)
    footnote(fig, "Short interest / shares outstanding (Compustat), latest report at least 10 days before month end. "
                  "Miller predicts the strongest effect among hard-to-short stocks. Newey-West t, 6 lags.")
    fig.savefig(OUT / "fig6_short_interest.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig7():
    mb = pd.read_csv(RES / "economic/by_mean_bin.csv", index_col=0)
    pr = pd.read_csv(RES / "economic/profile_by_mfd.csv", index_col=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.42})

    # left: where is disagreement? share of top-MFD-quintile stocks by mean-forecast bin
    x = mb.index.values
    sh = mb["share_top_mfd_q"].values * 100
    cols = [C[1] if i <= 2 else C[0] for i in x]
    a1.bar(x, sh, color=cols, width=0.8, edgecolor=SURF)
    a1.axhline(20, color=INK2, lw=1, ls=(0, (4, 3)))
    a1.text(20.6, 21.5, "20% if unrelated", ha="right", va="bottom", fontsize=9, color=INK2)
    for i in (1, 2):
        a1.text(i, sh[i - 1] + 1.5, f"{sh[i - 1]:.0f}%", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    a1.set_xticks([1, 5, 10, 15, 20], ["1\n(lowest)", "5", "10", "15", "20\n(highest)"])
    a1.set_xlabel("Forests' mean forecast, 5% bins")
    a1.set_ylabel("% of stocks in top MFD quintile")
    a1.set_ylim(0, 80)
    a1.grid(axis="y", color=GRID, lw=0.8); a1.set_axisbelow(True)
    a1.set_title("Disagreement piles up where the forecast is worst", loc="left", fontsize=11.5, color=INK, pad=10)

    # right: who are the high-MFD stocks? characteristic percentile ranks, lowest vs highest MFD quintile
    names = {"IVOL": "Idiosyncratic volatility", "MAX": "Max daily return (lottery)", "BETA": "Market beta",
             "ILLIQ": "Illiquidity", "AG": "Asset growth", "SIZE": "Size", "BM": "Book-to-market",
             "OP": "Operating profitability", "mean_forecast": "Forests' mean forecast"}
    keys = sorted(names, key=lambda k: -(pr.loc[5, k] - pr.loc[1, k]))
    y = np.arange(len(keys))
    lo, hi = pr.loc[1, keys].values * 100, pr.loc[5, keys].values * 100
    a2.axvline(50, color=INK2, lw=1)
    for i in y:
        a2.plot([lo[i], hi[i]], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    a2.plot(lo, y, "o", color=C[0], ms=9, mec=SURF, mew=1.5, zorder=3, label="Lowest MFD quintile")
    a2.plot(hi, y, "s", color=C[1], ms=9, mec=SURF, mew=1.5, zorder=3, label="Highest MFD quintile")
    a2.set_yticks(y, [names[k] for k in keys])
    a2.set_ylim(len(keys) - 0.5, -0.6)
    a2.set_xlim(25, 75)
    a2.set_xlabel("Average percentile rank within month (50 = typical stock)")
    a2.tick_params(axis="y", length=0); a2.spines["left"].set_visible(False)
    a2.grid(axis="x", color=GRID, lw=0.8); a2.set_axisbelow(True)
    a2.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=9.5, handletextpad=0.3)

    header(fig, "High-MFD stocks are the volatile, lottery-like, unprofitable stocks the forests expect to lose money",
           "Left: where disagreement sits in the forecast distribution. Right: what high- vs low-disagreement stocks look like.",
           y=0.95)
    footnote(fig, f"1976-08 to 2022-11, monthly cross-sections averaged over time. Next-month excess return in the lowest 5% "
                  f"mean-forecast bin: {mb.loc[1, 'ret']:+.2f}% vs {mb.loc[20, 'ret']:+.2f}% in the highest.")
    fig.savefig(OUT / "fig7_who_disagrees.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig7_split():
    """fig7a / fig7b: the two panels of fig7 as standalone half-slide figures (no figure header)."""
    mb = pd.read_csv(RES / "economic/by_mean_bin.csv", index_col=0)
    pr = pd.read_csv(RES / "economic/profile_by_mfd.csv", index_col=0)
    with plt.rc_context({"font.size": 13}):
        fig, a1 = plt.subplots(figsize=(8, 5.6))
        x = mb.index.values
        sh = mb["share_top_mfd_q"].values * 100
        a1.bar(x, sh, color=[C[1] if i <= 2 else C[0] for i in x], width=0.8, edgecolor=SURF)
        a1.axhline(20, color=INK2, lw=1.2, ls=(0, (4, 3)))
        a1.text(20.5, 21.5, "20% if unrelated", ha="right", va="bottom", fontsize=12, color=INK2)
        for i in (1, 2):
            a1.text(i, sh[i - 1] + 1.5, f"{sh[i - 1]:.0f}%", ha="center", va="bottom", fontsize=14, color=INK, fontweight="bold")
        a1.set_xticks([1, 5, 10, 15, 20], ["1\nlowest", "5", "10", "15", "20\nhighest"])
        a1.set_xlabel("Forests' mean forecast (5% bins)")
        a1.set_ylabel("% of stocks in the top MFD quintile")
        a1.set_ylim(0, 80)
        a1.grid(axis="y", color=GRID, lw=0.8); a1.set_axisbelow(True)
        fig.savefig(OUT / "fig7a_where_disagreement.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

        names = {"IVOL": "Idiosyncratic\nvolatility", "MAX": "Lottery-like\n(MAX)", "BETA": "Market\nbeta",
                 "ILLIQ": "Illiquidity", "AG": "Asset\ngrowth", "BM": "Book-to-\nmarket", "SIZE": "Size",
                 "OP": "Profitability", "mean_forecast": "Forests'\nmean forecast"}
        keys = sorted(names, key=lambda k: -(pr.loc[5, k] - pr.loc[1, k]))
        x = np.arange(len(keys))
        lo, hi = pr.loc[1, keys].values * 100, pr.loc[5, keys].values * 100
        w = 0.32
        fig, a2 = plt.subplots(figsize=(14, 4.6))
        for vals, off, col, lab in ((lo, -w / 2, C[0], "Lowest MFD quintile"), (hi, w / 2, C[1], "Highest MFD quintile")):
            a2.bar(x + off, vals - 50, bottom=50, width=w, color=col, edgecolor=SURF, label=lab)
        a2.axhline(50, color=INK2, lw=1.2)
        a2.set_xticks(x, [names[k] for k in keys], fontsize=13)
        a2.tick_params(axis="x", length=0)
        a2.set_ylim(25, 75)
        a2.set_ylabel("Percentile rank (50 = typical stock)")
        a2.grid(axis="y", color=GRID, lw=0.8); a2.set_axisbelow(True)
        n_up = int((hi > lo).sum())
        a2.axvspan(n_up - 0.5, len(keys) - 0.5, color=GRID, alpha=0.35, lw=0, zorder=0)
        a2.text((n_up - 1) / 2, 74, "Higher for high-MFD stocks", ha="center", va="top", fontsize=12, color=INK, fontweight="bold")
        a2.text((n_up + len(keys) - 1) / 2, 74, "Lower for high-MFD stocks", ha="center", va="top", fontsize=12, color=INK, fontweight="bold")
        a2.set_xlim(-0.6, len(keys) - 0.4)
        a2.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
        fig.savefig(OUT / "fig7b_who_disagrees.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def fig5_ridge_bars():
    """Slide version (no header): t-stat on MFD, alone vs + that model's own mean, forest vs ridge, spec (1)."""
    names = {"forest": "Random forest\n(paper)", "ridge": "Ridge"}
    t = {}
    for v in names:
        d = pd.read_csv(RES / f"paper_tables/{v}/table6.csv")
        d = d[(d["var"] == "std") & (d.spec == "(1)")]
        t[v] = (d[~d.with_mean].t.iloc[0], d[d.with_mean].t.iloc[0])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(-1.96, 1.96, color="#ecebe7", zorder=0)
    ax.text(len(names) - 0.45, 1.5, "not significant", ha="right", va="center", fontsize=12, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    w = 0.3
    for j, lab in enumerate(["MFD alone", "MFD + that model's own mean forecast"]):
        xs = x + (j - 0.5) * (w + 0.04)
        vals = [t[v][j] for v in names]
        ax.bar(xs, vals, width=w, color=C[j], label=lab, zorder=2)
        for xi, val in zip(xs, vals):
            ax.text(xi, val - 0.15, f"{val:.1f}", ha="center", va="top", fontsize=13, color=INK)
    ax.set_xticks(x, list(names.values()), fontsize=13)
    ax.set_ylim(-7.8, 2.2)
    ax.set_ylabel("t-stat on MFD", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=12)
    ax.set_xlim(-0.6, len(names) - 0.4)
    fig.savefig(OUT / "fig5_ridge_bars.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig7a_heatmap():
    """Slide version (no header): % of each mean-forecast quintile that falls in each MFD quintile."""
    from matplotlib.colors import LinearSegmentedColormap
    ct = pd.read_csv(RES / "economic/mfd_by_mean_quintile.csv", index_col=0)
    ct.columns = ct.columns.astype(int)
    M = ct.loc[[5, 4, 3, 2, 1]].values  # MFD Q5 on top
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#a9c9f0", "#2a78d6", "#123e78"])
    with plt.rc_context({"font.size": 14}):
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=50, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=15,
                        color="white" if v > 30 else INK, fontweight="bold" if v > 30 else "normal")
        ax.set_xticks(range(5), ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"])
        ax.set_yticks(range(5), ["Q5 (most)", "Q4", "Q3", "Q2", "Q1 (least)"])
        ax.set_xlabel("Forests' mean forecast quintile")
        ax.set_ylabel("MFD quintile")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", location="bottom", fraction=0.05, pad=0.2, aspect=40)
        cb.set_label("% of the column's stocks (20% if unrelated)", color=INK2)
        cb.outline.set_visible(False)
        fig.savefig(OUT / "fig7a_heatmap.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig6, fig7, fig7_split, fig5_ridge_bars, fig7a_heatmap):
        f()
        print("done", f.__name__)
