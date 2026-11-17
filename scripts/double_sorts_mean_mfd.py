#!/usr/bin/env python3
"""Conditional 5x5 double sorts of next-month excess returns on the ensemble
mean forecast and MFD (std of the 100 forecasts), both orders:

  A  mean quintiles first, then MFD quintiles within each mean quintile
  B  MFD quintiles first, then mean quintiles within each MFD quintile

Each cell: time-series average of the monthly portfolio return (% per month),
EW and VW (market equity at formation). Bottom row: second-sort Q5-Q1 within
each first-sort quintile, with Newey-West t (6 lags) and FF6 alpha. Right
column "Avg": average across first-sort quintiles (the second variable's
effect holding the first fixed). Sample 1976-08..2022-11, paper universe.
Ranks use the authors' dense-percentile convention (test_miller.decile_sort).

Usage (project root, mfd env): python scripts/double_sorts_mean_mfd.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tm", str(Path(__file__).with_name("test_miller.py")))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
OUT = ROOT / "results/double_sorts"
NQ = 5
LABEL = {"mean": "mean forecast", "std": "MFD"}
SHORT = {"mean": "Mean", "std": "MFD"}

# diverging red (negative) <-> gray midpoint <-> blue (positive), reference palette
CMAP = LinearSegmentedColormap.from_list(
    "div", ["#a3302f", "#e34948", "#f0efec", "#5598e7", "#184f95"])
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def qrank(s):
    pct = s.rank(method="dense", pct=True)
    pct = pct.where(pct != pct.min(), 1e-8)
    return np.ceil(pct * NQ).clip(lower=1).astype(int)


def double_sort(df, first, second, factors):
    d = df.dropna(subset=[first, second, "ret_exc_lead1m", "me"]).copy()
    d["q1"] = d.groupby("month")[first].transform(qrank)
    d["q2"] = d.groupby(["month", "q1"])[second].transform(qrank)
    fac = factors.copy() * 100
    fac.index = fac.index - 1  # factor month t+1 aligned with formation month t
    out = {}
    for w in ("ew", "vw"):
        if w == "ew":
            r = d.groupby(["month", "q1", "q2"])["ret_exc_lead1m"].mean()
        else:
            g = d.assign(rw=d.ret_exc_lead1m * d.me)
            r = g.groupby(["month", "q1", "q2"])["rw"].sum() / g.groupby(["month", "q1", "q2"])["me"].sum()
        r = (r * 100).unstack(["q1", "q2"])  # month x (q1, q2)
        cells = pd.DataFrame(index=range(1, NQ + 1), columns=list(range(1, NQ + 1)) + ["Avg"], dtype=float)
        hl = {}
        for q1 in range(1, NQ + 1):
            for q2 in range(1, NQ + 1):
                cells.loc[q2, q1] = r[(q1, q2)].mean()
            hl[q1] = r[(q1, NQ)] - r[(q1, 1)]
        for q2 in range(1, NQ + 1):
            cells.loc[q2, "Avg"] = sum(r[(q1, q2)] for q1 in range(1, NQ + 1)).div(NQ).mean()
        hl["Avg"] = sum(hl[q1] for q1 in range(1, NQ + 1)) / NQ
        stats = {}
        for k, s in hl.items():
            m, t = tm.nw_mean(s)
            a, ta = tm.nw_alpha(s, fac)
            stats[k] = {"ret": m, "t": t, "alpha_FF6": a, "t_alpha": ta}
        out[w] = (cells, pd.DataFrame(stats).T, len(r))
    return out


def panel_plot(ax, cells, hl, first, second, norm):
    data = cells.to_numpy(float)
    ax.imshow(data, cmap=CMAP, norm=norm, aspect="auto", extent=(-0.5, NQ + 0.5, NQ - 0.5, -0.5))
    ax.axvline(NQ - 0.5, color=SURF, lw=6)  # gap before Avg column
    for i in range(NQ):
        for j in range(NQ + 1):
            v = data[i, j]
            dark = abs(norm(v) - 0.5) > 0.32
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if dark else INK, fontweight="bold" if j == NQ else "normal")
    # H-L row as text under the grid (different statistic, not color-encoded)
    y_hl, y_a = NQ + 0.35, NQ + 1.1
    for j, k in enumerate(list(range(1, NQ + 1)) + ["Avg"]):
        h = hl.loc[k]
        ax.text(j, y_hl, f"{h.ret:+.2f}\n(t {h.t:+.2f})", ha="center", va="center", fontsize=9.5,
                color=INK, fontweight="bold" if k == "Avg" else "normal")
        ax.text(j, y_a, f"{h.alpha_FF6:+.2f}\n(t {h.t_alpha:+.2f})", ha="center", va="center",
                fontsize=8.5, color=INK2)
    f, s = SHORT[first], SHORT[second]
    ax.set_xlim(-0.5, NQ + 0.5)
    ax.set_ylim(NQ + 1.5, -0.5)
    ax.set_xticks(range(NQ + 1), [f"{f} Q1\n(low)"] + [f"{f} Q{q}" for q in range(2, NQ)] +
                  [f"{f} Q{NQ}\n(high)", f"Avg over\n{f} Q1–Q{NQ}"])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(list(range(NQ)) + [y_hl, y_a],
                  [f"{s} Q1 (low)"] + [f"{s} Q{q}" for q in range(2, NQ)] + [f"{s} Q{NQ} (high)",
                   f"{s} Q5−Q1\nreturn (t)", f"{s} Q5−Q1\nFF6 α (t)"])
    question = {"mean": "Does MFD predict returns, holding the mean forecast fixed?",
                "std": "Does the mean forecast predict returns, holding MFD fixed?"}[first]
    ax.annotate(question, (0, 1), xycoords="axes fraction", xytext=(0, 78), textcoords="offset points",
                ha="left", va="bottom", fontsize=12.5, fontweight="bold", color=INK)
    ax.annotate(f"Columns: {LABEL[first]} quintiles (sorted first, all stocks)\n"
                f"Rows: {LABEL[second]} quintiles within each column",
                (0, 1), xycoords="axes fraction", xytext=(0, 42), textcoords="offset points",
                ha="left", va="bottom", fontsize=10, color=INK2)
    ax.tick_params(length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = tm.load_panel(ROOT / "data/mfd_full/mfd_quantiles.pq")
    panel = panel[panel.month <= tm.PAPER_END]
    factors = tm.ff_factors()
    res = {("mean", "std"): double_sort(panel, "mean", "std", factors),
           ("std", "mean"): double_sort(panel, "std", "mean", factors)}

    lines = [f"Conditional 5x5 double sorts, {panel.month.min()}..{panel.month.max()}, "
             f"next-month excess return % per month; Q5-Q1 with NW(6) t and FF6 alpha"]
    for (first, second), r in res.items():
        for w in ("ew", "vw"):
            cells, hl, n = r[w]
            tag = f"{first}_then_{second}_{w}"
            cells.to_csv(OUT / f"{tag}_cells.csv")
            hl.to_csv(OUT / f"{tag}_hl.csv")
            lines.append(f"\n=== {LABEL[first]} first, then {LABEL[second]} [{w.upper()}], {n} months ===")
            lines.append("rows: " + LABEL[second] + " quintile; columns: " + LABEL[first] + " quintile")
            lines.append(cells.round(2).to_string())
            lines.append("Q5-Q1  " + "  ".join(f"{k}: {h.ret:+.2f} (t {h.t:+.2f}, α {h.alpha_FF6:+.2f}, t {h.t_alpha:+.2f})"
                                             for k, h in hl.iterrows()))
    txt = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(txt)
    print(txt)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF})
    for w in ("ew", "vw"):
        allv = np.concatenate([r[w][0].to_numpy(float).ravel() for r in res.values()])
        lim = np.nanmax(np.abs(allv))
        norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
        fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw={"wspace": 0.3})
        for ax, ((first, second), r) in zip(axes, res.items()):
            cells, hl, n = r[w]
            panel_plot(ax, cells, hl, first, second, norm)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes, shrink=0.6, pad=0.02)
        cb.set_label("next-month excess return, % per month", color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK2, length=0)
        fig.suptitle(f"Next-month returns in conditional double sorts ({'equal' if w == 'ew' else 'value'}-weighted), "
                     f"{panel.month.min()} to {panel.month.max()}", x=0.02, y=1.1, ha="left", fontsize=13.5, color=INK)
        fig.text(0.02, 0.01, "Cells: average monthly return. Q5−Q1: second-sort spread within each first-sort "
                 "quintile, Newey-West t (6 lags); FF6 α = FF5 + momentum alpha. Avg: mean across first-sort quintiles.",
                 fontsize=8.5, color=INK2)
        fig.savefig(OUT / f"double_sorts_{w}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
