#!/usr/bin/env python3
"""Slide figures from results/paper_tables/<variant>/ (made by scripts/paper_tables.py).

Per variant (forest, ridge, and any other variant folder present):
  <variant>_deciles.png   decile excess returns: plain MFD sort vs within mean-forecast quintiles
  <variant>_alphas.png    H-L excess return and alphas (CAPM, FF6, q, SY, DHS) with 95% CIs
  <variant>_fama_macbeth.png  MFD slope, paper's specs (1)-(3), without vs with mean forecast
Across variants:
  model_types.png         MFD slope without vs with its own mean forecast, one row per model type
                          (add a variant by running paper_tables.py --variant X; it appears here)

Colour encodes one thing everywhere: without (blue, filled circle) vs with the mean-forecast
control (orange, hollow diamond) -- slots 1-2 of the dataviz reference palette, also
distinguished by marker shape. Definitions: docs/definitions.md.

Usage (project root, mfd env): python scripts/plot_results.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/paper_tables"
OUT = ROOT / "results/figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_NO, C_MEAN = "#2a78d6", "#eb6834"   # palette slots 1-2
LBL_NO, LBL_MEAN = "MFD alone", "Controlling for the mean forecast"
NAMES = {"forest": "Random forest (paper)", "ridge": "Ridge"}
MODELS = ["XRet", "CAPM", "FF6", "HXZ", "SY", "DHS"]
MODEL_LABELS = ["Excess\nreturn", "CAPM", "FF6", "q-factor", "SY", "DHS"]
PAPER_FM = {"(1)": -0.35, "(2)": -0.31, "(3)": -0.24}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def ci(coef, t):
    se = np.abs(coef / t) if t != 0 else np.nan
    return 1.96 * se


def zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK2, lw=1, zorder=1)


def deciles(v):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, w, title in zip(axes, ("vw", "ew"), ("Value-weighted", "Equal-weighted")):
        a = pd.read_csv(IN / v / f"table3_{w}.csv", index_col=0).drop(index="H-L")
        b = pd.read_csv(IN / v / f"table3_mean_{w}.csv", index_col=0).drop(index="H-L")
        x = np.arange(1, 11)
        ax.plot(x, a["XRet"], color=C_NO, lw=2, marker="o", ms=8, mec=SURFACE, mew=1.5, label=LBL_NO, zorder=3)
        ax.plot(x, b["XRet"], color=C_MEAN, lw=2, marker="D", ms=8, mfc=SURFACE, mew=2, label=LBL_MEAN, zorder=3)
        zero(ax)
        ax.set_xticks(x, ["Low"] + [str(i) for i in range(2, 10)] + ["High"])
        ax.set_xlabel("MFD decile")
        hl_a = a["XRet"].iloc[-1] - a["XRet"].iloc[0]
        hl_b = b["XRet"].iloc[-1] - b["XRet"].iloc[0]
        ax.set_title(f"{title}   High-minus-low: {hl_a:+.2f} → {hl_b:+.2f}", loc="left")
        ax.set_xlim(0.5, 10.5)
    axes[0].set_ylabel("Next-month excess return (% per month)")
    axes[0].legend(loc="lower left")
    shrink = []
    for w in ("vw", "ew"):
        a0 = pd.read_csv(IN / v / f"table3_{w}.csv", index_col=0).loc["H-L", "XRet"]
        b0 = pd.read_csv(IN / v / f"table3_mean_{w}.csv", index_col=0).loc["H-L", "XRet"]
        shrink.append(1 - b0 / a0)
    fig.suptitle(f"{NAMES.get(v, v)} MFD: holding expected return fixed shrinks the high-minus-low spread by "
                 f"{min(shrink):.0%}–{max(shrink):.0%}", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.005, "Aug 1976 – Nov 2022. Orange: MFD deciles within mean-forecast quintiles "
             "(dependent 5×10 sort), averaged.", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def alphas(v):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    x = np.arange(len(MODELS))
    for ax, w, title in zip(axes, ("vw", "ew"), ("Value-weighted", "Equal-weighted")):
        for f, col, mk, dx, lab in ((f"table3_{w}.csv", C_NO, "o", -0.13, LBL_NO),
                                    (f"table3_mean_{w}.csv", C_MEAN, "D", 0.13, LBL_MEAN)):
            r = pd.read_csv(IN / v / f, index_col=0).loc["H-L"]
            est = np.array([r[m] for m in MODELS])
            err = np.array([ci(r[m], r["t" if m == "XRet" else "t_" + m]) for m in MODELS])
            ax.errorbar(x + dx, est, yerr=err, fmt=mk, color=col, ms=8, lw=2, capsize=0,
                        mfc=col if mk == "o" else SURFACE, mew=2, label=lab, zorder=3)
        zero(ax)
        ax.set_xticks(x, MODEL_LABELS)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("High-minus-low MFD (% per month)")
    axes[0].legend(loc="lower left")
    fig.suptitle(f"{NAMES.get(v, v)} MFD: high-minus-low return and alphas, with 95% intervals",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.005, "Aug 1976 – Nov 2022 (q-factor to 2021, SY to 2016, DHS to 2022). "
             "Factor models defined in docs/definitions.md.", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def fama_macbeth(v):
    t6 = pd.read_csv(IN / v / "table6.csv")
    t6 = t6[t6["var"] == "std"]
    specs = ["(1)", "(2)", "(3)"]
    spec_lbl = ["(1) BETA, SIZE,\nBM, MOM", "(2) + AG, OP", "(3) all 12\ncontrols"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(3)
    for wm, col, mk, dx, lab in ((False, C_NO, "o", -0.12, LBL_NO), (True, C_MEAN, "D", 0.12, LBL_MEAN)):
        s = t6[t6.with_mean == wm].set_index("spec").loc[specs]
        ax.errorbar(x + dx, s.coef, yerr=[ci(c, t) for c, t in zip(s.coef, s.t)], fmt=mk, color=col, ms=9,
                    lw=2, mfc=col if mk == "o" else SURFACE, mew=2, label=lab, zorder=3)
    if v == "forest":
        ax.scatter(x - 0.12, [PAPER_FM[s] for s in specs], marker="_", s=500, color=INK, lw=2,
                   label="Paper's estimate", zorder=4)
    zero(ax)
    ax.set_xticks(x, spec_lbl)
    ax.set_ylabel("MFD slope (% per month per 1 SD)")
    ax.legend(loc="lower right")
    ax.set_title(f"{NAMES.get(v, v)} MFD in Fama-MacBeth regressions, with 95% intervals", loc="left")
    fig.text(0.01, 0.005, "Aug 1976 – Nov 2022; paper's estimator (linearmodels, Newey-West). "
             "Controls defined in docs/definitions.md.", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def model_types(variants):
    rows = []
    for v in variants:
        t6 = pd.read_csv(IN / v / "table6.csv")
        for spec in ("(1)", "(3)"):
            for wm in (False, True):
                r = t6[(t6["var"] == "std") & (t6.spec == spec) & (t6.with_mean == wm)].iloc[0]
                rows.append({"v": v, "spec": spec, "wm": wm, "coef": r.coef, "t": r.t})
    d = pd.DataFrame(rows)
    order = [v for v in ("forest", "ridge") if v in variants] + [v for v in variants if v not in ("forest", "ridge")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 2.4 + 1.3 * len(order)), sharex=True, sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, spec, title in zip(axes, ("(1)", "(3)"), ("Spec (1): BETA, SIZE, BM, MOM", "Spec (3): all 12 controls")):
        for wm, col, mk, dy, lab in ((False, C_NO, "o", 0.12, LBL_NO), (True, C_MEAN, "D", -0.12, LBL_MEAN)):
            s = d[(d.spec == spec) & (d.wm == wm)].set_index("v").loc[order]
            ax.errorbar(s.coef, y + dy, xerr=[ci(c, t) for c, t in zip(s.coef, s.t)], fmt=mk, color=col, ms=9,
                        lw=2, mfc=col if mk == "o" else SURFACE, mew=2, label=lab, zorder=3)
            for yy, c, t in zip(y + dy, s.coef, s.t):
                ax.annotate(f"{c:+.2f} (t {t:+.1f})", (c + ci(c, t), yy), xytext=(6, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=10, color=INK2)
        zero(ax, "x")
        ax.set_yticks(y, [NAMES.get(v, v) for v in order])
        ax.set_title(title, loc="left")
        ax.set_xlabel("MFD slope (% per month per 1 SD)")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlim(right=0.32)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2)
    fig.suptitle("Across model types, MFD adds nothing once the same investors' mean forecast is controlled",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    variants = sorted(p.name for p in IN.iterdir() if (p / "table6.csv").exists())
    for v in variants:
        for name, fn in (("deciles", deciles), ("alphas", alphas), ("fama_macbeth", fama_macbeth)):
            f = fn(v)
            f.savefig(OUT / f"{v}_{name}.png", dpi=200)
            plt.close(f)
    f = model_types(variants)
    f.savefig(OUT / "model_types.png", dpi=200)
    plt.close(f)
    print(f"Wrote {len(variants) * 3 + 1} figures to {OUT} for variants {variants}")


if __name__ == "__main__":
    main()
