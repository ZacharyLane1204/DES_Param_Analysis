"""
degeneracy_scan.py  —  SNe Ia Cosmology Pipeline
===================================================
Posterior correlation / degeneracy scan.

With xi_mass_col, xi_sSFR_col, xi_sSFR_mass, omega, beta_alpha, gamma_alpha,
beta_gamma, etc. all potentially active together, some parameter pairs are
prone to near-degeneracy (host colour vs. SN colour, eta vs. xi_mass_col,
gamma vs. zeta if mass and sSFR are themselves correlated in your sample,
...). A near-degenerate pair means the DATA cannot actually separate the
two effects -- the posterior mean for either one, taken alone, is
misleading even if its marginal error bar looks fine.

This script computes the weighted posterior correlation matrix for every
active parameter in a saved run (reusing the same dyfunc.mean_and_cov
convention as run.py and compare_runs.py), flags any pair above a
threshold, and renders a correlation heatmap. Run this on your final
candidate model(s) before they go in a table.

Usage
-----
  python degeneracy_scan.py path/to/run_results.pkl
  python degeneracy_scan.py path/to/run_results.pkl --threshold 0.8

or:
  from degeneracy_scan import scan_degeneracies
  report = scan_degeneracies("run_results.pkl")
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dynesty import utils as dyfunc

from run    import load_results
from config import PARAM_DISPLAY


def weighted_correlation_matrix(results, active_names):
    """
    Weighted Pearson correlation matrix over every active parameter, using
    the nested-sampling importance weights (no equal-weight resampling
    needed -- dyfunc.mean_and_cov handles the weighting directly, same
    convention already used elsewhere in this pipeline).

    Returns
    -------
    corr : ndarray (k, k)   weighted correlation matrix, k = len(active_names)
    """
    weights = np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(results.samples, weights)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    denom[denom == 0] = np.nan   # guard a fixed/zero-variance parameter, if any slip through
    corr = cov / denom
    np.fill_diagonal(corr, 1.0)
    return corr


def scan_degeneracies(pkl_path, threshold=0.85, output_prefix=None, make_plot=True):
    """
    Load a saved run, compute its weighted posterior correlation matrix,
    flag any off-diagonal pair with |correlation| >= threshold, print a
    human-readable report, and (optionally) save a correlation heatmap.

    Parameters
    ----------
    pkl_path      : path to a "<...>_results.pkl" saved by run_sampler.
    threshold     : flag pairs at or above this absolute correlation.
        0.85-0.9 is a reasonable "worth a second look" line; anything
        above ~0.95 means the data essentially cannot separate the two
        parameters at all and one of them may need to be fixed or dropped.
    output_prefix : basename for the saved heatmap/CSV; defaults to the
        pkl filename with "_results.pkl" stripped.
    make_plot     : save a correlation heatmap PDF.

    Returns
    -------
    dict:
        corr_matrix  : pandas.DataFrame (active_names x active_names)
        flagged      : list of (name_i, name_j, corr) sorted by |corr| desc
    """
    results, active_names, param_specs, cfg = load_results(pkl_path)

    corr = weighted_correlation_matrix(results, active_names)
    corr_df = pd.DataFrame(corr, index=active_names, columns=active_names)

    flagged = []
    k = len(active_names)
    for i in range(k):
        for j in range(i + 1, k):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) >= threshold:
                flagged.append((active_names[i], active_names[j], float(c)))
    flagged.sort(key=lambda t: -abs(t[2]))

    if output_prefix is None:
        output_prefix = os.path.basename(pkl_path).replace("_results.pkl", "")

    print(f"\n{'='*60}")
    print(f"Degeneracy scan: {pkl_path}")
    print(f"Active parameters ({k}): {active_names}")
    if flagged:
        print(f"\n{len(flagged)} pair(s) at |correlation| >= {threshold}:")
        for name_i, name_j, c in flagged:
            severity = "SEVERE (data cannot separate these)" if abs(c) >= 0.95 \
                      else "worth a closer look"
            print(f"  {name_i:16s} <-> {name_j:16s}   r = {c:+.3f}   [{severity}]")
    else:
        print(f"\nNo pairs at |correlation| >= {threshold} — no strong "
              f"degeneracies flagged in this run's active parameter set.")
    print(f"{'='*60}\n")

    corr_df.to_csv(f"{output_prefix}_correlation_matrix.csv")

    if make_plot:
        display_labels = [PARAM_DISPLAY.get(p, {}).get("label", p) for p in active_names]
        fig, ax = plt.subplots(figsize=(0.6 * k + 3, 0.5 * k + 3))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(k)); ax.set_xticklabels(display_labels, rotation=90, fontsize=8)
        ax.set_yticks(range(k)); ax.set_yticklabels(display_labels, fontsize=8)
        for i in range(k):
            for j in range(k):
                if i != j and np.isfinite(corr[i, j]) and abs(corr[i, j]) >= threshold:
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                            color="black", fontsize=7,
                            fontweight="bold" if abs(corr[i, j]) >= 0.95 else "normal")
        fig.colorbar(im, ax=ax, label="weighted posterior correlation")
        ax.set_title(f"{output_prefix}: posterior correlation "
                     f"(flagged threshold |r| >= {threshold})", fontsize=10)
        fig.tight_layout()
        path = f"{output_prefix}_correlation_heatmap.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Correlation heatmap saved: {path}")

    return {"corr_matrix": corr_df, "flagged": flagged}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Weighted posterior correlation/degeneracy scan for a "
                    "saved dynesty run.")
    p.add_argument("pkl_path", help="Path to a '<...>_results.pkl' file.")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="Flag |correlation| at or above this value (default 0.85).")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    scan_degeneracies(args.pkl_path, threshold=args.threshold,
                      output_prefix=args.output_prefix, make_plot=not args.no_plot)