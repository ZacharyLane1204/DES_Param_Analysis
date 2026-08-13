"""
model_comparison_suite.py  —  SNe Ia Cosmology Pipeline
===========================================================
Orchestrates the headline compare_runs.py comparisons across a whole
category-selection pass, once experiment_runner.py / extra_runners.py
have produced the candidate pkls. This script does NOT fit anything
itself -- every pkl path below must already exist (see run.pkl_path_for /
your registry CSVs for where they live). Its only job is running the
right compare_runs calls in the right order and collecting them into one
report, so you don't hand-write the same compare_two_runs boilerplate
per category.

Comparisons run, per category in CATEGORY_SHORTLISTS
--------------------------------------------------------
  1. baseline vs. winner               (compare_two_runs)
  2. winner vs. runner-up              (compare_two_runs, only if >= 2
                                        candidates given for that category)
  3. baseline + shortlist tension grid (tension_matrix, up to the first 3
                                        candidates + baseline)

...then, once, across categories
-----------------------------------
  4. combined-winners model vs. baseline
  5. combined-winners model vs. EACH individual category winner

SETUP -- edit the paths below
--------------------------------
Every entry is a "<...>_results.pkl" path -- from pkl_path_for(run_name,
cfg) if you have the config in hand, or just the file under whatever
output_dir/run_tag you used. COMBINED_PKL is whatever your final
"everything that won" fit is -- e.g. the fullest entry out of
combo_ablation_checks.py, or your own final combined-model run.

Usage
-----
  python model_comparison_suite.py
  python model_comparison_suite.py --only sn_colour,host_colour

or:
  from model_comparison_suite import run_comparison_suite
  report = run_comparison_suite()
"""

import argparse

import pandas as pd

from compare_runs import compare_two_runs, tension_matrix

# ===========================================================================
# 1. PATHS  —  fill these in once experiment_runner.py / extra_runners.py
#    have run and you know which pkl is which.
# ===========================================================================

BASELINE_PKL = "Plots/baseline_results.pkl"

# category -> ordered list of (label, pkl_path), BEST FIRST. Only the
# first two entries are used for "winner vs runner-up"; only the first
# three (plus baseline) go into the shortlist tension_matrix grid.
CATEGORY_SHORTLISTS = {
    "sn_colour": [
        ("softbroken_sntau", "Plots/sn_col_model/sncolour_softbroken_sntau_results.pkl"),
        ("tanh_c0_sntau",    "Plots/sn_col_model/sncolour_tanh_c0_sntau_results.pkl"),
    ],
    "host_colour": [
        ("tanh_mass_step",  "Plots/host_col_model/host_colour_tanh_mass_step_results.pkl"),
    ],
    "host_mass": [
        ("mass_linear",     "Plots/mass/mass_linear_results.pkl"),
    ],
    "host_sSFR": [
        ("ssfr_tanh_F0ftau", "Plots/ssfr/ssfr_tanh_F0ftau_results.pkl"),
    ],
    "z_evolve": [
        ("gamma_z_power",   "Plots/evolution/gamma_z_power_results.pkl"),
    ],
    "stretch": [
        ("stretch_softbroken_x1tau", "Plots/stretch/stretch_softbroken_x1tau_results.pkl"),
    ],
    "interaction": [
        ("gamma_alpha",     "Plots/interaction/gamma_alpha_results.pkl"),
    ],
}

# The combined "everything that won" model -- produced separately (e.g. the
# fullest entry in combo_ablation_checks.py's COMBOS, or your own final fit).
COMBINED_PKL = "Plots/combo/interaction_sn_colour_host_results.pkl"

# category -> pkl for "combined vs. each individual winner". Defaults to
# each category's [0] shortlist entry, but kept separate/explicit in case
# you want to point at a different pkl than the shortlist winner.
INDIVIDUAL_WINNER_PKLS = {cat: candidates[0][1]
                          for cat, candidates in CATEGORY_SHORTLISTS.items()}


def run_comparison_suite(baseline_pkl=None, shortlists=None, combined_pkl=None,
                         individual_winners=None, output_prefix="model_comparison",
                         kde_max_dims=5, only=None):
    """
    Parameters
    ----------
    baseline_pkl       : defaults to BASELINE_PKL.
    shortlists         : defaults to CATEGORY_SHORTLISTS.
    combined_pkl        : defaults to COMBINED_PKL. Pass None/"" to skip
        steps 4-5 entirely (e.g. before you've fit the combined model yet).
    individual_winners  : defaults to INDIVIDUAL_WINNER_PKLS.
    output_prefix       : basename for every compare_two_runs/tension_matrix
        output file and the master summary CSV.
    kde_max_dims        : passed through to every compare_two_runs call.
    only                : optional iterable of category names to restrict to.

    Returns
    -------
    pandas.DataFrame, one row per comparison, with a "comparison" column
    ("baseline_vs_winner" | "winner_vs_runnerup" | "combined_vs_baseline" |
    "combined_vs_individual_winner") and every compare_two_runs summary
    field. Also saved to "<output_prefix>_summary.csv".
    """
    baseline_pkl        = baseline_pkl or BASELINE_PKL
    shortlists          = shortlists if shortlists is not None else CATEGORY_SHORTLISTS
    combined_pkl         = combined_pkl if combined_pkl is not None else COMBINED_PKL
    individual_winners   = (individual_winners if individual_winners is not None
                            else INDIVIDUAL_WINNER_PKLS)

    categories = list(shortlists.keys())
    if only is not None:
        only = set(only)
        categories = [c for c in categories if c in only]

    rows = []

    for category in categories:
        candidates = shortlists[category]
        if not candidates:
            continue
        winner_label, winner_pkl = candidates[0]

        # ---- 1. baseline vs. winner ----
        print(f"\n{'#'*60}\n# {category}: baseline vs. {winner_label}\n{'#'*60}")
        summary = compare_two_runs(
            baseline_pkl, winner_pkl,
            output_prefix=f"{output_prefix}_{category}_baseline_vs_winner",
            kde_max_dims=kde_max_dims)
        rows.append({"comparison": "baseline_vs_winner", "category": category,
                    "run_1": "baseline", "run_2": winner_label, **summary})

        # ---- 2. winner vs. runner-up ----
        if len(candidates) >= 2:
            runnerup_label, runnerup_pkl = candidates[1]
            print(f"\n{'#'*60}\n# {category}: {winner_label} vs. "
                 f"{runnerup_label}\n{'#'*60}")
            summary = compare_two_runs(
                winner_pkl, runnerup_pkl,
                output_prefix=f"{output_prefix}_{category}_winner_vs_runnerup",
                kde_max_dims=kde_max_dims)
            rows.append({"comparison": "winner_vs_runnerup", "category": category,
                        "run_1": winner_label, "run_2": runnerup_label, **summary})

        # ---- 3. baseline + shortlist tension grid ----
        grid_candidates = candidates[:3]
        labels = ["baseline"] + [c[0] for c in grid_candidates]
        paths  = [baseline_pkl] + [c[1] for c in grid_candidates]
        print(f"\n{'#'*60}\n# {category}: shortlist tension grid {labels}\n{'#'*60}")
        tension_matrix(paths, labels=labels,
                       output_prefix=f"{output_prefix}_{category}_shortlist_matrix")

    # ---- 4 & 5. combined vs. baseline, combined vs. each individual winner ----
    if combined_pkl:
        print(f"\n{'#'*60}\n# combined winners vs. baseline\n{'#'*60}")
        summary = compare_two_runs(
            baseline_pkl, combined_pkl,
            output_prefix=f"{output_prefix}_combined_vs_baseline",
            kde_max_dims=kde_max_dims)
        rows.append({"comparison": "combined_vs_baseline", "category": "combined",
                    "run_1": "baseline", "run_2": "combined", **summary})

        for category, winner_pkl in individual_winners.items():
            if only is not None and category not in only:
                continue
            print(f"\n{'#'*60}\n# combined winners vs. {category} "
                 f"individual winner\n{'#'*60}")
            summary = compare_two_runs(
                winner_pkl, combined_pkl,
                output_prefix=f"{output_prefix}_combined_vs_{category}",
                kde_max_dims=kde_max_dims)
            rows.append({"comparison": "combined_vs_individual_winner",
                        "category": category, "run_1": category,
                        "run_2": "combined", **summary})
    else:
        print("\ncombined_pkl not set -- skipping combined-vs-baseline / "
             "combined-vs-individual-winner steps (4-5).")

    report = pd.DataFrame(rows)
    report.to_csv(f"{output_prefix}_summary.csv", index=False)
    print(f"\nModel comparison suite summary saved: {output_prefix}_summary.csv")
    return report


def _parse_args():
    p = argparse.ArgumentParser(
        description="Orchestrate baseline-vs-winner / winner-vs-runnerup / "
                    "shortlist tension-grid / combined-vs-everything "
                    "compare_runs.py comparisons across every category.")
    p.add_argument("--output-prefix", default="model_comparison")
    p.add_argument("--kde-max-dims", type=int, default=5)
    p.add_argument("--only", default=None,
                   help="Comma-separated category names to run (default: "
                        "every category in CATEGORY_SHORTLISTS).")
    p.add_argument("--no-combined", action="store_true",
                   help="Skip steps 4-5 (combined-model comparisons).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    only = args.only.split(",") if args.only else None
    run_comparison_suite(output_prefix=args.output_prefix,
                         kde_max_dims=args.kde_max_dims, only=only,
                         combined_pkl=(None if args.no_combined else COMBINED_PKL))