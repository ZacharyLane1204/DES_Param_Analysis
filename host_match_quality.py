"""
host_match_quality.py  —  SNe Ia Cosmology Pipeline
======================================================
Host-match quality sensitivity check.

Host mass/colour/sSFR corrections are only as good as the SN-to-host
association. A SN assigned to the wrong galaxy poisons exactly the terms
this pipeline is trying to measure -- a DIFFERENT systematic from
CONFIG["mass_cut"], which asks "what is the host's mass" rather than "was
the host correctly identified at all".

This script fits a chosen model twice, identically except for the sample
cut:
  host_quality_cut = "all"     everything (your normal analysis sample)
  host_quality_cut = "strict"  only unambiguous host matches, using
                                 HOST_DDLR / HOST_CONFUSION / HOST_NMATCH
                                 thresholds set in CONFIG (host_ddlr_max,
                                host_confusion_max) -- see config.py and
                                run.py's load_and_filter_data()

then reports the parameter tension between the two fits via
compare_runs.compare_two_runs (both the Gaussian and, for a small enough
shared-parameter set, the nonparametric KDE tension), plus the usual Bayes
factor. If the host correction parameters shift meaningfully between the
two, that is a real systematic worth reporting; if not, this is a clean,
citable robustness statement.

Usage
-----
  python host_match_quality.py --tag best_model

or from another script:

  from host_match_quality import run_host_quality_check
  summary = run_host_quality_check(
      config_overrides={"run_tag": "best_model",
                        "model": {...}, "param_specs": {...}})
"""

import argparse
import copy

from config       import CONFIG
from run          import run_sampler, pkl_path_for
from compare_runs import compare_two_runs

# Host-match quality threshold used by every strict-cut fit launched from
# this script. DDLR is the SN-host separation in units of the host's
# directional light radius, so DDLR <= 2 is the standard "the SN lies
# inside the host's light profile" association criterion -- the value the
# analysis is standardised on (config.py's CONFIG["host_ddlr_max"] carries
# the same number). Anything looser (the old 4.0) admits associations that
# are as likely to be chance projections, which defeats the point of this
# check. Override per-invocation with --ddlr-max if you want to sweep it.
DDLR_MAX = 2.0


def run_host_quality_check(config_overrides=None, output_prefix="host_quality",
                           kde_max_dims=5):
    """
    Parameters
    ----------
    config_overrides : dict, layered on top of CONFIG — this is where you
        specify which model to test (config_overrides["model"] = {...}),
        any param_specs overrides, and a run_tag. Do NOT set
        "host_quality_cut" here; this function controls it directly so the
        two runs are guaranteed to differ ONLY in that one cut.
        "host_ddlr_max" IS respected if you set it — if absent it is
        forced to DDLR_MAX (2.0) rather than left to whatever CONFIG
        happens to carry, so a strict cut launched from here is always the
        DDLR<=2 cut regardless of config drift.
    output_prefix : passed through to compare_two_runs for the tension
        registry / overlay corner plot filenames.
    kde_max_dims : passed through to compare_two_runs.

    Returns
    -------
    dict summary from compare_two_runs (also written to
    "<output_prefix>_tension_registry.csv").
    """
    config_overrides = dict(config_overrides or {})
    base_tag = config_overrides.pop("run_tag", "host_quality_check")
    config_overrides.setdefault("host_ddlr_max", DDLR_MAX)

    pkl_paths = {}
    for cut in ("all", "strict"):
        cfg = copy.deepcopy(CONFIG)
        cfg.update(copy.deepcopy(config_overrides))
        cfg["host_quality_cut"] = cut
        cfg["run_tag"] = f"{base_tag}/host_quality_{cut}"

        print(f"\n{'#'*60}\n# host_quality_cut = '{cut}'"
              f"  (DDLR <= {cfg['host_ddlr_max']})\n{'#'*60}")
        results, sampler, active_names, data, run_name = run_sampler(cfg)
        pkl_paths[cut] = pkl_path_for(run_name, cfg)

    summary = compare_two_runs(
        pkl_paths["all"], pkl_paths["strict"],
        output_prefix=output_prefix, kde_max_dims=kde_max_dims)

    print("\nHost-match quality sensitivity check complete.")
    print(f"  All SNe        : {pkl_paths['all']}")
    print(f"  Strict matches : {pkl_paths['strict']}")
    if summary["gaussian_nsigma"] >= 2.0:
        print(f"  ** Gaussian tension {summary['gaussian_nsigma']} sigma -- "
              f"host-correction parameters shift meaningfully between the "
              f"full sample and the strict-host-match subsample. Treat this "
              f"as a real systematic, not sampling noise. **")
    else:
        print(f"  Gaussian tension {summary['gaussian_nsigma']} sigma -- "
              f"consistent with no host-match systematic at this level.")

    return summary


def _parse_args():
    p = argparse.ArgumentParser(
        description="Host-match quality sensitivity check: fit a model on "
                    "the full sample vs. a strict-host-match subsample and "
                    "report the parameter tension between the two.")
    p.add_argument("--tag", default="host_quality_check",
                   help="Run-tag prefix; the two runs are saved under "
                        "'<tag>/host_quality_all' and '<tag>/host_quality_strict'.")
    p.add_argument("--ddlr-max", type=float, default=None,
                   help=f"Override the strict cut's DDLR threshold "
                        f"(default: {DDLR_MAX}).")
    p.add_argument("--confusion-max", type=float, default=None,
                   help="Override CONFIG['host_confusion_max'] for the strict cut.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = {"run_tag": args.tag}
    if args.ddlr_max is not None:
        overrides["host_ddlr_max"] = args.ddlr_max
    if args.confusion_max is not None:
        overrides["host_confusion_max"] = args.confusion_max
    run_host_quality_check(config_overrides=overrides)