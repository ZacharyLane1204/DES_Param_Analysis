"""
rerun_prior_dominated.py  —  SNe Ia Cosmology Pipeline
==========================================================
Targeted follow-up on prior_shrinkage.py's output: refits specific
(run, parameter) pairs flagged "prior_dominated" with ONLY that parameter
switched to a uniform prior over its existing hard range -- everything
else about the run (model, other param_specs) stays exactly as it was.

Unlike uniform_priors_check.py (which always uniformizes alpha/beta/Om0
for a hand-picked BEST_MODELS set), this pulls the EXACT original config
for each flagged run_tag straight out of experiment_runner.py's
EXPERIMENTS list -- no reconstruction from the registry CSV (which only
logs categorical summaries: model family names, active_params, posterior
mean/std -- not the exact "fixed" values or prior types used), and no
risk of a transcription error on a long tag list. Any parameter can be
targeted, since different categories in this sweep turned out to be
prior-dominated on different parameters (Om0, Ode0, C0, M0, beta, alpha).

Two pieces:

1. targets_from_shrinkage() -- reads a prior_shrinkage.py CSV and builds
   a {run_tag: [param, ...]} dict, optionally filtered by tag prefix or
   an explicit run_tag list. Use this instead of hand-typing tag names.

2. run_degeneracy_prescan() -- NO refitting. Points degeneracy_scan.py at
   the EXISTING saved pkl for a list of already-completed runs (path
   reconstructed via pkl_path_for on that run's exact EXPERIMENTS config,
   same registry_file it originally used). For the host_col_model/mass/
   ssfr block (flagged only on C0/M0, never on the slope/amplitude
   parameters) this tells you which specific runs have a real C0-or-M0-
   vs-slope correlation worth an expensive refit, vs. which are just an
   expected, harmless pivot-parameter prior effect -- run this BEFORE
   deciding what to feed into (3).

3. run_targeted_uniform_refits() -- the actual expensive refit, on
   whatever targets dict you hand it (from (1), or curated by hand after
   looking at (2)'s results).

Usage
-----
  # No-refit prescan on the host_col_model/mass/ssfr block:
  python rerun_prior_dominated.py --prescan

  # Refit the evolution/* block + the 4 headline (alpha/beta) runs:
  python rerun_prior_dominated.py --refit evolution,headline

or:
  from rerun_prior_dominated import (targets_from_shrinkage,
                                     run_degeneracy_prescan,
                                     run_targeted_uniform_refits)
"""

import argparse
import copy

import pandas as pd

from run import run_sampler, pkl_path_for, load_and_filter_data  # noqa: F401 (load_and_filter_data re-exported for convenience)
from experiment_runner import EXPERIMENTS
import degeneracy_scan

_EXPERIMENTS_BY_TAG = {cfg["run_tag"]: cfg for cfg in EXPERIMENTS}


def targets_from_shrinkage(csv_path, prefix=None, run_tags=None,
                           flag="prior_dominated"):
    """
    Parameters
    ----------
    csv_path : path to a prior_shrinkage.py output CSV.
    prefix   : optional, only keep run_names starting with this string
        (e.g. "evolution/").
    run_tags : optional, only keep run_names in this explicit list/set.
    flag     : which flag substring to filter on (default "prior_dominated";
        "prior_in_tension" also works).

    Returns
    -------
    dict {run_tag: [param, param, ...]}
    """
    df = pd.read_csv(csv_path)
    flagged = df[df["flag"].str.contains(flag, na=False)]
    if prefix is not None:
        flagged = flagged[flagged["run_name"].str.startswith(prefix)]
    if run_tags is not None:
        flagged = flagged[flagged["run_name"].isin(set(run_tags))]

    targets = {}
    for _, r in flagged.iterrows():
        targets.setdefault(r["run_name"], []).append(r["param"])
    return targets


def run_degeneracy_prescan(run_tags, threshold=0.85, output_prefix_suffix="_prescan"):
    """
    NO refitting. Runs degeneracy_scan.scan_degeneracies on the EXISTING
    saved pkl for every run_tag in `run_tags`, using that run's exact
    original config (registry_file included) from EXPERIMENTS to
    reconstruct the pkl path. Raises immediately if any pkl is missing --
    that means the run never actually completed/saved, not a degeneracy
    result, and is worth investigating separately before trusting anything
    else about it in the registry.

    Returns
    -------
    pandas.DataFrame, one row per run_tag: n_degeneracies_flagged,
    degeneracies (name pairs + correlation), pkl_path.
    """
    rows = []
    for run_tag in run_tags:
        if run_tag not in _EXPERIMENTS_BY_TAG:
            raise KeyError(f"'{run_tag}' not found in experiment_runner.EXPERIMENTS.")
        cfg = _EXPERIMENTS_BY_TAG[run_tag]
        pkl_path = pkl_path_for(run_tag, cfg)

        print(f"\n{'#'*60}\n# degeneracy prescan: {run_tag}\n{'#'*60}")
        deg = degeneracy_scan.scan_degeneracies(
            pkl_path, threshold=threshold, make_plot=False,
            output_prefix=run_tag.replace("/", "_") + output_prefix_suffix)
        rows.append({
            "run_tag": run_tag, "pkl_path": pkl_path,
            "n_degeneracies_flagged": len(deg["flagged"]),
            "degeneracies": "; ".join(f"{a}<->{b}:{c:+.2f}"
                                      for a, b, c in deg["flagged"]),
        })

    report = pd.DataFrame(rows)
    report.to_csv("prior_dominated_degeneracy_prescan.csv", index=False)
    n_flagged = int((report["n_degeneracies_flagged"] > 0).sum())
    print(f"\n{'='*60}")
    print(f"{n_flagged}/{len(report)} run(s) show at least one degeneracy "
         f"pair >= |{threshold}| -- these are the ones actually worth a "
         f"targeted uniform-prior refit; the rest are very likely just an "
         f"expected, harmless pivot-parameter prior effect.")
    print(f"Full report: prior_dominated_degeneracy_prescan.csv")
    print(f"{'='*60}\n")
    return report


def run_targeted_uniform_refits(targets, tag_prefix="uniformcheck",
                                registry_file="run_uniformcheck_registry.csv"):
    """
    Parameters
    ----------
    targets       : dict {run_tag: [param, param, ...]} -- from
        targets_from_shrinkage() or hand-curated (e.g. after looking at
        run_degeneracy_prescan()'s results).
    tag_prefix    : every refit is saved under "<tag_prefix>/<original_tag>".
    registry_file : own registry CSV, kept separate from the main sweep.

    Returns
    -------
    pandas.DataFrame, one row per refit: original_tag, uniform_params,
    pkl_path.
    """
    missing = [t for t in targets if t not in _EXPERIMENTS_BY_TAG]
    if missing:
        raise KeyError(f"{len(missing)} target run_tag(s) not found in "
                       f"experiment_runner.EXPERIMENTS: {missing}")

    rows = []
    for run_tag, params in targets.items():
        base_cfg = _EXPERIMENTS_BY_TAG[run_tag]
        cfg = copy.deepcopy(base_cfg)
        cfg["run_tag"]      = f"{tag_prefix}/{run_tag}"
        cfg["registry_file"] = registry_file
        specs = copy.deepcopy(cfg["param_specs"])
        for p in params:
            specs[p]["prior"] = "uniform"   # keep existing range; mu/sigma
                                            # become irrelevant once uniform
        cfg["param_specs"] = specs

        print(f"\n{'#'*60}\n# {run_tag}  ->  uniform prior on {params}\n{'#'*60}")
        results, sampler, active_names, data, run_name = run_sampler(cfg)
        pkl_path = pkl_path_for(run_name, cfg)
        rows.append({"original_tag": run_tag,
                    "uniform_params": "|".join(params), "pkl_path": pkl_path})

    report = pd.DataFrame(rows)
    report.to_csv(f"{tag_prefix}_summary.csv", index=False)
    print(f"\nTargeted uniform-prior refit summary saved: "
         f"{tag_prefix}_summary.csv")
    return report


def _parse_args():
    p = argparse.ArgumentParser(
        description="Targeted uniform-prior refits / degeneracy prescan "
                    "for specific runs flagged prior_dominated by "
                    "prior_shrinkage.py.")
    p.add_argument("--shrinkage-csv",
                   default="run_publication_registry_prior_shrinkage.csv")
    p.add_argument("--prescan", action="store_true",
                   help="Run the no-refit degeneracy prescan on the "
                        "host_col_model/mass/ssfr C0/M0 block.")
    p.add_argument("--refit", default=None,
                   help="Comma-separated batch names to refit under uniform "
                        "priors: 'evolution' (21 runs, Om0), 'headline' "
                        "(4 runs, alpha/beta), or 'all_flagged' (every "
                        "prior_dominated row in the CSV).")
    p.add_argument("--tag-prefix", default="uniformcheck")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.prescan:
        host_block_tags = [t for t in targets_from_shrinkage(args.shrinkage_csv)
                           if t.startswith(("host_col_model/", "mass/", "ssfr/"))]
        run_degeneracy_prescan(host_block_tags)

    if args.refit:
        batches = args.refit.split(",")
        targets = {}
        if "evolution" in batches:
            targets.update(targets_from_shrinkage(args.shrinkage_csv,
                                                  prefix="evolution/"))
        if "headline" in batches:
            targets.update(targets_from_shrinkage(
                args.shrinkage_csv,
                run_tags=["sn_col_model/sncolour_dust_c0sntau",
                         "sn_col_model/sncolour_dust_sntau",
                         "sn_col_model/sncolour_softbroken_c0_sntau",
                         "stretch/stretch_softbroken_x10x1tau"]))
        if "all_flagged" in batches:
            targets.update(targets_from_shrinkage(args.shrinkage_csv))
        run_targeted_uniform_refits(targets, tag_prefix=args.tag_prefix)