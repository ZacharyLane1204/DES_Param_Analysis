"""
uniform_priors_check.py  —  SNe Ia Cosmology Pipeline
=========================================================
Uniform-prior reruns driven by prior_shrinkage.py / degeneracy_scan.py's
findings — NOT robustness checks on the chosen best model.

This is deliberately its own script, not part of experiment_runner.py or
extra_runners.py:

  - experiment_runner.py defines the informative-prior DEFAULT_PARAM_SPECS
    sweep -- swapping a parameter's prior shape there would silently change
    every entry that activates it, which is exactly what we want for a
    genuinely un-anchored parameter (see config.py's C0, now uniform by
    default) but NOT what we want for a targeted "does this specific
    combo's posterior survive without the informative prior" question.

  - extra_runners.py's "checks/" tag prefix is reserved for post-hoc
    robustness checks on the chosen BEST model (host-match quality, LOO-z,
    c-cuts, ...) -- see that file's own "checks/std_..." entries. A prior-
    shape comparison is a different kind of question (is the posterior
    prior- or data-dominated?) asked of specific combos flagged by
    prior_shrinkage.py's scan, not of "the" best model, so it does not
    belong under that prefix. This file's tags use "uniformpriors/"
    instead and never "checks/".

Every entry below reruns a combo already defined in experiment_runner.py
(or a close relative of one) with one or more parameters' prior swapped to
"uniform" over the SAME hard range -- i.e. only the prior SHAPE changes,
not its support. Compare each row here against its informative-prior
counterpart (same tag, minus the "uniformpriors/" prefix and prior
overrides, in experiment_runner.py's own registry) via compare_runs.py /
degeneracy_scan.py: a parameter estimate that shifts meaningfully once the
informative prior is removed means the posterior is prior-dominated
rather than data-dominated for that parameter, in that combo.

Own registry_file (run_uniformpriors_registry.csv), kept separate from
run_publication_registry.csv AND run_checks_registry.csv on purpose, same
reasoning as drilling_cones_checks.py / host_match_quality.py -- so these
never get mixed into, skipped against, or deduped with either.

SETUP -- edit ENTRIES below
------------------------------
Add one entry per (combo, parameters-to-uniformise) you want to test.
Each entry is exactly what you'd pass to experiment_runner.py's _build():
a tag (minus "uniformpriors/", added automatically), model overrides, and
param_overrides -- with the specific parameter(s) you want prior-shape-
tested given `{"prior": "uniform", ...}` (plus whatever `{"active": True,
...}` is needed to match the combo being tested).

Usage
-----
  python uniform_priors_check.py
  python uniform_priors_check.py --only baseline,Om0only_baseline
  python uniform_priors_check.py --dry-run

or:
  from uniform_priors_check import run_uniform_priors_check
  report = run_uniform_priors_check()
"""

import argparse
import copy

import pandas as pd

from config import CONFIG, DEFAULT_PARAM_SPECS
from run    import run_sampler, pkl_path_for

REGISTRY_FILE = "run_uniformpriors_registry.csv"


def _override(base_specs, **param_overrides):
    """Deep copy of base_specs with per-parameter overrides applied. Same
    helper as experiment_runner.py's -- duplicated locally rather than
    imported so this script has no dependency on experiment_runner.py's
    module-level state (thread-clamping env vars, etc.)."""
    specs = copy.deepcopy(base_specs)
    for name, updates in param_overrides.items():
        specs[name].update(updates)
    return specs


def _build(tag, param_overrides=None, config_overrides=None):
    """Build a complete config dict for one entry. Tag is prefixed with
    "uniformpriors/" automatically -- pass the bare descriptive name."""
    cfg = copy.deepcopy(CONFIG)
    cfg["run_tag"]     = f"uniformpriors/{tag}"
    cfg["param_specs"] = _override(DEFAULT_PARAM_SPECS, **(param_overrides or {}))
    cfg["registry_file"] = REGISTRY_FILE
    if config_overrides:
        cfg.update(config_overrides)
    return cfg


# ===========================================================================
# ENTRIES
# ===========================================================================
# alpha, beta, and Om0 use informative truncated_gaussian priors in
# DEFAULT_PARAM_SPECS (see config.py) -- centred on literature/SALT2
# training-sample values, not derived from this analysis. "baseline" and
# the full-combo entry below rerun the flat-LCDM baseline and the best-
# performing full model combo with those three priors swapped to uniform
# over the same hard range.
ENTRIES = [

    _build("baseline",
           param_overrides={"alpha": {"prior": "uniform"},
                             "beta":  {"prior": "uniform"},
                             "Om0":   {"prior": "uniform"}}),

    _build("gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear",
           config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken",
                                        "mass": "linear", "ssfr": "tanh",
                                        "host_colour": "none"}},
           param_overrides={"alpha": {"prior": "uniform"},
                             "beta":  {"prior": "uniform"},
                             "Om0":   {"prior": "uniform"},
                             "gamma_alpha": {"active": True, "fixed": None},
                             "c0": {"active": False, "fixed": 0},
                             "sn_tau": {"active": True, "fixed": 0.3},
                             "zeta":  {"active": True, "fixed": 0.0},
                             "F0":    {"active": True, "fixed": -10.5},
                             "ftau":  {"active": True, "fixed": 0.5},
                             "gamma": {"active": True, "fixed": 0.0},
                             "eta":   {"active": False, "fixed": 0.0}}),

    # Om0-only variant: cosmology is usually the parameter you care most
    # about not being prior-driven, so this isolates it from alpha/beta.
    _build("Om0only_baseline",
           param_overrides={"Om0": {"prior": "uniform"}}),

    # M0 uniform-prior check, interaction-active combo.
    # C0 is NOT overridden to "uniform" here -- it already defaults to a
    # uniform prior in DEFAULT_PARAM_SPECS now (see config.py), so every
    # experiment_runner.py entry that activates C0, including this combo's
    # own experiment_runner.py counterpart, already gets it. M0 keeps its
    # informative default there (see config.py's M0 docstring) because
    # unlike C0 it has genuine external anchoring -- but prior_shrinkage.py
    # + degeneracy_scan.py's prescan both flag it as prior-dominated
    # specifically in xi_mass_col-active ("*_inter_only_M0") combos
    # (M0<->xi_mass_col correlation ~-0.97, M0<->C0 ~+0.93), where the data
    # cannot cleanly separate "where the mass step sits" from "how strong
    # the mass-colour interaction is". This reruns exactly
    # experiment_runner.py's "host_col_model/host_colour_broken_mass_
    # tanh_inter_only_M0" combo with M0's prior swapped to uniform, so the
    # two runs are the direct informative-vs-uniform pair for the specific
    # combo where the degeneracy is worst.
    _build("M0_interaction_broken_mass_tanh",
           config_overrides={"model": {**CONFIG["model"],
                                        "host_colour": "broken",
                                        "mass": "tanh"}},
           param_overrides={"C0": {"active": True, "fixed": 0},
                             "M0": {"active": True, "fixed": 10.0,
                                    "prior": "uniform"},
                             "gamma": {"active": False, "fixed": 0},
                             "eta": {"active": False, "fixed": 0},
                             "xi_mass_col": {"active": True, "fixed": 0},
                             "tau": {"active": False, "fixed": 0.2}}),
]


def run_uniform_priors_check(only=None, dry_run=False):
    """
    Parameters
    ----------
    only    : optional iterable of bare tags (without "uniformpriors/",
        e.g. "baseline,Om0only_baseline") to restrict to.
    dry_run : print what would run without sampling.

    Returns
    -------
    pandas.DataFrame, one row per entry: run_tag, pkl_path, active_params.
    Also saved to "uniform_priors_check_summary.csv". Cross-reference each
    row against its informative-prior counterpart (same combo in
    experiment_runner.py's own registry) with compare_runs.compare_two_runs
    or prior_shrinkage.py's shrinkage/pull numbers.
    """
    entries = ENTRIES
    if only is not None:
        only = {f"uniformpriors/{t}" if not t.startswith("uniformpriors/") else t
               for t in only}
        entries = [cfg for cfg in entries if cfg["run_tag"] in only]

    rows = []
    for cfg in entries:
        active = [n for n, s in cfg["param_specs"].items() if s["active"]]
        print(f"\n{'='*60}\n{cfg['run_tag']}\n{'='*60}")
        print(f"  model: {cfg['model']}")
        print(f"  active params: {active}")
        if dry_run:
            rows.append({"run_tag": cfg["run_tag"], "pkl_path": "(dry-run)",
                        "active_params": "|".join(active)})
            continue

        results, sampler, active_names, data, run_name = run_sampler(cfg)
        rows.append({"run_tag": cfg["run_tag"],
                    "pkl_path": pkl_path_for(run_name, cfg),
                    "active_params": "|".join(active_names)})

    report = pd.DataFrame(rows)
    if not dry_run:
        report.to_csv("uniform_priors_check_summary.csv", index=False)
        print(f"\nUniform-priors check summary saved: "
             f"uniform_priors_check_summary.csv "
             f"(registry: {REGISTRY_FILE})")
    return report


def _parse_args():
    p = argparse.ArgumentParser(
        description="Uniform-prior reruns driven by prior_shrinkage.py / "
                    "degeneracy_scan.py findings -- NOT best-model "
                    "robustness checks (see extra_runners.py's checks/ for "
                    "those). Tags use 'uniformpriors/', own registry "
                    f"({REGISTRY_FILE}).")
    p.add_argument("--only", default=None,
                   help="Comma-separated bare tags (e.g. "
                        "'baseline,Om0only_baseline') to run. Default: "
                        "every entry in ENTRIES.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    only = args.only.split(",") if args.only else None
    run_uniform_priors_check(only=only, dry_run=args.dry_run)