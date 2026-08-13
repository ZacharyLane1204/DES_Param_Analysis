"""
combo_ablation_checks.py  —  SNe Ia Cosmology Pipeline
==========================================================
Final-stage robustness pass on a HAND-CHOSEN COMBINATION of winning
correction terms (interaction / SN-colour / host-galaxy model / ...) --
same incremental-combo idea you already use for "checks/std_..." entries
in extra_runners.py, but here each named TERM is a small reusable block
(model + param_overrides) and COMBOS lists which named terms to merge for
each ablation entry, so you don't hand-write every merged model/
param_overrides dict out in full the way extra_runners.py's literal
_build() calls do.

For EVERY combo below, this runs the same four robustness checks you'd
run on a single best-fit model at the end of extra_runners.py:
  1. Fit the combo once (also serves as its own baseline for the
     degeneracy scan).
  2. degeneracy_scan.scan_degeneracies   on that fit's saved pkl.
  3. host_match_quality.run_host_quality_check  (all vs. strict host match).
  4. loo_zbins.run_loo_zbins             (leave-one-z-bin-out predictive
                                          residuals).
  5. drilling-cones check (full sample vs. each sky cone) -- built by
     hand-replicating drilling_cones_checks.py's broad-uniform-FlatLambdaCDM
     override rather than importing that script's run(); see NOTE below.

NOTE on drilling_cones_checks.py: that script's BEST_MODEL/
BEST_PARAM_OVERRIDES are module-level constants meant to be hand-edited
once per invocation ("python drilling_cones_checks.py"), so its run()
can't safely be called in a loop with a different model each time. This
script instead calls drilling_cones.run_drilling_cones directly with the
same broad-uniform-Om0 override drilling_cones_checks.py builds, so every
combo below gets its own correctly-isolated cones check without touching
that script's globals.

SETUP -- edit TERMS and COMBOS below
--------------------------------------
TERMS: one entry per named correction block, same "model" + "param_
overrides" format as uniform_priors_check.py's BEST_MODELS / extra_
runners.py's _build(). Fill in your actual winning combo for each.

COMBOS: which TERMS to merge for each ablation entry. The example given
matches the procedure you described: interaction alone; interaction + SN
colour; interaction + SN colour + host; SN colour + host; SN colour
alone; host colour alone. Edit freely -- any subset of TERMS.keys() is
valid; list order doesn't matter, only membership.

Usage
-----
  python combo_ablation_checks.py
  python combo_ablation_checks.py --only interaction,interaction_sn_colour
  python combo_ablation_checks.py --skip-drilling-cones   # cheaper smoke test

or:
  from combo_ablation_checks import run_combo_checks
  report = run_combo_checks()
"""

import argparse
import copy

import pandas as pd

from config import CONFIG, DEFAULT_PARAM_SPECS
from run    import run_sampler, pkl_path_for

import degeneracy_scan
import host_match_quality
import loo_zbins
import drilling_cones

# ===========================================================================
# 1. NAMED TERMS  —  edit to match your actual category winners
# ===========================================================================
TERMS = {
    "interaction": {
        "model": {},                                              # e.g. {} if no model-family switch needed
        "param_overrides": {"gamma_alpha": {"active": True, "fixed": None}},
    },
    "sn_colour": {
        "model": {"sn_colour": "softbroken"},
        "param_overrides": {"sn_tau": {"active": True, "fixed": 0.3}},
    },
    "host": {
        "model": {"mass": "linear"},
        "param_overrides": {},
    },
    "host_colour": {
        "model": {"host_colour": "tanh"},
        "param_overrides": {},
    },
}

# ===========================================================================
# 2. COMBOS  —  which TERMS to merge for each ablation entry
# ===========================================================================
COMBOS = [
    ["interaction"],
    ["interaction", "sn_colour"],
    ["interaction", "sn_colour", "host"],
    ["sn_colour", "host"],
    ["sn_colour"],
    ["host_colour"],
]


def _combo_tag(term_names):
    return "combo/" + "_".join(term_names)


def _merge_terms(term_names):
    """Union the model-dict and param_overrides-dict of every named term.
    Raises on conflict rather than letting one term silently overwrite
    another -- two terms in the same combo both trying to set the same
    config['model'] key or the same param_specs field to DIFFERENT values
    is very likely a mistake in TERMS/COMBOS worth catching immediately."""
    model_overrides, param_overrides = {}, {}
    for t in term_names:
        term = TERMS[t]
        for k, v in term.get("model", {}).items():
            if k in model_overrides and model_overrides[k] != v:
                raise ValueError(f"Conflicting model['{k}'] between terms "
                                 f"in combo {term_names}: "
                                 f"{model_overrides[k]!r} vs {v!r}")
            model_overrides[k] = v
        for name, updates in term.get("param_overrides", {}).items():
            if name in param_overrides and param_overrides[name] != updates:
                raise ValueError(f"Conflicting param_overrides['{name}'] "
                                 f"between terms in combo {term_names}: "
                                 f"{param_overrides[name]!r} vs {updates!r}")
            param_overrides[name] = updates
    return model_overrides, param_overrides


def _build_combo_cfg(term_names, registry_file):
    model_overrides, param_overrides = _merge_terms(term_names)
    cfg = copy.deepcopy(CONFIG)
    cfg["run_tag"] = _combo_tag(term_names)
    cfg["model"]   = {**CONFIG["model"], **model_overrides}
    specs = copy.deepcopy(DEFAULT_PARAM_SPECS)
    for name, updates in param_overrides.items():
        specs[name].update(updates)
    cfg["param_specs"]   = specs
    cfg["registry_file"] = registry_file
    return cfg


def _broad_uniform_om0_overrides():
    """Same recipe as drilling_cones_checks.py's
    _broad_uniform_flatlcdm_overrides(only_om0=True) -- broad uniform Om0,
    w/Ode0 inactive so cosmology stays FlatLambdaCDM."""
    om0_lo, om0_hi = DEFAULT_PARAM_SPECS["Om0"]["range"]
    return {"Om0": {"active": True, "prior": "uniform", "range": [om0_lo, om0_hi]},
           "w":    {"active": False},
           "Ode0": {"active": False}}


def run_combo_checks(combos=None, only=None, registry_file="run_combo_registry.csv",
                     degeneracy_threshold=0.85, loo_n_bins=4,
                     run_host_quality=True, run_loo=True, run_drilling_cones=True,
                     drilling_cones_eps_deg=None, drilling_cones_min_samples=None,
                     drilling_cones_min_fit_size=None):
    """
    Parameters
    ----------
    combos                : defaults to COMBOS -- list of lists of TERMS keys.
    only                  : optional iterable of combo tags (underscore-
        joined term names, e.g. "interaction_sn_colour") to restrict to.
    registry_file          : own registry CSV for the combos' main fits --
        kept separate from the publication/checks registries on purpose
        (same reasoning as drilling_cones_checks.py / host_match_quality.py).
    degeneracy_threshold   : passed through to degeneracy_scan.scan_degeneracies.
    loo_n_bins             : passed through to loo_zbins.run_loo_zbins.
    run_host_quality / run_loo / run_drilling_cones : toggle each check off
        for a cheaper smoke test.
    drilling_cones_*       : passed through to drilling_cones.run_drilling_cones.

    Returns
    -------
    pandas.DataFrame, one row per combo: pkl_path, active_params,
    n_degeneracies_flagged, host_quality_gaussian_nsigma, n_loo_bins_flagged,
    n_cones_flagged / n_cones_fitted. Also saved to
    "combo_ablation_summary.csv".
    """
    combos = combos if combos is not None else COMBOS
    if only is not None:
        only = set(only)
        combos = [c for c in combos if "_".join(c) in only]

    rows = []
    for term_names in combos:
        tag = _combo_tag(term_names)
        print(f"\n{'='*60}\nCombo: {tag}  (terms: {term_names})\n{'='*60}")
        cfg = _build_combo_cfg(term_names, registry_file)

        # ---- 1. Fit the combo once ----
        results, sampler, active_names, data, run_name = run_sampler(cfg)
        pkl_path = pkl_path_for(run_name, cfg)
        row = {"combo": tag, "terms": "|".join(term_names),
              "pkl_path": pkl_path, "active_params": ",".join(active_names)}

        # ---- 2. Degeneracy scan ----
        deg = degeneracy_scan.scan_degeneracies(
            pkl_path, threshold=degeneracy_threshold,
            output_prefix=tag.replace("/", "_"))
        row["n_degeneracies_flagged"] = len(deg["flagged"])
        row["degeneracies"] = "; ".join(
            f"{a}<->{b}:{c:+.2f}" for a, b, c in deg["flagged"])

        # ---- 3. Host-match quality ----
        if run_host_quality:
            hq_overrides = {"run_tag": tag, "model": cfg["model"],
                           "param_specs": copy.deepcopy(cfg["param_specs"]),
                           "registry_file": registry_file}
            hq = host_match_quality.run_host_quality_check(
                config_overrides=hq_overrides,
                output_prefix=f"{tag.replace('/', '_')}_hostquality")
            row["host_quality_gaussian_nsigma"] = hq["gaussian_nsigma"]
            row["host_quality_lnB"] = hq["lnB"]

        # ---- 4. LOO redshift-bin CV ----
        if run_loo:
            loo_overrides = {"run_tag": tag, "model": cfg["model"],
                            "param_specs": copy.deepcopy(cfg["param_specs"]),
                            "registry_file": registry_file}
            loo_report = loo_zbins.run_loo_zbins(
                config_overrides=loo_overrides, n_bins=loo_n_bins,
                output_prefix=f"{tag.replace('/', '_')}_loo")
            n_flagged = int((loo_report["mean_residual"].abs()
                            > 2 * loo_report["mean_residual_err"]).sum())
            row["n_loo_bins_flagged"] = n_flagged

        # ---- 5. Drilling cones (broad uniform Om0, own registry) ----
        if run_drilling_cones:
            dc_specs = copy.deepcopy(cfg["param_specs"])
            for name, updates in _broad_uniform_om0_overrides().items():
                dc_specs[name].update(updates)
            dc_overrides = {"run_tag": tag, "model": cfg["model"],
                           "param_specs": dc_specs,
                           "registry_file": "run_drilling_cones_registry.csv",
                           "drilling_cones": True}
            dc_report = drilling_cones.run_drilling_cones(
                config_overrides=dc_overrides, eps_deg=drilling_cones_eps_deg,
                min_samples=drilling_cones_min_samples,
                min_fit_size=drilling_cones_min_fit_size,
                output_prefix=f"{tag.replace('/', '_')}_cones")
            if dc_report is not None:
                fitted = dc_report[~dc_report["skipped_too_few"]]
                n_flagged = int((pd.to_numeric(fitted["gaussian_nsigma"],
                                              errors="coerce") >= 2.0).sum())
                row["n_cones_flagged"] = n_flagged
                row["n_cones_fitted"]  = len(fitted)

        rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv("combo_ablation_summary.csv", index=False)
    print(f"\nCombo ablation summary saved: combo_ablation_summary.csv")
    return report


def _parse_args():
    p = argparse.ArgumentParser(
        description="Final robustness pass (degeneracy scan, host-match "
                    "quality, LOO-z-bins, drilling cones) on a family of "
                    "hand-chosen term combinations, same incremental-combo "
                    "idea as extra_runners.py's checks/std_... entries.")
    p.add_argument("--only", default=None,
                   help="Comma-separated combo tags (underscore-joined term "
                        "names, e.g. 'interaction,interaction_sn_colour') to "
                        "run. Default: run every entry in COMBOS.")
    p.add_argument("--registry-file", default="run_combo_registry.csv")
    p.add_argument("--skip-host-quality", action="store_true")
    p.add_argument("--skip-loo", action="store_true")
    p.add_argument("--skip-drilling-cones", action="store_true")
    p.add_argument("--loo-n-bins", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    only = args.only.split(",") if args.only else None
    run_combo_checks(only=only, registry_file=args.registry_file,
                     run_host_quality=not args.skip_host_quality,
                     run_loo=not args.skip_loo,
                     run_drilling_cones=not args.skip_drilling_cones,
                     loo_n_bins=args.loo_n_bins)