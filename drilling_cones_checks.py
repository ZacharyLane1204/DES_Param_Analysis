"""
drilling_cones_checks.py  —  SNe Ia Cosmology Pipeline
=========================================================
Standalone entry point for the drilling-cones systematic check, run with
broad uniform FlatLambdaCDM cosmology priors, feeding its OWN registry/CSV
files so it never mixes with the normal experiment_runner.py /
extra_runners.py sweeps or the publication/checks registries.

This does not duplicate any sampling/clustering/comparison logic — it is
a thin, opinionated config wrapper around drilling_cones.run_drilling_cones
(see that module's docstring for the actual algorithm). What this script
adds:

  1. A broad-uniform-prior FlatLambdaCDM param_specs override (Om0 uniform
     over its full hard range; w/Ode0 stay fixed/inactive so the cosmology
     stays FlatLambdaCDM, not free wCDM/non-flat). Set BROAD_UNIFORM_ONLY_
     OM0=False below to also broaden alpha/beta to uniform, matching the
     "checks/uniformpriors_*" experiments in extra_runners.py.
  2. Its own registry_file ("run_drilling_cones_registry.csv") for the
     baseline + per-cone fits, and its own summary CSV
     ("<tag>_drilling_cones.csv", via output_prefix) — kept separate from
     run_publication_registry.csv / run_checks_registry.csv on purpose, so
     this check's fits never get skipped/deduped against or mixed into a
     normal sweep's registry.
  3. A model_cfg you plug in below (BEST_MODEL) — set this to whatever your
     best-fitting model combination turned out to be from
     experiment_runner.py / extra_runners.py before running this script.

Usage
-----
  # 1. Edit BEST_MODEL / BEST_PARAM_OVERRIDES below to match your best fit.
  # 2. Run:
  python drilling_cones_checks.py --tag best_model_cones

Then build the LaTeX table with:
  python latex_tables.py --drilling-cones --drilling-cones-csv best_model_cones_drilling_cones.csv
"""

import argparse
import copy

from config import CONFIG, DEFAULT_PARAM_SPECS
import drilling_cones

# ===========================================================================
# 1.  BEST-FIT MODEL  —  edit this to match your chosen final model
# ===========================================================================
# Fill this in with whatever model combination came out on top from
# experiment_runner.py / extra_runners.py (checks/std_..., ssfr/..., etc.).
# Left as the plain baseline model (no sSFR/host-colour term) by default so
# this script runs out of the box; the FlatLambdaCDM / uniform-prior
# machinery below does not depend on what you put here.
BEST_MODEL = {
    "sn_colour":     CONFIG["model"]["sn_colour"],
    "x1_correction": CONFIG["model"]["x1_correction"],
    "mass":          CONFIG["model"]["mass"],
    "host_colour":   CONFIG["model"]["host_colour"],
    "ssfr":          CONFIG["model"]["ssfr"],
    "z_evolve":      CONFIG["model"]["z_evolve"],
}
BEST_PARAM_OVERRIDES = {}   # e.g. {"gamma_alpha": {"active": True, "fixed": None}}

# Set False to also broaden alpha/beta to uniform (matches
# "checks/uniformpriors_*" in extra_runners.py); True keeps only the
# cosmological parameter (Om0) broadened, which is the minimum needed for
# "broad uniform cosmology priors ... for FlatLambdaCDM".
BROAD_UNIFORM_ONLY_OM0 = True


def _broad_uniform_flatlcdm_overrides(only_om0=True):
    """
    param_specs overrides: Om0 uniform over its full hard range (a broad,
    uninformative cosmology prior), with w/Ode0 left inactive so the
    cosmology stays FlatLambdaCDM (see core.infer_cosmo_type — cosmo_type
    is inferred from which of w/Ode0 are active, not set explicitly).
    """
    om0_lo, om0_hi = DEFAULT_PARAM_SPECS["Om0"]["range"]
    overrides = {
        "Om0": {"active": True, "prior": "uniform", "range": [om0_lo, om0_hi]},
        "w":    {"active": False},
        "Ode0": {"active": False},
    }
    if not only_om0:
        overrides["alpha"] = {"prior": "uniform"}
        overrides["beta"]  = {"prior": "uniform"}
    return overrides


def _parse_args():
    p = argparse.ArgumentParser(
        description="Drilling-cones systematic check with broad uniform "
                    "FlatLambdaCDM cosmology priors on the chosen best-fit "
                    "model, writing its own registry/CSV.")
    p.add_argument("--tag", default="drilling_cones_uniform",
                   help="Run-tag / output-prefix base (default: "
                        "'drilling_cones_uniform').")
    p.add_argument("--registry-file", default="run_drilling_cones_registry.csv",
                   help="Registry CSV for the underlying baseline+cone fits "
                        "(kept separate from the publication/checks "
                        "registries on purpose).")
    p.add_argument("--eps-deg", type=float, default=None)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--min-fit-size", type=int, default=None)
    p.add_argument("--full-uniform", action="store_true",
                   help="Also broaden alpha/beta to uniform, not just Om0.")
    return p.parse_args()


def run(tag="drilling_cones_uniform", registry_file="run_drilling_cones_registry.csv",
       eps_deg=None, min_samples=None, min_fit_size=None, only_om0=True):
    param_overrides = _broad_uniform_flatlcdm_overrides(only_om0=only_om0)
    param_overrides.update(copy.deepcopy(BEST_PARAM_OVERRIDES))

    param_specs = copy.deepcopy(DEFAULT_PARAM_SPECS)
    for name, updates in param_overrides.items():
        param_specs[name].update(updates)

    config_overrides = {
        "run_tag":       tag,
        "model":         dict(BEST_MODEL),
        "param_specs":   param_specs,
        "registry_file": registry_file,
        "drilling_cones": True,
    }

    return drilling_cones.run_drilling_cones(
        config_overrides=config_overrides,
        eps_deg=eps_deg, min_samples=min_samples, min_fit_size=min_fit_size,
        output_prefix=tag)


if __name__ == "__main__":
    args = _parse_args()
    report = run(tag=args.tag, registry_file=args.registry_file,
                eps_deg=args.eps_deg, min_samples=args.min_samples,
                min_fit_size=args.min_fit_size,
                only_om0=not args.full_uniform)
    if report is not None:
        print(f"\nDrilling-cones (uniform-prior FlatLambdaCDM) check complete. "
             f"CSV: {args.tag}_drilling_cones.csv")