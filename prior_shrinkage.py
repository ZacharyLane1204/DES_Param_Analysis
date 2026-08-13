"""
prior_shrinkage.py  —  SNe Ia Cosmology Pipeline
===================================================
Cheap, no-refit diagnostic for prior dominance, read directly off a run
registry CSV (run_publication_registry.csv / run_checks_registry.csv /
run_uniformpriors_registry.csv / ...).

Every _registry_row (see run.py) already stores <param>_mean/<param>_std
for every active parameter of every run. For any parameter with a
gaussian-family prior (gaussian / truncated_gaussian -- the only prior
types with a genuine "mu"/"sigma" to shrink from: alpha, beta, Om0, Ode0,
M under DEFAULT_PARAM_SPECS), that's enough to compute, with ZERO
additional sampling:

    shrinkage = 1 - posterior_std / prior_sigma
        ~1  -> data pin the parameter down far tighter than the prior
              (prior irrelevant to the point estimate)
        ~0  -> posterior as wide as the prior (posterior ~ prior; the DATA
              are not constraining this parameter here -- the point
              estimate is essentially the prior's, not the data's)
        <0  -> posterior WIDER than the prior (rare; usually a poorly
              converged/multimodal run rather than genuine prior
              dominance -- cross-check with run.diagnose_modes before
              trusting shrinkage on that row at all)

    pull = (posterior_mean - prior_mu) / prior_sigma
        how many prior-sigma the data have pulled the parameter away from
        the prior's central value. Large |pull| together with high
        shrinkage means the data disagree with the prior and are strong
        enough to win that disagreement -- worth checking whether the
        prior's mu is still appropriate for this sample, independent of
        whether it's "dominating" the fit.

IMPORTANT CAVEAT: this reads DEFAULT_PARAM_SPECS's mu/sigma as "the"
prior for every row, which is only valid for rows that used the ordinary
informative prior. Runs whose param_specs explicitly overrode a
parameter to a different prior (most obviously the "uniformpriors/*" rows
from uniform_priors_check.py) do NOT have a meaningful shrinkage/pull
relative to a gaussian they were never sampled under. This script
auto-skips any run_name containing "uniformpriors" as a best-effort
guard, but the underlying issue is that the registry does not currently
log the ACTUAL prior type/mu/sigma used
per run, only the resulting posterior mean/std -- worth adding to
_registry_row in run.py if you want this trusted at scale rather than
eyeballing which rows are exempt. If you keep other deliberately-non-
default priors in your normal registries, pass reference_specs=<your
dict> to override what "the prior" means for a given call.

Usage
-----
  python prior_shrinkage.py run_publication_registry.csv
  python prior_shrinkage.py run_checks_registry.csv --flag-shrinkage 0.2

or:
  from prior_shrinkage import scan_registry
  report = scan_registry("run_publication_registry.csv")
"""

import argparse

import pandas as pd

from config import DEFAULT_PARAM_SPECS

_GAUSSIAN_FAMILY = {"gaussian", "truncated_gaussian"}


def _reference_priors(reference_specs=None):
    specs = reference_specs or DEFAULT_PARAM_SPECS
    return {name: spec for name, spec in specs.items()
           if spec.get("prior") in _GAUSSIAN_FAMILY}


def scan_registry(registry_path, reference_specs=None, flag_shrinkage=0.2,
                  flag_pull=3.0, skip_pattern="uniformpriors",
                  output_csv=None):
    """
    Parameters
    ----------
    registry_path   : path to any run.py-format registry CSV.
    reference_specs : dict, defaults to DEFAULT_PARAM_SPECS -- what "the
        prior" means for every row. Only entries with prior in
        {"gaussian","truncated_gaussian"} are checked (see module
        docstring for why the other prior families are skipped -- their
        "mu"/"sigma"/"scale" fields don't have the same meaning).
    flag_shrinkage  : rows with shrinkage below this are flagged
        "prior_dominated" (default 0.2 -- posterior within 20% of the
        prior's own width).
    flag_pull       : rows with |pull| above this are flagged
        "prior_in_tension" (default 3.0 prior-sigma).
    skip_pattern    : run_name substring that marks a row as having used
        a non-default prior for these parameters (best-effort guard, see
        module docstring's caveat) -- matching rows are skipped entirely,
        not silently mis-scored.
    output_csv      : defaults to "<registry_path stem>_prior_shrinkage.csv".

    Returns
    -------
    pandas.DataFrame, one row per (run_name, parameter) scored.
    """
    df  = pd.read_csv(registry_path)
    ref = _reference_priors(reference_specs)

    rows = []
    n_skipped = 0
    for _, r in df.iterrows():
        run_name = str(r["run_name"])
        if skip_pattern and skip_pattern in run_name:
            n_skipped += 1
            continue
        active = (str(r["active_params"]).split("|")
                 if pd.notna(r.get("active_params")) else [])
        for name in active:
            if name not in ref:
                continue
            mean_col, std_col = f"{name}_mean", f"{name}_std"
            if mean_col not in df.columns or std_col not in df.columns:
                continue
            post_mean, post_std = r.get(mean_col), r.get(std_col)
            if pd.isna(post_mean) or pd.isna(post_std):
                continue

            prior_mu, prior_sigma = ref[name]["mu"], ref[name]["sigma"]
            shrinkage = 1.0 - (float(post_std) / prior_sigma)
            pull      = (float(post_mean) - prior_mu) / prior_sigma

            flags = []
            if shrinkage < flag_shrinkage:
                flags.append("prior_dominated")
            if abs(pull) > flag_pull:
                flags.append("prior_in_tension")

            rows.append({
                "run_name": run_name, "param": name,
                "prior_type": ref[name]["prior"],
                "prior_mu": prior_mu, "prior_sigma": prior_sigma,
                "post_mean": post_mean, "post_std": post_std,
                "shrinkage": round(shrinkage, 4),
                "pull":      round(pull, 4),
                "flag": "|".join(flags) if flags else "",
            })

    report = pd.DataFrame(rows)
    if len(report):
        report = report.sort_values("shrinkage").reset_index(drop=True)

    output_csv = output_csv or registry_path.replace(".csv", "_prior_shrinkage.csv")
    report.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Prior shrinkage scan: {registry_path}")
    print(f"{len(df)} run(s) in registry, {n_skipped} skipped "
         f"(run_name matched '{skip_pattern}')")
    n_runs_scored = report["run_name"].nunique() if len(report) else 0
    print(f"{len(report)} (run, parameter) row(s) scored across "
         f"{n_runs_scored} run(s)")

    if len(report):
        dominated = report[report["flag"].str.contains("prior_dominated", na=False)]
        tension   = report[report["flag"].str.contains("prior_in_tension", na=False)]
        if len(dominated):
            print(f"\n{len(dominated)} row(s) flagged prior_dominated "
                 f"(shrinkage < {flag_shrinkage}) -- candidates for a full "
                 f"uniform-prior refit (see uniform_priors_check.py):")
            for _, r in dominated.iterrows():
                print(f"  {r['run_name']:50s} {r['param']:12s} "
                     f"shrinkage={r['shrinkage']:+.3f}  pull={r['pull']:+.2f}")
        else:
            print(f"\nNo rows flagged prior_dominated (shrinkage < "
                 f"{flag_shrinkage}) -- every scored run's posterior is "
                 f"meaningfully tighter than its prior for these parameters.")
        if len(tension):
            print(f"\n{len(tension)} row(s) flagged prior_in_tension "
                 f"(|pull| > {flag_pull} prior-sigma):")
            for _, r in tension.iterrows():
                print(f"  {r['run_name']:50s} {r['param']:12s} "
                     f"pull={r['pull']:+.2f} sigma  (shrinkage={r['shrinkage']:+.3f})")
    print(f"\nFull report saved: {output_csv}")
    print(f"{'='*60}\n")

    return report


def _parse_args():
    p = argparse.ArgumentParser(
        description="No-refit prior-dominance scan over a run.py registry "
                    "CSV -- shrinkage and pull for every gaussian-family "
                    "active parameter of every run already in the registry.")
    p.add_argument("registry_path")
    p.add_argument("--flag-shrinkage", type=float, default=0.2)
    p.add_argument("--flag-pull", type=float, default=3.0)
    p.add_argument("--skip-pattern", default="uniformpriors")
    p.add_argument("--output-csv", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    scan_registry(args.registry_path, flag_shrinkage=args.flag_shrinkage,
                 flag_pull=args.flag_pull, skip_pattern=args.skip_pattern,
                 output_csv=args.output_csv)