"""
z_uncertainty_check.py  —  SNe Ia Cosmology Pipeline
========================================================
Monte-Carlo redshift-uncertainty propagation check, run over EVERY competing
model in best_model.COMBOS, in parallel.

Rather than marginalising redshift uncertainty into the likelihood by
quadrature -- which risks double-counting against MUERR (which already carries
a peculiar-velocity contribution via MUERR_VPEC) and is awkward because z
enters mu_theory through a distance integral rather than a cheap algebraic
profile -- this refits each model N times, each time with data["z"]
independently perturbed by a Gaussian draw of width zerr_col ("zHDERR" by
default) BEFORE anything is computed from it. Both mu_theory(z) and any
z_evolve correction in compute_mu_corr then consume the SAME perturbed z, so
the perturbation propagates consistently through everything downstream.

WHAT IT REPORTS
---------------
For every model and every active parameter:
  * baseline_mean / baseline_posterior_std -- from one UNPERTURBED fit.
  * mc_std   : scatter of the recovered posterior mean across realizations.
  * mc_std_as_frac_of_posterior_std -- THE headline number. If the MC scatter
    is small next to the parameter's own posterior width, redshift uncertainty
    is not adding meaningfully to the error budget beyond what MUERR and the
    covariance already capture.
  * bias_nsigma : (mc_mean - baseline_mean) / baseline_posterior_std -- catches
    a systematic SHIFT, which a scatter statistic on its own cannot see.

A separate cosmology-focused summary pulls Om0 (plus w/Ode0 where active) out
for every model side by side: "does redshift uncertainty move my cosmology, and
does the answer depend on which standardisation model I chose?"

CHOOSING N
----------
The precision of an estimate of a standard deviation from N samples is roughly
1/sqrt(2(N-1)):

      N=8  -> 27%     N=16 -> 18%     N=24 -> 15%
      N=32 -> 13%     N=48 -> 10%     N=64 ->  9%

The headline statistic is compared against a 20% threshold, so an N whose own
uncertainty is 27% cannot resolve it -- N=8 (the old default) was too noisy to
support the conclusion it was being used to draw. N=24 is the default here: 15%
precision, and 9 models x 25 fits = 225 runs, which parallelises comfortably.
Use --n-realizations 8 for a smoke test, 48 if a referee wants ~10%.

DOUBLE-COUNTING PECULIAR VELOCITY
---------------------------------
zHDERR already carries the peculiar-velocity / flow-model uncertainty, and
MUERR_VPEC is the SAME physical contribution already folded into the covariance
in magnitude space. Leaving both in place counts it twice: once as a z shift
here, once as a magnitude-space term. By default this check therefore rebuilds
the covariance with MUERR_VPEC removed in quadrature
(sqrt(muerr^2 - muerr_vpec^2)) for the baseline AND every realization, so PV
uncertainty enters through the z perturbation only. --include-muerr-vpec
restores the old double-counted behaviour for a before/after comparison.

NAMING
------
    z_uncert/base_<combo>        unperturbed baseline
    z_uncert/z_<NNN>_<combo>     realization NNN (seed = seed0 + NNN)

Realization n uses seed0 + n for EVERY model, so model A's realization 7 and
model B's realization 7 see the same z draw -- the models are compared on
identical perturbations rather than on independent noise.

USAGE
-----
    python z_uncertainty_check.py --workers 16                    # all models, N=24
    python z_uncertainty_check.py --n-realizations 8 --workers 8  # smoke test
    python z_uncertainty_check.py --only mass_linear_ssfr_tanh --n-realizations 48
    python z_uncertainty_check.py --best-model-only --workers 8
    python z_uncertainty_check.py --dry-run
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dynesty import utils as dyfunc

from config import CONFIG, DEFAULT_PARAM_SPECS
from run import load_and_filter_data, run_sampler, pkl_path_for
from experiment_naming import ExperimentRegistry
from loo_zbins import _refactorise_covariance
from best_model import COMBOS, BEST_COMBO, merge_terms as _merge_terms, combo_tag
from parallel_runner import Job, run_jobs, add_parallel_args


# ===========================================================================
# 1.  NAMING / DEFAULTS
# ===========================================================================
# Every run tag this script can produce, in one editable place.
TAG_TEMPLATES = {
    "baseline":    "z_uncert/base_{combo}",
    "realization": "z_uncert/z_{n:03d}_{combo}",
}

OUT_DIR = "z_uncertainty"
REGISTRY = "run_z_uncertainty_registry.csv"

# See "CHOOSING N" in the module docstring.
DEFAULT_N_REALIZATIONS = 24

# Headline threshold: MC scatter this large a fraction of the posterior width
# means redshift uncertainty is a non-negligible part of the error budget.
FLAG_FRAC = 0.20


def _tag(kind, **kw):
    return TAG_TEMPLATES[kind].format(**kw)


# ===========================================================================
# 2.  WORKER ENTRY POINT
# ===========================================================================
# Must be module-level: parallel_runner uses spawn, which pickles the callable
# by qualified name.

def job_z_fit(cfg, seed=None, zerr_col=None, exclude_muerr_vpec=True,
              muerr_vpec_col=None):
    """One fit: unperturbed when seed is None, otherwise z-perturbed.

    The worker loads and filters the data itself rather than receiving a
    preloaded tuple. The covariance alone is ~26 MB for this sample, and
    pickling that to each of a few hundred jobs would cost far more than
    re-reading the CSV per worker.

    The geometric covariance is NOT rebuilt per realization: it does not depend
    on z. It IS rebuilt once when exclude_muerr_vpec removes MUERR_VPEC from
    muerr, because that does change the diagonal.
    """
    df, data, cov_mat, inv_cov, log_det, C_sum, keep_idx = load_and_filter_data(cfg)
    zerr_col = zerr_col or cfg.get("col_zerr", "zHDERR")

    if exclude_muerr_vpec:
        muerr_vpec_col = muerr_vpec_col or cfg.get("col_muerr_vpec", "MUERR_VPEC")
        if muerr_vpec_col not in df.columns:
            raise KeyError(
                f"exclude_muerr_vpec=True needs column '{muerr_vpec_col}', "
                f"which is not in the data. Pass --include-muerr-vpec to skip "
                f"the correction (at the cost of double-counting peculiar "
                f"velocity against {zerr_col}).")
        mv = np.asarray(df[muerr_vpec_col].values, dtype=float)
        # clip at 0: floating-point noise could otherwise make a tiny negative
        # variance where muerr_vpec == muerr.
        muerr_novpec = np.sqrt(np.clip(data["muerr"] ** 2 - mv ** 2, 0.0, None))
        data = dict(data)
        data["muerr"] = muerr_novpec
        inv_cov, log_det, C_sum = _refactorise_covariance(
            cov_mat, muerr_novpec, cfg.get("sigma_int", 0.0))

    if seed is not None:
        if zerr_col not in df.columns:
            raise KeyError(f"z perturbation needs column '{zerr_col}', which "
                           f"is not in the data.")
        rng = np.random.default_rng(seed)
        data = dict(data)
        # Clip keeps z physical. At DES redshifts a zHDERR-sized draw never
        # comes near the bound, so this is a guard, not a distortion.
        data["z"] = np.clip(
            data["z"] + rng.normal(0.0, np.asarray(df[zerr_col].values, float)),
            1e-4, None)

    preloaded = (df, data, cov_mat, inv_cov, log_det, C_sum, keep_idx)
    results, _, active_names, _, run_name = run_sampler(cfg, preloaded=preloaded)

    w = np.exp(results.logwt - results.logz[-1])
    w /= w.sum()
    mean, cov = dyfunc.mean_and_cov(results.samples, w)
    std = np.sqrt(np.diag(cov))

    out = {"run_tag": cfg.get("run_tag"),
           "run_name": run_name,
           "pkl_path": pkl_path_for(run_name, cfg),
           "seed": seed,
           "n_sne": int(len(data["z"])),
           "logz": float(results.logz[-1]),
           "logz_err": float(results.logzerr[-1]),
           "active_params": ",".join(active_names)}
    for name, m, s in zip(active_names, mean, std):
        out[f"{name}_mean"] = float(m)
        out[f"{name}_std"] = float(s)
    return out


# ===========================================================================
# 3.  CONFIG CONSTRUCTION
# ===========================================================================

def build_combo_cfg(term_names, registry, registry_file):
    """One combo's baseline config, built through the shared ExperimentRegistry.

    Using the registry (rather than hand-assembling a dict) means a combo that
    repeats an earlier tag, resolves to an identical model fingerprint, or
    activates a parameter its own model makes degenerate is caught HERE, before
    N expensive fits, by the same guards experiment_runner.py uses.
    """
    model_overrides, param_overrides = _merge_terms(term_names)
    return registry.build(_tag("baseline", combo=combo_tag(term_names)),
                          param_overrides=param_overrides,
                          config_overrides={"model": {**CONFIG["model"],
                                                      **model_overrides},
                                            "registry_file": registry_file})


# ===========================================================================
# 4.  DRIVER
# ===========================================================================

def run_z_uncertainty_check(combos=None, only=None,
                            n_realizations=DEFAULT_N_REALIZATIONS, seed0=2000,
                            zerr_col=None, exclude_muerr_vpec=True,
                            muerr_vpec_col=None, registry_file=REGISTRY,
                            n_workers=None, sequential=False,
                            log_dir="logs/z_uncert", out_dir=OUT_DIR,
                            dry_run=False, capture_output=True):
    """Run the MC z-uncertainty check across every competing model.

    Returns a dict of DataFrames: realizations, summary, cosmology.
    """
    combos = list(combos) if combos is not None else list(COMBOS)
    if only:
        want = set(only)
        combos = [c for c in combos if combo_tag(c) in want]
        if not combos:
            raise SystemExit(
                f"No combo matched --only {sorted(want)}.\nAvailable: "
                + ", ".join(combo_tag(c) for c in COMBOS))

    os.makedirs(out_dir, exist_ok=True)
    registry = ExperimentRegistry(CONFIG, DEFAULT_PARAM_SPECS)

    common = {"zerr_col": zerr_col,
              "exclude_muerr_vpec": exclude_muerr_vpec,
              "muerr_vpec_col": muerr_vpec_col}

    jobs, meta = [], []
    for term_names in combos:
        ctag = combo_tag(term_names)
        base_cfg = build_combo_cfg(term_names, registry, registry_file)

        jobs.append(Job(label=base_cfg["run_tag"], func=job_z_fit,
                        kwargs={"cfg": base_cfg, "seed": None, **common},
                        group="baseline", cost=2.0))
        meta.append({"combo": ctag, "kind": "baseline",
                     "tag": base_cfg["run_tag"], "realization": -1, "seed": None})

        for n in range(n_realizations):
            cfg_n = copy.deepcopy(base_cfg)
            cfg_n["run_tag"] = _tag("realization", combo=ctag, n=n)
            jobs.append(Job(label=cfg_n["run_tag"], func=job_z_fit,
                            kwargs={"cfg": cfg_n, "seed": seed0 + n, **common},
                            group="realization", cost=1.0))
            meta.append({"combo": ctag, "kind": "realization",
                         "tag": cfg_n["run_tag"], "realization": n,
                         "seed": seed0 + n})

    meta = pd.DataFrame(meta)
    meta.to_csv(os.path.join(out_dir, "z_job_plan.csv"), index=False)

    prec = 100.0 / np.sqrt(2 * max(n_realizations - 1, 1))
    results = run_jobs(
        jobs, n_workers=n_workers, log_dir=log_dir, sequential=sequential,
        capture_output=capture_output, dry_run=dry_run,
        title="z-uncertainty Monte Carlo",
        summary_name="z_uncertainty_summary.log",
        extra_lines=[
            f"Models            : {len(combos)}",
            f"Realizations      : {n_realizations} per model "
            f"(+1 unperturbed baseline)",
            f"Scatter precision : ~{prec:.0f}% "
            f"(threshold being tested is {FLAG_FRAC:.0%})",
            f"zerr column       : {zerr_col or CONFIG.get('col_zerr', 'zHDERR')}",
            f"MUERR_VPEC        : "
            + ("removed from covariance (no double-count)" if exclude_muerr_vpec
               else "KEPT (double-counted against zHDERR)"),
        ])
    if dry_run:
        return {"plan": meta}

    if prec > 100 * FLAG_FRAC:
        print(f"\n[warn] With N={n_realizations} the scatter itself is only "
              f"known to ~{prec:.0f}%, which is coarser than the {FLAG_FRAC:.0%} "
              f"threshold being tested.\n       Treat any flag as indicative "
              f"only; use --n-realizations {DEFAULT_N_REALIZATIONS} or more for "
              f"a publishable number.")

    by_tag = {r.label: r.value for r in results
              if r.status == "ok" and r.value is not None}
    return _assemble(combos, by_tag, n_realizations, out_dir)


def _assemble(combos, by_tag, n_realizations, out_dir):
    real_rows, summary_rows = [], []

    for term_names in combos:
        ctag = combo_tag(term_names)
        base = by_tag.get(_tag("baseline", combo=ctag))
        if base is None:
            print(f"[warn] {ctag}: baseline fit failed -- no summary possible "
                  f"for this model (everything is measured relative to it).")
            continue
        params = [p for p in base["active_params"].split(",") if p]

        runs = []
        for n in range(n_realizations):
            v = by_tag.get(_tag("realization", combo=ctag, n=n))
            if v is None:
                continue
            row = {"combo": ctag, "realization": n, "seed": v["seed"],
                   "n_sne": v["n_sne"], "logz": v["logz"],
                   "pkl_path": v["pkl_path"]}
            for p in params:
                row[p] = v.get(f"{p}_mean")
            runs.append(row)
        real_rows.extend(runs)

        if len(runs) < 2:
            print(f"[warn] {ctag}: only {len(runs)} successful realization(s) "
                  f"-- a scatter cannot be estimated.")
            continue

        rdf = pd.DataFrame(runs)
        for p in params:
            base_mean = base.get(f"{p}_mean", np.nan)
            base_std = base.get(f"{p}_std", np.nan)
            vals = pd.to_numeric(rdf.get(p), errors="coerce").dropna()
            if len(vals) < 2:
                continue
            mc_std = float(vals.std(ddof=1))
            mc_mean = float(vals.mean())
            usable = bool(base_std and np.isfinite(base_std) and base_std > 0)
            frac = mc_std / base_std if usable else np.nan
            bias = (mc_mean - base_mean) / base_std if usable else np.nan
            summary_rows.append({
                "combo": ctag, "param": p, "n_realizations": int(len(vals)),
                "baseline_mean": base_mean, "baseline_posterior_std": base_std,
                "mc_mean": mc_mean, "mc_std": mc_std,
                "mc_std_as_frac_of_posterior_std": frac,
                "bias_nsigma": bias,
                "flagged": bool(np.isfinite(frac) and frac >= FLAG_FRAC),
            })

    realizations = pd.DataFrame(real_rows)
    summary = pd.DataFrame(summary_rows)
    realizations.to_csv(os.path.join(out_dir, "z_uncertainty_realizations.csv"),
                        index=False)
    summary.to_csv(os.path.join(out_dir, "z_uncertainty_summary.csv"),
                   index=False)

    # Cosmology-focused view: the columns a reader actually wants first.
    cosmo_params = ["Om0", "w", "Ode0"]
    cosmology = pd.DataFrame()
    if len(summary):
        cosmology = (summary[summary["param"].isin(cosmo_params)]
                     .sort_values(["param", "combo"]).reset_index(drop=True))
        if len(cosmology):
            cosmology.to_csv(
                os.path.join(out_dir, "z_uncertainty_cosmology.csv"), index=False)

    _print_summary(summary, cosmology, n_realizations, out_dir)
    _plot(summary, out_dir)
    return {"realizations": realizations, "summary": summary,
            "cosmology": cosmology}


# ===========================================================================
# 5.  REPORTING
# ===========================================================================

def _print_summary(summary, cosmology, n_realizations, out_dir):
    if not len(summary):
        print("\nNo summary rows produced -- every fit failed. "
              "Check the per-job logs.")
        return

    print("\n" + "=" * 104)
    print(f"  COSMOLOGY IMPACT OF REDSHIFT UNCERTAINTY   "
          f"({n_realizations} realizations per model)")
    print("=" * 104)
    if len(cosmology):
        print(f"  {'model':<44s} {'param':<6s} {'baseline':>11s} "
              f"{'post.sigma':>11s} {'MC scatter':>11s} {'ratio':>7s} "
              f"{'bias':>8s}")
        print("  " + "-" * 100)
        for _, r in cosmology.iterrows():
            flag = "  <-- matters" if r["flagged"] else ""
            print(f"  {r['combo']:<44s} {r['param']:<6s} "
                  f"{r['baseline_mean']:>11.5f} "
                  f"{r['baseline_posterior_std']:>11.5f} "
                  f"{r['mc_std']:>11.5f} "
                  f"{r['mc_std_as_frac_of_posterior_std']:>7.3f} "
                  f"{r['bias_nsigma']:>+7.2f}s{flag}")
    else:
        print("  (no cosmological parameter was active in any model)")

    flagged = summary[summary["flagged"]]
    print("\n" + "=" * 104)
    if len(flagged):
        print(f"  ** {len(flagged)} model/parameter pair(s) have MC scatter "
              f">= {FLAG_FRAC:.0%} of their posterior width **")
        for _, r in flagged.iterrows():
            print(f"     {r['combo']:<44s} {r['param']:<14s} "
                  f"ratio={r['mc_std_as_frac_of_posterior_std']:.3f}  "
                  f"bias={r['bias_nsigma']:+.2f} sigma")
        print("  For these, redshift / peculiar-velocity uncertainty is a "
              "non-negligible part of the error\n  budget and should be quoted "
              "as a systematic.")
    else:
        print(f"  All parameters in all models show MC scatter < "
              f"{FLAG_FRAC:.0%} of their posterior width.")
        print("  Redshift uncertainty is not adding meaningfully to the error "
              "budget beyond what MUERR\n  and the covariance already capture.")

    biased = summary[summary["bias_nsigma"].abs() >= 0.5]
    if len(biased):
        print(f"\n  Note: {len(biased)} parameter(s) also show a mean SHIFT "
              f">= 0.5 sigma between the perturbed\n  ensemble and the "
              f"unperturbed baseline (bias_nsigma) -- a systematic offset, not "
              f"just scatter.\n  A pure-noise perturbation should not move the "
              f"mean, so this is worth investigating.")
    print("=" * 104)
    print(f"  Outputs in {os.path.abspath(out_dir)}/")
    print("    z_uncertainty_realizations.csv  per-realization recovered means")
    print("    z_uncertainty_summary.csv       per model/parameter statistics")
    print("    z_uncertainty_cosmology.csv     cosmology parameters only")
    print("    z_uncertainty.pdf               scatter vs posterior width\n")


def _plot(summary, out_dir):
    """One panel per model: MC scatter beside each parameter's posterior width."""
    if not len(summary):
        return
    combos = list(dict.fromkeys(summary["combo"]))
    n = len(combos)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.8 * nrow),
                             squeeze=False)
    flat = axes.ravel()
    for ax, ctag in zip(flat, combos):
        sub = (summary[summary["combo"] == ctag]
               .sort_values("mc_std_as_frac_of_posterior_std", ascending=False))
        x = np.arange(len(sub))
        ax.bar(x, sub["mc_std"], width=0.4, color="steelblue",
               label="MC scatter (z perturbed)")
        ax.bar(x + 0.4, sub["baseline_posterior_std"], width=0.4,
               color="grey", alpha=0.75, label="posterior sigma (unperturbed)")
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels(sub["param"], rotation=45, ha="right", fontsize=8)
        ax.set_title(ctag, fontsize=8)
        ax.set_ylabel("std")
        ax.set_yscale("log")
    for ax in flat[n:]:
        ax.axis("off")
    flat[0].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "z_uncertainty.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


# ===========================================================================
# 6.  CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo redshift-uncertainty propagation check across "
                    "every competing model: refit N times with z perturbed by "
                    "its own uncertainty, then compare the scatter -- and any "
                    "systematic shift -- to the unperturbed posterior width.")
    p.add_argument("--only", default=None,
                   help="Comma-separated combo tags to run "
                        "(default: all of best_model.COMBOS).")
    p.add_argument("--best-model-only", action="store_true",
                   help="Run only best_model.BEST_COMBO.")
    p.add_argument("--list", action="store_true",
                   help="List available combo tags and exit.")
    p.add_argument("--n-realizations", type=int, default=DEFAULT_N_REALIZATIONS,
                   help=f"Perturbed refits per model (default "
                        f"{DEFAULT_N_REALIZATIONS}; see CHOOSING N in the "
                        f"module docstring).")
    p.add_argument("--seed0", type=int, default=2000,
                   help="Base RNG seed. Realization n uses seed0 + n for every "
                        "model, so all models see identical z draws.")
    p.add_argument("--zerr-col", default=None,
                   help="Redshift-uncertainty column to perturb by. Default "
                        "'zHDERR' (includes peculiar-velocity uncertainty); "
                        "use 'zCMBERR' for a measurement-noise-only variant.")
    p.add_argument("--include-muerr-vpec", action="store_true",
                   help="Do NOT remove MUERR_VPEC from the covariance first. "
                        "Reproduces the older behaviour, which double-counts "
                        "peculiar velocity against zHDERR.")
    p.add_argument("--registry-file", default=REGISTRY)
    p.add_argument("--out-dir", default=OUT_DIR)
    add_parallel_args(p, default_log_dir="logs/z_uncert")
    return p.parse_args()


# The __main__ guard is REQUIRED, not stylistic: parallel_runner spawns
# workers, and spawn re-imports this module in every child. Without the guard
# each child would re-run the whole check.
if __name__ == "__main__":
    args = _parse_args()
    if args.list:
        print("Available combos (best_model.COMBOS):")
        for c in COMBOS:
            print(f"  {combo_tag(c)}")
        raise SystemExit(0)

    run_z_uncertainty_check(
        combos=[BEST_COMBO] if args.best_model_only else None,
        only=[s.strip() for s in args.only.split(",")] if args.only else None,
        n_realizations=args.n_realizations,
        seed0=args.seed0,
        zerr_col=args.zerr_col,
        exclude_muerr_vpec=not args.include_muerr_vpec,
        registry_file=args.registry_file,
        n_workers=args.workers,
        sequential=args.sequential,
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        capture_output=not args.no_capture,
    )
