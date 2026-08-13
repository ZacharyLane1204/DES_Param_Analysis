"""
z_uncertainty_check.py  —  SNe Ia Cosmology Pipeline
========================================================
Monte-Carlo redshift-uncertainty propagation check -- option (b) from the
z-uncertainty discussion: rather than marginalising z uncertainty into
the likelihood via quadrature (which risks double-counting against
MUERR, which already appears to include a peculiar-velocity contribution
via MUERR_VPEC, and is much more expensive since z enters mu_theory's
cosmology distance integral rather than a cheap algebraic profile), this
refits a chosen model N_REALIZATIONS times, each time with data["z"]
independently perturbed by a Gaussian draw of width zerr_col (default
"zHDERR") BEFORE anything is computed from it. Both mu_theory(z) AND any
z_evolve correction in compute_mu_corr consume the SAME perturbed z, so
the check propagates consistently through everything downstream of
redshift, not just the cosmology distance modulus.

Reports the SCATTER of each active parameter's recovered posterior mean
across realizations, against that same parameter's own posterior width
from a single unperturbed fit -- the number that answers "does redshift
measurement uncertainty matter": if the MC scatter is small relative to
the reported posterior sigma, z uncertainty is not adding meaningfully to
your error budget beyond what MUERR/the covariance already capture.

Like loo_zbins.py, this does NOT touch config["data_file"] or create new
CSVs, and does NOT refactorise the covariance per realization -- the
geometric covariance (host photometric errors etc.) does not depend on z,
only data["z"] itself changes, so the original inv_cov_mat/log_det_const/
C_sum from load_and_filter_data are reused unmodified for every
realization. Each realization is handed to run_sampler via `preloaded=`,
so it reuses the exact same sampling/saving/plotting/registry code path
as a normal run.

Usage
-----
  python z_uncertainty_check.py --tag best_model --n-realizations 8

or:
  from z_uncertainty_check import run_z_uncertainty_check
  report = run_z_uncertainty_check(config_overrides={"run_tag": "best_model",
                                                     "model": {...}},
                                   n_realizations=8)
"""

import argparse
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dynesty import utils as dyfunc

from config    import CONFIG
from run       import load_and_filter_data, run_sampler, pkl_path_for, load_results
from loo_zbins import _refactorise_covariance


def run_z_uncertainty_check(config_overrides=None, n_realizations=8, seed0=2000,
                            zerr_col=None, output_prefix=None, baseline_pkl=None,
                            exclude_muerr_vpec=True, muerr_vpec_col=None):
    """
    Parameters
    ----------
    config_overrides : dict layered on top of CONFIG -- specify the model
        to test (config_overrides["model"] = {...}), param_specs
        overrides, and a run_tag. Keep zlo/zhi/other cuts identical to
        baseline_pkl's if you pass one, so the comparison is apples-to-
        apples.
    n_realizations : number of independent Gaussian z-perturbation
        realizations to refit (drawn fresh per SN, independent across SNe
        and across realizations).
    seed0          : base RNG seed; realization i uses seed0 + i.
    zerr_col       : CSV column with the per-SN redshift uncertainty to
        draw perturbations from. Defaults to CONFIG.get("col_zerr",
        "zHDERR") -- the Hubble-diagram redshift's total uncertainty.
        zHDERR is intentionally the default: it is the column that
        actually carries the peculiar-velocity/flow-model correction's
        uncertainty (it is NOT just measurement noise -- zHDERR correlates
        with VPECERR at r~0.7 in this sample and sits a near-constant
        ~0.0008 above zCMBERR, matching a peculiar-velocity floor added in
        quadrature). zCMBERR/zHELERR are the redshift BEFORE that
        correction is applied and so do not carry the uncertainty this
        check is meant to probe -- use zerr_col="zCMBERR" only if you
        deliberately want a pure-measurement-noise version of this check
        for comparison, not as the headline result.
    exclude_muerr_vpec : bool -- since zerr_col="zHDERR" (the default)
        already carries the peculiar-velocity uncertainty, and MUERR_VPEC
        is the SAME physical contribution already folded into the
        covariance in magnitude space (see config.py's col_muerr / the
        data file's MUERR = f(MUERR_RAW, MUERR_VPEC, ...)), leaving both
        in place double-counts peculiar-velocity uncertainty: once as a
        z-shift here, once as a magnitude-space term in every covariance
        entry. When True (default), this check rebuilds the covariance
        for the baseline AND every realization with MUERR_VPEC's
        contribution removed in quadrature from muerr
        (sqrt(muerr**2 - muerr_vpec**2)) BEFORE folding in the geometric
        covariance, so peculiar-velocity uncertainty enters through the
        z-perturbation only -- the honest way to ask "does treating PV
        uncertainty as a redshift shift instead of a linearised mu-space
        term change my answer". Set False to reproduce the old (double-
        counted) behaviour, e.g. for a direct before/after comparison.
    muerr_vpec_col : CSV column holding the magnitude-space peculiar-
        velocity error term. Defaults to CONFIG.get("col_muerr_vpec",
        "MUERR_VPEC"). Only used when exclude_muerr_vpec=True.
    output_prefix  : basename for the summary CSV/plot; defaults to run_tag.
    baseline_pkl   : path to an existing UN-perturbed fit's
        "<...>_results.pkl" on identical cuts/model, to compare each
        realization's posterior mean against that run's own posterior
        WIDTH. If None, this fits one additional unperturbed baseline
        itself first. NOTE: if you pass an existing baseline_pkl, make
        sure it was fit with the SAME exclude_muerr_vpec setting as this
        call, or the "posterior width" you're comparing MC scatter
        against will itself be double-counted PV uncertainty and every
        ratio in the summary will be biased low.

    Returns
    -------
    dict:
        realizations : pandas.DataFrame, one row per realization (seed,
            pkl_path, <param>_mean for every active parameter)
        summary      : pandas.DataFrame, one row per parameter:
            baseline_mean, baseline_posterior_std, mc_mean_across_
            realizations, mc_std_across_realizations,
            mc_std_as_frac_of_posterior_std  <- the number to read
        baseline_pkl : the unperturbed-fit pkl used for comparison
    """
    config_overrides = dict(config_overrides or {})
    # This check's baseline + every MC realization are systematic-check
    # fits, not part of the science case -- default them to their own
    # registry so they never land in run_publication_registry.csv
    # (CONFIG's default) unless explicitly overridden, matching
    # injection_test.py / extra_runners.py's pattern.
    config_overrides.setdefault("registry_file", "run_z_uncertainty_registry.csv")
    base_tag = config_overrides.pop("run_tag", "z_uncertainty")
    output_prefix = output_prefix or base_tag.replace("/", "_")

    base_cfg = copy.deepcopy(CONFIG)
    base_cfg.update(copy.deepcopy(config_overrides))
    zerr_col = zerr_col or base_cfg.get("col_zerr", "zHDERR")

    print(f"\n{'='*60}\nz-uncertainty MC check: loading full filtered "
         f"sample...\n{'='*60}")
    df, data, cov_mat, inv_cov_mat, log_det_const, C_sum, keep_idx = \
        load_and_filter_data(base_cfg)
    # cov_mat here is the pure geometric (pre-muerr, pre-sigma_int)
    # covariance -- see load_and_filter_data's docstring/return contract.

    if zerr_col not in df.columns:
        raise KeyError(f"z_uncertainty_check requires column '{zerr_col}' "
                       f"in the data CSV (see zerr_col=/config['col_zerr']).")
    zerr = df[zerr_col].values
    z0   = data["z"].copy()

    # ---- Remove MUERR_VPEC's contribution to avoid double-counting PV ----
    # See exclude_muerr_vpec's docstring above. This only matters when
    # zerr_col itself carries peculiar-velocity uncertainty (true for the
    # default "zHDERR"; not true for a pure-measurement column like
    # "zCMBERR"/"zHELERR", where there is nothing to remove and this block
    # is a harmless near-no-op).
    if exclude_muerr_vpec:
        muerr_vpec_col = muerr_vpec_col or base_cfg.get("col_muerr_vpec", "MUERR_VPEC")
        if muerr_vpec_col not in df.columns:
            raise KeyError(f"exclude_muerr_vpec=True requires column "
                           f"'{muerr_vpec_col}' in the data CSV (see "
                           f"muerr_vpec_col=/config['col_muerr_vpec'], or "
                           f"pass exclude_muerr_vpec=False to skip this).")
        muerr_vpec  = df[muerr_vpec_col].values
        muerr_full  = data["muerr"].copy()
        muerr_novpec = np.sqrt(np.clip(muerr_full**2 - muerr_vpec**2, 0.0, None))
        n_clipped = int(np.sum(muerr_full**2 < muerr_vpec**2))
        print(f"exclude_muerr_vpec=True: removing '{muerr_vpec_col}' from "
             f"muerr in quadrature before refactorising the covariance "
             f"(mean muerr {muerr_full.mean():.5f} -> {muerr_novpec.mean():.5f}"
             f"{f'; {n_clipped} SNe clipped at 0 -- MUERR_VPEC exceeded MUERR' if n_clipped else ''})."
             f" Peculiar-velocity uncertainty will enter this check only "
             f"through the z-perturbation below.")
        data["muerr"] = muerr_novpec
        sigma_int = base_cfg.get("sigma_int", 0.0)
        inv_cov_mat, log_det_const, C_sum = _refactorise_covariance(
            cov_mat, muerr_novpec, sigma_int)

    # ---- 0. Unperturbed baseline (unless one is already supplied) ----
    if baseline_pkl is None:
        base_fit_cfg = copy.deepcopy(base_cfg)
        base_fit_cfg["run_tag"] = f"{base_tag}/z_unperturbed"
        preloaded0 = (df, data, cov_mat, inv_cov_mat, log_det_const, C_sum, keep_idx)
        print(f"\n{'#'*60}\n# Unperturbed baseline\n{'#'*60}")
        _, _, _, _, run_name0 = run_sampler(base_fit_cfg, preloaded=preloaded0)
        baseline_pkl = pkl_path_for(run_name0, base_fit_cfg)

    results0, active0, _, _ = load_results(baseline_pkl)
    weights0 = np.exp(results0.logwt - results0.logz[-1])
    mean0, cov0 = dyfunc.mean_and_cov(results0.samples, weights0)
    std0 = np.sqrt(np.diag(cov0))
    baseline_means = {name: float(m) for name, m in zip(active0, mean0)}
    baseline_stds  = {name: float(s) for name, s in zip(active0, std0)}

    # ---- 1..N. Perturbed realizations ----
    rows = []
    for i in range(n_realizations):
        rng = np.random.default_rng(seed0 + i)
        z_perturbed = z0 + rng.normal(0.0, zerr)
        z_perturbed = np.clip(z_perturbed, 1e-4, None)   # keep z physical

        data_i = dict(data)
        data_i["z"] = z_perturbed

        cfg_i = copy.deepcopy(base_cfg)
        cfg_i["run_tag"] = f"{base_tag}/z_mc{i}"
        preloaded_i = (df, data_i, cov_mat, inv_cov_mat, log_det_const, C_sum, keep_idx)

        print(f"\n{'#'*60}\n# Realization {i+1}/{n_realizations} "
             f"(seed={seed0 + i})\n{'#'*60}")
        results_i, _, active_i, _, run_name_i = run_sampler(cfg_i, preloaded=preloaded_i)
        pkl_i = pkl_path_for(run_name_i, cfg_i)

        weights_i = np.exp(results_i.logwt - results_i.logz[-1])
        mean_i, _ = dyfunc.mean_and_cov(results_i.samples, weights_i)

        row = {"realization": i, "seed": seed0 + i, "pkl_path": pkl_i}
        for name, m in zip(active_i, mean_i):
            row[f"{name}_mean"] = float(m)
        rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(f"{output_prefix}_z_uncertainty_mc.csv", index=False)

    # ---- Scatter summary: MC scatter vs. the baseline's own posterior width ----
    summary_rows = []
    for name in active0:
        col = f"{name}_mean"
        if col not in report.columns:
            continue
        mc_std   = float(report[col].std(ddof=1))
        base_std = baseline_stds.get(name, np.nan)
        frac = mc_std / base_std if base_std and np.isfinite(base_std) else np.nan
        summary_rows.append({
            "param": name,
            "baseline_mean": baseline_means.get(name, np.nan),
            "baseline_posterior_std": base_std,
            "mc_mean_across_realizations": float(report[col].mean()),
            "mc_std_across_realizations": mc_std,
            "mc_std_as_frac_of_posterior_std": frac,
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        "mc_std_as_frac_of_posterior_std", ascending=False).reset_index(drop=True)
    summary.to_csv(f"{output_prefix}_z_uncertainty_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"z-uncertainty MC summary ({n_realizations} realizations, "
         f"zerr_col='{zerr_col}'):")
    for _, r in summary.iterrows():
        ratio = r["mc_std_as_frac_of_posterior_std"]
        flag = (" ** >=20% of posterior width -- z uncertainty matters "
               "here **" if np.isfinite(ratio) and ratio >= 0.2 else "")
        print(f"  {r['param']:14s} MC scatter = {r['mc_std_across_realizations']:.5f}  "
             f"posterior std = {r['baseline_posterior_std']:.5f}  "
             f"ratio = {ratio:.3f}{flag}")

    n_flagged = int((summary["mc_std_as_frac_of_posterior_std"] >= 0.2).sum())
    if n_flagged:
        print(f"\n** {n_flagged} parameter(s) show MC scatter >= 20% of "
             f"their posterior width -- redshift measurement/peculiar-"
             f"velocity uncertainty is a non-negligible contributor for "
             f"these; consider whether MUERR_VPEC's fiducial-cosmology "
             f"assumption needs revisiting for this parameter. **")
    else:
        print(f"\nAll parameters show MC scatter < 20% of their posterior "
             f"width -- redshift uncertainty is not adding meaningfully to "
             f"the error budget beyond what MUERR/the covariance already "
             f"capture.")
    print(f"\nPer-realization CSV: {output_prefix}_z_uncertainty_mc.csv")
    print(f"Summary CSV:         {output_prefix}_z_uncertainty_summary.csv")
    print(f"{'='*60}\n")

    # ---- Plot: MC scatter vs. posterior width, per parameter ----
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary))
    ax.bar(x,       summary["mc_std_across_realizations"], width=0.4,
          label="MC scatter (z perturbation)", color="steelblue")
    ax.bar(x + 0.4, summary["baseline_posterior_std"],      width=0.4,
          label="posterior std (unperturbed)", color="grey", alpha=0.7)
    ax.set_xticks(x + 0.2); ax.set_xticklabels(summary["param"], rotation=45, ha="right")
    ax.set_ylabel("std")
    ax.set_title(f"{output_prefix}: z-uncertainty MC scatter vs. posterior width")
    ax.legend()
    fig.tight_layout()
    plot_path = f"{output_prefix}_z_uncertainty.pdf"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    return {"realizations": report, "summary": summary, "baseline_pkl": baseline_pkl}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo redshift-uncertainty propagation check: "
                    "refit N times with z perturbed by its own uncertainty "
                    "and compare the scatter to the unperturbed posterior "
                    "width.")
    p.add_argument("--tag", default="z_uncertainty")
    p.add_argument("--n-realizations", type=int, default=8)
    p.add_argument("--seed0", type=int, default=2000)
    p.add_argument("--zerr-col", default=None,
                   help="Redshift-uncertainty column to perturb by. Default "
                        "'zHDERR' (carries peculiar-velocity uncertainty). "
                        "Use 'zCMBERR' for a pure-measurement-noise variant.")
    p.add_argument("--baseline-pkl", default=None)
    p.add_argument("--include-muerr-vpec", action="store_true",
                   help="Do NOT remove MUERR_VPEC from the covariance "
                        "before this check (reproduces the old double-"
                        "counted-against-zHDERR behaviour; default is to "
                        "remove it -- see exclude_muerr_vpec docstring).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_z_uncertainty_check(config_overrides={"run_tag": args.tag},
                            n_realizations=args.n_realizations,
                            seed0=args.seed0, zerr_col=args.zerr_col,
                            baseline_pkl=args.baseline_pkl,
                            exclude_muerr_vpec=not args.include_muerr_vpec)