"""
injection_test.py  —  SNe Ia Cosmology Pipeline
==================================================
Recovery/injection test: does this pipeline correctly recover a KNOWN
correction model from data built to look like your real sample?

Rather than synthesising z / x1 / c / host properties from scratch (which
risks missing real structure — selection effects, correlations between
host mass and sSFR, etc.), this script keeps EVERY real column from your
actual data — z, x1, c, logM, host_colour, logsSFR, all their measurement
errors, delta_bias, the full covariance — and replaces ONLY x0 with a
synthetic value manufactured so that the TRUE generating model (a chosen
set of parameter values you specify) is known exactly. Realistic,
correlated noise consistent with your actual covariance matrix (geometric
+ MUERR + sigma_int, exactly as build_covariance/load_and_filter_data
construct it for a real fit) is added on top.

How the synthetic x0 is built
------------------------------
compute_mu_corr's distance-modulus expression is additively separable in
x0 — every other term (host correction, colour/stretch corrections, all
interaction terms) is independent of x0:

    mu_corr = -2.5*log10(x0)  +  A(z, x1, c, logM, host_colour, logsSFR; theta)

So rather than re-deriving A by hand (risking it drifting out of sync with
core.py's actual formula as new interaction terms get added), we call
compute_mu_corr ONCE with x0 fixed to 1 (so -2.5*log10(1) = 0) to read off
A directly, then solve for the x0 that produces any target mu_corr:

    x0_synthetic = 10 ** ( -0.4 * (mu_corr_target - A) )

mu_corr_target = mu_theory(z, Om0_true) + M_fiducial + correlated_noise,
with correlated_noise drawn once from the SAME covariance a real fit on
this dataset would use.

The pipeline is then run on this data exactly as run_sampler normally
would (via its `preloaded=` hook), and the recovered posterior is compared
to the injected truth via a Gaussian Mahalanobis "recovery tension" (the
same statistic compare_runs.gaussian_tension uses for two real runs,
applied here against a zero-covariance "truth point").

Usage
-----
  from injection_test import run_injection_test
  report = run_injection_test(
      config_overrides={"model": {"mass": "step", "host_colour": "linear",
                                  "ssfr": "none", "sn_colour": "linear",
                                  "x1_correction": "linear", "z_evolve": "power"}},
      true_params={"alpha": 0.15, "beta": 3.0, "gamma": 0.08, "M0": 10.0,
                  "tau": 0.2, "eta": 0.0, "xi_mass_col": 0.0, "Om0": 0.3,
                  ... one entry per ACTIVE parameter in that model_cfg ...})

or from the command line with a JSON truth file:
  python injection_test.py --tag injection_check --truth truth.json
"""

import argparse
import copy
import json

import numpy as np
from scipy.linalg import cholesky

from config       import CONFIG, DEFAULT_PARAM_SPECS
from run          import (load_and_filter_data, run_sampler, pkl_path_for,
                          build_param_getter, infer_cosmo_type)
from core         import compute_mu_corr, mu_theory
from compare_runs import gaussian_tension, enclosed_prob_to_nsigma
from dynesty      import utils as dyfunc


def _cosmo_kwargs(true_params, cosmo_type):
    Om0 = true_params["Om0"]
    if cosmo_type == "FlatLambdaCDM":
        return {"Om0": Om0, "cosmo_type": cosmo_type}
    elif cosmo_type == "wCDM":
        return {"Om0": Om0, "w": true_params["w"], "cosmo_type": cosmo_type}
    elif cosmo_type == "LambdaCDM":
        return {"Om0": Om0, "Ode0": true_params["Ode0"], "cosmo_type": cosmo_type}
    raise ValueError(f"Unknown cosmo_type '{cosmo_type}'")


def _full_true_params(param_specs, active_names, user_truth):
    """
    Build the complete parameter dict compute_mu_corr/mu_theory need: user-
    supplied truth values for every ACTIVE parameter, fixed values (from
    param_specs) for every inactive one. Raises a clear error listing any
    active parameter the caller forgot to specify a truth for, rather than
    silently defaulting it (a silently-defaulted "truth" would make the
    recovery test meaningless for that parameter).
    """
    missing = [n for n in active_names if n not in user_truth]
    if missing:
        raise ValueError(
            f"true_params is missing a truth value for active parameter(s) "
            f"{missing}. Every active parameter needs an injected truth for "
            f"this test to mean anything — see param_specs for what's active "
            f"under your chosen model_cfg.")
    full = {name: spec["fixed"] for name, spec in param_specs.items()}
    full.update({n: user_truth[n] for n in active_names})
    # Allow the caller to also override specific FIXED parameters (e.g. w,
    # Ode0 for a non-flat truth) via true_params even if they're inactive.
    for n, v in user_truth.items():
        full[n] = v
    return full


def build_injected_dataset(config_overrides, true_params, M_fiducial=-19.3,
                           sigma_int_inject=None, seed=0):
    """
    Load the real (filtered) dataset and replace x0 with a synthetic value
    consistent with `true_params`, keeping every other column real.

    Returns
    -------
    fold_cfg   : the config actually used (model_cfg resolved, etc.)
    preloaded  : (df, data, cov_mat, inv_cov_mat, log_det_const, C_sum,
                 keep_idx) tuple, with data["x0"] replaced — ready to pass
                 straight to run_sampler(fold_cfg, preloaded=preloaded).
    true_params_full : the complete parameter dict actually injected
                       (active truths + fixed values), for later comparison.
    """
    cfg = copy.deepcopy(CONFIG)
    cfg.update(copy.deepcopy(config_overrides))

    df, data, cov_mat, inv_cov_mat, log_det_const, C_sum, keep_idx = \
        load_and_filter_data(cfg)

    param_specs  = copy.deepcopy(cfg.get("param_specs", DEFAULT_PARAM_SPECS))
    model_cfg    = cfg["model"]
    cosmo_type   = infer_cosmo_type(param_specs)
    active_names = [n for n, s in param_specs.items() if s["active"]]

    true_full = _full_true_params(param_specs, active_names, true_params)
    sigma_int = (cfg.get("sigma_int", 0.0) if sigma_int_inject is None
                else sigma_int_inject)

    # ---- A(z, x1, c, logM, host_colour, logsSFR; theta) via x0=1 trick ----
    data_x0_one = dict(data)
    data_x0_one["x0"] = np.ones_like(data["x0"])
    A = compute_mu_corr(data_x0_one, true_full, model_cfg)

    # ---- Correlated noise from the SAME covariance a real fit would use ----
    rng      = np.random.default_rng(seed)
    muerr    = data["muerr"]
    cov_full = cov_mat + np.diag(muerr**2)
    if sigma_int > 0:
        cov_full = cov_full + np.diag(np.full(len(muerr), sigma_int**2))
    L     = cholesky(cov_full, lower=True)
    noise = L @ rng.standard_normal(len(muerr))

    mu_th_true    = mu_theory(data["z"], **_cosmo_kwargs(true_full, cosmo_type))
    mu_corr_target = mu_th_true + M_fiducial + noise
    
    print(f"Injected fiducial M = {M_fiducial:.4f}, sigma_int = {sigma_int:.4f}, "
          f"seed = {seed}, noise std = {np.std(noise):.4f}")
    print(f"Injected mu_corr_target = {mu_corr_target}")
    
    # raise ValueError("DEBUG: stop here to check injected mu_corr_target and noise std")

    x0_synth = 10.0 ** (-0.4 * (mu_corr_target - A))

    synthetic_data       = dict(data)
    synthetic_data["x0"] = x0_synth

    preloaded = (df, synthetic_data, cov_mat, inv_cov_mat, log_det_const,
                C_sum, keep_idx)
    return cfg, preloaded, true_full, active_names

def run_injection_test(config_overrides=None, true_params=None,
                       M_fiducial=-19.3, sigma_int_inject=None, seed=0,
                       nsigma_flag=2.0):
    """
    Full injection/recovery test: build synthetic data with a known truth,
    fit it with run_sampler exactly as a real analysis would, and report
    per-parameter pulls plus an overall Mahalanobis "recovery tension"
    (same statistic compare_runs.gaussian_tension uses for two real runs,
    here applied against the injected truth treated as a zero-covariance
    point).

    Parameters
    ----------
    config_overrides : dict — must include "model" (the model_cfg to test)
        and normally a "run_tag"; anything else layers on top of CONFIG
        exactly as for a normal run_sampler call.
    true_params : dict, truth values for every ACTIVE parameter under that
        model_cfg (see _full_true_params — missing entries raise an error
        rather than silently defaulting).
    M_fiducial : injected fiducial absolute magnitude offset (arbitrary —
        M is analytically marginalised by the fit either way, so this
        value itself is never checked, only the OTHER parameters are).
    sigma_int_inject : intrinsic scatter to inject; defaults to
        CONFIG["sigma_int"]/config_overrides["sigma_int"] if not given, so
        the injected noise model matches what the fit itself assumes.
    seed : noise RNG seed — fixed for reproducibility.
    nsigma_flag : per-parameter pulls at or above this are flagged in the
        printed report.

    Returns
    -------
    dict: per-parameter recovered mean/std/truth/pull, overall Mahalanobis
    nsigma, and the pkl path of the fit (for further inspection with
    compare_runs.py / degeneracy_scan.py).
    """
    config_overrides = dict(config_overrides or {})
    if "model" not in config_overrides:
        raise ValueError('config_overrides must include "model" (the '
                         "model_cfg to inject and recover).")
    # Injection/recovery fits are synthetic-data sanity checks, not part of
    # the science case -- default them to their own registry so they never
    # land in run_publication_registry.csv (CONFIG's default) unless the
    # caller explicitly asks for that via config_overrides["registry_file"].
    config_overrides.setdefault("registry_file", "run_injection_registry.csv")
    true_params = dict(true_params or {})
    base_tag = config_overrides.pop("run_tag", "injection_test")

    cfg, preloaded, true_full, active_names = build_injected_dataset(
        config_overrides, true_params, M_fiducial=M_fiducial,
        sigma_int_inject=sigma_int_inject, seed=seed)
    cfg["run_tag"] = base_tag

    print(f"\n{'='*60}\nInjection test: fitting synthetic data "
          f"(model={cfg['model']}, seed={seed})\n{'='*60}")
    print("Injected truth (active parameters):")
    for n in active_names:
        print(f"  {n:16s} = {true_full[n]}")

    results, sampler, active_names_fit, data_fit, run_name = run_sampler(
        cfg, preloaded=preloaded)

    weights = np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(results.samples, weights)
    std = np.sqrt(np.diag(cov))
    truth_vec = np.array([true_full[n] for n in active_names_fit])

    tension = gaussian_tension(mean, cov, truth_vec, np.zeros_like(cov))

    per_param = []
    print(f"\n{'Parameter':16s} {'truth':>10s} {'recovered':>12s} "
          f"{'std':>9s} {'pull':>7s}")
    for i, n in enumerate(active_names_fit):
        pull = (mean[i] - truth_vec[i]) / std[i] if std[i] > 0 else np.nan
        flag = "  **" if abs(pull) >= nsigma_flag else ""
        print(f"{n:16s} {truth_vec[i]:10.4f} {mean[i]:12.4f} "
              f"{std[i]:9.4f} {pull:7.2f}{flag}")
        per_param.append({"param": n, "truth": float(truth_vec[i]),
                          "recovered_mean": float(mean[i]),
                          "recovered_std": float(std[i]),
                          "pull": float(pull)})

    print(f"\nOverall recovery tension: {tension['nsigma']:.2f} sigma "
          f"(chi2/dof = {tension['Q']:.2f}/{tension['dof']}, "
          f"p={tension['pvalue']:.4g})")
    if tension["nsigma"] >= nsigma_flag:
        print(f"** Recovery tension >= {nsigma_flag} sigma — the pipeline "
              f"did NOT cleanly recover the injected truth. Check the "
              f"per-parameter pulls above for which parameter(s) are "
              f"driving it before trusting this model on real data. **")
    else:
        print(f"Recovery consistent with the injected truth within "
              f"{nsigma_flag} sigma.")

    return {"per_param": per_param, "overall_nsigma": tension["nsigma"],
            "overall_Q": tension["Q"], "overall_dof": tension["dof"],
            "overall_pvalue": tension["pvalue"],
            "pkl_path": pkl_path_for(run_name, cfg)}


def _param_specs_from_overrides(param_overrides):
    """
    Build a full param_specs dict (every entry from DEFAULT_PARAM_SPECS,
    with the given per-parameter fields overridden) -- same pattern as
    experiment_runner.py / extra_runners.py's _override() helper, exposed
    here so a --truth JSON file can activate/deactivate/reprior parameters
    beyond CONFIG's defaults (e.g. to inject+recover a richer model like
    "gamma_alpha + sSFR tanh + linear mass", not just the plain baseline).
    """
    specs = copy.deepcopy(DEFAULT_PARAM_SPECS)
    for name, updates in (param_overrides or {}).items():
        specs[name].update(updates)
    return specs


def _parse_args():
    p = argparse.ArgumentParser(
        description="Recovery/injection test: inject a known correction "
                    "model into real host/SN-property data with a "
                    "synthetic x0, refit, and check recovery.")
    p.add_argument("--tag", default="injection_test",
                   help="Base run-tag / output-file prefix. Each seed's "
                        "outputs get '_seed_<n>' appended before the "
                        "extension (e.g. Plots/<tag>_seed_1_results.pkl, "
                        "..._seed_1_corner.pdf), so runs are easy to tell "
                        "apart and to feed into compare_runs.py / "
                        "degeneracy_scan.py pairwise.")
    p.add_argument("--truth", required=True,
                   help='Path to a JSON file: {"model": {...}, '
                        '"true_params": {...}, "param_overrides": {...} '
                        '(optional), "config_overrides": {...} (optional)} '
                        "-- see injection_test.py docstring / example JSON "
                        "files for the full format.")
    p.add_argument("--seed", type=int, default=1,
                   help="Starting seed (default 1). With --num-seeds N, "
                        "runs seeds --seed, --seed+1, ..., --seed+N-1.")
    p.add_argument("--num-seeds", type=int, default=1,
                   help="Number of seeds to run sequentially, starting at "
                        "--seed (default 1, i.e. just the one seed).")
    return p.parse_args()


def _build_config_overrides(args, spec, seed):
    """Per-seed config_overrides dict: same base config, but run_tag gets
    '_seed_<n>' appended so every output file (pkl, corner plot, Hubble
    plot -- see run.pkl_path_for / run_sampler's output_prefix, which is
    derived entirely from run_tag) is uniquely named per seed."""
    config_overrides = {"run_tag": f"{args.tag}_seed_{seed}",
                        "model": spec["model"]}

    # Optional: activate/deactivate/reprior parameters beyond CONFIG's
    # defaults (e.g. turn on gamma_alpha, switch to a tanh sSFR term, ...).
    # Every parameter this makes ACTIVE must have a matching entry in
    # true_params, or run_injection_test raises a clear error listing
    # what's missing -- see _full_true_params.
    if "param_overrides" in spec:
        config_overrides["param_specs"] = _param_specs_from_overrides(
            spec["param_overrides"])

    # Optional: any other CONFIG field, most usefully "nlive_mode"/"nlive"
    # to keep a smoke-test fast (CONFIG's own default is "publication",
    # i.e. ndim x 500 live points -- likely overkill for "does this run at
    # all and recover something sane").
    config_overrides.update(spec.get("config_overrides", {}))
    return config_overrides


def _print_multi_seed_summary(rows):
    """rows: list of (seed, overall_nsigma, worst_param, worst_pull)."""
    print(f"\n{'='*60}\nMulti-seed injection test summary\n{'='*60}")
    print(f"{'seed':>6s} {'overall nsigma':>15s} {'worst param':>16s} {'worst pull':>11s}")
    for seed, nsigma, worst_name, worst_pull in rows:
        flag = "  **" if nsigma >= 2.0 else ""
        print(f"{seed:6d} {nsigma:15.2f} {worst_name:>16s} {worst_pull:11.2f}{flag}")
    n_flagged = sum(1 for _, nsigma, _, _ in rows if nsigma >= 2.0)
    print(f"\n{n_flagged}/{len(rows)} seed(s) at or above 2.0 sigma overall tension.")
    if len(rows) == 1:
        print("Only one seed run -- a single flagged seed is not on its own "
             "strong evidence of a bug (see injection_test.py docstring). "
             "Rerun with --num-seeds 3-5 to check whether it's consistent.")
    elif n_flagged >= 1 and n_flagged < len(rows):
        print("Flagged on some seeds but not others -- check whether the "
             "SAME parameter(s) recur across the flagged seeds (a real "
             "issue) vs. different ones each time (sampling noise / a "
             "parameter degeneracy -- see degeneracy_scan.py).")
    elif n_flagged == len(rows) and len(rows) > 1:
        print("Flagged on every seed -- this looks like a persistent bias, "
             "not noise. Check the recurring parameter(s) above against "
             "their prior (an informative prior centred away from the "
             "injected truth biases the average recovery -- see the "
             "*_uniformpriors.json truth files for an isolating test) and "
             "against core.py's compute_mu_corr for this model.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = _parse_args()
    with open(args.truth) as f:
        spec = json.load(f)

    summary_rows = []
    for i in range(args.num_seeds):
        seed = args.seed + i
        config_overrides = _build_config_overrides(args, spec, seed)
        print(f"\n{'#'*60}\n# Seed {seed}  ({i + 1}/{args.num_seeds})\n{'#'*60}")
        report = run_injection_test(
            config_overrides=config_overrides,
            true_params=spec["true_params"], seed=seed)

        worst = max(report["per_param"], key=lambda r: abs(r["pull"]))
        summary_rows.append((seed, report["overall_nsigma"],
                            worst["param"], worst["pull"]))

    if args.num_seeds > 1:
        _print_multi_seed_summary(summary_rows)