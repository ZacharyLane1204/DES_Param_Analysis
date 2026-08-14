"""
run.py  —  SNe Ia Cosmology Pipeline
======================================
Orchestration layer.  Imports physics from core.py and settings from config.py.
You should not need to edit this file; change config.py instead.

Usage
-----
  python run.py                        # uses CONFIG from config.py
  python run.py --tag wCDM_first_look  # appends a tag to the run name
  python run.py --config my_config.py  # (future: load an alternate config)

Run naming
----------
Each run gets a unique name of the form:

    YYYYMMDD_HHMMSS[_tag]

e.g.  20240601_143022  or  20240601_143022_wCDM_first_look

All output files for a run (pkl, corner pdf, Hubble pdf) share this prefix.
A summary row is appended to run_registry.csv (path set in CONFIG) after
every successful run.

Run registry columns
--------------------
  run_name, timestamp, cosmo_type, sn_colour_model, mass_model,
  host_colour_model, z_evolve_model, sigma_int, nlive, dlogz,
  active_params,          ← comma-separated list of sampled params
  <param>_active,         ← one column per known param (True/False)
  <param>_prior,          ← prior type string (or "fixed")
  <param>_fixed,          ← fixed value if inactive, else ""
  logZ, logZ_err
"""

import os
import sys
import copy
import math
import time
import pickle
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats import gaussian_kde
from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import utils as dyfunc
import dynesty.plotting as dyplot

# ---- project imports ----
from config import CONFIG, DEFAULT_PARAM_SPECS, PARAM_DISPLAY
from core   import (build_covariance, build_param_getter, make_prior_transform,
                    make_loglike, mu_theory, compute_mu_corr, infer_cosmo_type, get_best_fit,
                    X1_CORRECTION_MODELS, SSFR_MODELS, gauss_hermite_nodes)
import core as _core   # kept separate so we can write _core.Z_PIVOT_RUNTIME at runtime

def effective_sample_size(weights):
    """Kish (1965) ESS: 1 / sum(w_i^2) for normalised weights."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return 1.0 / np.sum(w**2)

def diagnose_modes(results, active_names, param_idx=None,
                   bandwidth=0.05, min_mode_weight=0.05,
                   warn_threshold=100):
    """
    Estimate effective sample size per posterior mode.

    Parameters
    ----------
    results        : dynesty Results object
    active_names   : list of parameter names
    param_idx      : which parameter to use for 1D mode detection (default: 0).
                     Pick the one you expect to be bimodal from the corner plot.
    bandwidth      : KDE bandwidth as fraction of parameter range (default 0.05)
    min_mode_weight: ignore modes with < this fraction of total weight (default 5%)
    warn_threshold : print a warning if any mode has ESS below this (default 100)

    Returns
    -------
    dict with keys: 'total_ess', 'modes' (list of dicts per mode)
    """
    # --- Resample to equal weights ---
    samples_eq = dyfunc.resample_equal(results.samples,
                                       np.exp(results.logwt - results.logz[-1]))
    n_eq = len(samples_eq)

    # --- Raw importance weights for ESS calculations ---
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / weights.sum()
    total_ess = effective_sample_size(weights)

    print(f"Total samples (nested):    {len(results.samples)}")
    print(f"Equal-weight resampled:    {n_eq}")
    print(f"Total ESS:                 {total_ess:.0f}")

    if param_idx is None:
        param_idx = 0
    pname = active_names[param_idx]

    # --- Detect modes via KDE on the chosen 1D marginal ---
    x = samples_eq[:, param_idx]
    x_range = x.max() - x.min()
    bw = bandwidth * x_range

    kde = gaussian_kde(x, bw_method=bw / np.std(x))
    x_grid = np.linspace(x.min() - 0.1 * x_range,
                         x.max() + 0.1 * x_range, 2000)
    kde_vals = kde(x_grid)

    # Find local maxima in the KDE
    peaks = []
    for i in range(1, len(kde_vals) - 1):
        if kde_vals[i] > kde_vals[i-1] and kde_vals[i] > kde_vals[i+1]:
            peaks.append((x_grid[i], kde_vals[i]))

    if len(peaks) == 0:
        peaks = [(x_grid[np.argmax(kde_vals)], kde_vals.max())]

    # Find valleys between peaks to use as mode boundaries
    # Sort peaks by location
    peaks.sort(key=lambda p: p[0])
    boundaries = [x.min() - 0.5 * x_range]
    for i in range(len(peaks) - 1):
        lo, hi = peaks[i][0], peaks[i+1][0]
        valley_idx = np.argmin(kde_vals[(x_grid >= lo) & (x_grid <= hi)])
        valley_x = x_grid[(x_grid >= lo) & (x_grid <= hi)][valley_idx]
        boundaries.append(valley_x)
    boundaries.append(x.max() + 0.5 * x_range)

    # --- Assign each equal-weight sample to a mode ---
    mode_assignments = np.digitize(x, boundaries) - 1
    mode_assignments = np.clip(mode_assignments, 0, len(peaks) - 1)

    # --- Per-mode diagnostics ---
    modes_out = []
    print(f"\nMode detection on parameter: {pname}  (param_idx={param_idx})")
    print(f"{'Mode':>6}  {'Peak':>8}  {'Frac':>6}  {'ESS':>8}  {'Status'}")
    print("-" * 50)

    for m_idx, (peak_x, peak_density) in enumerate(peaks):
        mask = (mode_assignments == m_idx)
        frac = mask.sum() / n_eq

        if frac < min_mode_weight:
            print(f"{m_idx:>6}  {peak_x:>8.3f}  {frac:>6.1%}  {'—':>8}  IGNORED (< {min_mode_weight:.0%} weight)")
            continue

        # ESS for this mode: use equal-weight count as proxy
        # (A more precise calculation would use the original importance
        #  weights for samples near this mode, but equal-weight count
        #  is a conservative lower bound and straightforward to interpret.)
        mode_ess = int(mask.sum())

        # Also compute mean and std of all parameters in this mode
        mode_samples = samples_eq[mask]
        mode_means = {active_names[i]: float(np.mean(mode_samples[:, i]))
                      for i in range(len(active_names))}
        mode_stds  = {active_names[i]: float(np.std(mode_samples[:, i]))
                      for i in range(len(active_names))}

        status = "OK"
        if mode_ess < warn_threshold:
            status = f"WARNING: ESS < {warn_threshold}"

        print(f"{m_idx:>6}  {peak_x:>8.3f}  {frac:>6.1%}  {mode_ess:>8d}  {status}")
        for name in active_names:
            print(f"         {name:15s}: {mode_means[name]:+.4f} ± {mode_stds[name]:.4f}")
        print()

        modes_out.append({"peak": peak_x, "fraction": frac, "ess": mode_ess,
                          "means": mode_means, "stds": mode_stds, "samples": mode_samples,})

    return {"total_ess": total_ess, "modes": modes_out}

# ===========================================================================
# 1.  RUN NAME GENERATION
# ===========================================================================

def generate_run_name(tag=""):
    """
    Return a unique run identifier.

    - Tag provided  → use the tag as-is (e.g. "cosmo/flatwCDM").
      The tag is the unique identifier; no timestamp is added.
    - No tag        → fall back to a plain timestamp "YYYYMMDD_HHMMSS".
    """
    if tag:
        return tag
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pkl_path_for(run_name, config):
    """
    Reconstruct the "<output_prefix>_results.pkl" path for a given run_name
    and config, matching run_sampler's own output_prefix construction
    exactly (including the run_name-as-subdirectory convention, e.g.
    run_name="cosmo/flatwCDM"). Lets other scripts (compare_runs.py callers,
    injection tests, host-quality checks, LOO-CV, ...) locate a run's saved
    pickle without duplicating that path logic themselves.
    """
    out_dir    = config.get("output_dir", ".")
    run_subdir = os.path.dirname(run_name)
    run_stem   = os.path.basename(run_name)
    full_dir   = os.path.join(out_dir, run_subdir) if run_subdir else out_dir
    output_prefix = os.path.join(full_dir, run_stem)
    return f"{output_prefix}_results.pkl"

# ===========================================================================
# 2.  RUN REGISTRY
# ===========================================================================

_ALL_PARAM_NAMES = list(DEFAULT_PARAM_SPECS.keys())  # All parameter names we track in the registry.

def _registry_row(run_name, config, param_specs, active_names, results,
                  nlive_used, data, inv_cov_mat, model_cfg, cosmo_type):
    """Build a single flat dict representing one row in the registry."""

    N   = len(data["z"])
    k   = len(active_names)
    dof = N - k

    # --- Best-fit sample ---
    best       = get_best_fit(results, active_names)
    get_params = build_param_getter(param_specs, active_names)
    theta_best = np.array([best[name] for name in active_names])
    params     = get_params(theta_best)

    # --- Residual at best fit ---
    # Use the same _cosmo_kwargs helper that make_loglike uses — this avoids
    # the silent bug where params["Ode0"] is always present in the dict (from
    # build_param_getter's fixed_vals) and params.get("Ode0", 0.7) never falls
    # back, so non-LambdaCDM runs would get Ode0=0.6824 (the stale fixed default)
    # instead of the cosmology-consistent 1-Om0.
    from core import _cosmo_kwargs
    mu_cosmo = mu_theory(data["z"], **_cosmo_kwargs(params, cosmo_type))
    mu_corr  = compute_mu_corr(data, params, model_cfg)
    delta    = mu_corr - mu_cosmo

    # --- MAP value of M (from Goliath B/C) ---
    B        = np.sum(inv_cov_mat @ delta)
    C_scalar = np.sum(inv_cov_mat)
    M_hat    = B / C_scalar
    delta   -= M_hat

    # --- Goodness of fit ---
    chi2     = float(delta @ inv_cov_mat @ delta)
    chi2_dof = chi2 / dof
    logL_max = float(np.max(results.logl))
    AIC      = 2 * k - 2 * logL_max
    BIC      = k * np.log(N) - 2 * logL_max

    # --- Weighted posterior means for active params ---
    weights      = np.exp(results.logwt - results.logz[-1])
    mean, cov_pm = dyfunc.mean_and_cov(results.samples, weights)
    param_means  = dict(zip(active_names, mean))
    param_stds   = dict(zip(active_names, np.sqrt(np.diag(cov_pm))))

    # --- Build flat row ---
    logZ_val = float(results.logz[-1])
    logZ_err = float(results.logzerr[-1])

    row = {"run_name":    run_name,
           "timestamp":   pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
           # Sampler
           "nlive":       nlive_used,
           "ndim":        k,
           "N_sne":       N,
           "dof":         dof,
           "logZ":        round(logZ_val, 3),
           "logZ_err":    round(logZ_err, 3),
           # Fit quality
           "chi2":        round(chi2,     3),
           "chi2_dof":    round(chi2_dof, 4),
           "logL_max":    round(logL_max, 4),
           "AIC":         round(AIC,      3),
           "BIC":         round(BIC,      3),
           "M_hat":       round(M_hat,    5),
           # Model choices
           "cosmo_type":          cosmo_type,
           "sn_colour_model":     model_cfg["sn_colour"],
           "x1_correction_model": model_cfg.get("x1_correction", "linear"),
           "mass_model":          model_cfg["mass"],
           "host_colour_model":   model_cfg["host_colour"],
           "ssfr_model":          model_cfg.get("ssfr", "none"),
           "z_evolve_model":      model_cfg["z_evolve"],
           # Data filter settings recorded for reproducibility
           "zlo":         config.get("zlo",      None),
           "zhi":         config.get("zhi",      None),
           "x1_range":    str(config.get("x1_range", None)),
           "c_range":     str(config.get("c_range",  None)),
           "idsurvey":    config.get("idsurvey", False),
           "mass_cut":    config.get("mass_cut", "all"),
           "host_quality_cut": config.get("host_quality_cut", "all"),
           "obs_z_type":  config.get("obs_z_type", "all"),
           "active_params":     "|".join(active_names),
           # Which active parameters were sampled under a prior SHAPE other
           # than the DEFAULT_PARAM_SPECS one, as "name:prior_type" pairs
           # (e.g. "alpha:uniform|beta:uniform|Om0:uniform").
           #
           # Without this, prior_shrinkage.py has no way to know which rows
           # it may legitimately score: it reads DEFAULT_PARAM_SPECS's
           # mu/sigma as "the" prior for every row, which is simply wrong
           # for any run that overrode a parameter to a uniform prior --
           # e.g. experiment_runner.py's whole "evolution/" section, which
           # now uses broad uniform alpha/beta/Om0 precisely BECAUSE the
           # earlier scan flagged them, and would otherwise be re-flagged
           # forever as "prior_dominated" against a gaussian they were
           # never sampled under. The old best-effort guard was a
           # run_name substring match ("uniformpriors"), which those tags
           # do not contain.
           "prior_overrides": "|".join(
               f"{n}:{param_specs[n].get('prior')}"
               for n in active_names
               if n in DEFAULT_PARAM_SPECS
               and param_specs[n].get("prior") != DEFAULT_PARAM_SPECS[n].get("prior")),
           # How the host mass/colour/sSFR measurement errors were treated.
           # Recorded per run so that host-error checks (extra_runners.py's
           # "hosterr/" section) can be told apart in the registry, and so
           # that any run predating these switches is not silently compared
           # against one that uses them.
           "n_gh_nodes":      config.get("n_gh_nodes", 20),
           "host_colour_err": data.get("host_colour_err_mode", "unknown"),
           "ssfr_err_max":    config.get("ssfr_err_max", None),
           "host_var_penalty": bool(config.get("host_var_penalty", False)),
           # Posterior means and stds for every active parameter
           **{f"{n}_mean": round(param_means[n], 5) for n in active_names},
           **{f"{n}_std":  round(param_stds[n],  5) for n in active_names},
    }

    return row

def update_registry(run_name, config, param_specs, active_names, results,
                    nlive_used, data, inv_cov_mat, model_cfg, cosmo_type):
    """
    Append or update a row in the run registry CSV.
    If run_name already exists the row is overwritten in-place.
    Creates the file on the first call.
    """
    registry_path = config.get("registry_file", "run_registry.csv")
    row = _registry_row(run_name, config, param_specs, active_names, results,
                        nlive_used, data, inv_cov_mat, model_cfg, cosmo_type)
    df_new = pd.DataFrame([row])

    if os.path.isfile(registry_path):
        existing = pd.read_csv(registry_path)
        # Add any new columns that didn't exist in the old file
        for col in df_new.columns:
            if col not in existing.columns:
                existing[col] = ""
        # Overwrite if run_name already present, otherwise append
        if run_name in existing["run_name"].values:
            existing.loc[existing["run_name"] == run_name, df_new.columns] = df_new.values
            combined = existing
        else:
            combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new

    combined.to_csv(registry_path, index=False)
    print(f"Registry updated: {registry_path}  (run: {run_name})")

# ===========================================================================
# 3.  I/O UTILITIES
# ===========================================================================

def save_results(results, active_names, param_specs, config, output_prefix):
    """Save dynesty results + metadata to a pickle."""
    path = f"{output_prefix}_results.pkl"
    with open(path, "wb") as f:
        pickle.dump({"results":      results,
                     "active_names": active_names,
                     "param_specs":  param_specs,
                     "config":       config}, f)
    print(f"Results saved: {path}")

def load_results(path):
    """
    Load previously saved results.

    Returns
    -------
    results, active_names, param_specs, config
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["results"], d["active_names"], d.get("param_specs", {}), d.get("config", {})

# ===========================================================================
# 4.  SUMMARY PRINTER
# ===========================================================================

def _hpd_summary(samples, weights, credible=0.6827):
    """
    Highest-posterior-density (HPD) interval around the KDE mode.

    Returns (mode, err_lo, err_hi) where err_lo and err_hi are the distances
    from the mode to the lower and upper HPD bounds respectively.

    For a Gaussian posterior this is identical to the 16/50/84 percentile
    result.  For skewed, peaked, or prior-railing posteriors it gives a
    tighter and more meaningful summary than the median-based approach.

    Parameters
    ----------
    samples  : 1D array of posterior samples for one parameter
    weights  : importance weights (from dynesty results, normalised to sum=1)
    credible : probability to enclose (default 0.6827 = 1-sigma)

    Returns
    -------
    mode   : float  location of KDE peak
    err_lo : float  distance below mode to lower HPD bound
    err_hi : float  distance above mode to upper HPD bound
    flagged: bool   True if distribution appears bimodal (median in a trough)
    """
    from scipy.stats import gaussian_kde

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    # KDE with importance weights
    try:
        kde = gaussian_kde(samples, weights=weights)
    except Exception:
        # Fallback if KDE fails (e.g. all samples identical)
        q16, q50, q84 = np.percentile(samples, [16, 50, 84])
        return q50, q50 - q16, q84 - q50, False

    x_grid   = np.linspace(samples.min(), samples.max(), 2000)
    kde_vals = kde(x_grid)
    dx       = x_grid[1] - x_grid[0]

    # Mode
    peak_idx = int(np.argmax(kde_vals))
    mode_x   = x_grid[peak_idx]

    # Bimodality flag: is the density at the weighted median < 30% of peak?
    sw       = np.sort(samples)
    ww       = weights[np.argsort(samples)]
    median_x = float(np.interp(0.5, np.cumsum(ww) / np.sum(ww), sw))
    density_at_median = float(kde(median_x)[0])
    flagged  = density_at_median < 0.30 * kde_vals[peak_idx]

    # HPD: binary-search for density threshold d* such that
    # integral of kde above d* equals the target credible fraction
    total_mass  = float(np.trapz(kde_vals, x_grid))
    target_mass = credible * total_mass

    d_lo, d_hi = 0.0, float(kde_vals.max())
    for _ in range(80):
        d_mid = 0.5 * (d_lo + d_hi)
        mass  = float(np.sum(kde_vals[kde_vals >= d_mid]) * dx)
        if mass > target_mass:
            d_lo = d_mid
        else:
            d_hi = d_mid
    d_thresh = 0.5 * (d_lo + d_hi)

    # Find HPD lower and upper bounds (outermost crossings of the threshold)
    above     = kde_vals >= d_thresh
    crossings = np.where(np.diff(above.astype(int)))[0]

    if len(crossings) < 2:
        # Degenerate case: whole range is above threshold → use percentiles
        q16, q84 = np.percentile(samples, [16, 84])
        return mode_x, mode_x - q16, q84 - mode_x, flagged

    lo_bound = float(x_grid[crossings[0]])
    hi_bound = float(x_grid[crossings[-1] + 1])

    err_lo = max(mode_x - lo_bound, 0.0)
    err_hi = max(hi_bound - mode_x, 0.0)
    return mode_x, err_lo, err_hi, flagged


def _print_summary(results, active_names):
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / weights.sum()

    print("\n=== Posterior Summary (HPD mode ± 1σ) ===")
    for i, name in enumerate(active_names):
        mode, err_lo, err_hi, flagged = _hpd_summary(samples[:, i], weights)
        flag_str = "  [BIMODAL]" if flagged else ""
        print(f"  {name:15s} = {mode:+.4f}  +{err_hi:.4f}  -{err_lo:.4f}{flag_str}")

    print(f"\n  log Z = {results.logz[-1]:.3f}  ±  {results.logzerr[-1]:.3f}")


# ===========================================================================
# 5.  PLOTS
# ===========================================================================

def _format_title(name, mode, err_lo, err_hi, sigfigs=3, flagged=False):
    """Return a LaTeX title string using HPD mode and 1-sigma bounds."""
    label = PARAM_DISPLAY.get(name, {}).get("label", name)
    fmt   = f".{sigfigs}f"
    if flagged:
        return f"{label} $= {mode:{fmt}}^{{+{err_hi:{fmt}}}}_{{-{err_lo:{fmt}}}}$ [multi-modal]"
    return (f"{label} $= {mode:{fmt}}"
            f"^{{+{err_hi:{fmt}}}}_{{-{err_lo:{fmt}}}}$")


def _transform_gaussweight_beta(samples, active_names, data):
    """
    For the asymm_gauss_weight colour model, replace the raw beta column in
    the samples array with beta_eff = beta * mean_w, where mean_w is the mean
    Gaussian weight across the dataset for each posterior sample.

      w_i(c) = exp(-0.5 * ((c_i - c0) / sn_tau)^2)
      mean_w  = (1/N) * sum_i w_i(c)
      beta_eff = beta * mean_w

    beta_eff is the effective mean colour correction amplitude, comparable to
    beta in the linear model.  Without this transformation, beta in the
    gaussweight model is inflated because down-weighting outlier colours
    forces beta up to compensate at typical colours.

    This operates on a COPY of the samples array so the original results
    object is not mutated.

    Parameters
    ----------
    samples      : ndarray (N_samples, N_params)  — will be copied
    active_names : list of parameter name strings
    data         : data dict containing data["c"]

    Returns
    -------
    samples_out  : ndarray (N_samples, N_params)  — beta column replaced
    label_out    : str  — new LaTeX label for beta axis
    """
    samples_out = samples.copy()
    c_data      = data["c"]                              # (N_sn,)

    beta_idx = active_names.index("beta")
    # c0 and sn_tau may or may not be active; fall back to fixed defaults
    c0_idx   = active_names.index("c0")   if "c0"     in active_names else None
    tau_idx  = active_names.index("sn_tau") if "sn_tau" in active_names else None

    beta_samp = samples_out[:, beta_idx]
    c0_samp   = samples_out[:, c0_idx]   if c0_idx  is not None else np.zeros(len(samples_out))
    tau_samp  = samples_out[:, tau_idx]  if tau_idx is not None else np.ones(len(samples_out))

    mean_w = np.array([
        np.mean(np.exp(-0.5 * ((c_data - c0) / st) ** 2))
        for c0, st in zip(c0_samp, tau_samp)
    ])

    samples_out[:, beta_idx] = beta_samp * mean_w
    return samples_out, r"$\beta_{\rm eff}$"


def plot_corner(results, active_names, output_prefix="dynesty_run",
                labels_override=None, model_cfg=None, data=None):
    """
    Corner plot of the posterior using dynesty's built-in plotter.

    Titles show the HPD mode and 1-sigma bounds (highest-posterior-density
    interval) rather than the 16/50/84 percentiles.  For Gaussian posteriors
    these are identical.  For skewed or peaked posteriors the HPD gives a
    tighter, more meaningful summary centred on the actual best-fit value.

    Bimodal distributions are flagged in the title.  Vertical dashed lines
    on the diagonal show the HPD mode and ±1σ bounds.

    Special transformations applied before plotting
    ------------------------------------------------
    asymm_gauss_weight:
        The raw beta is replaced with beta_eff = beta * mean_w (see
        _transform_gaussweight_beta).  This makes beta comparable to the
        linear model.  The beta axis label changes to β_eff.

    sn_colour_dust:
        No transformation needed here — the normalisation at c_ref is
        already built into sn_colour_dust in core.py, so beta already
        represents the correction at the reference colour across all
        posterior samples.

    Parameters
    ----------
    results       : dynesty results object
    active_names  : list of sampled parameter names
    output_prefix : path prefix for the saved PDF
    labels_override : optional list of LaTeX label strings (overrides PARAM_DISPLAY)
    model_cfg     : model selection dict from config (needed for gaussweight transform)
    data          : data dict (needed for gaussweight transform)
    """
    samples = results.samples.copy()    # copy so we can transform without mutation
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / weights.sum()
    ndim    = len(active_names)

    # ---- Special beta transformation for asymm_gauss_weight ----
    # Must happen before the labels/titles loop so that HPD is computed on
    # the transformed (comparable) beta_eff, not the raw inflated beta.
    beta_label_override = None
    if (model_cfg is not None and data is not None
            and model_cfg.get("sn_colour") == "asymm_gauss_weight"
            and "beta" in active_names):
        samples, beta_label_override = _transform_gaussweight_beta(
            samples, active_names, data)

    labels  = []
    titles  = []
    hpd_vals = []   # store (mode, err_lo, err_hi) for drawing lines
    for i, name in enumerate(active_names):
        disp  = PARAM_DISPLAY.get(name, {})
        # Use beta_eff label for gaussweight, otherwise use labels_override or PARAM_DISPLAY
        if name == "beta" and beta_label_override is not None:
            label = beta_label_override
        elif labels_override:
            label = labels_override[i]
        else:
            label = disp.get("label", name)
        sf    = disp.get("sigfigs", 3)
        labels.append(label)

        mode, err_lo, err_hi, flagged = _hpd_summary(samples[:, i], weights)
        hpd_vals.append((mode, err_lo, err_hi))
        titles.append(_format_title(name, mode, err_lo, err_hi,
                                    sigfigs=sf, flagged=flagged))

    fig, axes = dyplot.cornerplot(
        results,
        labels=labels,
        color="steelblue",
        show_titles=False,          # we draw our own HPD-based titles below
        title_kwargs={"fontsize": 11},
        label_kwargs={"fontsize": 11},
        quantiles=None,             # suppress dynesty's percentile lines
        smooth=0.02,
        fig=plt.subplots(ndim, ndim, figsize=(3 * ndim, 3 * ndim)),
    )

    # Draw HPD mode and ±1σ bounds as vertical lines on diagonal panels
    for i, (mode, err_lo, err_hi) in enumerate(hpd_vals):
        ax = axes[i, i]
        ax.axvline(mode,           color="steelblue", lw=1.5, ls="-")
        ax.axvline(mode - err_lo,  color="steelblue", lw=1.0, ls="--")
        ax.axvline(mode + err_hi,  color="steelblue", lw=1.0, ls="--")
        ax.set_title(titles[i], fontsize=10, pad=4)

    # logZ in the figure suptitle
    logz     = results.logz[-1]
    logz_err = results.logzerr[-1]
    fig.suptitle(rf"$\ln\mathcal{{Z}} = {logz:.4f} \pm {logz_err:.4f}$",
                 fontsize=13, y=1.01)

    path = f"{output_prefix}_corner.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Corner plot saved: {path}")
    plt.close(fig)

def _cosmo_kwargs(params, cosmo_type):
    """
    Return the keyword arguments for mu_theory that are appropriate for
    the requested cosmology type, reading values from params only when
    that parameter is actually meaningful for the model.
    """
    Om0 = params["Om0"]
    if cosmo_type == "FlatLambdaCDM":
        return {"Om0": Om0, "cosmo_type": cosmo_type}
    elif cosmo_type == "wCDM":
        return {"Om0": Om0, "w": params["w"], "cosmo_type": cosmo_type}
    elif cosmo_type == "LambdaCDM":
        return {"Om0": Om0, "Ode0": params["Ode0"], "cosmo_type": cosmo_type}
    else:
        raise ValueError(f"Unknown cosmo_type '{cosmo_type}'")


def plot_hubble_diagram(results, active_names, data, param_specs, model_cfg,
                        cosmo_type, output_prefix="dynesty_run", n_posterior=200):
    """
    Hubble diagram with two panels.

    Top panel
    ---------
    z vs mu — theory curve at best-fit cosmology, SN data points, and a
    shaded ±1σ band from n_posterior weighted posterior draws.

    Bottom panel
    ------------
    Hubble residuals  Δμ = μ_SN - μ_theory  with ±0.15 mag reference lines.
    """
    samples  = results.samples
    weights  = np.exp(results.logwt - results.logz[-1])
    weights /= weights.sum()

    get_params = build_param_getter(param_specs, active_names)

    def weighted_median(values, wts):
        sorter = np.argsort(values)
        cumwt  = np.cumsum(wts[sorter])
        cumwt /= cumwt[-1]
        return np.interp(0.5, cumwt, values[sorter])

    median_theta  = np.array([weighted_median(samples[:, i], weights) for i in range(samples.shape[1])])
    median_params = get_params(median_theta)

    z_data = data["z"]

    mu_corr_med = compute_mu_corr(data, median_params, model_cfg)

    z_theory   = np.linspace(z_data.min() * 0.9, z_data.max() * 1.05, 500)
    mu_th_med  = mu_theory(z_theory, **_cosmo_kwargs(median_params, cosmo_type))
    mu_th_data = mu_theory(z_data,   **_cosmo_kwargs(median_params, cosmo_type))

    M_hat  = np.mean(mu_corr_med - mu_th_data)
    mu_sn  = mu_corr_med - M_hat
    resid  = mu_sn - mu_th_data

    rng      = np.random.default_rng(42)
    draw_idx = rng.choice(len(samples), size=n_posterior, p=weights)

    mu_th_draws = np.zeros((n_posterior, len(z_theory)))
    for k, idx in enumerate(draw_idx):
        p = get_params(samples[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mu_th_draws[k] = mu_theory(z_theory, **_cosmo_kwargs(p, cosmo_type))
            except Exception:
                mu_th_draws[k] = np.nan

    band_lo = np.nanpercentile(mu_th_draws, 16, axis=0)
    band_hi = np.nanpercentile(mu_th_draws, 84, axis=0)

    fig, (ax_top, ax_res) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    cosmo_label_parts = [rf"$\Omega_m={median_params['Om0']:.3f}$"]
    if cosmo_type == "LambdaCDM":
        cosmo_label_parts.append(rf"$\Omega_{{\Lambda}}={median_params['Ode0']:.3f}$")
    elif cosmo_type == "wCDM":
        cosmo_label_parts.append(rf"$w={median_params['w']:.3f}$")
    cosmo_label = "Posterior median (" + ", ".join(cosmo_label_parts) + ")"

    muerr_data = data.get("muerr")
    ax_top.errorbar(z_data, mu_sn, yerr=muerr_data, fmt="o", color="steelblue",
                    markersize=2.5, alpha=0.55, elinewidth=0.6,
                    capsize=0, label="DES SNe Ia", zorder=2)
    ax_top.plot(z_theory, mu_th_med, color="crimson", lw=1.8,
                label=cosmo_label, zorder=3)
    ax_top.fill_between(z_theory, band_lo, band_hi,
                        color="crimson", alpha=0.15, label=r"$1\sigma$ posterior band")
    ax_top.set_ylabel(r"Distance modulus $\mu$", fontsize=13)
    ax_top.legend(fontsize=10, loc="upper left")
    ax_top.set_title("Hubble Diagram", fontsize=14)

    ax_res.axhline(0, color="crimson", lw=1.4, zorder=3)
    ax_res.errorbar(z_data, resid, yerr=muerr_data, fmt="o", color="steelblue",
                    markersize=2.5, alpha=0.55, elinewidth=0.6, capsize=0, zorder=2)
    ax_res.axhline(+0.15, color="grey", lw=0.8, ls="--")
    ax_res.axhline(-0.15, color="grey", lw=0.8, ls="--")
    ax_res.set_ylabel(r"$\Delta\mu$", fontsize=12)
    ax_res.set_xlabel(r"Redshift $z$", fontsize=13)
    ax_res.set_ylim(-0.6, 0.6)

    for ax in (ax_top, ax_res):
        ax.set_xscale("log")
        ax.set_xlim(z_data.min() * 0.85, z_data.max() * 1.1)

    path = f"{output_prefix}_hubble.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Hubble diagram saved: {path}")
    plt.close(fig)

# ===========================================================================
# 6.  MAIN SAMPLER ENTRY POINT
# ===========================================================================

def load_and_filter_data(config):
    """
    Load the data CSV, apply every configured filter (redshift, stretch,
    colour, survey, mass_cut, host_quality_cut, muerr_cut), build the
    host-property quadrature draws, and precompute the factorised
    covariance matrix.

    This is split out of run_sampler() so that other scripts (injection/
    recovery tests, host-quality sensitivity checks, leave-one-bin-out
    cross-validation, ...) can build the EXACT SAME `data` dict and
    covariance a real run.py fit would use, without duplicating ~280 lines
    of filtering/quadrature/Cholesky logic and risking the two copies
    drifting apart.

    Parameters
    ----------
    config : dict — see CONFIG in config.py. Only the data-loading/filter
        keys are read here (data_file, cov_file, col_*, zlo/zhi, x1_range,
        c_range, idsurvey, mass_cut, host_quality_cut, obs_z_type,
        muerr_cut, n_gh_nodes, sigma_int, c_centre/c_ref/x1_centre
        overrides). Model/prior keys are read later, by run_sampler itself.

    Returns
    -------
    df            : pandas.DataFrame  the filtered data table
    data          : dict              see compute_mu_corr's docstring
    cov_mat       : ndarray (N, N)    covariance BEFORE muerr/sigma_int are
                                      folded in (rarely needed directly —
                                      most callers want inv_cov_mat)
    inv_cov_mat   : ndarray (N, N)    inverse of the full covariance
                                      (geometric + muerr + sigma_int)
    log_det_const : float             precomputed log-determinant term
    C_sum         : float             precomputed sum(inv_cov_mat)
    keep_idx      : ndarray           indices into the ORIGINAL unfiltered
                                      CSV that survived every cut, in case
                                      a caller needs to cross-reference rows
                                      (e.g. to build a held-out complement
                                      set — see loo_zbins.py)
    """

    # ---- Load data ----
    df = pd.read_csv(config["data_file"], sep=r'\s+')

    # keep_idx tracks which original rows survive every filter so the
    # covariance matrix can be sliced to the same subset.
    keep_idx = np.arange(len(df))

    # -- Redshift cuts --
    zlo = config.get("zlo", None)
    zhi = config.get("zhi", None)
    if zlo is not None:
        mask_z   = df[config["col_z"]].values >= zlo
        keep_idx = keep_idx[mask_z]
        df       = df.iloc[mask_z].reset_index(drop=True)
        print(f"Redshift cut (zlo={zlo}): {mask_z.sum()} -> {len(df)} SNe retained")
    if zhi is not None:
        mask_z   = df[config["col_z"]].values <= zhi
        keep_idx = keep_idx[mask_z]
        df       = df.iloc[mask_z].reset_index(drop=True)
        print(f"Redshift cut (zhi={zhi}): {mask_z.sum()} -> {len(df)} SNe retained")

    # -- Stretch (x1) cut --
    x1_range = config.get("x1_range", None)
    if x1_range is not None:
        x1_lo, x1_hi = float(x1_range[0]), float(x1_range[1])
        mask_x1  = ((df[config["col_x1"]].values >= x1_lo)
                    & (df[config["col_x1"]].values <= x1_hi))
        n_before = len(df)
        keep_idx = keep_idx[mask_x1]
        df       = df.iloc[mask_x1].reset_index(drop=True)
        print(f"Stretch cut (x1 in [{x1_lo}, {x1_hi}]): "
              f"{n_before} -> {len(df)} SNe retained")

    # -- Colour (c) cut --
    c_range = config.get("c_range", None)
    if c_range is not None:
        c_lo, c_hi = float(c_range[0]), float(c_range[1])
        mask_c   = ((df[config["col_c"]].values >= c_lo)
                    & (df[config["col_c"]].values <= c_hi))
        n_before = len(df)
        keep_idx = keep_idx[mask_c]
        df       = df.iloc[mask_c].reset_index(drop=True)
        print(f"Colour cut (c in [{c_lo}, {c_hi}]): "
              f"{n_before} -> {len(df)} SNe retained")

    # -- Survey filter --
    if config.get("idsurvey", False):
        if "IDSURVEY" not in df.columns:
            raise KeyError("idsurvey=True but column 'IDSURVEY' not found in data CSV.")
        mask_survey = df["IDSURVEY"].isin([10, 150]).values
        keep_idx    = keep_idx[mask_survey]
        n_before    = len(df)
        df          = df.iloc[mask_survey].reset_index(drop=True)
        print(f"Survey filter (DES=10, Foundation=150): {n_before} -> {len(df)} SNe retained")

    # -- Mass sub-sample --
    mass_cut = config.get("mass_cut", "all").lower()
    if mass_cut not in ("all", "high", "low"):
        raise ValueError(f"mass_cut must be 'all', 'high', or 'low'; got '{mass_cut}'")
    if mass_cut != "all":
        logM_col    = config["col_logM"]
        mask_notnan = df[logM_col].notna().values
        if mass_cut == "high":
            mask_mass = mask_notnan & (df[logM_col].values >= 10.0)
        else:  # low
            mask_mass = mask_notnan & (df[logM_col].values < 10.0)
        keep_idx  = keep_idx[mask_mass]
        n_before  = len(df)
        df        = df.iloc[mask_mass].reset_index(drop=True)
        print(f"Mass cut ({mass_cut}, logM {'>=10' if mass_cut=='high' else '<10'}): "
              f"{n_before} -> {len(df)} SNe retained")

    # -- Host-match quality cut --
    # Distinct systematic from mass_cut above: this asks "was the SN
    # correctly matched to its host galaxy at all", not "what is that
    # host's mass". A wrong host association poisons the mass/colour/sSFR
    # terms directly, so a strict-match run is a standard robustness check
    # for any host-correction analysis (see host_match_quality.py).
    host_quality_cut = config.get("host_quality_cut", "all").lower()
    if host_quality_cut not in ("all", "strict"):
        raise ValueError(
            f"host_quality_cut must be 'all' or 'strict'; got '{host_quality_cut}'")
    if host_quality_cut == "strict":
        ddlr_max      = config.get("host_ddlr_max", 2.0)
        confusion_max = config.get("host_confusion_max", 0.1)
        missing_cols  = [c for c in ("HOST_DDLR", "HOST_CONFUSION", "HOST_NMATCH")
                        if c not in df.columns]
        if missing_cols:
            raise KeyError(
                f"host_quality_cut='strict' requires columns {missing_cols} "
                f"in the data CSV, but they were not found.")
        mask_quality = (
            (df["HOST_DDLR"].values      >= 0)      # HOST_DDLR == -9 is the
                                                      # catalog's "no host
                                                      # match found at all"
                                                      # sentinel (confirmed:
                                                      # every such row also
                                                      # has HOST_OBJID == 0).
                                                      # A real DDLR is always
                                                      # >= 0, so this excludes
                                                      # the sentinel — do NOT
                                                      # rely on .notna() here,
                                                      # missing DDLR is coded
                                                      # as a numeric -9, not
                                                      # NaN, so notna() alone
                                                      # silently lets every
                                                      # "no host" row through.
            & (df["HOST_DDLR"].values      <= ddlr_max)
            & (df["HOST_CONFUSION"].values <= confusion_max)
            & (df["HOST_NMATCH"].values    == 1)
        )
        keep_idx  = keep_idx[mask_quality]
        n_before  = len(df)
        df        = df.iloc[mask_quality].reset_index(drop=True)
        print(f"Host-match quality cut (strict, DDLR<={ddlr_max}, "
              f"confusion<={confusion_max}, NMATCH==1): "
              f"{n_before} -> {len(df)} SNe retained")

    # -- Host redshift observation type (spectroscopic vs photometric) --
    # A third, distinct systematic axis from mass_cut/host_quality_cut: this
    # asks whether the HOST galaxy's redshift is spectroscopic or only
    # photometric, a common source of redshift-accuracy / line-of-sight
    # systematics (spec-z hosts tend to be brighter/lower-z; photo-z hosts
    # carry larger, occasionally biased, redshift uncertainty).
    #   "all"  -> no cut on this axis (default)
    #   "spec" -> keep only SNe whose host has a valid HOST_ZSPEC > 0
    #   "phot" -> keep only SNe with a valid host sky position but NO
    #             HOST_ZSPEC (i.e. photometric-only host redshift)
    # A host position is "valid" if HOST_RA > 0 and HOST_DEC > -90 (the
    # catalog's sentinel values for "no host association"); rows without a
    # valid host position are excluded from BOTH "spec" and "phot".
    obs_z_type = config.get("obs_z_type", "all").lower()
    if obs_z_type not in ("all", "spec", "phot"):
        raise ValueError(f"obs_z_type must be 'all', 'spec', or 'phot'; got '{obs_z_type}'")
    if obs_z_type != "all":
        ra_col    = config.get("col_host_ra", "HOST_RA")
        dec_col   = config.get("col_host_dec", "HOST_DEC")
        zspec_col = config.get("col_host_zspec", "HOST_ZSPEC")
        missing_cols = [c for c in (ra_col, dec_col, zspec_col) if c not in df.columns]
        if missing_cols:
            raise KeyError(
                f"obs_z_type='{obs_z_type}' requires columns {missing_cols} "
                f"in the data CSV, but they were not found.")
        mask_valid_pos = (df[ra_col].values > 0) & (df[dec_col].values > -90)
        has_specz = (mask_valid_pos & df[zspec_col].notna().values
                    & (df[zspec_col].values > 0))
        if obs_z_type == "spec":
            mask_obsz = has_specz
        else:  # "phot"
            mask_obsz = mask_valid_pos & ~has_specz
        keep_idx  = keep_idx[mask_obsz]
        n_before  = len(df)
        df        = df.iloc[mask_obsz].reset_index(drop=True)
        print(f"obs_z_type ({obs_z_type}): {n_before} -> {len(df)} SNe retained")

    # ---- Load covariance early (moved up from its old spot just before the
    # Cholesky step) so its diagonal can be QA'd BEFORE logM_knots,
    # c_centre/c_ref, x1_centre, and the quadrature draws get computed below
    # -- those are all sample statistics, so a bad row needs to be dropped
    # before they're derived, not after.
    cov_mat = build_covariance(config["cov_file"])
    if len(keep_idx) < cov_mat.shape[0]:
        cov_mat = cov_mat[np.ix_(keep_idx, keep_idx)]
        print(f"Covariance matrix subsetted to {cov_mat.shape[0]} x {cov_mat.shape[1]} "
              f"(matching {len(keep_idx)} retained SNe)")

    # muerr is normally extracted much later (just before the Cholesky
    # step); pulled forward here too so muerr_cut below can act on the same
    # "geometric + measurement error" diagonal you've been checking by hand
    # (cov_mat + diag(muerr**2)), not the geometric term alone. Stashed into
    # data["muerr"] below when `data` is built — not re-read a second time.
    muerr = df[config["col_muerr"]].values

    # -- Optional diagonal-covariance QC cut (muerr_cut) --
    # Off by default (see CONFIG). A handful of SNe can carry a
    # pathological diagonal entry in the raw geometric (STAT+SYS)
    # covariance -- a numerical instability in a systematic-derivative
    # term for that one object -- uncorrelated with FITPROB/FITCHI2/SNR/
    # MUERR (see injection_test.py seed=13 investigation). Set muerr_cut
    # per-experiment (e.g. in extra_runners.py) to test threshold
    # sensitivity; the default run/sweep is unaffected either way.
    muerr_cut = config.get("muerr_cut", None)
    if muerr_cut is not None:
        diag_std = np.sqrt(np.diag(cov_mat) + muerr**2)
        mask_bad_diag = diag_std >= muerr_cut
        if mask_bad_diag.any():
            id_col = config.get("col_cid", "CID")
            bad_ids = (df.loc[mask_bad_diag, id_col].tolist()
                      if id_col in df.columns else np.flatnonzero(mask_bad_diag).tolist())
            print(f"muerr_cut ({muerr_cut} mag): dropping {mask_bad_diag.sum()} "
                  f"SN(e) with diag std >= {muerr_cut}: {bad_ids}")
            mask_good = ~mask_bad_diag
            n_before  = len(df)
            keep_idx  = keep_idx[mask_good]
            df        = df.iloc[mask_good].reset_index(drop=True)
            cov_mat   = cov_mat[np.ix_(mask_good, mask_good)]
            muerr     = muerr[mask_good]
            print(f"muerr_cut applied: {n_before} -> {len(df)} SNe retained")

    data = {"z":           df[config["col_z"]].values,
            "x0":          df[config["col_x0"]].values,
            "x1":          df[config["col_x1"]].values,
            "c":           df[config["col_c"]].values,
            "logM":        df[config["col_logM"]].values,
            "delta_bias":  df[config["col_delta_bias"]].values,
            "host_colour": df[config["col_host_colour"]].values,
            "muerr":       muerr}

    # ---- Load sSFR column (NaN where absent or missing) ----
    col_ssfr = config.get("col_logsSFR", None)
    if col_ssfr and col_ssfr in df.columns:
        data["logsSFR"] = df[col_ssfr].values.astype(float)
        n_ssfr_valid    = np.sum(np.isfinite(data["logsSFR"]))
        print(f"sSFR column      : {col_ssfr}  "
              f"({n_ssfr_valid}/{len(df)} SNe have finite sSFR measurements)")
    else:
        data["logsSFR"] = np.zeros(len(df))   # no correction for any SN
        if config.get("model", {}).get("ssfr", "none") != "none":
            print(f"  Warning: ssfr model is not 'none' but column "
                  f"'{col_ssfr}' not found — logsSFR set to zero for all SNe.")
        else:
            print(f"sSFR column      : absent (ssfr model = none; logsSFR = 0)")

    # ---- Host-property error marginalisation (Gauss-Hermite quadrature) ----
    # Builds fixed (N, K) quadrature draws around each SN's logM / host_colour
    # / logsSFR point estimate, using that SN's own measurement error.  These
    # are consumed by core.compute_mu_corr in place of the bare point
    # estimate, marginalising measurement uncertainty out of the mass/
    # host-colour/sSFR profile functions without touching the covariance
    # matrix (see core.py's "HOST-PROPERTY ERROR MARGINALISATION" section for
    # the full rationale). Built once here, not inside the likelihood, so the
    # quadrature nodes are identical for every likelihood call in this run —
    # deterministic and reproducible across reruns of the same config.
    n_gh_nodes = int(config.get("n_gh_nodes", 20))
    gh_eps, gh_weights = gauss_hermite_nodes(n_gh_nodes)
    data["gh_weights"] = gh_weights
    print(f"Host error marginalisation: {n_gh_nodes} Gauss-Hermite nodes")

    def _quadrature_draws(col_name, point_vals, label, err_override=None,
                          err_max=None):
        """
        Return an (N, K) array of quadrature abscissas around point_vals.

        - Missing/unconfigured error column -> zero error for every SN
          (every node collapses to the point estimate; reproduces the old
          point-estimate-only behaviour exactly, at zero extra cost).
        - Missing/non-finite error for an individual SN -> treated as
          zero error for that SN only (exact point estimate).
        - NaN point estimates (e.g. SNe with no host sSFR measurement) are
          preserved as NaN in every draw, so the downstream profile
          functions' existing np.isfinite(...) -> 0.0 handling (e.g.
          ssfr_tanh) still fires correctly under quadrature.
        - err_override supplies a derived error array in place of a data
          column (used for the host colour, whose error column is not
          populated in the DES metadata).
        - err_max masks SNe whose quoted error exceeds the threshold: the
          point estimate becomes NaN, so that SN contributes nothing to
          this profile.  The SN is deliberately NOT dropped from the
          sample, so every model is still compared on identical objects.
        """
        n = len(point_vals)
        if err_override is not None:
            err = np.where(np.isfinite(err_override) & (err_override >= 0),
                           err_override, 0.0)
        elif col_name and col_name in df.columns:
            err = df[col_name].values.astype(float)
            err = np.where(np.isfinite(err) & (err >= 0), err, 0.0)
            n_err = int(np.sum(err > 0))
            print(f"  {label:14s}: {col_name}  ({n_err}/{n} SNe have finite, "
                  f"positive measurement error)")
        else:
            err = np.zeros(n)
            print(f"  {label:14s}: absent -> treated as exact (zero error)")

        point_vals = np.asarray(point_vals, dtype=float)
        if err_max is not None:
            bad = err > err_max
            if bad.any():
                print(f"  {label:14s}: masking {int(bad.sum())}/{n} SNe with "
                      f"quoted error > {err_max} (point estimate -> NaN; "
                      f"SNe retained in the sample)")
                point_vals = np.where(bad, np.nan, point_vals)

        pv          = np.where(np.isfinite(point_vals), point_vals, 0.0)
        draws       = pv[:, None] + err[:, None] * gh_eps[None, :]
        missing_pt  = ~np.isfinite(point_vals)
        draws[missing_pt, :] = np.nan
        return draws

    data["logM_draws"] = _quadrature_draws(
        config.get("col_logM_err"), data["logM"], "logM err")

    # ---- Host colour error -------------------------------------------------
    # HOST_COLOR_ERR is present in the DES metadata but never populated (it is
    # -999 for every SN), so the host colour would otherwise be the only host
    # property treated as exactly measured, while mass and sSFR are smoothed
    # by their errors.  That asymmetry is not physical: it would hand the host
    # colour models an unearned advantage.
    #
    # HOST_COLOR and HOST_LOGMASS come out of the same SED fit to the same
    # host photometry, so their uncertainties are physically linked.  We
    # therefore derive a host colour error from HOST_LOGMASS_ERR using the
    # slope of the mass-to-light/colour relation (Taylor et al. 2011,
    # log(M*/L_i) = 1.15 + 0.70 (g-i), so d logM / d colour = 0.70):
    #
    #     sigma_colour ~= sigma_logM / host_colour_err_mass_slope
    #
    # This is conservative (an upper bound), because sigma_logM also absorbs
    # distance, luminosity and model terms that do not come from the colour.
    # It is a DERIVED quantity, never a measured one — set
    # host_colour_err_from_logmass=False to switch it off.
    hc_err_col = config.get("col_host_colour_err")
    hc_err_raw = (df[hc_err_col].values.astype(float)
                  if hc_err_col and hc_err_col in df.columns else None)
    hc_usable  = (hc_err_raw is not None
                  and np.any(np.isfinite(hc_err_raw) & (hc_err_raw > 0)))
    hc_derived = None
    if config.get("host_colour_err_from_logmass", True) and not hc_usable:
        slope = float(config.get("host_colour_err_mass_slope", 0.70))
        logm_err = df[config["col_logM_err"]].values.astype(float) \
            if config.get("col_logM_err") in df.columns else None
        if logm_err is not None and slope > 0:
            hc_derived = np.where(np.isfinite(logm_err) & (logm_err > 0),
                                  logm_err / slope, 0.0)
            print(f"  {'host colour err':14s}: column '{hc_err_col}' is not "
                  f"populated; DERIVED as HOST_LOGMASS_ERR / {slope} "
                  f"(Taylor+2011 mass-colour slope) — median "
                  f"{np.median(hc_derived[hc_derived > 0]):.4f} mag")
    data["host_colour_err_derived"] = hc_derived is not None
    if hc_derived is not None:
        slope = float(config.get("host_colour_err_mass_slope", 0.70))
        data["host_colour_err_mode"] = f"logmass/{slope:.2f}"
    elif hc_usable:
        data["host_colour_err_mode"] = "column"
    else:
        data["host_colour_err_mode"] = "none"
    data["host_colour_draws"] = _quadrature_draws(
        hc_err_col, data["host_colour"], "host colour err",
        err_override=hc_derived)

    # ---- sSFR error --------------------------------------------------------
    # HOST_LOGsSFR_ERR is bimodal: a well-measured population plus a pileup of
    # failure-mode values around 10 dex, far larger than the entire population
    # spread (~2.4 dex).  Those carry no information, so their point estimates
    # are masked (set to NaN) rather than capped — capping would keep a
    # meaningless value and give it artificial weight.  The SNe stay in the
    # sample so evidences remain comparable across models.
    data["logsSFR_draws"] = _quadrature_draws(
        config.get("col_logsSFR_err"), data["logsSFR"], "sSFR err",
        err_max=config.get("ssfr_err_max", None))

    # ---- Precompute data-derived quantities stored in data dict ----

    # logM_knots: 25th/50th/75th percentile of logM, used by mass_spline.
    # Fixed at data-load time — identical for all runs on this dataset.
    data["logM_knots"] = np.percentile(data["logM"], [25, 50, 75])
    print(f"Mass spline knots: {data['logM_knots']}")

    # c_centre: the mean SALT2 colour, stored in the data dict so that
    # sn_colour_quadratic can subtract it internally when called.
    # data["c"] is NEVER modified here — it stays as the raw SALT2 colour
    # for every model.  The linear, broken, tanh, and all other colour models
    # read data["c"] directly without any centring.  Only sn_colour_quadratic
    # applies the centring, inside its own function, so it has no effect on
    # any other model regardless of which model is selected for a given run.
    # Override in config with "c_centre": <float> for cross-dataset consistency.
    c_centre = float(config.get("c_centre", np.mean(data["c"])))
    data["c_centre"]  = c_centre
    print(f"SN colour centre : c_centre = {c_centre:.4f}  "
          f"(stored for sn_colour_quadratic only; data['c'] is unchanged)")

    # c_ref: median |c| used to normalise sn_colour_dust so beta retains its
    # linear-model interpretation at the reference colour.  Computed once from
    # data; override in config with "c_ref": <float> for cross-run consistency.
    c_ref = float(np.median(np.abs(data["c"])))
    data["c_ref"]  = c_ref
    print(f"SN colour ref    : c_ref    = {c_ref:.4f}  "
          f"(normalisation point for sn_colour_dust)")

    x1_ref = float(np.median(np.abs(data["x1"])))
    data["x1_ref"] = x1_ref
    print(f"Stretch ref      : x1_ref = {x1_ref:.4f}  "
        f"(normalisation point for x1_correction_powerlaw)")

    # x1_centre: sample mean of x1, used by x1_correction_quadratic to centre
    # the quadratic term.  data["x1"] is NEVER modified — centring happens
    # inside x1_correction_quadratic at call time, so it does not affect any
    # other x1 correction model.
    # Override in config with "x1_centre": <float> for cross-dataset consistency.
    x1_centre = float(config.get("x1_centre", np.mean(data["x1"])))
    data["x1_centre"] = x1_centre
    print(f"Stretch centre   : x1_centre = {x1_centre:.4f}  "
          f"(stored for x1_correction_quadratic only; data['x1'] is unchanged)")

    # ---- Set redshift pivot from data  ----------------------------------------
    # The z-evolution functions in core.py (z_evolve_power/log/zz) are normalised
    # so that the evolution factor equals exactly 1 at z = Z_PIVOT.  This means
    # alpha/beta/gamma are interpreted as the nuisance corrections at the *median*
    # redshift of the sample, not at z=0 which lies outside the data entirely.
    #
    # Anchoring to z=0 forces the sampler to extrapolate the evolution function to
    # a point with no constraining data, creating a near-perfect degeneracy between
    # the baseline value (alpha) and the slope (a): many (alpha, a) pairs produce
    # the same correction at all observed redshifts.  Setting the pivot inside the
    # data — at the median z — breaks this degeneracy because now alpha is pinned
    # to where the data is densest, and a describes departures above and below that.
    #
    # This value is written into core.Z_PIVOT once per run, before ptform/logl
    # are constructed, so all three evolution functions see the same pivot.
    # If you change datasets the pivot updates automatically.
    z_median = float(np.median(data["z"]))
    _core.Z_PIVOT_RUNTIME = z_median
    print(f"Redshift pivot   : z_pivot = {z_median:.4f}  (median of {len(data['z'])} SNe)")

    # ---- Add remaining diagonal noise terms to covariance ----
    # muerr already extracted/filtered above (and stored in data["muerr"]);
    # no need to re-read df[config["col_muerr"]] here.
    #
    # IMPORTANT: this must NOT reassign `cov_mat` itself. `cov_mat` is
    # documented (see this function's Returns docstring) and relied upon
    # by callers that slice a SUBSET of it and refactorise their own
    # noise terms per-subset (loo_zbins.py's _refactorise_covariance,
    # reused by drilling_cones.py) as the covariance BEFORE muerr/sigma_int
    # are folded in — i.e. the pure geometric (host photometric errors
    # etc.) term. Folding muerr/sigma_int into `cov_mat` here and
    # returning that would silently double-count them in every caller
    # that then adds its own per-subset diag(muerr**2) on top (previously
    # a real bug: every loo_zbins.py fold and every drilling_cones.py cone
    # had its diagonal inflated by an extra muerr**2 + sigma_int**2 term).
    # `cov_full` below is a separate array used only for this function's
    # own Cholesky factorisation / inv_cov_mat; `cov_mat` itself is
    # returned untouched.
    cov_full = cov_mat + np.diag(muerr**2)

    sigma_int = config.get("sigma_int", 0.0)
    if sigma_int > 0:
        cov_full = cov_full + np.diag(np.full(len(muerr), sigma_int**2))

    # ---- Cholesky-factorise once; reuse for log-det and inverse ----
    # For N=1820 the full inverse is ~26 MB (float64) — acceptable to store.
    # The likelihood hot path then does a single BLAS dgemv per call, which
    # is faster than two cho_solve triangular solves per call at this size.
    # log-det is computed from the Cholesky diagonal (numerically superior
    # to np.linalg.slogdet for near-singular matrices).
    N = cov_full.shape[0]
    try:
        chol_fac      = cho_factor(cov_full, lower=True)
        log_det_const = (2.0 * np.sum(np.log(np.diag(chol_fac[0])))
                         + N * np.log(2.0 * np.pi))
        # Solve C * inv_cov_mat = I  →  inv_cov_mat = C^{-1}
        inv_cov_mat   = cho_solve(chol_fac, np.eye(N))
    except LinAlgError:
        raise ValueError("Covariance matrix is not positive definite after "
                         "adding muerr/sigma_int diagonal terms.")

    C_sum = float(np.sum(inv_cov_mat))

    # Stashed for the optional host-variance likelihood path, which needs the
    # full covariance (muerr/sigma_int included, host variance excluded) so it
    # can add a parameter-dependent diagonal and refactorise per call.
    data["cov_full"] = cov_full

    # `cov_mat` here is still the pure geometric covariance (pre-muerr,
    # pre-sigma_int) as documented — callers that need the full covariance
    # should use inv_cov_mat, or refactorise a subset of `cov_mat` via
    # loo_zbins._refactorise_covariance exactly once, not twice.
    return df, data, cov_mat, inv_cov_mat, log_det_const, C_sum, keep_idx

def _make_progress_printer(interval_s):
    """
    Return a dynesty ``print_func`` that emits one compact line every
    ``interval_s`` seconds instead of rewriting a progress bar several
    times a second.

    Why this exists: experiment_runner.py redirects each run's stdout to
    ``logs/<tag>.log``.  dynesty's default progress writer assumes a live
    terminal and repaints its status line continuously using carriage
    returns.  Sent to a file instead, every repaint is appended verbatim,
    so a single long publication run buries its actual output (setup
    banner, parameter summary, warnings) under hundreds of thousands of
    near-identical progress lines and the log grows to hundreds of MB.

    Throttling to a fixed wall-clock interval keeps the log readable and
    small while still letting `tail -f` confirm a run is alive and show
    how logz and dlogz are converging.  The first call always prints, so
    a log shows sampling has started without waiting a full interval,
    and the final iteration always prints so the last line reflects the
    true end state rather than whenever the last tick happened to land.

    Parameters
    ----------
    interval_s : float
        Minimum seconds between printed lines.

    Notes
    -----
    dynesty calls this as
    ``print_func(results, niter, ncall, add_live_it=, dlogz=, stop_val=,
    nbatch=, logl_min=, logl_max=)``.  Two details of that interface are
    easy to get wrong:

    * ``results`` is an ``IteratorResult`` during the main sampling loop
      but an ``IteratorResultShort`` inside a dynamic batch, and the
      short form has a *different* field order and carries no ``logz``
      or ``delta_logz``.  Fields are therefore read by name only, never
      by position, and simply omitted when absent.
    * During the final add-live-points phase dynesty calls this once per
      live point, which for a publication run is thousands of calls in
      quick succession.  That phase gets a single announcement line
      rather than being exempted from throttling.
    """
    state = {"last": 0.0, "started": time.time(), "announced_live": False}

    def _field(results, name):
        """Read a field by name, or None if this result type lacks it."""
        return getattr(results, name, None)

    def print_func(results, niter, ncall, add_live_it=None, dlogz=None,
                   stop_val=None, nbatch=None, logl_min=-np.inf,
                   logl_max=np.inf, pbar=None):
        now = time.time()
        first = state["last"] == 0.0

        # add_live_it is set only while dynesty appends the final live
        # points -- once per point, so this must not bypass throttling.
        # One line marks the start of that phase; the real end-of-run
        # numbers come from _print_summary immediately afterwards.
        if add_live_it is not None:
            if state["announced_live"]:
                return
            state["announced_live"] = True
        elif not first and (now - state["last"]) < interval_s:
            return
        state["last"] = now

        logz  = _field(results, "logz")
        logzv = _field(results, "logzvar")
        eff   = _field(results, "eff")
        dlz   = _field(results, "delta_logz")

        elapsed = now - state["started"]
        hrs, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)

        parts = [f"[{hrs:d}:{mins:02d}:{secs:02d}]",
                 f"iter={niter}", f"ncall={ncall}"]
        if nbatch is not None:
            parts.append(f"batch={nbatch}")
        if logz is not None and np.isfinite(logz):
            err = ""
            if logzv is not None and logzv > 0:
                err = f" +/- {math.sqrt(logzv):.3f}"
            parts.append(f"logz={logz:.3f}{err}")
        if dlz is not None and np.isfinite(dlz) and dlz < 1e6:
            # dynesty seeds delta_logz with a ~1e300 sentinel on the first
            # iteration, which is finite but meaningless -- printing it
            # dumps 300 digits into the log.
            target = f" (target {dlogz:g})" if dlogz is not None else ""
            parts.append(f"dlogz={dlz:.4f}{target}")
        if stop_val is not None and np.isfinite(stop_val):
            parts.append(f"stop={stop_val:.3f}")
        if eff is not None:
            parts.append(f"eff={eff:.1f}%")
        if add_live_it is not None:
            parts.append("[adding live points]")

        print("  " + "  ".join(parts), flush=True)

    return print_func


def run_sampler(config, preloaded=None):
    """
    Load data, build the likelihood, run dynesty nested sampling, save all
    outputs, update the run registry, and return the results.

    Parameters
    ----------
    config    : dict  — see CONFIG in config.py
    preloaded : optional tuple (df, data, cov_mat, inv_cov_mat,
        log_det_const, C_sum, keep_idx), exactly as returned by
        load_and_filter_data(config). When given, run_sampler uses this
        directly instead of calling load_and_filter_data(config) itself —
        every other step (sampling, saving, plotting, registry) is
        unchanged and fully reused. This lets other scripts hand run_sampler
        a MODIFIED or SUBSETTED version of the normal data (a training-only
        subset for leave-one-bin-out cross-validation, or synthetic x0 for
        an injection/recovery test) without duplicating any of the ~250
        lines of sampling/output logic below — see loo_zbins.py and
        injection_test.py. The caller is responsible for ensuring `data`,
        `cov_mat`, `inv_cov_mat` etc. are mutually consistent (same row
        order/count) — load_and_filter_data's own output already satisfies
        this by construction; if you subset it further, subset every array
        the same way, including data["*_draws"] (N, K) along axis 0 only.

    Returns
    -------
    results      : dynesty Results object
    sampler      : NestedSampler (for manual post-processing)
    active_names : list[str]
    data         : dict  {z, x0, x1, c, logM, delta_bias, host_colour}
    run_name     : str   unique identifier for this run
    """

    # ---- Generate run name and output directory ----
    tag      = config.get("run_tag") or ""
    run_name = generate_run_name(tag)
    out_dir  = config.get("output_dir", ".")

    os.makedirs(out_dir, exist_ok=True)

    # run_name may contain slashes (e.g. "cosmo/flatwCDM") which imply
    # subdirectories under out_dir.
    run_subdir    = os.path.dirname(run_name)
    run_stem      = os.path.basename(run_name)
    full_dir      = os.path.join(out_dir, run_subdir) if run_subdir else out_dir
    os.makedirs(full_dir, exist_ok=True)
    output_prefix = os.path.join(full_dir, run_stem)

    print(f"\n{'='*60}")
    print(f"Run name: {run_name}")
    print(f"Outputs:  {output_prefix}_*")
    print(f"{'='*60}\n")

    # ---- Load data (filters, quadrature draws, covariance factorisation) ----
    if preloaded is not None:
        df, data, cov_mat, inv_cov_mat, log_det_const, C_sum, keep_idx = preloaded
    else:
        df, data, cov_mat, inv_cov_mat, log_det_const, C_sum, keep_idx = \
            load_and_filter_data(config)


    # ---- Resolve active parameters ----
    param_specs = copy.deepcopy(config.get("param_specs", DEFAULT_PARAM_SPECS))
    model_cfg   = config["model"]

    cosmo_type   = infer_cosmo_type(param_specs)
    active_names = [name for name, spec in param_specs.items() if spec["active"]]
    ndim         = len(active_names)

    print(f"Active parameters ({ndim}): {active_names}")
    print(f"Cosmology (inferred) : {cosmo_type}")
    print(f"Models               : {model_cfg}")
    print(f"Marginalise M        : {not param_specs['M']['active']}")

    config["cosmo_type"] = cosmo_type  # Stash inferred cosmo_type so the registry row can record it

    # ---- Build and run sampler ----
    ptform = make_prior_transform(param_specs, active_names)

    # Optional: propagate host measurement error as a VARIANCE as well as a
    # bias (see core.compute_mu_corr / cov_log_likelihood_hetero).  This makes
    # the covariance parameter dependent, so it must be refactorised on every
    # likelihood call — O(N^3) instead of O(N^2).  Off by default; enable it
    # for targeted systematic checks, not for production sweeps.
    host_var = bool(config.get("host_var_penalty", False))
    cov_base = None
    if host_var:
        cov_base = data.get("cov_full")
        if cov_base is None:
            raise ValueError("host_var_penalty is enabled but the full "
                             "covariance was not stashed by "
                             "load_and_filter_data (stale preloaded data?).")
        if cov_base.shape[0] != len(data["z"]):
            raise ValueError(
                f"host_var_penalty: stashed covariance is "
                f"{cov_base.shape[0]}x{cov_base.shape[0]} but there are "
                f"{len(data['z'])} SNe. Callers that subset the sample "
                f"(loo_zbins, drilling_cones) must refactorise their own "
                f"covariance subset before enabling this.")
        print("Host variance penalty: ON — covariance refactorised per "
              "likelihood call (much slower; intended for systematic checks)")

    logl   = make_loglike(data, inv_cov_mat, log_det_const, C_sum,
                          param_specs, active_names, model_cfg, cosmo_type,
                          cov_base=cov_base)

    # ---- Sampler mode ----
    # Set "sampler_mode": "dynamic" in CONFIG (or per-experiment config_overrides)
    # to use DynamicNestedSampler instead of the default static NestedSampler.
    #
    # Dynamic nested sampling concentrates additional live-point batches on the
    # regions that contribute most to logZ uncertainty, giving ~30-50% tighter
    # evidence estimates for the same number of likelihood evaluations.  It is
    # the recommended mode for publication model-comparison runs.
    #
    # Static mode is faster for exploratory runs and identical in API behaviour.
    _sampler_mode = config.get("sampler_mode", "static")

    # ---- nlive autoscaling ----
    # ndim = total number of active (sampled) parameters for this run.
    # e.g. baseline (alpha, beta, gamma, Om0) → ndim=4 → 1200 publication, 200 exploratory
    #
    # Priority:
    #   1. Explicit config["nlive"] integer  → use exactly as given
    #   2. config["nlive_mode"] == "publication"  → ndim * 500
    #   3. default / "exploratory"               → ndim * 50
    #
    # For dynamic mode, nlive controls nlive_init (the initial static pass).
    # The dynamic batches are then sized by nlive_batch (default: nlive_init // 4,
    # minimum 25).  Both can be overridden explicitly via config keys.
    _nlive_explicit = config.get("nlive")
    _nlive_mode     = config.get("nlive_mode", "exploratory")
    if _nlive_explicit:
        nlive     = int(_nlive_explicit)
        nlive_src = "user-set"
    elif _nlive_mode == "publication":
        nlive     = ndim * 500
        nlive_src = f"publication (ndim x 500 = {ndim} x 500)"
    else:
        nlive     = ndim * 50
        nlive_src = f"exploratory (ndim x 50 = {ndim} x 50)"
    print(f"Live points      : {nlive}  [{nlive_src}]")
    print(f"Sampler mode     : {_sampler_mode}")

    _bound  = config.get("bound",  "multi")
    _sample = config.get("sample", "rslice")
    _dlogz  = config.get("dlogz",  1e-3)
    _verbose = config.get("verbose", True)

    # ---- Progress reporting --------------------------------------------
    # progress_interval is the minimum wall-clock gap (seconds) between
    # dynesty progress lines.  Runs are logged to files, not terminals, so
    # the default is a half-hourly heartbeat rather than dynesty's
    # continuous repaint; see _make_progress_printer for why that matters.
    #   > 0  : throttled heartbeat (default 1800 = every 30 min)
    #   == 0 : dynesty's own uncapped progress bar (interactive use)
    # Setting config["verbose"] = False silences progress entirely.
    _progress_interval = float(config.get("progress_interval", 1800))
    _print_func = (_make_progress_printer(_progress_interval)
                   if _verbose and _progress_interval > 0 else None)
    if _verbose:
        if _print_func is not None:
            print(f"Progress updates : every {_progress_interval / 60:g} min")
        else:
            print("Progress updates : continuous (dynesty default)")
    else:
        print("Progress updates : disabled")

    if _sampler_mode == "dynamic":
        # ── nlive_batch sizing ────────────────────────────────────────────────
        # nlive_batch is the number of live points added per refinement batch
        # AFTER the initial static pass (nlive_init).  It controls the
        # granularity of logZ improvement per round.
        #
        # The right scale is ndim, not nlive_init.  Here is why:
        #
        #   - Each batch targets the shell where logZ uncertainty is largest.
        #     The sampler needs enough points to adequately cover that shell in
        #     ndim dimensions.  A shell in ndim-D space needs O(ndim) points to
        #     be resolved — fewer gives noisy logZ improvement per batch.
        #
        #   - nlive_init // 4 (our old formula) is a ratio-based heuristic that
        #     accidentally scales with ndim*500 in publication mode.  For ndim=6
        #     this gives 450 — far more than needed per batch, making each round
        #     expensive and preventing the dlogz criterion from stopping cleanly
        #     between rounds.
        #
        #   - ndim * 5 is the principled lower bound: 5 points per dimension per
        #     batch is enough to move the logZ estimate meaningfully, while keeping
        #     batches short so dlogz can exit promptly.  Minimum 25 guards against
        #     very low-ndim models.
        #
        # Batch count estimate (for intuition, not used in code):
        #   total_refinement_samples ≈ nlive_batch x maxbatch
        #   A typical publication run needs ~nlive_init extra samples total
        #   (i.e. the dynamic pass roughly doubles the static pass).
        #   → expected batches ≈ nlive_init / nlive_batch = ndim*500 / (ndim*5) = 60
        #   This is independent of ndim — a sensible, predictable run length.
        #
        # Examples with publication mode (nlive_init = ndim x 500):
        #   ndim=4  → nlive_init=1200, nlive_batch=20→25(min), ~48 batches
        #   ndim=6  → nlive_init=1800, nlive_batch=30,          ~60 batches
        #   ndim=10 → nlive_init=5000, nlive_batch=50,          ~60 batches
        #
        # Override with config["nlive_batch"] for manual control.
        _nlive_batch = int(config.get("nlive_batch", max(30, ndim * 5)))

        # ── maxbatch autoscaling ──────────────────────────────────────────────
        # maxbatch caps the number of refinement rounds as a wall-time safety
        # valve.  dlogz will usually terminate the run before this is reached.
        #
        # Rule: allow enough batches to add ~1x nlive_init worth of additional
        # samples (i.e. the dynamic phase is at most as expensive as the initial
        # static pass).  This is always sufficient for publication evidence
        # accuracy and prevents runaway refinement on pathological posteriors.
        #
        # maxbatch = ceil(nlive_init / nlive_batch) ≈ 500/5 = 60 for all ndim.
        # Override with config["maxbatch"] to hard-cap wall time.
        import math
        _maxbatch_auto = math.ceil(nlive / _nlive_batch)
        _maxbatch = config.get("maxbatch", _maxbatch_auto)

        print(f"  nlive_init     : {nlive}")
        print(f"  nlive_batch    : {_nlive_batch}  (ndim x 5 = {ndim} x 5, min 25)")
        print(f"  maxbatch       : {_maxbatch}  "
              f"({'user-set' if 'maxbatch' in config else f'auto = ceil({nlive}/{_nlive_batch})'})")
        print(f"  max refinement : {_nlive_batch * _maxbatch} samples  "
              f"({_nlive_batch * _maxbatch / nlive:.1f} x nlive_init)")
        print(f"  dlogz_init     : {_dlogz}")

        sampler = DynamicNestedSampler(
            logl, ptform, ndim,
            bound=_bound,
            sample=_sample,
        )
        sampler.run_nested(
            nlive_init=nlive,
            nlive_batch=_nlive_batch,
            maxbatch=_maxbatch,
            dlogz_init=_dlogz,
            print_progress=_verbose,
            print_func=_print_func,
        )
        results    = sampler.results
        nlive_used = nlive  # registry records the init nlive

    else:
        # Static nested sampler — default behaviour, unchanged from before.
        sampler = NestedSampler(logl, ptform, ndim, nlive=nlive, bound=_bound, sample=_sample)
        
        sampler.run_nested(dlogz=_dlogz, maxiter=config.get("maxiter", None),
                           print_progress=_verbose, print_func=_print_func)
        
        results   = sampler.results
        nlive_used = sampler.results.nlive

    # ---- Post-run output ----
    _print_summary(results, active_names)
    save_results(results, active_names, param_specs, config, output_prefix=output_prefix)

    # BUG FIX: pass the local `run_name` variable directly, not config.get("run_name")
    # config never has a "run_name" key — generate_run_name() returns it as a local.
    update_registry(run_name = run_name, config = config, param_specs = param_specs,
                    active_names = active_names, results = results, nlive_used = nlive_used,
                    data = data, inv_cov_mat = inv_cov_mat, model_cfg = model_cfg, cosmo_type = cosmo_type)
    
    # diag = diagnose_modes(results, active_names, param_idx=eta_idx)

    plot_corner(results, active_names, output_prefix=output_prefix,
                model_cfg=model_cfg, data=data)
    plot_hubble_diagram(results, active_names, data, param_specs,
                        model_cfg, cosmo_type, output_prefix=output_prefix)

    return results, sampler, active_names, data, run_name

# ===========================================================================
# 7.  COMMAND-LINE ENTRY POINT
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Run SNe Ia nested sampling")
    p.add_argument("--tag", default="",
                   help="Short label appended to the auto-generated run name")
    p.add_argument("--nlive", type=int, default=None,
                   help="Override number of live points")
    p.add_argument("--dlogz", type=float, default=None,
                   help="Override stopping criterion dlogz")
    p.add_argument("--progress-interval", type=float, default=None,
                   dest="progress_interval",
                   help="Seconds between dynesty progress lines "
                        "(default 1800 = every 30 min; 0 = dynesty's "
                        "continuous progress bar)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress dynesty progress output entirely")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    cfg = copy.deepcopy(CONFIG)

    if args.tag:
        cfg["run_tag"] = args.tag
    if args.nlive is not None:
        cfg["nlive"] = args.nlive
    if args.dlogz is not None:
        cfg["dlogz"] = args.dlogz
    if args.progress_interval is not None:
        cfg["progress_interval"] = args.progress_interval
    if args.quiet:
        cfg["verbose"] = False

    results, sampler, active_names, data, run_name = run_sampler(cfg)