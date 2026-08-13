"""
loo_zbins.py  —  SNe Ia Cosmology Pipeline
=============================================
Leave-one-redshift-bin-out cross-validation.

Fits your model N times, holding out one redshift bin each time, and checks
whether the fit (trained on every OTHER bin) correctly predicts the
held-out bin's Hubble residuals. A single global fit can hide z-dependent
systematics — an evolving host population, a redshift-dependent selection
effect, a bias-correction mismatch that only shows up at certain z — that
this catches directly: if the held-out residuals are systematically offset
or trend with host property WITHIN the held-out bin, the global fit is
missing something as a function of redshift.

This does NOT create separate CSV files or otherwise touch config["data_file"]
— that would break the alignment between row order and the covariance
matrix (build_covariance's rows correspond 1:1 to config["data_file"]'s
rows). Instead it calls load_and_filter_data(config) ONCE to get a
fully-filtered (df, data, cov_mat, ...) tuple where that alignment is
already correctly established (load_and_filter_data has already subsetted
cov_mat to match df's row order), then splits THAT in-memory tuple into
per-fold training/held-out sets, refactorises the training covariance for
each fold, and hands it to run_sampler via its `preloaded=` argument — so
every fold reuses the exact same sampling/saving/plotting/registry code
path as a normal run, just on a data subset.

Note on the mass_spline / c_centre / c_ref / x1_centre / x1_ref knots:
these are recomputed ONCE from the full filtered sample (before any fold
split) and held FIXED across all folds, rather than refit per-fold. This
is a deliberate choice — letting spline knots wobble between folds would
make the folds' models subtly different from each other, confounding the
z-dependence question this test is actually asking.

Usage
-----
  python loo_zbins.py --tag your_best_model --n-bins 4

or:
  from loo_zbins import run_loo_zbins
  report = run_loo_zbins(config_overrides={"run_tag": "best_model",
                                           "model": {...}}, n_bins=4)
"""

import argparse
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from config import CONFIG
from run    import (load_and_filter_data, run_sampler, pkl_path_for,
                    build_param_getter, compute_mu_corr, mu_theory,
                    infer_cosmo_type, DEFAULT_PARAM_SPECS)
import copy as _copy


# Keys in the data dict that vary per-SN (row axis 0) and must be sliced
# along with df when building a fold. Every other key (gh_weights,
# logM_knots, c_centre, c_ref, x1_ref, x1_centre) is a fixed, data-derived
# CONSTANT shared across all folds — see module docstring.
_PER_ROW_1D_KEYS = ["z", "x0", "x1", "c", "logM", "delta_bias",
                    "host_colour", "logsSFR", "muerr"]
_PER_ROW_2D_KEYS = ["logM_draws", "host_colour_draws", "logsSFR_draws"]


def _subset_data(data, mask):
    """Return a new data dict restricted to rows where mask is True."""
    out = dict(data)
    for k in _PER_ROW_1D_KEYS:
        if k in data:
            out[k] = data[k][mask]
    for k in _PER_ROW_2D_KEYS:
        if k in data:
            out[k] = data[k][mask, :]
    return out


def _refactorise_covariance(cov_mat_geo, muerr, sigma_int):
    """
    Given the geometric (pre-noise) covariance for a subset of SNe and
    their muerr, rebuild the full covariance (+ diagonal noise terms) and
    refactorise it — same recipe as load_and_filter_data's tail end, but
    callable per-fold on an already-subsetted matrix.
    """
    cov = cov_mat_geo + np.diag(muerr**2)
    if sigma_int and sigma_int > 0:
        cov = cov + np.diag(np.full(len(muerr), sigma_int**2))
    N = cov.shape[0]
    try:
        chol_fac      = cho_factor(cov, lower=True)
        log_det_const = (2.0 * np.sum(np.log(np.diag(chol_fac[0])))
                         + N * np.log(2.0 * np.pi))
        inv_cov_mat   = cho_solve(chol_fac, np.eye(N))
    except LinAlgError:
        raise ValueError("Fold covariance is not positive definite — this "
                         "fold's held-out bin may be too small/ill-conditioned; "
                         "try fewer bins.")
    C_sum = float(np.sum(inv_cov_mat))
    return inv_cov_mat, log_det_const, C_sum


def run_loo_zbins(config_overrides=None, n_bins=4, output_prefix=None):
    """
    Parameters
    ----------
    config_overrides : dict layered on top of CONFIG — specify the model to
        test (config_overrides["model"] = {...}), param_specs overrides,
        and a run_tag. Redshift cuts (zlo/zhi) in config_overrides apply to
        the WHOLE sample before binning, same as any other run.
    n_bins : number of redshift bins (quantile-based, so each fold trains
        on roughly (n_bins-1)/n_bins of the data and holds out ~1/n_bins).
    output_prefix : basename for the summary CSV/plot; defaults to the
        run_tag.

    Returns
    -------
    pandas.DataFrame, one row per fold: z_lo, z_hi, n_heldout,
    mean_residual, mean_residual_err (bootstrap), and per-fold posterior
    means for every active parameter (for eyeballing whether they wander
    fold to fold).
    """
    config_overrides = dict(config_overrides or {})
    base_tag = config_overrides.pop("run_tag", "loo_zbins")
    output_prefix = output_prefix or base_tag.replace("/", "_")

    base_cfg = _copy.deepcopy(CONFIG)
    base_cfg.update(_copy.deepcopy(config_overrides))

    print(f"\n{'='*60}\nLeave-one-bin-out CV: loading full filtered sample...\n{'='*60}")
    df, data, cov_mat, _, _, _, keep_idx = load_and_filter_data(base_cfg)
    sigma_int = base_cfg.get("sigma_int", 0.0)

    z = data["z"]
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    edges[0]  -= 1e-8   # include the minimum point in bin 0
    edges[-1] += 1e-8   # include the maximum point in the last bin
    bin_id = np.digitize(z, edges) - 1   # 0..n_bins-1

    param_specs  = _copy.deepcopy(base_cfg.get("param_specs", DEFAULT_PARAM_SPECS))
    model_cfg    = base_cfg["model"]
    cosmo_type   = infer_cosmo_type(param_specs)
    active_names = [name for name, spec in param_specs.items() if spec["active"]]
    get_params   = build_param_getter(param_specs, active_names)

    def _cosmo_kwargs(params):
        Om0 = params["Om0"]
        if cosmo_type == "FlatLambdaCDM":
            return {"Om0": Om0, "cosmo_type": cosmo_type}
        elif cosmo_type == "wCDM":
            return {"Om0": Om0, "w": params["w"], "cosmo_type": cosmo_type}
        elif cosmo_type == "LambdaCDM":
            return {"Om0": Om0, "Ode0": params["Ode0"], "cosmo_type": cosmo_type}
        raise ValueError(f"Unknown cosmo_type '{cosmo_type}'")

    rows = []
    for b in range(n_bins):
        train_mask = bin_id != b
        held_mask  = bin_id == b
        n_train, n_held = int(train_mask.sum()), int(held_mask.sum())
        if n_held == 0:
            continue

        print(f"\n{'#'*60}\n# Fold {b+1}/{n_bins}: z in "
              f"[{edges[b]:.4f}, {edges[b+1]:.4f}]  "
              f"(train={n_train}, held-out={n_held})\n{'#'*60}")

        train_data  = _subset_data(data, train_mask)
        train_dfsub = df.iloc[train_mask].reset_index(drop=True)
        train_cov_geo = cov_mat[np.ix_(train_mask, train_mask)]
        inv_cov_train, log_det_train, C_sum_train = _refactorise_covariance(
            train_cov_geo, train_data["muerr"], sigma_int)

        fold_cfg = _copy.deepcopy(base_cfg)
        fold_cfg["run_tag"] = f"{base_tag}/loo_bin{b}"
        preloaded = (train_dfsub, train_data, train_cov_geo,
                    inv_cov_train, log_det_train, C_sum_train,
                    keep_idx[train_mask])

        results, sampler, active_names_fold, _, run_name = run_sampler(
            fold_cfg, preloaded=preloaded)

        # ---- Evaluate the held-out bin using this fold's posterior ----
        weights = np.exp(results.logwt - results.logz[-1])
        weights /= weights.sum()

        def weighted_median(values, wts):
            sorter = np.argsort(values)
            cumwt  = np.cumsum(wts[sorter]); cumwt /= cumwt[-1]
            return np.interp(0.5, cumwt, values[sorter])

        median_theta  = np.array([weighted_median(results.samples[:, i], weights)
                                  for i in range(results.samples.shape[1])])
        median_params = get_params(median_theta)

        # M is analytically marginalised, so it must be estimated from the
        # TRAINING data (same recipe as run.py's plot_hubble_diagram) —
        # using the held-out bin to estimate its own M would defeat the
        # point of a predictive check.
        mu_corr_train = compute_mu_corr(train_data, median_params, model_cfg)
        mu_th_train   = mu_theory(train_data["z"], **_cosmo_kwargs(median_params))
        M_hat         = float(np.mean(mu_corr_train - mu_th_train))

        held_data   = _subset_data(data, held_mask)
        mu_corr_held = compute_mu_corr(held_data, median_params, model_cfg)
        mu_th_held   = mu_theory(held_data["z"], **_cosmo_kwargs(median_params))
        resid_held   = mu_corr_held - mu_th_held - M_hat

        # Simple bootstrap on the held-out mean residual for an error bar
        # (does not use the full covariance's off-diagonal terms — a
        # deliberately conservative/approximate uncertainty, fine for a
        # diagnostic plot; do not over-interpret its precise value).
        rng = np.random.default_rng(1000 + b)
        boot = [resid_held[rng.integers(0, n_held, n_held)].mean()
                for _ in range(2000)]
        mean_resid, mean_resid_err = float(np.mean(resid_held)), float(np.std(boot))

        row = {"fold": b, "z_lo": edges[b], "z_hi": edges[b + 1],
              "n_train": n_train, "n_heldout": n_held,
              "mean_residual": round(mean_resid, 5),
              "mean_residual_err": round(mean_resid_err, 5),
              "pkl_path": pkl_path_for(run_name, fold_cfg)}
        for name in active_names_fold:
            row[f"{name}_mean"] = round(float(median_params[name]), 5)
        rows.append(row)

        flag = " ** >2sigma from zero **" if abs(mean_resid) > 2 * mean_resid_err else ""
        print(f"Fold {b+1} held-out mean residual: {mean_resid:+.4f} +/- "
              f"{mean_resid_err:.4f} mag{flag}")

    report = pd.DataFrame(rows)
    report.to_csv(f"{output_prefix}_loo_zbins.csv", index=False)
    print(f"\nLOO-CV summary saved: {output_prefix}_loo_zbins.csv")

    # ---- Summary plot: held-out residual vs redshift bin ----
    fig, ax = plt.subplots(figsize=(7, 4))
    z_mid = 0.5 * (report["z_lo"] + report["z_hi"])
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.errorbar(z_mid, report["mean_residual"], yerr=report["mean_residual_err"],
               fmt="o", color="steelblue", capsize=3)
    ax.set_xlabel("redshift bin (held-out)")
    ax.set_ylabel(r"held-out mean $\Delta\mu$ (mag)")
    ax.set_title(f"{output_prefix}: leave-one-z-bin-out predictive residual")
    fig.tight_layout()
    plot_path = f"{output_prefix}_loo_zbins.pdf"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"LOO-CV plot saved: {plot_path}")

    n_flagged = int((report["mean_residual"].abs()
                     > 2 * report["mean_residual_err"]).sum())
    if n_flagged:
        print(f"\n** {n_flagged}/{len(report)} bin(s) show a held-out mean "
              f"residual more than 2 sigma from zero -- the global model "
              f"may be missing z-dependent structure. See "
              f"{plot_path} and consider a z_evolve variant. **")
    else:
        print(f"\nAll {len(report)} held-out bins consistent with zero mean "
              f"residual within ~2 sigma -- no strong evidence of missed "
              f"z-dependent structure from this check.")

    return report


def _parse_args():
    p = argparse.ArgumentParser(
        description="Leave-one-redshift-bin-out cross-validation.")
    p.add_argument("--tag", default="loo_zbins")
    p.add_argument("--n-bins", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_loo_zbins(config_overrides={"run_tag": args.tag}, n_bins=args.n_bins)