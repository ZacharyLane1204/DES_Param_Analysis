"""
drilling_cones.py  —  SNe Ia Cosmology Pipeline
==================================================
Line-of-sight / sky-position ("drilling cones") systematic check.

Surveys like DES-SN observe a handful of fixed deep-field pointings —
clustering SNe by host sky position on the unit sphere (via DBSCAN, exactly
as in the exploratory snippet this script formalises) recovers those
pointings directly. This script:

  1. Fits the FULL sample once (the "standard" baseline posterior) —
     reusing whatever mass_cut / host_quality_cut / obs_z_type / z cuts are
     already set in your config, so this is an apples-to-apples baseline.
  2. Clusters the same (already-filtered) sample's host RA/Dec into cones
     via DBSCAN on the unit sphere.
  3. Refits each cone with >= cone_min_fit_size SNe as its own posterior.
  4. Compares every cone's posterior against the FULL-sample baseline via
     compare_runs.compare_two_runs (Gaussian + KDE tension, same universal
     sigma scale used throughout this pipeline) — this is the "multi-
     dimensional match to the standard-model posterior" the check is for:
     does any single field/cone pull the fit away from what the combined
     sample says, i.e. is there a line-of-sight bias hiding in one field?
  5. Saves a Mollweide sky plot coloured by cluster, annotated with each
     cone's tension against the baseline, plus a CSV summary.

SAFETY GATE: this does nothing unless config["drilling_cones"] is True
(default False) — see config.py. This keeps it out of any normal
experiment_runner.py/extra_runners.py sweep unless explicitly turned on
for a specific check.

Usage
-----
  from drilling_cones import run_drilling_cones
  report = run_drilling_cones(config_overrides={
      "run_tag": "best_model", "model": {...}, "drilling_cones": True})

or:
  python drilling_cones.py --tag best_model
  (the CLI always forces drilling_cones=True, since running the script
  directly is itself the explicit opt-in)
"""

import argparse
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from config       import CONFIG
from run          import load_and_filter_data, run_sampler, pkl_path_for
from compare_runs import compare_two_runs
from loo_zbins     import _subset_data, _refactorise_covariance


def find_sky_clusters(df, ra_col="HOST_RA", dec_col="HOST_DEC",
                      eps_deg=0.7, min_samples=20):
    """
    DBSCAN clustering of host sky positions on the unit sphere — same
    recipe as the exploratory notebook snippet this formalises.

    Rows with an invalid host position (RA <= 0 or Dec <= -90, the
    catalog's "no host" sentinel) are excluded from clustering and get
    label -1 (same convention DBSCAN uses for noise/unclustered points),
    so they end up correctly excluded from every per-cone fit downstream.

    Returns
    -------
    labels : ndarray (len(df),) int — cluster id per row, aligned to df's
        row order; -1 for invalid position OR DBSCAN-flagged noise points.
    """
    valid = (df[ra_col].values > 0) & (df[dec_col].values > -90)
    labels = np.full(len(df), -1, dtype=int)

    ra  = np.radians(df.loc[valid, ra_col].values)
    dec = np.radians(df.loc[valid, dec_col].values)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    coords = np.column_stack((x, y, z))

    theta = np.radians(eps_deg)
    eps   = 2 * np.sin(theta / 2)   # chord length on the unit sphere

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels[valid] = db.fit_predict(coords)
    return labels


def plot_cones(df, labels, tension_by_cluster, ra_col, dec_col, output_prefix):
    """Mollweide sky plot coloured by cluster, annotated with each cone's
    Gaussian tension against the full-sample baseline."""
    valid = labels >= 0
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="mollweide")

    ra_wrapped = np.where(df[ra_col].values > 180,
                          df[ra_col].values - 360, df[ra_col].values)

    sc = ax.scatter(np.radians(ra_wrapped[valid]), np.radians(df[dec_col].values[valid]),
                    s=3, alpha=0.6, c=labels[valid], cmap="tab10")
    ax.scatter(np.radians(ra_wrapped[~valid]), np.radians(df[dec_col].values[~valid]),
              s=1, alpha=0.15, c="grey", label="no valid host / unclustered")

    for cid, tension in tension_by_cluster.items():
        m = labels == cid
        if m.sum() == 0:
            continue
        ra_c  = np.radians(np.median(ra_wrapped[m]))
        dec_c = np.radians(np.median(df[dec_col].values[m]))
        label = f"cone {cid}\n{tension:.1f}$\\sigma$" if np.isfinite(tension) else f"cone {cid}\n(skipped)"
        colour = "red" if (np.isfinite(tension) and tension >= 2.0) else "black"
        ax.annotate(label, (ra_c, dec_c), fontsize=8, color=colour, ha="center",
                   fontweight="bold" if colour == "red" else "normal")

    ax.grid(True, alpha=0.3)
    ax.set_title(f"{output_prefix}: sky cones and tension vs. full-sample baseline")
    fig.tight_layout()
    path = f"{output_prefix}_cones_skymap.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Cone sky map saved: {path}")
    return path


def run_drilling_cones(config_overrides=None, eps_deg=None, min_samples=None,
                       min_fit_size=None, output_prefix=None, kde_max_dims=5):
    """
    Parameters
    ----------
    config_overrides : dict layered on top of CONFIG. Must include
        config_overrides["drilling_cones"] = True (or have
        CONFIG["drilling_cones"] already True) — otherwise this function
        prints a message and returns None immediately (see module
        docstring's SAFETY GATE).
    eps_deg, min_samples, min_fit_size : override CONFIG["cone_eps_deg"] /
        CONFIG["cone_min_samples"] / CONFIG["cone_min_fit_size"] if given.
    output_prefix : basename for the summary CSV/skymap; defaults to run_tag.

    Returns
    -------
    pandas.DataFrame, one row per fitted cone: cluster_id, n_sne,
    ra_centre, dec_centre, gaussian_nsigma, kde_nsigma, lnB, pkl_path.
    None if the safety gate is not satisfied.
    """
    config_overrides = dict(config_overrides or {})
    if not config_overrides.get("drilling_cones", CONFIG.get("drilling_cones", False)):
        print('drilling_cones is False -- skipping (set '
             'config_overrides={"drilling_cones": True, ...} to run this '
             'check explicitly; see config.py).')
        return None

    base_tag = config_overrides.pop("run_tag", "drilling_cones")
    output_prefix = output_prefix or base_tag.replace("/", "_")

    cfg = copy.deepcopy(CONFIG)
    cfg.update(copy.deepcopy(config_overrides))
    sigma_int = cfg.get("sigma_int", 0.0)
    ra_col    = cfg.get("col_host_ra", "HOST_RA")
    dec_col   = cfg.get("col_host_dec", "HOST_DEC")
    eps_deg      = eps_deg      if eps_deg      is not None else cfg.get("cone_eps_deg", 0.7)
    min_samples  = min_samples  if min_samples  is not None else cfg.get("cone_min_samples", 20)
    min_fit_size = min_fit_size if min_fit_size is not None else cfg.get("cone_min_fit_size", 50)

    print(f"\n{'='*60}\nDrilling cones: loading full filtered sample...\n{'='*60}")
    df, data, cov_mat, inv_cov_mat, log_det_const, C_sum, keep_idx = \
        load_and_filter_data(cfg)

    # ---- 1. Fit the full sample once as the baseline ----
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg["run_tag"] = f"{base_tag}/all"
    print(f"\n{'#'*60}\n# Baseline: full sample (N={len(df)})\n{'#'*60}")
    results_base, _, active_base, _, run_name_base = run_sampler(
        baseline_cfg, preloaded=(df, data, cov_mat, inv_cov_mat,
                                 log_det_const, C_sum, keep_idx))
    baseline_pkl = pkl_path_for(run_name_base, baseline_cfg)

    # ---- 2. Cluster host sky positions ----
    if ra_col not in df.columns or dec_col not in df.columns:
        raise KeyError(f"drilling_cones requires columns '{ra_col}'/'{dec_col}' "
                       f"in the data CSV.")
    labels = find_sky_clusters(df, ra_col, dec_col, eps_deg=eps_deg,
                               min_samples=min_samples)
    cluster_ids = sorted(c for c in np.unique(labels) if c >= 0)
    print(f"\nFound {len(cluster_ids)} sky cluster(s) (eps={eps_deg} deg, "
          f"min_samples={min_samples}); "
          f"{int(np.sum(labels < 0))} SNe unclustered/no valid host position.")

    # ---- 3-4. Fit each cone, compare to baseline ----
    rows = []
    tension_by_cluster = {}
    for cid in cluster_ids:
        mask_c = labels == cid
        n_c = int(mask_c.sum())
        ra_centre  = float(np.median(df.loc[mask_c, ra_col]))
        dec_centre = float(np.median(df.loc[mask_c, dec_col]))
        if n_c < min_fit_size:
            print(f"\nCone {cid}: {n_c} SNe < cone_min_fit_size={min_fit_size} "
                 f"-- skipping fit.")
            tension_by_cluster[cid] = np.nan
            rows.append({"cluster_id": cid, "n_sne": n_c,
                        "ra_centre": ra_centre, "dec_centre": dec_centre,
                        "gaussian_nsigma": "", "kde_nsigma": "", "lnB": "",
                        "pkl_path": "", "skipped_too_few": True})
            continue

        print(f"\n{'#'*60}\n# Cone {cid}: N={n_c}, RA~{ra_centre:.1f}, "
              f"Dec~{dec_centre:.1f}\n{'#'*60}")
        cone_data     = _subset_data(data, mask_c)
        cone_cov_geo  = cov_mat[np.ix_(mask_c, mask_c)]
        inv_cov_cone, log_det_cone, C_sum_cone = _refactorise_covariance(
            cone_cov_geo, cone_data["muerr"], sigma_int)

        cone_cfg = copy.deepcopy(cfg)
        cone_cfg["run_tag"] = f"{base_tag}/cone{cid}"
        preloaded_cone = (df.loc[mask_c].reset_index(drop=True), cone_data,
                         cone_cov_geo, inv_cov_cone, log_det_cone, C_sum_cone,
                         keep_idx[mask_c])
        _, _, _, _, run_name_cone = run_sampler(cone_cfg, preloaded=preloaded_cone)
        cone_pkl = pkl_path_for(run_name_cone, cone_cfg)

        summary = compare_two_runs(baseline_pkl, cone_pkl,
                                   output_prefix=f"{output_prefix}_cone{cid}",
                                   kde_max_dims=kde_max_dims, make_plot=False)
        tension_by_cluster[cid] = summary["gaussian_nsigma"]

        rows.append({"cluster_id": cid, "n_sne": n_c,
                    "ra_centre": ra_centre, "dec_centre": dec_centre,
                    "gaussian_nsigma": summary["gaussian_nsigma"],
                    "kde_nsigma": summary["kde_nsigma"],
                    "lnB": summary["lnB"], "pkl_path": cone_pkl,
                    "skipped_too_few": False})

    report = pd.DataFrame(rows)
    report.to_csv(f"{output_prefix}_drilling_cones.csv", index=False)
    print(f"\nDrilling-cones summary saved: {output_prefix}_drilling_cones.csv")

    plot_cones(df, labels, tension_by_cluster, ra_col, dec_col, output_prefix)

    fitted = report[~report["skipped_too_few"]]
    flagged = fitted[pd.to_numeric(fitted["gaussian_nsigma"], errors="coerce") >= 2.0]
    if len(flagged):
        print(f"\n** {len(flagged)}/{len(fitted)} fitted cone(s) show >= 2 "
             f"sigma tension against the full-sample baseline -- possible "
             f"line-of-sight/field-dependent systematic. See "
             f"{output_prefix}_cones_skymap.pdf and the flagged cone_id(s): "
             f"{list(flagged['cluster_id'])}. **")
    else:
        print(f"\nAll {len(fitted)} fitted cones consistent with the full-"
             f"sample baseline within 2 sigma -- no strong evidence of a "
             f"line-of-sight bias from this check.")

    return report


def _parse_args():
    p = argparse.ArgumentParser(
        description="Line-of-sight / sky-cone drilling systematic check "
                    "(DBSCAN-clustered host sky positions vs. full-sample "
                    "baseline). Running this script directly always forces "
                    "drilling_cones=True.")
    p.add_argument("--tag", default="drilling_cones")
    p.add_argument("--eps-deg", type=float, default=None)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--min-fit-size", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_drilling_cones(
        config_overrides={"run_tag": args.tag, "drilling_cones": True},
        eps_deg=args.eps_deg, min_samples=args.min_samples,
        min_fit_size=args.min_fit_size)