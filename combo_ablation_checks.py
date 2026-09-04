"""
combo_ablation_checks.py  —  SNe Ia Cosmology Pipeline
==========================================================
Full robustness pass over EVERY competing model in best_model.COMBOS.

For each combo this runs, in order:
  1. Main fit               -- the combo on the full sample.
  2. Degeneracy scan        -- on that fit's saved pkl.
  3. Host-match quality     -- all-hosts vs. strict-host-match refit + tension.
  4. LOO redshift-bin CV    -- one refit per held-out z bin + predictive
                               residual on the held-out bin.
  5. Drilling cones         -- full-sample reference + one refit per sky cone,
                               each cone individually labelled, under a
                               broad-uniform Om0 prior so the cosmology is
                               free to move.

WHAT CHANGED (and why)
----------------------
The previous version looped over combos SERIALLY and, inside each combo,
called host_match_quality.run_host_quality_check / loo_zbins.run_loo_zbins /
drilling_cones.run_drilling_cones, each of which looped serially over its own
refits. For 9 combos that is of order 200 strictly sequential nested-sampling
runs -- days of wall time on an idle machine.

This version DECOMPOSES the whole pass into individual sampling jobs and runs
them through parallel_runner.run_jobs, so `--workers N` gives near-linear
speed-up and every job gets its own log file. The trade-off is that the three
check modules' driver functions are no longer called; their *primitives*
(_subset_data, _refactorise_covariance, find_sky_clusters, plot_cones,
compare_two_runs) are imported and reused instead, so the science is identical
while the scheduling is ours. Those modules remain fully usable standalone.

PHASES
------
  Phase 0 (serial, parent) : load the sample ONCE, derive the z-bin edges and
                             the sky-cone assignment, and build every job.
                             These are properties of the DATA CUTS, not of the
                             model, so they are identical for every combo --
                             deriving them once guarantees every combo is
                             ablated on exactly the same bins and cones, which
                             is what makes the deltas comparable.
  Phase 1 (parallel)       : every nested-sampling fit.
  Phase 2 (parallel)       : every posterior comparison (host-quality tension,
                             per-cone tension) and the degeneracy scans, which
                             can only start once Phase 1's pkls exist.
  Phase 3 (serial, parent) : assemble the summary CSVs and the cone sky maps.

NAMING AND FILING
-----------------
Every run tag is built from TAG_TEMPLATES below, so all of a combo's runs are
sectioned by CHECK TYPE and sort together on disk:

    combo/fit_<combo>
    combo/host_qual_<combo>_<all|strict>
    combo/loo_zbin_<combo>_b<NN>
    combo/drilling_all_<combo>          <- full-sample reference for the cones
    combo/drilling_<NN>_<combo>         <- cone NN, labelled in cone_plan.csv

run.py's pkl_path_for() treats everything before the last "/" as a
subdirectory, so these all land in "<output_dir>/combo/". Set
--section-dirs to file them into per-check subdirectories
("<output_dir>/combo/host_qual/..." etc.) instead.

Cone numbers are NOT DBSCAN's raw cluster ids: DBSCAN ids are unstable if the
sample or eps changes. Phase 0 assigns each surviving cone a stable, contiguous
index ordered by descending SN count and records the mapping (index, raw
cluster id, RA/Dec centre, N) in "<out>/cone_plan.csv", so a tag like
"combo/drilling_03_..." is always resolvable back to a real patch of sky.

USAGE
-----
    # cheapest smoke test: main fits only, no sub-checks
    python combo_ablation_checks.py --fits-only --workers 8

    # everything, 16 workers
    python combo_ablation_checks.py --workers 16

    # one combo, everything
    python combo_ablation_checks.py --only mass_linear_ssfr_tanh --workers 8

    # see the job list (including every cone) without running anything
    python combo_ablation_checks.py --dry-run
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
from dynesty import utils as dyfunc

from config import CONFIG, DEFAULT_PARAM_SPECS
from run import (load_and_filter_data, run_sampler, pkl_path_for,
                 build_param_getter, compute_mu_corr, mu_theory,
                 infer_cosmo_type)
from experiment_naming import ExperimentRegistry
from loo_zbins import _subset_data, _refactorise_covariance
from drilling_cones import find_sky_clusters, plot_cones
from compare_runs import compare_two_runs
from best_model import (TERMS, COMBOS, BASELINE_COMBO,
                        merge_terms as _merge_terms, combo_tag)
from parallel_runner import Job, run_jobs, add_parallel_args, safe_label

import degeneracy_scan


# ===========================================================================
# 1.  NAMING
# ===========================================================================
# {combo} is best_model.combo_tag(...) (underscore-joined term names).
# Edit here to change the whole naming scheme in one place; nothing below
# hard-codes a tag.
TAG_TEMPLATES = {
    "fit":        "combo/fit_{combo}",
    "host_qual":  "combo/host_qual_{combo}_{cut}",
    "loo_zbin":   "combo/loo_zbin_{combo}_b{bin:02d}",
    "drill_ref":  "combo/drilling_all_{combo}",
    "drill_cone": "combo/drilling_{cone:02d}_{combo}",
}

# --section-dirs variant: same names, but each check gets its own directory
# under <output_dir>/combo/ so the Plots tree is browsable per check.
TAG_TEMPLATES_SECTIONED = {
    "fit":        "combo/fit/{combo}",
    "host_qual":  "combo/host_qual/{combo}_{cut}",
    "loo_zbin":   "combo/loo_zbin/{combo}_b{bin:02d}",
    "drill_ref":  "combo/drilling/{combo}_all",
    "drill_cone": "combo/drilling/{combo}_c{cone:02d}",
}

OUT_DIR = "combo_ablation"          # CSV/plot outputs (not the pkl tree)
REGISTRY_MAIN = "run_combo_registry.csv"
REGISTRY_CONES = "run_drilling_cones_registry.csv"


def _tag(kind, templates, **kw):
    return templates[kind].format(**kw)


# ===========================================================================
# 2.  CONFIG CONSTRUCTION
# ===========================================================================

def build_combo_cfg(term_names, registry_file, registry, templates):
    """Build one combo's main-fit config through the shared ExperimentRegistry.

    Using the registry (rather than hand-assembling a dict) means a combo that
    (a) repeats an earlier tag, (b) resolves to an identical model + active-
    parameter fingerprint under a different tag, or (c) activates a parameter
    that its own model makes degenerate (gamma with mass="none", a linear
    model's own zero-point) is caught HERE, before an expensive fit, by the
    same guards experiment_runner.py and extra_runners.py use.

    NOTE: no host_colour fallback is injected. An older version forced
    "host_colour": "linear" for every combo that did not name it. Because eta
    (host_colour's amplitude) defaults to active=False, fixed=0.035, that
    applied an UNFITTED +0.035*HOST_COLOR mag correction to every combo -- one
    the sampler never saw and the Occam factor never charged for, contaminating
    every ablation delta. Combos now inherit CONFIG["model"]["host_colour"]
    ("none"); a combo that wants host colour must say so via a TERMS entry that
    ALSO activates eta.
    """
    model_overrides, param_overrides = _merge_terms(term_names)
    tag = _tag("fit", templates, combo=combo_tag(term_names))
    return registry.build(tag,
                          param_overrides=param_overrides,
                          config_overrides={"model": {**CONFIG["model"],
                                                      **model_overrides},
                                            "registry_file": registry_file})


def broad_uniform_om0_overrides():
    """Broad uniform Om0, w/Ode0 inactive so the cosmology stays FlatLambdaCDM.

    Same recipe as drilling_cones_checks.py's
    _broad_uniform_flatlcdm_overrides(only_om0=True). The cones check exists to
    ask "does the cosmology move between patches of sky", so Om0 must NOT carry
    the informative prior the publication runs use -- otherwise the prior, not
    the data, sets every cone's Om0 and every cone looks reassuringly
    consistent for entirely the wrong reason.
    """
    om0_lo, om0_hi = DEFAULT_PARAM_SPECS["Om0"]["range"]
    return {"Om0": {"active": True, "prior": "uniform", "range": [om0_lo, om0_hi]},
            "w":    {"active": False},
            "Ode0": {"active": False}}


def _with(cfg, **overrides):
    out = copy.deepcopy(cfg)
    out.update(overrides)
    return out


# ===========================================================================
# 3.  WORKER ENTRY POINTS
# ===========================================================================
# These MUST be module-level: parallel_runner uses spawn, which pickles the
# function by qualified name. Each returns a small picklable dict.

def job_sample(cfg, subset_idx=None, expect_n=None, loo=None):
    """Run one nested-sampling fit, optionally restricted to a positional subset.

    subset_idx : list of positional indices into the FILTERED sample (the order
        load_and_filter_data returns). The parent derives these once so every
        combo shares identical bins/cones; the worker re-loads the data itself
        rather than having a 26 MB covariance matrix pickled to it per job.
    expect_n   : the parent's filtered-sample size. Asserted here so that if a
        config difference ever made a worker's filtered sample a different
        length, the positional indices would raise instead of silently
        selecting the wrong supernovae.
    loo        : dict(fold, z_lo, z_hi, held_idx) -- when present, the held-out
        bin's predictive residual is evaluated with this fit's posterior.
    """
    df, data, cov_mat, inv_cov, log_det, C_sum, keep_idx = load_and_filter_data(cfg)
    n_full = len(data["z"])
    if expect_n is not None and n_full != expect_n:
        raise RuntimeError(
            f"Filtered sample is {n_full} SNe but the parent derived its "
            f"subset indices from {expect_n}. Positional indices would select "
            f"the wrong SNe -- refusing to run {cfg.get('run_tag')!r}.")

    if subset_idx is None:
        preloaded = (df, data, cov_mat, inv_cov, log_det, C_sum, keep_idx)
        sub_data = data
    else:
        mask = np.zeros(n_full, dtype=bool)
        mask[np.asarray(subset_idx, dtype=int)] = True
        sub_data = _subset_data(data, mask)
        sub_cov = cov_mat[np.ix_(mask, mask)]
        inv_s, logdet_s, csum_s = _refactorise_covariance(
            sub_cov, sub_data["muerr"], cfg.get("sigma_int", 0.0))
        preloaded = (df.loc[mask].reset_index(drop=True), sub_data, sub_cov,
                     inv_s, logdet_s, csum_s, keep_idx[mask])

    results, _, active_names, _, run_name = run_sampler(cfg, preloaded=preloaded)
    out = {"run_tag": cfg.get("run_tag"), "run_name": run_name,
           "pkl_path": pkl_path_for(run_name, cfg),
           "active_params": ",".join(active_names),
           "n_sne": int(len(sub_data["z"])),
           "logz": float(results.logz[-1]),
           "logz_err": float(results.logzerr[-1])}

    # Posterior mean/std of every active parameter. Recorded on EVERY fit
    # (not just the main one) because the drilling-cones table has to show how
    # the recovered cosmology moves from cone to cone, and the LOO table how it
    # moves fold to fold -- both need Om0 (and friends) per sub-run, not just a
    # tension statistic.
    w = np.exp(results.logwt - results.logz[-1])
    w /= w.sum()
    mean, cov = dyfunc.mean_and_cov(results.samples, w)
    std = np.sqrt(np.diag(cov))
    for name, m, s in zip(active_names, mean, std):
        out[f"{name}_mean"] = float(m)
        out[f"{name}_std"] = float(s)

    if loo is not None:
        out.update(_evaluate_loo_fold(results, active_names, cfg, data,
                                      sub_data, loo))
    return out


def _evaluate_loo_fold(results, active_names, cfg, full_data, train_data, loo):
    """Predictive residual of the held-out z bin under this fold's posterior.

    M (the absolute-magnitude zero point) is analytically marginalised in the
    likelihood, so it has to be re-estimated from the TRAINING data -- using
    the held-out bin to estimate its own zero point would absorb exactly the
    offset this check is trying to detect and guarantee a null result.
    """
    weights = np.exp(results.logwt - results.logz[-1])
    weights /= weights.sum()

    def weighted_median(values, wts):
        order = np.argsort(values)
        cw = np.cumsum(wts[order])
        cw /= cw[-1]
        return np.interp(0.5, cw, values[order])

    specs = cfg["param_specs"]
    get_params = build_param_getter(specs, active_names)
    theta = np.array([weighted_median(results.samples[:, i], weights)
                      for i in range(results.samples.shape[1])])
    params = get_params(theta)

    cosmo_type = infer_cosmo_type(specs)
    if cosmo_type == "FlatLambdaCDM":
        ck = {"Om0": params["Om0"], "cosmo_type": cosmo_type}
    elif cosmo_type == "wCDM":
        ck = {"Om0": params["Om0"], "w": params["w"], "cosmo_type": cosmo_type}
    elif cosmo_type == "LambdaCDM":
        ck = {"Om0": params["Om0"], "Ode0": params["Ode0"], "cosmo_type": cosmo_type}
    else:
        raise ValueError(f"Unknown cosmo_type '{cosmo_type}'")

    model_cfg = cfg["model"]
    M_hat = float(np.mean(compute_mu_corr(train_data, params, model_cfg)
                          - mu_theory(train_data["z"], **ck)))

    held_mask = np.zeros(len(full_data["z"]), dtype=bool)
    held_mask[np.asarray(loo["held_idx"], dtype=int)] = True
    held = _subset_data(full_data, held_mask)
    resid = (compute_mu_corr(held, params, model_cfg)
             - mu_theory(held["z"], **ck) - M_hat)

    # Bootstrap error bar on the held-out mean. Deliberately ignores the
    # covariance's off-diagonal terms -- a conservative diagnostic scale, not a
    # calibrated uncertainty. Do not quote it as one.
    rng = np.random.default_rng(1000 + int(loo["fold"]))
    n = len(resid)
    boot = [resid[rng.integers(0, n, n)].mean() for _ in range(2000)]

    row = {"fold": int(loo["fold"]), "z_lo": float(loo["z_lo"]),
           "z_hi": float(loo["z_hi"]), "n_heldout": int(n),
           "mean_residual": float(np.mean(resid)),
           "mean_residual_err": float(np.std(boot))}
    # Suffixed "_median" so these (weighted-median point estimates, used to
    # predict the held-out bin) never collide with the "_mean"/"_std"
    # posterior summaries job_sample records for every fit.
    for name in active_names:
        row[f"{name}_median"] = float(params[name])
    return row


def job_compare(pkl_a, pkl_b, output_prefix, kde_max_dims=5, make_plot=False):
    """Posterior tension between two completed runs (Phase 2)."""
    return compare_two_runs(pkl_a, pkl_b, output_prefix=output_prefix,
                            kde_max_dims=kde_max_dims, make_plot=make_plot)


def job_degeneracy(pkl_path, output_prefix, threshold=0.85):
    """Degeneracy scan on a completed run (Phase 2)."""
    deg = degeneracy_scan.scan_degeneracies(pkl_path, threshold=threshold,
                                            output_prefix=output_prefix)
    return {"n_flagged": len(deg["flagged"]),
            "flagged": "; ".join(f"{a}<->{b}:{c:+.2f}" for a, b, c in deg["flagged"])}


# ===========================================================================
# 4.  PHASE 0  —  derive the shared bins/cones and build the job list
# ===========================================================================

def derive_sample_plan(base_cfg, n_bins, eps_deg, min_samples, min_fit_size,
                       out_dir=OUT_DIR):
    """Load the sample once and derive the z-bin and sky-cone partitions.

    Both partitions depend only on the DATA CUTS, which are identical for every
    combo, so deriving them once here (a) guarantees all combos are ablated on
    exactly the same bins and cones -- the precondition for their deltas being
    comparable at all -- and (b) avoids every worker re-running DBSCAN.
    """
    df, data, cov_mat, _, _, _, _ = load_and_filter_data(base_cfg)
    n_full = len(data["z"])
    z = data["z"]

    # ---- z bins (equal-count quantiles) ----
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-8            # include the min in bin 0
    edges[-1] += 1e-8           # include the max in the last bin
    bin_id = np.digitize(z, edges) - 1
    zbins = []
    for b in range(n_bins):
        held = np.flatnonzero(bin_id == b)
        if held.size == 0:
            continue
        zbins.append({"fold": b, "z_lo": float(edges[b]), "z_hi": float(edges[b + 1]),
                      "held_idx": held.tolist(),
                      "train_idx": np.flatnonzero(bin_id != b).tolist(),
                      "n_heldout": int(held.size)})

    # ---- sky cones ----
    ra_col = base_cfg.get("col_host_ra", "HOST_RA")
    dec_col = base_cfg.get("col_host_dec", "HOST_DEC")
    cones, labels = [], None
    if ra_col in df.columns and dec_col in df.columns:
        labels = find_sky_clusters(df, ra_col, dec_col, eps_deg=eps_deg,
                                   min_samples=min_samples)
        raw_ids = sorted(c for c in np.unique(labels) if c >= 0)
        # Stable, contiguous cone numbers ordered by descending size. DBSCAN's
        # own ids depend on row order and eps and would silently re-label cones
        # between runs, making "cone 3" mean different sky on different days.
        raw_ids.sort(key=lambda c: -int(np.sum(labels == c)))
        for k, cid in enumerate(raw_ids):
            mask = labels == cid
            idx = np.flatnonzero(mask)
            cones.append({
                "cone": k,
                "dbscan_id": int(cid),
                "n_sne": int(idx.size),
                "ra_centre": float(np.median(df.loc[mask, ra_col])),
                "dec_centre": float(np.median(df.loc[mask, dec_col])),
                "idx": idx.tolist(),
                "fitted": bool(idx.size >= min_fit_size),
            })
        for c in cones:
            c["label"] = (f"c{c['cone']:02d} (RA {c['ra_centre']:.1f}, "
                          f"Dec {c['dec_centre']:+.1f}, N={c['n_sne']})")

    os.makedirs(out_dir, exist_ok=True)
    if cones:
        pd.DataFrame([{k: v for k, v in c.items() if k != "idx"} for c in cones]) \
            .to_csv(os.path.join(out_dir, "cone_plan.csv"), index=False)
    pd.DataFrame([{k: v for k, v in b.items()
                   if k not in ("held_idx", "train_idx")} for b in zbins]) \
        .to_csv(os.path.join(out_dir, "zbin_plan.csv"), index=False)

    return {"n_full": n_full, "zbins": zbins, "cones": cones,
            "labels": labels, "df": df, "ra_col": ra_col, "dec_col": dec_col}


def build_jobs(combos, plan, templates, registry_file, do_host_quality,
               do_loo, do_cones, min_fit_size, deep_combos=None,
               cone_combos=None):
    """Every Phase-1 sampling job, plus the bookkeeping needed for Phase 2.

    deep_combos
        Optional set of combo tags. When given, only these combos receive the
        LOO z-bin folds; every combo still gets its main fit and its
        host-quality pair. This is the main cost lever for LOO: it is n_bins
        fits per model, while the main fit is the only one needed to RANK the
        ladder. Rank first on all of them, then validate the two or three
        that are actually in contention.

    cone_combos
        Combo tags that get drilling cones, chosen INDEPENDENTLY of
        deep_combos. Defaults to the BASELINE model alone.

        A cone asks whether the cosmology recovered from one patch of sky
        differs from the all-sky one. That is a question about line-of-sight
        structure in THE DATA, not about the standardisation model, so the
        number that means something is the baseline one. Run it on nine
        models before you have that and you get nine numbers with no
        reference to read them against, at nine times the cost -- cones are
        n_cones + 1 fits per model and are the single most expensive check
        here. Establish it on the baseline, then repeat on the adopted model
        once the ladder has chosen one.
    """
    registry = ExperimentRegistry(CONFIG, DEFAULT_PARAM_SPECS)
    jobs, meta = [], []
    n_full = plan["n_full"]

    for term_names in combos:
        ctag = combo_tag(term_names)
        deep = deep_combos is None or ctag in deep_combos
        cones_here = cone_combos is None or ctag in cone_combos
        cfg = build_combo_cfg(term_names, registry_file, registry, templates)

        # ---- 1. main fit ----
        jobs.append(Job(label=cfg["run_tag"], func=job_sample,
                        kwargs={"cfg": cfg, "expect_n": n_full},
                        group="fit", cost=10.0))
        meta.append({"combo": ctag, "terms": "|".join(term_names),
                     "kind": "fit", "tag": cfg["run_tag"]})

        # ---- 2. host-match quality (all vs. strict) ----
        # host_quality_cut changes which SNe survive the cut, so these two
        # runs load their own (differently sized) samples -- no expect_n, no
        # subset indices.
        if do_host_quality:
            for cut in ("all", "strict"):
                hq = _with(cfg,
                           run_tag=_tag("host_qual", templates, combo=ctag, cut=cut),
                           host_quality_cut=cut,
                           host_ddlr_max=cfg.get("host_ddlr_max", 2.0))
                jobs.append(Job(label=hq["run_tag"], func=job_sample,
                                kwargs={"cfg": hq}, group="host_qual", cost=8.0))
                meta.append({"combo": ctag, "kind": f"host_qual_{cut}",
                             "tag": hq["run_tag"]})

        # ---- 3. LOO redshift bins ----
        if do_loo and deep:
            for b in plan["zbins"]:
                lcfg = _with(cfg, run_tag=_tag("loo_zbin", templates,
                                               combo=ctag, bin=b["fold"]))
                jobs.append(Job(
                    label=lcfg["run_tag"], func=job_sample,
                    kwargs={"cfg": lcfg, "subset_idx": b["train_idx"],
                            "expect_n": n_full,
                            "loo": {"fold": b["fold"], "z_lo": b["z_lo"],
                                    "z_hi": b["z_hi"], "held_idx": b["held_idx"]}},
                    group="loo_zbin", cost=7.0))
                meta.append({"combo": ctag, "kind": "loo_zbin",
                             "fold": b["fold"], "tag": lcfg["run_tag"]})

        # ---- 4. drilling cones (broad-uniform Om0, own registry) ----
        if do_cones and cones_here and plan["cones"]:
            dc_specs = copy.deepcopy(cfg["param_specs"])
            for name, upd in broad_uniform_om0_overrides().items():
                dc_specs[name].update(upd)

            ref = _with(cfg, run_tag=_tag("drill_ref", templates, combo=ctag),
                        param_specs=dc_specs, registry_file=REGISTRY_CONES,
                        drilling_cones=True)
            jobs.append(Job(label=ref["run_tag"], func=job_sample,
                            kwargs={"cfg": ref, "expect_n": n_full},
                            group="drilling", cost=9.0))
            meta.append({"combo": ctag, "kind": "drill_ref", "tag": ref["run_tag"]})

            for c in plan["cones"]:
                if not c["fitted"]:
                    continue
                ccfg = _with(cfg, run_tag=_tag("drill_cone", templates,
                                               combo=ctag, cone=c["cone"]),
                             param_specs=copy.deepcopy(dc_specs),
                             registry_file=REGISTRY_CONES, drilling_cones=True)
                jobs.append(Job(
                    label=ccfg["run_tag"], func=job_sample,
                    kwargs={"cfg": ccfg, "subset_idx": c["idx"],
                            "expect_n": n_full},
                    group="drilling",
                    # Cones are small subsamples, so they sample much faster
                    # than a full-sample fit; dispatch them last.
                    cost=1.0 + 4.0 * c["n_sne"] / max(n_full, 1)))
                meta.append({"combo": ctag, "kind": "drill_cone",
                             "cone": c["cone"], "cone_label": c["label"],
                             "n_sne": c["n_sne"], "ra_centre": c["ra_centre"],
                             "dec_centre": c["dec_centre"], "tag": ccfg["run_tag"]})

    return jobs, pd.DataFrame(meta)


# ===========================================================================
# 5.  DRIVER
# ===========================================================================

def run_combo_checks(combos=None, only=None, registry_file=REGISTRY_MAIN,
                     n_workers=None, sequential=False, log_dir="logs/combo",
                     out_dir=OUT_DIR, dry_run=False, capture_output=True,
                     section_dirs=False, degeneracy_threshold=0.85,
                     loo_n_bins=4, run_host_quality=True, run_loo=True,
                     run_drilling_cones=True, cones_eps_deg=None,
                     cones_min_samples=None, cones_min_fit_size=None,
                     deep_only=None, cone_only=None, cones_all=False):
    """Run the full ablation pass. Returns a dict of summary DataFrames."""
    combos = combos if combos is not None else COMBOS
    if only:
        only = set(only)
        combos = [c for c in combos if combo_tag(c) in only]
        if not combos:
            raise SystemExit(f"No combo matched --only {sorted(only)}. "
                             f"Available: {[combo_tag(c) for c in COMBOS]}")

    # Drilling cones default to the BASELINE model alone -- see build_jobs'
    # cone_combos docstring. --cone-only names a different set; --cones-all
    # runs every combo (the old behaviour, ~n_cones+1 fits x 9).
    if cones_all:
        cone_combos = None
    elif cone_only:
        cone_combos = set(cone_only)
    else:
        cone_combos = {combo_tag(BASELINE_COMBO)}

    deep_combos = None
    if deep_only:
        deep_combos = set(deep_only)
        known = {combo_tag(c) for c in combos}
        unknown = deep_combos - known
        if unknown:
            raise SystemExit(f"--deep-only names combos that are not being "
                             f"run: {sorted(unknown)}. Available: {sorted(known)}")

    templates = TAG_TEMPLATES_SECTIONED if section_dirs else TAG_TEMPLATES
    os.makedirs(out_dir, exist_ok=True)

    base_cfg = copy.deepcopy(CONFIG)
    base_cfg["registry_file"] = registry_file
    eps = cones_eps_deg if cones_eps_deg is not None else base_cfg.get("cone_eps_deg", 0.7)
    minsamp = (cones_min_samples if cones_min_samples is not None
               else base_cfg.get("cone_min_samples", 20))
    minfit = (cones_min_fit_size if cones_min_fit_size is not None
              else base_cfg.get("cone_min_fit_size", 50))

    # ---- Phase 0 ----
    print(f"\n{'=' * 74}\n  Phase 0: deriving shared z bins and sky cones "
          f"(once, for all combos)\n{'=' * 74}")
    plan = derive_sample_plan(base_cfg, loo_n_bins, eps, minsamp, minfit, out_dir)
    n_fit_cones = sum(1 for c in plan["cones"] if c["fitted"])
    print(f"  Filtered sample : {plan['n_full']} SNe")
    print(f"  z bins          : {len(plan['zbins'])}")
    print(f"  Sky cones       : {len(plan['cones'])} found, {n_fit_cones} "
          f"with >= {minfit} SNe (fitted)")
    for c in plan["cones"]:
        print(f"      {c['label']}{'' if c['fitted'] else '   [SKIPPED: too few]'}")

    if run_drilling_cones and cone_combos is not None:
        present = {combo_tag(c) for c in combos} & cone_combos
        missing = cone_combos - {combo_tag(c) for c in combos}
        if missing:
            raise SystemExit(f"--cone-only names combos that are not being "
                             f"run: {sorted(missing)}")
        print(f"\n  Drilling cones restricted to {len(present)}/{len(combos)} "
              f"combo(s): {', '.join(sorted(present))}")
        print("  (--cones-all runs every combo; see BASELINE_COMBO in "
              "best_model.py for why this is the default.)")

    if deep_combos is not None:
        print(f"\n  LOO z-bin folds restricted to "
              f"{len(deep_combos)}/{len(combos)} combo(s):")
        for t in sorted(deep_combos):
            print(f"      {t}")
        print("  All combos still get their main fit and host-quality pair.")

    jobs, meta = build_jobs(combos, plan, templates, registry_file,
                            run_host_quality, run_loo, run_drilling_cones,
                            minfit, deep_combos=deep_combos,
                            cone_combos=cone_combos)
    meta.to_csv(os.path.join(out_dir, "job_plan.csv"), index=False)

    # ---- Phase 1 ----
    res1 = run_jobs(jobs, n_workers=n_workers, log_dir=log_dir,
                    sequential=sequential, capture_output=capture_output,
                    dry_run=dry_run, title="Phase 1/2: nested-sampling fits",
                    summary_name="phase1_summary.log",
                    extra_lines=[f"Combos: {len(combos)}",
                                 f"Registry: {registry_file}"])
    if dry_run:
        print("Job plan written to "
              f"{os.path.join(out_dir, 'job_plan.csv')}\n")
        return {"plan": meta}

    by_tag = {r.label: r for r in res1}

    def pkl_of(tag):
        r = by_tag.get(tag)
        return r.value["pkl_path"] if (r and r.status == "ok" and r.value) else None

    # ---- Phase 2: comparisons + degeneracy scans ----
    jobs2, meta2 = [], []
    for _, m in meta.iterrows():
        ctag = m["combo"]
        if m["kind"] == "fit":
            p = pkl_of(m["tag"])
            if p:
                jobs2.append(Job(label=f"degeneracy/{ctag}", func=job_degeneracy,
                                 kwargs={"pkl_path": p,
                                         "output_prefix": os.path.join(out_dir, f"deg_{ctag}"),
                                         "threshold": degeneracy_threshold},
                                 group="degeneracy"))
                meta2.append({"combo": ctag, "kind": "degeneracy",
                              "label": f"degeneracy/{ctag}"})
        elif m["kind"] == "drill_cone":
            a = pkl_of(_tag("drill_ref", templates, combo=ctag))
            b = pkl_of(m["tag"])
            if a and b:
                lbl = f"cone_tension/{ctag}_c{int(m['cone']):02d}"
                jobs2.append(Job(label=lbl, func=job_compare,
                                 kwargs={"pkl_a": a, "pkl_b": b,
                                         "output_prefix": os.path.join(
                                             out_dir, f"cone_{ctag}_c{int(m['cone']):02d}")},
                                 group="cone_tension"))
                meta2.append({"combo": ctag, "kind": "cone_tension",
                              "cone": int(m["cone"]),
                              "cone_label": m.get("cone_label", ""),
                              "n_sne": m.get("n_sne"), "label": lbl})

    if run_host_quality:
        for term_names in combos:
            ctag = combo_tag(term_names)
            a = pkl_of(_tag("host_qual", templates, combo=ctag, cut="all"))
            b = pkl_of(_tag("host_qual", templates, combo=ctag, cut="strict"))
            if a and b:
                lbl = f"host_tension/{ctag}"
                jobs2.append(Job(label=lbl, func=job_compare,
                                 kwargs={"pkl_a": a, "pkl_b": b,
                                         "output_prefix": os.path.join(
                                             out_dir, f"hostqual_{ctag}")},
                                 group="host_tension"))
                meta2.append({"combo": ctag, "kind": "host_tension", "label": lbl})

    res2 = run_jobs(jobs2, n_workers=n_workers, log_dir=log_dir,
                    sequential=sequential, capture_output=capture_output,
                    title="Phase 2/2: posterior comparisons and degeneracy scans",
                    summary_name="phase2_summary.log") if jobs2 else []
    by_label2 = {r.label: r for r in res2}

    # ---- Phase 3: assemble ----
    out = _assemble(combos, meta, meta2, by_tag, by_label2, plan, templates, out_dir)
    return out


def _val(res):
    return res.value if (res is not None and res.status == "ok" and res.value) else None


def _assemble(combos, meta, meta2, by_tag, by_label2, plan, templates, out_dir):
    """Write the four summary CSVs and the per-combo cone sky maps."""
    meta2 = pd.DataFrame(meta2)

    # ---- evidence of every fit ----
    ev_rows = []
    for _, m in meta.iterrows():
        v = _val(by_tag.get(m["tag"]))
        row = {k: m.get(k) for k in ("combo", "kind", "tag", "fold", "cone",
                                     "cone_label") if k in m}
        row["status"] = "ok" if v else "FAILED"
        if v:
            row.update({"n_sne": v["n_sne"], "logz": v["logz"],
                        "logz_err": v["logz_err"], "pkl_path": v["pkl_path"],
                        "active_params": v["active_params"]})
        ev_rows.append(row)
    evidence = pd.DataFrame(ev_rows)
    evidence.to_csv(os.path.join(out_dir, "combo_all_runs.csv"), index=False)

    # ---- LOO ----
    loo_rows = []
    for _, m in meta[meta["kind"] == "loo_zbin"].iterrows():
        v = _val(by_tag.get(m["tag"]))
        if v:
            loo_rows.append({"combo": m["combo"], "tag": m["tag"], **{
                k: v[k] for k in v if k not in ("run_tag", "run_name")}})
    loo = pd.DataFrame(loo_rows)
    if len(loo):
        loo["flagged_2sigma"] = (loo["mean_residual"].abs()
                                 > 2 * loo["mean_residual_err"])
        loo.to_csv(os.path.join(out_dir, "combo_loo_zbins.csv"), index=False)

    # ---- host quality ----
    hq_rows = []
    if len(meta2):
        for _, m in meta2[meta2["kind"] == "host_tension"].iterrows():
            v = _val(by_label2.get(m["label"]))
            if v:
                hq_rows.append({"combo": m["combo"],
                                "gaussian_nsigma": v.get("gaussian_nsigma"),
                                "kde_nsigma": v.get("kde_nsigma"),
                                "lnB": v.get("lnB")})
    hostq = pd.DataFrame(hq_rows)
    if len(hostq):
        hostq.to_csv(os.path.join(out_dir, "combo_host_quality.csv"), index=False)

    # ---- cones ----
    cone_rows = []
    if len(meta2):
        for _, m in meta2[meta2["kind"] == "cone_tension"].iterrows():
            v = _val(by_label2.get(m["label"]))
            fit = _val(by_tag.get(_tag("drill_cone", templates,
                                       combo=m["combo"], cone=int(m["cone"]))))
            ref = _val(by_tag.get(_tag("drill_ref", templates, combo=m["combo"])))
            fit = fit or {}
            ref = ref or {}
            row = {
                "combo": m["combo"], "cone": int(m["cone"]),
                "cone_label": m.get("cone_label", ""), "n_sne": m.get("n_sne"),
                "gaussian_nsigma": (v or {}).get("gaussian_nsigma"),
                "kde_nsigma": (v or {}).get("kde_nsigma"),
                "lnB": (v or {}).get("lnB"),
                "cone_logz": fit.get("logz"), "cone_logz_err": fit.get("logz_err"),
                "ref_logz": ref.get("logz"), "ref_logz_err": ref.get("logz_err"),
            }
            # Per-cone recovered cosmology (and every other shared parameter)
            # against the full-sample reference, with the SAME broad-uniform
            # priors. This is the actual question the cones check asks: does
            # the cosmology recovered from one patch of sky differ from the
            # cosmology recovered from all of it? A tension sigma alone does
            # not tell you which way, or by how much.
            for p in sorted({k[:-5] for k in fit if k.endswith("_mean")}
                            & {k[:-5] for k in ref if k.endswith("_mean")}):
                row[f"cone_{p}"] = fit.get(f"{p}_mean")
                row[f"cone_{p}_err"] = fit.get(f"{p}_std")
                row[f"ref_{p}"] = ref.get(f"{p}_mean")
                row[f"ref_{p}_err"] = ref.get(f"{p}_std")
                d_err = np.hypot(fit.get(f"{p}_std", np.nan),
                                 ref.get(f"{p}_std", np.nan))
                row[f"delta_{p}"] = fit.get(f"{p}_mean", np.nan) - ref.get(f"{p}_mean", np.nan)
                # ---- error on the difference: two ways, both reported -----
                # The cone's SNe are a SUBSET of the reference's, so the two
                # posteriors are positively correlated and the naive
                # quadrature sum sqrt(s_cone^2 + s_ref^2) is the WRONG error
                # on their difference. It is too large, so the sigma it gives
                # is too small and a real line-of-sight offset can hide in it.
                #
                # For nested samples where the subset's information is part
                # of the reference's, the standard result is
                #     Var(theta_sub - theta_ref) = s_sub^2 - s_ref^2
                # (Var of the difference between an estimator and a more
                # precise estimator that CONTAINS it -- the cross term
                # cancels the smaller variance rather than adding to it).
                # That is _nsigma below, and it is the one to quote.
                # _nsigma_quad is kept only because it is what a reader will
                # compute by hand from the four columns above and wonder why
                # it disagrees.
                #
                # abs() guards the case s_ref > s_cone, which is unphysical
                # under nesting and means one of the two chains has not
                # converged; it is flagged rather than silently NaN'd.
                s_c = fit.get(f"{p}_std", np.nan)
                s_r = ref.get(f"{p}_std", np.nan)
                var_d = (s_c ** 2 - s_r ** 2) if (np.isfinite(s_c)
                                                  and np.isfinite(s_r)) else np.nan
                row[f"delta_{p}_var_negative"] = bool(np.isfinite(var_d)
                                                      and var_d <= 0)
                sd = np.sqrt(abs(var_d)) if np.isfinite(var_d) else np.nan
                row[f"delta_{p}_err"] = sd
                row[f"delta_{p}_nsigma"] = (row[f"delta_{p}"] / sd
                                            if sd and np.isfinite(sd) else np.nan)
                row[f"delta_{p}_nsigma_quad"] = (row[f"delta_{p}"] / d_err
                                                 if d_err and np.isfinite(d_err) else np.nan)
            cone_rows.append(row)
    cones = pd.DataFrame(cone_rows)
    if len(cones):
        cones.to_csv(os.path.join(out_dir, "combo_drilling_cones.csv"), index=False)
        # Per-combo sky map, coloured by that combo's per-cone tension.
        if plan["labels"] is not None:
            id_of = {c["cone"]: c["dbscan_id"] for c in plan["cones"]}
            for ctag, grp in cones.groupby("combo"):
                tension = {id_of[int(r["cone"])]: r["gaussian_nsigma"]
                           for _, r in grp.iterrows() if int(r["cone"]) in id_of}
                try:
                    plot_cones(plan["df"], plan["labels"], tension,
                               plan["ra_col"], plan["dec_col"],
                               os.path.join(out_dir, f"cones_{ctag}"))
                except Exception as exc:      # plotting must never kill a run
                    print(f"[warn] cone sky map failed for {ctag}: {exc}")

    # ---- degeneracy ----
    deg_rows = []
    if len(meta2):
        for _, m in meta2[meta2["kind"] == "degeneracy"].iterrows():
            v = _val(by_label2.get(m["label"]))
            if v:
                deg_rows.append({"combo": m["combo"],
                                 "n_degeneracies_flagged": v["n_flagged"],
                                 "degeneracies": v["flagged"]})
    deg = pd.DataFrame(deg_rows)

    # ---- headline per-combo summary ----
    rows = []
    for term_names in combos:
        ctag = combo_tag(term_names)
        fit = _val(by_tag.get(_tag("fit", templates, combo=ctag)))
        row = {"combo": ctag, "terms": "|".join(term_names)}
        if fit:
            row.update({"logz": fit["logz"], "logz_err": fit["logz_err"],
                        "n_sne": fit["n_sne"], "pkl_path": fit["pkl_path"],
                        "active_params": fit["active_params"],
                        "ndim": len(fit["active_params"].split(","))})
        if len(deg):
            d = deg[deg["combo"] == ctag]
            if len(d):
                row["n_degeneracies_flagged"] = int(d.iloc[0]["n_degeneracies_flagged"])
                row["degeneracies"] = d.iloc[0]["degeneracies"]
        if len(hostq):
            h = hostq[hostq["combo"] == ctag]
            if len(h):
                row["host_quality_gaussian_nsigma"] = h.iloc[0]["gaussian_nsigma"]
                row["host_quality_lnB"] = h.iloc[0]["lnB"]
        if len(loo):
            l = loo[loo["combo"] == ctag]
            if len(l):
                row["n_loo_bins"] = len(l)
                row["n_loo_bins_flagged"] = int(l["flagged_2sigma"].sum())
                row["max_abs_loo_residual"] = float(l["mean_residual"].abs().max())
        if len(cones):
            c = cones[cones["combo"] == ctag]
            if len(c):
                ns = pd.to_numeric(c["gaussian_nsigma"], errors="coerce")
                row["n_cones_fitted"] = len(c)
                row["n_cones_flagged"] = int((ns >= 2.0).sum())
                row["max_cone_nsigma"] = float(ns.max()) if ns.notna().any() else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    if "logz" in summary.columns and summary["logz"].notna().any():
        best = summary["logz"].max()
        summary["dlnZ_vs_best"] = summary["logz"] - best
        summary = summary.sort_values("logz", ascending=False).reset_index(drop=True)
    summary_path = os.path.join(out_dir, "combo_ablation_summary.csv")
    summary.to_csv(summary_path, index=False)

    _print_summary(summary, out_dir)
    return {"summary": summary, "all_runs": evidence, "loo": loo,
            "host_quality": hostq, "cones": cones, "degeneracy": deg}


def _print_summary(summary, out_dir):
    print(f"\n{'=' * 100}")
    print("  COMBO ABLATION SUMMARY  (sorted by evidence)")
    print(f"{'=' * 100}")
    # (column, header, width, format spec). The spec is complete -- note the
    # sign flag must come BEFORE the width ("<+9.3f", not "<9+.3f", which is
    # not a valid format specifier).
    cols = [("combo", "combo", 44, "<44s"),
            ("ndim", "ndim", 5, "<5.0f"),
            ("logz", "lnZ", 12, "<12.3f"),
            ("dlnZ_vs_best", "dlnZ", 9, "<+9.3f"),
            ("n_degeneracies_flagged", "deg", 6, "<6.0f"),
            ("host_quality_gaussian_nsigma", "host_ns", 9, "<9.2f"),
            ("n_loo_bins_flagged", "loo!", 6, "<6.0f"),
            ("n_cones_flagged", "cones!", 7, "<7.0f")]
    print("  " + "  ".join(f"{h:<{w}}" for _c, h, w, _f in cols))
    print("  " + "  ".join("-" * w for _c, _h, w, _f in cols))
    for _, r in summary.iterrows():
        cells = []
        for name, _h, w, spec in cols:
            v = r.get(name)
            if v is None or (not isinstance(v, str) and pd.isna(v)):
                cells.append(f"{'-':<{w}}")
            elif spec.endswith("s"):
                cells.append(f"{str(v):{spec}}")
            else:
                cells.append(f"{float(v):{spec}}")
        print("  " + "  ".join(cells))
    print(f"{'=' * 100}")
    print(f"  deg     = correlated parameter pairs flagged by the degeneracy scan")
    print(f"  host_ns = all-hosts vs. strict-host-match posterior tension (sigma)")
    print(f"  loo!    = held-out z bins whose mean residual is > 2 sigma from zero")
    print(f"  cones!  = sky cones in >= 2 sigma tension with the full-sample fit")
    print(f"{'=' * 100}")
    print(f"  Outputs in {os.path.abspath(out_dir)}/:")
    for f in ("combo_ablation_summary.csv", "combo_all_runs.csv",
              "combo_loo_zbins.csv", "combo_host_quality.csv",
              "combo_drilling_cones.csv", "cone_plan.csv", "zbin_plan.csv",
              "job_plan.csv"):
        p = os.path.join(out_dir, f)
        if os.path.isfile(p):
            print(f"    {p}")
    print()


def _parse_args():
    p = argparse.ArgumentParser(
        description="Parallel robustness pass (fit, degeneracy scan, host-match "
                    "quality, LOO z-bins, labelled drilling cones) over every "
                    "competing model in best_model.COMBOS.")
    p.add_argument("--only", default=None,
                   help="Comma-separated combo tags (underscore-joined term "
                        "names) to run. Default: every entry in COMBOS.")
    p.add_argument("--list", action="store_true",
                   help="List the available combo tags and exit.")
    p.add_argument("--registry-file", default=REGISTRY_MAIN)
    p.add_argument("--out-dir", default=OUT_DIR,
                   help=f"Directory for summary CSVs/plots (default: {OUT_DIR}/).")
    p.add_argument("--section-dirs", action="store_true",
                   help="File each check into its own subdirectory "
                        "(<output_dir>/combo/host_qual/... etc.) instead of "
                        "prefixing the run name.")
    p.add_argument("--deep-only", default=None,
                   help="Comma-separated combo tags that should receive the "
                        "LOO z-bin folds (n_bins fits each). Drilling cones "
                        "are controlled separately by --cone-only. "
                        "Every combo still gets its main fit and its "
                        "host-quality pair, so the ladder is still fully "
                        "ranked. LOO + cones are ~n_bins + n_cones + 1 fits "
                        "per model and dominate the total, so restricting "
                        "them to the two or three models actually in "
                        "contention is the cheapest large saving available. "
                        "Suggested two-pass use: run --fits-only over "
                        "everything, read the ranking, then re-run with "
                        "--deep-only <winner>,<runner-up>,mass_linear.")
    p.add_argument("--cone-only", default=None,
                   help="Comma-separated combo tags to run drilling cones "
                        "for. Default: best_model.BASELINE_COMBO alone. A "
                        "cone measures line-of-sight structure in the DATA, "
                        "so the baseline number is the one that means "
                        "something; expand to the adopted model once the "
                        "ablation ladder has chosen one.")
    p.add_argument("--cones-all", action="store_true",
                   help="Run drilling cones for EVERY combo (n_cones + 1 "
                        "fits per model -- the most expensive thing in this "
                        "script). Only worth it if the baseline cone result "
                        "turns out to be model dependent.")
    p.add_argument("--skip-host-quality", action="store_true")
    p.add_argument("--skip-loo", action="store_true")
    p.add_argument("--skip-drilling-cones", action="store_true")
    p.add_argument("--fits-only", action="store_true",
                   help="Shorthand for all three --skip-* flags: run only each "
                        "combo's main fit. The cheapest meaningful smoke test.")
    p.add_argument("--loo-n-bins", type=int, default=4)
    p.add_argument("--degeneracy-threshold", type=float, default=0.85)
    p.add_argument("--cones-eps-deg", type=float, default=None)
    p.add_argument("--cones-min-samples", type=int, default=None)
    p.add_argument("--cones-min-fit-size", type=int, default=None)
    add_parallel_args(p, default_log_dir="logs/combo")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.list:
        print("Available combos (best_model.COMBOS):")
        for c in COMBOS:
            print(f"  {combo_tag(c):<52s}  terms: {', '.join(c)}")
        raise SystemExit(0)

    skip_all = args.fits_only
    run_combo_checks(
        only=args.only.split(",") if args.only else None,
        registry_file=args.registry_file,
        n_workers=args.workers, sequential=args.sequential,
        log_dir=args.log_dir, out_dir=args.out_dir, dry_run=args.dry_run,
        capture_output=not args.no_capture, section_dirs=args.section_dirs,
        degeneracy_threshold=args.degeneracy_threshold,
        loo_n_bins=args.loo_n_bins,
        run_host_quality=not (args.skip_host_quality or skip_all),
        run_loo=not (args.skip_loo or skip_all),
        run_drilling_cones=not (args.skip_drilling_cones or skip_all),
        deep_only=([t.strip() for t in args.deep_only.split(",") if t.strip()]
                   if args.deep_only else None),
        cone_only=([t.strip() for t in args.cone_only.split(",") if t.strip()]
                   if args.cone_only else None),
        cones_all=args.cones_all,
        cones_eps_deg=args.cones_eps_deg,
        cones_min_samples=args.cones_min_samples,
        cones_min_fit_size=args.cones_min_fit_size)
