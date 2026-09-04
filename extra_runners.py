"""
extra_runners.py  —  SNe Ia Cosmology Pipeline
==============================================
Post-hoc systematic checks on the COMPETING MODELS, run in parallel.

Every model that survives the main search (best_model.COMBOS) is refit under a
fixed list of check categories. Because the same set of models is pushed
through every category, each category answers the question "does this
systematic change WHICH model wins, or only what that model recovers?" -- which
is a stronger statement than refitting a single chosen model would give.

THE CATEGORIES
--------------
  a  std       reference fit -- CONFIG's default flat LambdaCDM on the full
                sample. Every other category's delta is measured against the
                SAME model's std entry, so a delta isolates the systematic and
                not the model. Also carries the host-measurement-error
                matched-pair family (hosterr/*) for the best model.
  b  lcdm      non-flat LambdaCDM: Ode0 sampled (curvature free), w = -1.
  c  wcdm      flat wCDM: w sampled, Ode0 = 1 - Om0.
  d  zlow      z < 0.1 only.
  e  zhigh     z >= 0.1 only.
  f  survey    survey filter on: keep only DES (IDSURVEY 10) and Foundation
                (150), dropping the other low-z surveys.
  g  masslow   host log(M*) < 10 only.
  h  masshigh  host log(M*) >= 10 only.
  i  specz     spectroscopic host redshifts only.
  j  photz     photometric host redshifts only.
  k  x1cut     tightened stretch cut, x1 in [-2, 2].
  l  ccut      tightened colour cut, c in [-0.2, 0.2].

Categories b and c are cosmology variants: std is already FLAT LambdaCDM (the
CONFIG default), so "lcdm" here means the non-flat extension. Making std and
lcdm the same fit would be caught by the registry's duplicate-fingerprint guard
rather than silently doubling the compute.

Categories d-l all keep the default flat LambdaCDM cosmology, so their delta
against std is purely the effect of the sample cut.

SAMPLE SIZES (DES 5yr, 1820 SNe after the standard cuts)
--------------------------------------------------------
    zlow     201     zhigh   1619     survey  1740
    masslow  594     masshigh 1226    x1cut   1670     ccut    1668
    specz / photz depend on the host-redshift flags in the metadata.

zlow is the one to read carefully. Those 201 SNe are almost entirely the
external low-z samples, and at z < 0.1 the deceleration lever arm is tiny, so
Om0 there is essentially set by its prior rather than by the data. Read the
zlow row as a check on the STANDARDISATION parameters (alpha/beta/gamma and the
host terms), not as an independent cosmology measurement.

COMPARISON WITH THE OTHER DRIVERS
---------------------------------
Host-match quality, leave-one-redshift-bin-out and drilling cones used to be
reachable from here through --host-quality-check / --loo-zbins /
--drilling-cones. They now live in combo_ablation_checks.py, which runs them
for every competing model in parallel with proper per-check naming. Running
them from both places would have two processes appending to the same registry
files. Use:

    python combo_ablation_checks.py --workers 16

NAMING
------
    checks/<category>_<combo>        e.g. checks/wcdm_mass_linear_ssfr_tanh
    hosterr/<best_combo>_<variant>   e.g. hosterr/mass_linear_ssfr_tanh_varpen

matching experiment_runner.py's "<category>/<description>" convention, with
"checks/" and "hosterr/" reserved for this file (see
experiment_naming.CATEGORY_PREFIXES).

USAGE
-----
    python extra_runners.py --list                     # every planned run
    python extra_runners.py --dry-run --workers 16     # plan, then exit
    python extra_runners.py --workers 16               # everything
    python extra_runners.py --categories std,wcdm,lcdm --workers 12
    python extra_runners.py --categories zlow,zhigh --only mass_linear_ssfr_tanh
    python extra_runners.py --no-hosterr --workers 16
    python extra_runners.py --publication --workers 32 # nlive = ndim x 500
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
from dynesty import utils as dyfunc

from config import CONFIG, DEFAULT_PARAM_SPECS
from run import run_sampler, pkl_path_for
from experiment_naming import ExperimentRegistry
from best_model import (COMBOS, BEST_COMBO, merge_terms as _merge_terms,
                        combo_tag, best_model_overrides)
from parallel_runner import Job, run_jobs, add_parallel_args


# ===========================================================================
# 1.  NAMING / DEFAULTS
# ===========================================================================
TAG_TEMPLATES = {
    "check":   "checks/{category}_{combo}",
    "hosterr": "hosterr/{combo}_{variant}",
}

OUT_DIR = "extra_runners"
REGISTRY = "run_checks_registry.csv"

# The reference category every other category's delta is measured against.
REFERENCE_CATEGORY = "std"


def _tag(kind, **kw):
    return TAG_TEMPLATES[kind].format(**kw)


# ===========================================================================
# 2.  CHECK CATEGORIES
# ===========================================================================
# key -> (letter, title, config_overrides, param_overrides, note)
#
# config_overrides are top-level CONFIG fields (data cuts). param_overrides are
# activations layered ON TOP of whatever the combo's own terms set, so a
# category can free w or Ode0 without any combo needing to know about it.
#
# Every one of these is applied to EVERY combo in best_model.COMBOS, so adding
# a category here adds len(COMBOS) runs. Keep that in mind before adding one.

CATEGORIES = {
    "std": {
        "letter": "a",
        "title": "Reference fit (flat LambdaCDM, full sample)",
        "config": {},
        "params": {},
        "note": "The baseline for every other category. Also the category the "
                "hosterr/* matched pairs attach to.",
    },
    "lcdm": {
        "letter": "b",
        "title": "Non-flat LambdaCDM (Ode0 free)",
        "config": {},
        # Ode0 active makes the cosmology non-flat. std is ALREADY flat
        # LambdaCDM (CONFIG's default), so activating Ode0 is what makes this
        # category a distinct fit rather than a duplicate of std.
        "params": {"Ode0": {"active": True, "fixed": None},
                   "w": {"active": False, "fixed": -1.0}},
        "note": "Curvature free. Compare Om0 against std to see how much of "
                "the Om0 constraint was flatness rather than data.",
    },
    "wcdm": {
        "letter": "c",
        "title": "Flat wCDM (w free)",
        "config": {},
        "params": {"w": {"active": True, "fixed": None},
                   "Ode0": {"active": False, "fixed": None}},
        "note": "Dark-energy equation of state free. A standardisation model "
                "that only wins under w = -1 is not a robust winner.",
    },
    "zlow": {
        "letter": "d",
        "title": "Low redshift only (z < 0.1)",
        # zhi is an upper bound (run.py keeps z <= zhi), so z < 0.1 is zhi=0.1.
        # The OLD version of this file had this inverted -- its "zlow_" entries
        # set zlo=0.1, which keeps the HIGH-z half.
        "config": {"zhi": 0.1},
        "params": {},
        "note": "~201 SNe, almost all external low-z. Om0 is prior-dominated "
                "here; read this row for alpha/beta/gamma, not cosmology.",
    },
    "zhigh": {
        "letter": "e",
        "title": "High redshift only (z >= 0.1)",
        "config": {"zlo": 0.1},
        "params": {},
        "note": "~1619 SNe, essentially the DES sample on its own.",
    },
    "survey": {
        "letter": "f",
        "title": "Survey filter (DES + Foundation only)",
        "config": {"idsurvey": True},
        "params": {},
        "note": "Drops the ~80 SNe from the other low-z surveys (IDSURVEY "
                "63/64/65/66/5), which carry the most heterogeneous "
                "photometric calibration.",
    },
    "masslow": {
        "letter": "g",
        "title": "Low-mass hosts only (log M* < 10)",
        "config": {"mass_cut": "low"},
        "params": {},
        "note": "Splitting on host mass and refitting the host terms is "
                "partly circular by construction -- a mass step fitted within "
                "one side of the split has little left to constrain it. The "
                "informative comparison is alpha/beta between g and h.",
    },
    "masshigh": {
        "letter": "h",
        "title": "High-mass hosts only (log M* >= 10)",
        "config": {"mass_cut": "high"},
        "params": {},
        "note": "See masslow.",
    },
    "specz": {
        "letter": "i",
        "title": "Spectroscopic host redshifts only",
        "config": {"obs_z_type": "spec"},
        "params": {},
        "note": "Removes photo-z redshift error as a systematic entirely.",
    },
    "photz": {
        "letter": "j",
        "title": "Photometric host redshifts only",
        "config": {"obs_z_type": "phot"},
        "params": {},
        "note": "The complement of specz. Compare the pair rather than "
                "reading either alone.",
    },
    "x1cut": {
        "letter": "k",
        "title": "Tight stretch cut (x1 in [-2, 2])",
        "config": {"x1_range": [-2, 2]},
        "params": {},
        "note": "Trims the stretch tails, which is where any non-linear x1 "
                "correction has most of its leverage.",
    },
    "ccut": {
        "letter": "l",
        "title": "Tight colour cut (c in [-0.2, 0.2])",
        "config": {"c_range": [-0.2, 0.2]},
        "params": {},
        "note": "Trims the colour tails, which is where the softbroken / "
                "broken SN-colour models differ most from linear.",
    },
}

CATEGORY_ORDER = list(CATEGORIES)


# ===========================================================================
# 3.  HOST MEASUREMENT ERROR MATCHED PAIRS  (part of category "std")
# ===========================================================================
# Every entry below fits the SAME model on the SAME SNe and changes only how
# the host mass / colour / sSFR MEASUREMENT ERRORS are treated, so the lnZ
# differences are attributable to that choice alone.
#
#   host_colour_err_from_logmass
#       HOST_COLOR_ERR is -999 for every SN in the DES metadata. Left alone,
#       host colour would be the only host property treated as exactly
#       measured while mass and sSFR are smoothed by their errors -- an
#       asymmetry that flatters the host-colour models. With this on, the
#       colour error is derived from HOST_LOGMASS_ERR / slope via the
#       Taylor+2011 mass-to-light/colour relation.
#
#   ssfr_err_max
#       HOST_LOGsSFR_ERR is bimodal with a failure-mode pileup near 10 dex,
#       larger than the entire ~2.4 dex population spread. Above this
#       threshold the sSFR point estimate is masked to NaN. The SN is KEPT so
#       evidences stay comparable across variants.
#
#   host_var_penalty
#       Adds Var[f] to the covariance diagonal: the extra SCATTER the host
#       measurement error injects into mu, not just the bias correction the
#       quadrature already applies. Expensive -- the covariance becomes
#       parameter-dependent and must be refactorised on every likelihood call.
#
# HOSTERR_BEST is DERIVED from best_model.BEST_COMBO, so these pairs stay
# like-for-like with the rest of the pipeline automatically. Edit
# best_model.py, never this block.
_best_model_overrides, _best_param_overrides = best_model_overrides()
HOSTERR_BEST = {
    "label": combo_tag(BEST_COMBO),
    "model": _best_model_overrides,
    "param_overrides": _best_param_overrides,
}

# The reference treatment is whatever config.py defaults to:
# host_colour_err_from_logmass=True, ssfr_err_max=2.5, host_var_penalty=False.
#
# There is deliberately NO ("ref", {}) entry here. The reference treatment on
# the best model IS checks/std_<best_combo>, which category "a" already fits;
# adding it again under a hosterr/ name would be the same fit run twice (the
# registry's duplicate-fingerprint guard raises on exactly this). Compare every
# variant below against checks/std_<best_combo> -- HOSTERR_REFERENCE_TAG.
HOSTERR_VARIANTS = [
    ("varpen",             {"host_var_penalty": True}),
    ("nocolourerr",        {"host_colour_err_from_logmass": False}),
    ("nocolourerr_varpen", {"host_colour_err_from_logmass": False,
                            "host_var_penalty": True}),
    # Literature slope spans ~0.5-1.15; a smaller slope means a LARGER derived
    # colour error, so 0.50 is the pessimistic end.
    ("slope050",           {"host_colour_err_mass_slope": 0.50}),
    ("slope115",           {"host_colour_err_mass_slope": 1.15}),
    ("ssfrmask20",         {"ssfr_err_max": 2.0}),
    ("ssfrmask30",         {"ssfr_err_max": 3.0}),
    ("nossfrmask",         {"ssfr_err_max": None}),
    # All host measurement error off: point estimates only.
    ("noerrors",           {"col_logM_err": None,
                            "col_host_colour_err": None,
                            "col_logsSFR_err": None,
                            "host_colour_err_from_logmass": False,
                            "ssfr_err_max": None}),
    # Quadrature convergence. Discontinuous ("step") profiles converge slowly
    # under Gauss-Hermite, the second moment more slowly than the first. If the
    # best model uses a step, compare gh80 against ref before trusting any
    # host-error delta at the 0.1 lnZ level.
    ("gh80",               {"n_gh_nodes": 80}),
    ("gh80_varpen",        {"n_gh_nodes": 80, "host_var_penalty": True}),
]

# The run every hosterr/* variant is measured against. Same model, same SNe,
# reference error treatment -- it is category "a"'s entry for the best model.
HOSTERR_REFERENCE_TAG = _tag("check", category=REFERENCE_CATEGORY,
                             combo=HOSTERR_BEST["label"])


# ===========================================================================
# 4.  WORKER ENTRY POINT
# ===========================================================================
# Module-level: parallel_runner spawns workers and pickles the callable by
# qualified name.

def job_fit(cfg):
    """Run one nested-sampling fit and return a small picklable summary."""
    results, _, active_names, data, run_name = run_sampler(cfg)

    w = np.exp(results.logwt - results.logz[-1])
    w /= w.sum()
    mean, cov = dyfunc.mean_and_cov(results.samples, w)
    std = np.sqrt(np.diag(cov))

    # n_sne is recorded on every run because most of these categories cut the
    # sample: without it a reader cannot tell whether a category's lnZ moved
    # because the model fits worse or because it is fitting 200 SNe instead of
    # 1820.
    out = {"run_tag": cfg.get("run_tag"), "run_name": run_name,
           "pkl_path": pkl_path_for(run_name, cfg),
           "n_params": len(active_names),
           "active_params": ",".join(active_names),
           "logz": float(results.logz[-1]),
           "logz_err": float(results.logzerr[-1]),
           "n_sne": int(len(data["z"]))}
    # Posterior summaries: the check tables report how each systematic moves
    # the recovered parameters, not just the evidence.
    for name, m, s in zip(active_names, mean, std):
        out[f"{name}_mean"] = float(m)
        out[f"{name}_std"] = float(s)
    return out


# ===========================================================================
# 5.  EXPERIMENT CONSTRUCTION
# ===========================================================================

def build_experiments(categories=None, combos=None, registry_file=REGISTRY,
                      include_hosterr=True, nlive_mode=None):
    """Build every planned run as a list of dicts.

    Each entry: {category, letter, combo, variant, tag, cfg}.

    All configs are built through one shared ExperimentRegistry, so a category
    that accidentally reproduces another category's exact fit (e.g. an "lcdm"
    that forgot to free Ode0 and is therefore identical to "std") raises here,
    before a hundred expensive fits, rather than silently doubling the compute.
    """
    categories = list(categories) if categories else list(CATEGORY_ORDER)
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown category/categories {unknown}. "
                         f"Available: {CATEGORY_ORDER}")
    combos = list(combos) if combos is not None else list(COMBOS)

    registry = ExperimentRegistry(CONFIG, DEFAULT_PARAM_SPECS)
    plan = []

    for key in categories:
        cat = CATEGORIES[key]
        for term_names in combos:
            ctag = combo_tag(term_names)
            model_over, param_over = _merge_terms(term_names)
            # Category params layer ON TOP of the combo's own term params, so
            # a category can free w/Ode0 without any combo knowing about it.
            params = {**param_over, **cat["params"]}
            cfg_over = {"model": {**CONFIG["model"], **model_over},
                        "registry_file": registry_file,
                        **cat["config"]}
            tag = _tag("check", category=key, combo=ctag)
            cfg = registry.build(tag, param_overrides=params,
                                 config_overrides=cfg_over)
            if nlive_mode:
                cfg["nlive_mode"] = nlive_mode
            plan.append({"category": key, "letter": cat["letter"],
                         "combo": ctag, "variant": "", "tag": tag, "cfg": cfg})

    if include_hosterr and REFERENCE_CATEGORY in categories:
        label = HOSTERR_BEST["label"]
        for variant, overrides in HOSTERR_VARIANTS:
            cfg_over = {"model": {**CONFIG["model"], **HOSTERR_BEST["model"]},
                        "registry_file": registry_file, **overrides}
            tag = _tag("hosterr", combo=label, variant=variant)
            cfg = registry.build(
                tag, param_overrides=HOSTERR_BEST["param_overrides"],
                config_overrides=cfg_over)
            if nlive_mode:
                cfg["nlive_mode"] = nlive_mode
            plan.append({"category": "hosterr", "letter": "a",
                         "combo": label, "variant": variant, "tag": tag,
                         "cfg": cfg})

    # Keep new entries honest about the "<category>/" convention without
    # renaming the historical tags this registry never sees.
    registry.validate_category_prefixes()
    return plan


def _nlive_display(cfg, cli_mode=None):
    """The nlive that will actually be used, for the plan listing."""
    if cfg.get("nlive"):
        return int(cfg["nlive"])
    mode = cli_mode or cfg.get("nlive_mode", "exploratory")
    n = sum(1 for s in cfg["param_specs"].values() if s["active"])
    return n * 500 if mode == "publication" else n * 50


# ===========================================================================
# 6.  DRIVER
# ===========================================================================

def run_extra_checks(categories=None, combos=None, only=None,
                     include_hosterr=True, nlive_mode=None,
                     registry_file=REGISTRY, n_workers=None, sequential=False,
                     log_dir="logs/checks", out_dir=OUT_DIR, dry_run=False,
                     capture_output=True):
    """Run the selected check categories over the selected models, in parallel.

    Returns a dict of DataFrames: evidence, deltas.
    """
    if only:
        want = set(only)
        combos = [c for c in (combos if combos is not None else COMBOS)
                  if combo_tag(c) in want]
        if not combos:
            raise SystemExit(
                f"No combo matched --only {sorted(want)}.\nAvailable: "
                + ", ".join(combo_tag(c) for c in COMBOS))

    plan = build_experiments(categories, combos, registry_file,
                             include_hosterr, nlive_mode)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{k: v for k, v in p.items() if k != "cfg"} for p in plan]) \
        .to_csv(os.path.join(out_dir, "extra_runners_plan.csv"), index=False)

    jobs = []
    for p in plan:
        n_active = sum(1 for s in p["cfg"]["param_specs"].values() if s["active"])
        # host_var_penalty refactorises the covariance on every likelihood call,
        # so those runs are far slower than their parameter count suggests.
        cost = n_active * (6.0 if p["cfg"].get("host_var_penalty") else 1.0)
        jobs.append(Job(label=p["tag"], func=job_fit, kwargs={"cfg": p["cfg"]},
                        group=p["category"], cost=cost))

    cat_lines = []
    for key in (categories or CATEGORY_ORDER):
        if key in CATEGORIES:
            cat_lines.append(f"    {CATEGORIES[key]['letter']}) "
                             f"{key:<9s} {CATEGORIES[key]['title']}")
    if include_hosterr and REFERENCE_CATEGORY in (categories or CATEGORY_ORDER):
        cat_lines.append(f"    a) hosterr   Host measurement-error matched "
                         f"pairs ({HOSTERR_BEST['label']})")

    results = run_jobs(
        jobs, n_workers=n_workers, log_dir=log_dir, sequential=sequential,
        capture_output=capture_output, dry_run=dry_run,
        title="Extra systematic checks",
        summary_name="extra_runners_summary.log",
        extra_lines=[f"Models        : {len(combos) if combos is not None else len(COMBOS)}",
                     f"nlive mode    : {nlive_mode or 'per-experiment'}",
                     "Categories    :"] + cat_lines)
    if dry_run:
        return {"plan": pd.DataFrame(
            [{k: v for k, v in p.items() if k != "cfg"} for p in plan])}

    by_tag = {r.label: r.value for r in results
              if r.status == "ok" and r.value is not None}
    return _assemble(plan, by_tag, out_dir)


def _assemble(plan, by_tag, out_dir):
    rows = []
    for p in plan:
        v = by_tag.get(p["tag"])
        row = {"category": p["category"], "letter": p["letter"],
               "combo": p["combo"], "variant": p["variant"], "tag": p["tag"],
               "title": CATEGORIES.get(p["category"], {}).get(
                   "title", "Host measurement-error variant"),
               "status": "ok" if v else "failed"}
        if v:
            row.update({k: val for k, val in v.items() if k != "run_tag"})
        rows.append(row)
    evidence = pd.DataFrame(rows)
    evidence.to_csv(os.path.join(out_dir, "extra_runners_evidence.csv"),
                    index=False)

    # ---- Deltas against each model's own std reference ----
    # Measured WITHIN a model, never across models: two categories fit
    # different SNe, so their absolute lnZ values are not comparable, but the
    # SHIFT a systematic induces is comparable between models.
    deltas = []
    ok = evidence[evidence["status"] == "ok"]
    ref = ok[ok["category"] == REFERENCE_CATEGORY].set_index("combo")
    param_cols = [c for c in ok.columns if c.endswith("_mean")]
    for _, r in ok.iterrows():
        if r["category"] in (REFERENCE_CATEGORY, "hosterr"):
            continue
        if r["combo"] not in ref.index:
            continue
        b = ref.loc[r["combo"]]
        d = {"category": r["category"], "letter": r["letter"],
             "combo": r["combo"], "tag": r["tag"],
             "ref_tag": b["tag"],
             "logz": r["logz"], "ref_logz": b["logz"],
             # Recorded but NOT to be read as model comparison: a subsample
             # category fits fewer SNe, so its lnZ is smaller for reasons that
             # have nothing to do with the model. Only the parameter shifts
             # below are interpretable across a sample cut.
             "delta_logz_NOT_COMPARABLE_ACROSS_CUTS": r["logz"] - b["logz"]}
        # Is this category a strict SUBSAMPLE of the reference, or the same
        # SNe fitted under a different model? It decides which error bar on
        # the parameter shift is correct, so it is recorded explicitly.
        n_r, n_b = r.get("n_sne"), b.get("n_sne")
        nested = bool(pd.notna(n_r) and pd.notna(n_b) and float(n_r) < float(n_b))
        d["n_sne"] = n_r
        d["ref_n_sne"] = n_b
        d["nested_in_ref"] = nested
        for pc in param_cols:
            base = pc[:-5]
            sc = f"{base}_std"
            if pd.isna(r.get(pc)) or pd.isna(b.get(pc)):
                continue
            shift = r[pc] - b[pc]
            s_r = float(r.get(sc, np.nan))
            s_b = float(b.get(sc, np.nan))
            # A subsample fit and the full-sample fit SHARE SNe, so their
            # errors are positively correlated and the quadrature sum
            # sqrt(s_sub^2 + s_full^2) is not the error on their difference.
            # It is too big, which makes the tension look smaller than it is.
            # For a nested subsample the correct variance is
            #     Var(theta_sub - theta_full) = s_sub^2 - s_full^2 ,
            # which is what nsigma uses when nested_in_ref is True.
            #
            # For same-sample categories (lcdm, wcdm -- identical SNe, only
            # the cosmology model differs) neither expression is right and
            # there is no closed form, because the correlation depends on how
            # the extra parameter projects onto this one. Quadrature is used
            # there and is CONSERVATIVE, so a shift that is significant under
            # it is genuinely significant; a shift that is not may still be.
            if nested and np.isfinite(s_r) and np.isfinite(s_b):
                var_d = s_r ** 2 - s_b ** 2
                denom = np.sqrt(abs(var_d)) if np.isfinite(var_d) else np.nan
                d[f"delta_{base}_var_negative"] = bool(var_d <= 0)
            else:
                denom = np.sqrt(s_r ** 2 + s_b ** 2)
            d[f"delta_{base}"] = shift
            d[f"delta_{base}_err"] = denom
            d[f"delta_{base}_nsigma"] = (shift / denom
                                         if denom and np.isfinite(denom) and denom > 0
                                         else np.nan)
            d[f"delta_{base}_nsigma_quad"] = (
                shift / np.sqrt(s_r ** 2 + s_b ** 2)
                if np.isfinite(s_r) and np.isfinite(s_b)
                and (s_r or s_b) else np.nan)
        deltas.append(d)
    deltas = pd.DataFrame(deltas)
    if len(deltas):
        deltas.to_csv(os.path.join(out_dir, "extra_runners_deltas.csv"),
                      index=False)

    # ---- Host measurement-error matched pairs ----
    # Unlike the subsample categories, these DO fit the same SNe with the same
    # model and differ only in the error treatment, so here delta lnZ IS a
    # genuine Bayes factor for that treatment choice.
    hosterr = pd.DataFrame()
    he = ok[ok["category"] == "hosterr"]
    ref_row = ok[ok["tag"] == HOSTERR_REFERENCE_TAG]
    if len(he) and len(ref_row):
        r0 = ref_row.iloc[0]
        rows = [{"variant": r["variant"], "tag": r["tag"],
                 "n_sne": r.get("n_sne"), "logz": r["logz"],
                 "logz_err": r["logz_err"],
                 "ref_tag": HOSTERR_REFERENCE_TAG,
                 "ref_logz": r0["logz"],
                 "delta_logz_vs_ref": r["logz"] - r0["logz"]}
                for _, r in he.iterrows()]
        hosterr = pd.DataFrame(rows).sort_values("delta_logz_vs_ref",
                                                 ascending=False)
        hosterr.to_csv(os.path.join(out_dir, "extra_runners_hosterr.csv"),
                       index=False)
    elif len(he):
        print(f"[warn] host-error variants ran but their reference "
              f"({HOSTERR_REFERENCE_TAG}) did not succeed -- no delta lnZ "
              f"can be reported for them.")

    # ---- Within-category model ranking ----
    # THIS is the comparable one: within a single category every model is fit
    # on exactly the same SNe, so lnZ differences are genuine Bayes factors.
    ranks = []
    for cat, grp in ok[ok["category"] != "hosterr"].groupby("category"):
        best = grp["logz"].max()
        for _, r in grp.sort_values("logz", ascending=False).iterrows():
            ranks.append({"category": cat, "letter": r["letter"],
                          "combo": r["combo"], "n_sne": r.get("n_sne"),
                          "logz": r["logz"], "logz_err": r["logz_err"],
                          "delta_logz_vs_best_in_category": r["logz"] - best})
    ranking = pd.DataFrame(ranks)
    if len(ranking):
        ranking.to_csv(os.path.join(out_dir, "extra_runners_ranking.csv"),
                       index=False)

    _print_summary(evidence, ranking, hosterr, out_dir)
    return {"evidence": evidence, "deltas": deltas, "ranking": ranking,
            "hosterr": hosterr}


def _print_summary(evidence, ranking, hosterr, out_dir):
    n_ok = int((evidence["status"] == "ok").sum())
    print("\n" + "=" * 96)
    print(f"  EXTRA SYSTEMATIC CHECKS  --  {n_ok}/{len(evidence)} fits succeeded")
    print("=" * 96)

    if len(ranking):
        print("  Model ranking WITHIN each category (same SNe, so these lnZ "
              "differences are real\n  Bayes factors; lnZ values are NOT "
              "comparable BETWEEN categories that cut the sample\n  "
              "differently):\n")
        for cat in ranking["category"].drop_duplicates():
            grp = ranking[ranking["category"] == cat]
            title = CATEGORIES.get(cat, {}).get("title", cat)
            letter = grp["letter"].iloc[0]
            print(f"  [{letter}] {cat} -- {title}")
            for _, r in grp.iterrows():
                mark = "  <-- best" if abs(
                    r["delta_logz_vs_best_in_category"]) < 1e-9 else ""
                print(f"       {r['combo']:<46s} lnZ = {r['logz']:>10.3f} "
                      f"  dlnZ = {r['delta_logz_vs_best_in_category']:>+8.3f}"
                      f"{mark}")
            print()

    if len(hosterr):
        print("  Host measurement-error treatment (same model, same SNe, so "
              "dlnZ IS a Bayes factor):\n")
        for _, r in hosterr.iterrows():
            print(f"       {r['variant']:<22s} lnZ = {r['logz']:>10.3f} "
                  f"  dlnZ vs reference = {r['delta_logz_vs_ref']:>+8.3f}")
        print()

    failed = evidence[evidence["status"] != "ok"]
    if len(failed):
        print(f"  {len(failed)} fit(s) FAILED -- see the per-job logs:")
        for t in failed["tag"]:
            print(f"     {t}")
        print()

    print(f"  Outputs in {os.path.abspath(out_dir)}/")
    print("    extra_runners_plan.csv      every run that was planned")
    print("    extra_runners_evidence.csv  lnZ + posterior summaries per run")
    print("    extra_runners_deltas.csv    parameter shifts vs each model's std")
    print("    extra_runners_ranking.csv   model ranking within each category")
    print("    extra_runners_hosterr.csv   host-error variants vs their reference")
    print("=" * 96 + "\n")


# ===========================================================================
# 7.  CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Post-hoc systematic checks on every competing model, in "
                    "parallel. Categories a-l are listed in the module "
                    "docstring and in --list-categories.")
    p.add_argument("--categories", default=None,
                   help="Comma-separated category keys "
                        f"(default: all of {','.join(CATEGORY_ORDER)}).")
    p.add_argument("--only", default=None,
                   help="Comma-separated combo tags "
                        "(default: all of best_model.COMBOS).")
    p.add_argument("--no-hosterr", action="store_true",
                   help="Skip the host measurement-error matched pairs.")
    p.add_argument("--list", action="store_true",
                   help="List every planned run with its parameter count and "
                        "nlive, then exit.")
    p.add_argument("--list-categories", action="store_true",
                   help="List the check categories and exit.")
    p.add_argument("--registry-file", default=REGISTRY)
    p.add_argument("--out-dir", default=OUT_DIR)
    nl = p.add_mutually_exclusive_group()
    nl.add_argument("--publication", action="store_true",
                    help="Force nlive_mode='publication' (ndim x 500).")
    nl.add_argument("--explore", action="store_true",
                    help="Force nlive_mode='exploratory' (ndim x 50).")
    add_parallel_args(p, default_log_dir="logs/checks")
    return p.parse_args()


# The __main__ guard is REQUIRED, not stylistic: parallel_runner spawns
# workers and spawn re-imports this module in every child.
if __name__ == "__main__":
    args = _parse_args()

    if args.list_categories:
        print(f"{'':2s}{'key':<10s} {'title'}")
        for key in CATEGORY_ORDER:
            c = CATEGORIES[key]
            print(f"{c['letter']}) {key:<10s} {c['title']}")
            print(f"   {'':<10s} {c['note']}")
        print(f"a) {'hosterr':<10s} Host measurement-error matched pairs "
              f"({HOSTERR_BEST['label']}, {len(HOSTERR_VARIANTS)} variants)")
        raise SystemExit(0)

    _mode = ("publication" if args.publication
             else "exploratory" if args.explore else None)
    _cats = [s.strip() for s in args.categories.split(",")] if args.categories else None
    _only = [s.strip() for s in args.only.split(",")] if args.only else None

    if args.list:
        _combos = ([c for c in COMBOS if combo_tag(c) in set(_only)]
                   if _only else None)
        _plan = build_experiments(_cats, _combos, args.registry_file,
                                  not args.no_hosterr, _mode)
        print(f"{'idx':>4}  {'tag':<62} {'cat':<9} params  nlive")
        print(f"{'---':>4}  {'---':<62} {'---':<9} ------  -----")
        for i, p in enumerate(_plan):
            n = sum(1 for s in p["cfg"]["param_specs"].values() if s["active"])
            print(f"{i:>4}  {p['tag']:<62} {p['category']:<9} {n:>6}  "
                  f"{_nlive_display(p['cfg'], _mode)}")
        print(f"\n{len(_plan)} run(s) planned.")
        raise SystemExit(0)

    run_extra_checks(
        categories=_cats,
        only=_only,
        include_hosterr=not args.no_hosterr,
        nlive_mode=_mode,
        registry_file=args.registry_file,
        n_workers=args.workers,
        sequential=args.sequential,
        log_dir=args.log_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        capture_output=not args.no_capture,
    )
