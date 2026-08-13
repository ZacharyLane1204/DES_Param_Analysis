"""
experiment_runner.py  —  SNe Ia Cosmology Pipeline
==============================================
Define every run variant here as a small dict of overrides on top of the
base CONFIG / DEFAULT_PARAM_SPECS from config.py.  Then run all of them
(or a named subset) without ever touching config.py.

Usage
-----
  # Run everything (sequentially)
  python experiment_runner.py

  # Run only experiments whose tag matches a pattern
  python experiment_runner.py --tag flat_lcdm
  python experiment_runner.py --tag nuisance

  # Dry-run: print what would be run without sampling
  python experiment_runner.py --dry-run

  # Run a single experiment by index (0-based)
  python experiment_runner.py --index 2

  # Run a range of indices (useful for splitting across server jobs)
  python experiment_runner.py --index 0-9
  python experiment_runner.py --index 10-19
  
  nice -n 19 python experiment_runner.py --workers 80 --index 0-79 && nice -n 19 python experiment_runner.py --workers 80 --index 80-159 && 
  nice -n 19 python experiment_runner.py --workers 80 --index 160-239 && nice -n 19 python experiment_runner.py --workers 80 --index 240-266
"""

import copy
import argparse
import sys
import os
from datetime import datetime

# ===========================================================================
# THREAD CLAMPING  —  must happen BEFORE any numerical library is imported
# ===========================================================================
# NumPy / OpenBLAS / MKL / OMP read their thread-count env vars at import
# time, not at call time.  Setting them here — at the top of the main module,
# before the `from config import …` line triggers numpy — is the only
# reliable way to ensure the *parent* process itself is single-threaded.
#
# Why this matters for multiprocessing:
#   ProcessPoolExecutor spawns worker processes by forking (Linux default)
#   or spawning.  With fork, the child inherits the parent's already-
#   initialised OpenBLAS thread pool.  If the parent has N threads, each
#   child also gets N threads → N_workers × N CPU cores consumed.
#   Setting the vars here clamps the parent pool to 1, so every forked
#   child also starts with 1 thread.  The redundant os.environ assignment
#   in _run_one() is kept as a belt-and-braces guard for spawn-mode.
#
# With this in place each worker process uses exactly 1 CPU thread,
# so you can safely run --workers K and consume exactly K cores total.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_var] = "1"

# Optional: threadpoolctl provides a runtime limit that survives dlopen
# of new BLAS libraries loaded after the env vars are read.  Import it
# now (while no threads are active yet) so the limit is set process-wide.
try:
    from threadpoolctl import threadpool_limits as _tpl
    _tpl(1)
except Exception:
    pass  # threadpoolctl not installed or broken — env vars above are sufficient
# ===========================================================================

from config import CONFIG, DEFAULT_PARAM_SPECS
from run import run_sampler

def _M(ssfr="none", mass=None, host_colour=None, sn_colour=None):
    """Return a model dict with the given overrides on top of CONFIG['model']."""
    m = dict(CONFIG["model"])
    m["ssfr"] = ssfr
    if mass        is not None: m["mass"]        = mass
    if host_colour is not None: m["host_colour"] = host_colour
    if sn_colour   is not None: m["sn_colour"]   = sn_colour
    return m

# ===========================================================================
# HELPERS
# ===========================================================================

def _override(base_specs, **param_overrides):
    """
    Return a deep copy of base_specs with per-parameter overrides applied.

    Each key in param_overrides is a parameter name; the value is a dict of
    fields to update, e.g.:

        _override(base, Om0={"active": False}, w={"active": True})

    Only the listed fields are changed — all other fields for that parameter
    are inherited from base_specs unchanged.
    """
    specs = copy.deepcopy(base_specs)
    for name, updates in param_overrides.items():
        specs[name].update(updates)
    return specs


def _build(tag, param_overrides=None, config_overrides=None):
    """
    Build a complete config dict for one experiment.

    Parameters
    ----------
    tag              : str   — unique human-readable label (appended to run name)
    param_overrides  : dict  — {param_name: {field: value, ...}, ...}
    config_overrides : dict  — top-level CONFIG fields to override, e.g.
                               {"sigma_int": 0.1, "nlive": 2000}
    """
    cfg = copy.deepcopy(CONFIG)
    cfg["run_tag"]    = tag
    cfg["param_specs"] = _override(DEFAULT_PARAM_SPECS, **(param_overrides or {}))
    if config_overrides:
        cfg.update(config_overrides)
    return cfg

_REG = {"registry_file": "run_publication_registry.csv"}

# ===========================================================================
# BROAD UNIFORM PRIORS FOR THE REDSHIFT-EVOLUTION SWEEP
# ===========================================================================
# prior_shrinkage.py's scan of run_publication_registry.csv flagged 20 of
# the 36 "evolution/*" runs as prior_dominated (shrinkage < 0.2), every one
# of them on Om0, with alpha close behind -- i.e. across almost the whole
# redshift-evolution section the posterior for the SALT3/cosmology nuisance
# parameters was no tighter than the informative truncated_gaussian prior
# it started from. That leaves the section's uniformly negative Delta ln Z
# ambiguous: it could mean the data genuinely disfavour redshift evolution,
# or it could just mean the informative priors were carrying the fit and
# the extra exponent bought nothing on top of them.
#
# These overrides settle that for the standardisation coefficients, which
# are given UNIFORM priors over ranges deliberately far wider than both
# their informative sigma and their previous hard clips:
#
#   alpha  truncated_gaussian(0.17, 0.05) on [0.04, 0.26]  ->  U[0.0, 0.5]
#   beta   truncated_gaussian(3.12, 0.50) on [1.5,  6.5]   ->  U[0.0, 8.0]
#
# so their posteriors are free to move anywhere the data support, and a
# remaining Delta ln Z deficit cannot be blamed on their priors.
#
# Om0 DELIBERATELY KEEPS ITS INFORMATIVE CMB-LEVEL PRIOR.
# ------------------------------------------------------
# It is the one parameter where "prior dominated" is the intended state
# rather than a warning. Om0 sets the shape of the distance-redshift
# relation, which is the *same* thing the evolution exponents modulate: a
# free Om0 and a free alpha(z)/beta(z) exponent are close to degenerate
# over the DES redshift range, so they simply trade against each other.
# Freeing Om0 would let it absorb the very redshift dependence the sweep
# exists to measure, and the exponents would then come back consistent
# with zero for a reason that has nothing to do with the data. The
# constraint is external (CMB), it is legitimately much tighter than
# SNe alone can deliver, and the whole point of imposing it is to hold
# the background cosmology fixed so any residual z-dependence has to show
# up in the standardisation terms.
#
# So the Om0 flags prior_shrinkage.py raises on this section are expected
# and should not be "fixed"; they are recorded rather than acted on.
#
# Note that widening alpha and beta still costs evidence through the
# Occam factor, so ln Z from these runs is NOT comparable to the
# informative-prior rows elsewhere in run_publication_registry.csv --
# compare them against "evolution/baseline_broaduniform", which is the
# no-evolution model fitted under these exact priors, and which exists
# for precisely that reason.
#
# The evolution exponents a/b/g keep their arcsinh priors deliberately (see
# the section comment below).
_ZEVO_BROAD_UNIFORM = {
    "alpha": {"prior": "uniform", "range": [0.0, 0.5]},
    "beta":  {"prior": "uniform", "range": [0.0, 8.0]},
    # Om0 is intentionally absent -- see the block comment above.
}


def _zevo(tag, z_evolve, *exponents, **param_overrides):
    """
    Build one redshift-evolution experiment.

    Every entry in the "evolution/" section goes through this helper so the
    broad uniform alpha/beta priors (_ZEVO_BROAD_UNIFORM) are applied
    identically to all of them -- writing them out per-entry invited exactly
    the kind of silent drift where one row keeps the informative prior and
    its Delta ln Z is then quietly incomparable to its neighbours'.

    Parameters
    ----------
    tag       : full run tag, e.g. "evolution/alpha_beta_z_power".
    z_evolve  : the z_evolve model name ("power"/"log"/"zz"/"linear"/
                "exp"/"step").
    exponents : which evolution exponents to activate, any of "a" (alpha
                evolution), "b" (beta), "g" (gamma). Pass none for the
                matched-prior no-evolution reference.
    param_overrides : any further per-parameter overrides, merged last so a
                caller can still override the broad uniform block if needed.
    """
    overrides = {name: dict(spec) for name, spec in _ZEVO_BROAD_UNIFORM.items()}
    for e in exponents:
        overrides[e] = {"active": True, "fixed": 0}
    for name, updates in param_overrides.items():
        overrides.setdefault(name, {}).update(updates)
    return _build(tag,
                  config_overrides={"model": {**CONFIG["model"],
                                              "z_evolve": z_evolve}},
                  param_overrides=overrides)


# ===========================================================================
# EXPERIMENT DEFINITIONS
# ===========================================================================
# Each entry is a call to _build().  Add / remove entries freely.
# The tag becomes part of the run name and the registry CSV.
#
# Convention used here:
#   cosmo/          — cosmological model variants (FlatLambdaCDM/wCDM/LambdaCDM,
#                     free Om0, ...)
#   nuisance/       — SALT2 nuisance parameter variants (alpha/beta on/off, ...)
#   sn_col_model/   — SN colour correction functional-form variants
#   host_col_model/ — host-colour correction functional-form variants
#   mass/           — host-mass step functional form variants
#   ssfr/           — host specific-star-formation-rate term variants
#   evolution/      — redshift-evolution (z_evolve) variants
#   stretch/        — SN stretch (x1) correction variants
#   interaction/    — cross-term / interaction variants (gamma_alpha, xi_*, ...)
#   baseline/       — the single reference/baseline model everything else compares to
#
# "scatter/" (intrinsic-scatter sigma_int variants) is a reserved prefix with
# no LIVE entries right now -- both its _build() calls below are commented
# out. Uncomment them (or add new scatter/... entries) rather than inventing
# a different prefix for the same category.
#
# This list is meant to track whatever prefixes are actually LIVE below —
# if you add a new top-level category, add its prefix here too. Don't check
# this by grepping the raw source for `_build("prefix/` — that matches
# commented-out lines too (this list's own "scatter/: 2 entries" line was
# wrong for exactly that reason until this pass). Check what's actually
# live instead:
#   python -c "
#   import sys, types
#   sys.modules['run'] = types.ModuleType('run')
#   sys.modules['run'].run_sampler = lambda *a, **k: None
#   import experiment_runner as er
#   from collections import Counter
#   print(Counter(t['run_tag'].split('/')[0] for t in er.EXPERIMENTS))"
#
# NOTE: "checks/" is reserved for extra_runners.py and the other
# post-hoc systematic-check scripts (host_match_quality.py, loo_zbins.py,
# drilling_cones_checks.py, z_uncertainty_check.py, combo_ablation_checks.py)
# — do not add "checks/..." entries here. Uniform-prior reworkings driven by
# the prior_shrinkage.py / degeneracy_scan.py analysis (e.g. C0's prior) are
# folded directly into DEFAULT_PARAM_SPECS in config.py instead, so they
# become the new default for every relevant entry below rather than living
# as a separate parallel "checks/uniformpriors_*" run — see config.py's
# DEFAULT_PARAM_SPECS docstring for the current C0/M0 prior rationale.
#
# You can use any tag scheme you like; these are just strings. Every tag
# must be unique across the whole EXPERIMENTS list below — a repeated tag
# means the second entry silently overwrites (--rerun) or is silently
# skipped (default) instead of running as its own experiment. See
# combo_ablation_checks.py / this file's own git history for what that
# looks like when it goes unnoticed.
# ===========================================================================

EXPERIMENTS = [

            # -----------------------------------------------------------------------
            # BASELINE
            # -----------------------------------------------------------------------
            _build("baseline"),

            # -----------------------------------------------------------------------
            # COSMOLOGY VARIANTS
            # -----------------------------------------------------------------------

            _build("cosmo/nonflatLCDM", param_overrides={"Ode0": {"active": True, "fixed": 0.6824}}),
            _build("cosmo/flatwCDM", param_overrides={"w": {"active": True, "fixed": -1.0}}),
            _build("cosmo/Om0_free", param_overrides={"Om0": {"active": True, "prior": "uniform", "range": [0.2, 0.6]}}),
            
            # -----------------------------------------------------------------------
            # NUISANCE PARAMETER VARIANTS
            # -----------------------------------------------------------------------

            _build("nuisance/no_alpha", param_overrides={"alpha": {"active": False, "fixed": 0.0}}),

            _build("nuisance/no_beta", param_overrides={"beta": {"active": False, "fixed": 0.0}}),

            _build("nuisance/no_gamma", param_overrides={"gamma": {"active": False, "fixed": 0.0}}),

            _build("nuisance/no_alpha_beta", param_overrides={"alpha": {"active": False, "fixed": 0.0},
                                                              "beta":  {"active": False, "fixed": 0.0}}),

            _build("nuisance/no_gamma_beta", param_overrides={"gamma": {"active": False, "fixed": 0.0},
                                                              "beta":  {"active": False, "fixed": 0.0}}),
            
            _build("nuisance/no_alpha_gamma", param_overrides={"gamma": {"active": False, "fixed": 0.0},
                                                               "alpha":  {"active": False, "fixed": 0.0}}),

            # -----------------------------------------------------------------------
            # REDSHIFT EVOLUTION VARIANTS
            # -----------------------------------------------------------------------
            # BROAD UNIFORM NUISANCE/COSMOLOGY PRIORS -- see _ZEVO_BROAD_UNIFORM
            # above. Every entry in this section is built through _zevo(), so
            # alpha, beta and Om0 are uniform over deliberately over-wide
            # ranges in all of them, and no entry can silently drift back to
            # the informative defaults. The evolution exponents a/b/g keep
            # their arcsinh priors -- they are the parameters under test, and
            # changing their prior at the same time would confound "the
            # nuisance priors were doing the work" with "the exponent prior
            # was doing the work".
            #
            # Compare these against evolution/baseline_broaduniform (the
            # no-evolution reference fitted under the SAME broad uniform
            # priors), NOT against the top-level "baseline" row, which still
            # uses the informative priors and therefore has a systematically
            # different Occam factor.

            # Matched-prior reference: z_evolve is irrelevant here because no
            # exponent is active, so this is the flat, no-evolution model
            # fitted under exactly the broad uniform priors every entry below
            # uses. Delta ln Z against THIS row isolates the evidence for
            # redshift evolution from the evidence lost to widening the
            # priors.
            _zevo("evolution/baseline_broaduniform", "power"),

            _zevo("evolution/alpha_z_power", "power", "a"),
            _zevo("evolution/beta_z_power", "power", "b"),
            _zevo("evolution/gamma_z_power", "power", "g"),
            _zevo("evolution/alpha_beta_z_power", "power", "a", "b"),
            _zevo("evolution/beta_gamma_z_power", "power", "b", "g"),
            _zevo("evolution/all_z_power", "power", "a", "b", "g"),

            _zevo("evolution/alpha_z_log", "log", "a"),
            _zevo("evolution/beta_z_log", "log", "b"),
            _zevo("evolution/gamma_z_log", "log", "g"),
            _zevo("evolution/alpha_beta_z_log", "log", "a", "b"),
            _zevo("evolution/beta_gamma_z_log", "log", "b", "g"),
            _zevo("evolution/all_z_log", "log", "a", "b", "g"),

            _zevo("evolution/alpha_z_zz", "zz", "a"),
            _zevo("evolution/beta_z_zz", "zz", "b"),
            _zevo("evolution/gamma_z_zz", "zz", "g"),
            _zevo("evolution/alpha_beta_z_zz", "zz", "a", "b"),
            _zevo("evolution/beta_gamma_z_zz", "zz", "b", "g"),
            _zevo("evolution/all_z_zz", "zz", "a", "b", "g"),

            # ---- linear-in-z evolution (first-order Taylor around z_pivot) ----
            _zevo("evolution/alpha_z_linear", "linear", "a"),
            _zevo("evolution/beta_z_linear", "linear", "b"),
            _zevo("evolution/gamma_z_linear", "linear", "g"),
            _zevo("evolution/alpha_beta_z_linear", "linear", "a", "b"),
            _zevo("evolution/beta_gamma_z_linear", "linear", "b", "g"),
            _zevo("evolution/all_z_linear", "linear", "a", "b", "g"),

            # ---- exp-in-z evolution ----
            _zevo("evolution/alpha_z_exp", "exp", "a"),
            _zevo("evolution/beta_z_exp", "exp", "b"),
            _zevo("evolution/gamma_z_exp", "exp", "g"),
            _zevo("evolution/alpha_beta_z_exp", "exp", "a", "b"),
            _zevo("evolution/beta_gamma_z_exp", "exp", "b", "g"),
            _zevo("evolution/all_z_exp", "exp", "a", "b", "g"),

            # ---- step-in-z evolution ----
            _zevo("evolution/alpha_z_step", "step", "a"),
            _zevo("evolution/beta_z_step", "step", "b"),
            _zevo("evolution/gamma_z_step", "step", "g"),
            _zevo("evolution/alpha_beta_z_step", "step", "a", "b"),
            _zevo("evolution/beta_gamma_z_step", "step", "b", "g"),
            _zevo("evolution/all_z_step", "step", "a", "b", "g"),

          # -----------------------------------------------------------------------
            # Stretch MODEL VARIANTS  (same parameters, different model function)
            # -----------------------------------------------------------------------
            # x1_tau is the transition width for tanh and softbroken models.
            # It has a log_normal prior peaking near 0.3 mag so the linear
            # limit (large x1_tau) is always reachable by the data.
            
            # Stretch quadratic
            _build("stretch/stretch_quadratic",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "quadratic"}},
                   param_overrides={"x1_0": {"active": True, "fixed": 0}}),

            # Stretch tanh            
            _build("stretch/stretch_tanh",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "tanh"}},
                   param_overrides={"x1_0": {"active": False, "fixed": 0.0},}),
            
            _build("stretch/stretch_tanh_x10",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "tanh"}},
                   param_overrides={"x1_0": {"active": True, "fixed": 0.0},}), 
            
            _build("stretch/stretch_tanh_x10x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "tanh"}},
                   param_overrides={"x1_0": {"active": True, "fixed": 0.0},
                                    "x1_tau": {"active": True, "fixed": 0.3}}),      
                         
            # Stretch soft broken            
            _build("stretch/stretch_softbroken",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "softbroken"}},
                   param_overrides={"x1_0": {"active": False, "fixed": 0.0},
                                    "x1_tau": {"active": False, "fixed": 0.3}}),
            
            _build("stretch/stretch_softbroken_x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "softbroken"}},
                   param_overrides={"x1_0": {"active": False, "fixed": 0.0},
                                    "x1_tau": {"active": True, "fixed": 0.3}}),
            
            _build("stretch/stretch_softbroken_x10x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "softbroken"}},
                   param_overrides={"x1_0": {"active": True, "fixed": 0.0},
                                    "x1_tau": {"active": True, "fixed": 0.3}}),                 
            
            # Stretch step broken            
            _build("stretch/stretch_stepbroken",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "stepbroken"}},
                   param_overrides={"x1_0": {"active": False, "fixed": 0.0},
                                    "x1_tau": {"active": False, "fixed": 0.3}}),
            
            _build("stretch/stretch_stepbroken_x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "stepbroken"}},
                   param_overrides={"x1_0": {"active": False, "fixed": 0.0},
                                    "x1_tau": {"active": True, "fixed": 0.3}}),
            
            _build("stretch/stretch_stepbroken_x10x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "stepbroken"}},
                   param_overrides={"x1_0": {"active": True, "fixed": 0.0},
                                    "x1_tau": {"active": True, "fixed": 0.3}}),      

            # Stretch Asymmetric Weight            
            _build("stretch/stretch_asymm",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "asymm_gauss_weight"}},
                   param_overrides={"x1_0": {"active": False},
                                    "x1_tau": {"active": False}}),
            
            _build("stretch/stretch_asymm_x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "asymm_gauss_weight"}},
                   param_overrides={"x1_0": {"active": False},
                                    "x1_tau": {"active": True}}),
            
            _build("stretch/stretch_asymm_x10x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "asymm_gauss_weight"}},
                   param_overrides={"x1_0": {"active": True},
                                    "x1_tau": {"active": True}}),   

            # Stretch Power-law         
            _build("stretch/stretch_powerlaw",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "powerlaw"}},
                   param_overrides={"x1_0": {"active": False},
                                    "x1_tau": {"active": False}}),
            
            _build("stretch/stretch_powerlaw_x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "powerlaw"}},
                   param_overrides={"x1_0": {"active": False},
                                    "x1_tau": {"active": True}}),
            
            _build("stretch/stretch_powerlaw_x10x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "powerlaw"}},
                   param_overrides={"x1_0": {"active": True},
                                    "x1_tau": {"active": True}}),   

            # Stretch Double-broken      
            _build("stretch/stretch_doublebroken",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "doublebroken"}},
                   param_overrides={"x1_0": {"active": False},
                                    "x1_tau": {"active": False}}),
            
            _build("stretch/stretch_doublebroken_x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "doublebroken"}},
                   param_overrides={"x1_0": {"active": False},
                                    "x1_tau": {"active": True}}),
            
            _build("stretch/stretch_doublebroken_x10x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "doublebroken"}},
                   param_overrides={"x1_0": {"active": True},
                                    "x1_tau": {"active": True}}),   

            # Stretch Sigmoid      
            _build("stretch/stretch_sigmoid",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "sigmoid"}},
                   param_overrides={"x1_0": {"active": False},
                                    "x1_tau": {"active": False}}),
            
            _build("stretch/stretch_sigmoid_x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "sigmoid"}},
                   param_overrides={"x1_0": {"active": False},
                                    "x1_tau": {"active": True}}),
            
            _build("stretch/stretch_sigmoid_x10x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "sigmoid"}},
                   param_overrides={"x1_0": {"active": True},
                                    "x1_tau": {"active": True}}), 

            # -----------------------------------------------------------------------
            # SN COLOUR MODEL VARIANTS  (same parameters, different model function)
            # -----------------------------------------------------------------------
            # sn_tau is the transition width for tanh and softbroken models.
            # It has a log_normal prior peaking near 0.3 mag so the linear
            # limit (large sn_tau) is always reachable by the data.
            
            # SN Colour quadratic
            _build("sn_col_model/sncolour_quadratic",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "quadratic"}},
                   param_overrides={"c0": {"active": True, "fixed": 0}}),

            # SN Colour tanh            
            _build("sn_col_model/sncolour_tanh",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "tanh"}},
                   param_overrides={"c0": {"active": False, "fixed": 0.0},}),
            
            _build("sn_col_model/sncolour_tanh_c0",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "tanh"}},
                   param_overrides={"c0": {"active": True, "fixed": 0.0},}), 
            
            _build("sn_col_model/sncolour_tanh_c0_sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "tanh"}},
                   param_overrides={"c0": {"active": True, "fixed": 0.0},
                                    "sn_tau": {"active": True, "fixed": 0.3}}),      
            
            # SN Colour broken        
            _build("sn_col_model/sncolour_broken",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "broken"}},
                   param_overrides={"c0": {"active": True, "fixed": 0.0},}),
                         
            # SN Colour soft broken            
            _build("sn_col_model/sncolour_softbroken",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}},
                   param_overrides={"c0": {"active": False, "fixed": 0.0},
                                    "sn_tau": {"active": False, "fixed": 0.3}}),
            
            _build("sn_col_model/sncolour_softbroken_sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}},
                   param_overrides={"c0": {"active": False, "fixed": 0.0},
                                    "sn_tau": {"active": True, "fixed": 0.3}}),
            
            _build("sn_col_model/sncolour_softbroken_c0_sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}},
                   param_overrides={"c0": {"active": True, "fixed": 0.0},
                                    "sn_tau": {"active": True, "fixed": 0.3}}),                 
            
            # SN Colour dust
            # c0 is the dust model's colour pivot. It is deliberately NOT
            # fitted in the two entries below: c0 is fixed to 1.0 so the
            # pivot is held at a single reference value and the dust model
            # is tested purely through beta (and, for _sntau, the power-law
            # exponent). This also removes what used to be a silent
            # duplicate -- "sncolour_dust" and "sncolour_dust_c0" carried
            # byte-identical overrides, so the two tags described the same
            # fit. The "_c0"/"_c0sntau" entries further down are now the
            # only ones that sample c0, which is what their names claim.
            _build("sn_col_model/sncolour_dust",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "dust"}},
                   param_overrides={"c0": {"active": False, "fixed": 1.0}}),

            _build("sn_col_model/sncolour_dust_c0",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "dust"}},
                   param_overrides={"c0": {"active": True, "prior": "truncated_gaussian",
                                           "range": [0.3, 2.0], "mu": 1.0, "sigma": 0.3,
                                           "fixed": 1.0}}),

            _build("sn_col_model/sncolour_dust_sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "dust"}},
                   param_overrides={"c0": {"active": False, "fixed": 1.0},
                                    "sn_tau": {"active": True}}),
            
            _build("sn_col_model/sncolour_dust_c0sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "dust"}},
                   param_overrides={"c0": {"active": True, "prior": "truncated_gaussian",
                                           "range": [0.3, 2.0], "mu": 1.0, "sigma": 0.3,
                                           "fixed": 1.0}, 
                                    "sn_tau": {"active": True}}),
            
            # SN Colour step broken            
            _build("sn_col_model/sncolour_stepbroken",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "stepbroken"}},
                   param_overrides={"c0": {"active": False, "fixed": 0.0},
                                    "sn_tau": {"active": False, "fixed": 0.3}}),
            
            _build("sn_col_model/sncolour_stepbroken_sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "stepbroken"}},
                   param_overrides={"c0": {"active": False, "fixed": 0.0},
                                    "sn_tau": {"active": True, "fixed": 0.3}}),
            
            _build("sn_col_model/sncolour_stepbroken_c0_sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "stepbroken"}},
                   param_overrides={"c0": {"active": True, "fixed": 0.0},
                                    "sn_tau": {"active": True, "fixed": 0.3}}),    

            # SN Colour gaussian weight            
            _build("sn_col_model/sncolour_gaussweight",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "asymm_gauss_weight"}},
                   param_overrides={"c0": {"active": False, "fixed": 0.0},
                                    "sn_tau": {"active": False, "fixed": 0.3}}),
            
            _build("sn_col_model/sncolour_gaussweight_sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "asymm_gauss_weight"}},
                   param_overrides={"c0": {"active": False, "fixed": 0.0},
                                    "sn_tau": {"active": True, "fixed": 0.3}}),
            
            _build("sn_col_model/sncolour_gaussweight_c0_sntau",
                   config_overrides={"model": {**CONFIG["model"], "sn_colour": "asymm_gauss_weight"}},
                   param_overrides={"c0": {"active": True, "fixed": 0.0},
                                    "sn_tau": {"active": True, "fixed": 0.3}}),    
            
            # -----------------------------------------------------------------------
            # HOST COLOUR MODEL VARIANTS  (same parameters, different model function)
            # -----------------------------------------------------------------------
            
            # Mass Steps
            _build("host_col_model/host_colour_linear_mass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                              "host_colour": "linear"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_quadratic_mass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                                 "host_colour": "quadratic"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                               "host_colour": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_tanh_mass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                            "host_colour": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_broken_mass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                            "host_colour": "broken"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),            

            _build("host_col_model/host_colour_asymm_mass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                            "host_colour": "asymm"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            # Mass Steps
            _build("host_col_model/host_colour_linear_doublemass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                              "host_colour": "linear", 
                                                                                              "mass": "double_step"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "M0": {"active": False, "fixed": 9.5}, 
                                    "M1": {"active": False, "fixed": 10.5}}),
            
            _build("host_col_model/host_colour_quadratic_doublemass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                                 "host_colour": "quadratic", 
                                                                                                 "mass": "double_step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "M0": {"active": False, "fixed": 9.5}, 
                                    "M1": {"active": False, "fixed": 10.5}}),
            
            _build("host_col_model/host_colour_sigmoid_doublemass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                               "host_colour": "sigmoid", 
                                                                                               "mass": "double_step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "M0": {"active": False, "fixed": 9.5}, 
                                    "M1": {"active": False, "fixed": 10.5}}),
            
            _build("host_col_model/host_colour_tanh_doublemass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                              "host_colour": "tanh", 
                                                                                              "mass": "double_step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "M0": {"active": False, "fixed": 9.5}, 
                                    "M1": {"active": False, "fixed": 10.5}}),
            
            _build("host_col_model/host_colour_broken_doublemass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                            "host_colour": "broken", 
                                                                                            "mass": "double_step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "M0": {"active": False, "fixed": 9.5}, 
                                    "M1": {"active": False, "fixed": 10.5}}),            

            _build("host_col_model/host_colour_asymm_doublemass_step", config_overrides={"model": {**CONFIG["model"], 
                                                                                            "host_colour": "asymm", 
                                                                                            "mass": "double_step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "M0": {"active": False, "fixed": 9.5}, 
                                    "M1": {"active": False, "fixed": 10.5}}),
            
            # Mass None (i.e. no mass step, but still host-colour dependence)
            _build("host_col_model/host_colour_linear_mass_none", config_overrides={"model": {**CONFIG["model"], 
                                                                                              "host_colour": "linear", 
                                                                                              "mass": "none"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "gamma": {"active": False, "fixed": 0}}),
            
            _build("host_col_model/host_colour_quadratic_mass_none", config_overrides={"model": {**CONFIG["model"], 
                                                                                                 "host_colour": "quadratic", 
                                                                                                 "mass": "none"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "gamma": {"active": False, "fixed": 0}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_none", config_overrides={"model": {**CONFIG["model"], 
                                                                                               "host_colour": "sigmoid", 
                                                                                               "mass": "none"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "gamma": {"active": False, "fixed": 0}}),
            
            _build("host_col_model/host_colour_tanh_mass_none", config_overrides={"model": {**CONFIG["model"], 
                                                                                            "host_colour": "tanh", 
                                                                                            "mass": "none"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "gamma": {"active": False, "fixed": 0}}),
            
            _build("host_col_model/host_colour_broken_mass_none", config_overrides={"model": {**CONFIG["model"], 
                                                                                            "host_colour": "broken",
                                                                                            "mass": "none"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0},
                                    "gamma": {"active": False, "fixed": 0}}),            

            _build("host_col_model/host_colour_asymm_mass_none", config_overrides={"model": {**CONFIG["model"], 
                                                                                            "host_colour": "asymm", 
                                                                                            "mass": "none"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0},
                                    "gamma": {"active": False, "fixed": 0}}),       
            
            # Mass Linear 
            _build("host_col_model/host_colour_linear_mass_linear", config_overrides={"model": {**CONFIG["model"], 
                                                                                                "host_colour": "linear", 
                                                                                                "mass": "linear"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_quadratic_mass_linear", config_overrides={"model": {**CONFIG["model"], 
                                                                                                   "host_colour": "quadratic", 
                                                                                                   "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_linear", config_overrides={"model": {**CONFIG["model"], 
                                                                                                 "host_colour": "sigmoid", 
                                                                                                 "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_tanh_mass_linear", config_overrides={"model": {**CONFIG["model"], 
                                                                                              "host_colour": "tanh", 
                                                                                              "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_broken_mass_linear", config_overrides={"model": {**CONFIG["model"], 
                                                                                              "host_colour": "broken", 
                                                                                              "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_asymm_mass_linear", config_overrides={"model": {**CONFIG["model"], 
                                                                                              "host_colour": "asymm", 
                                                                                              "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0}}),
            
            # Mass Steps with non-fixed width (htau)
            _build("host_col_model/host_colour_sigmoid_mass_step_htau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                    "host_colour": "sigmoid",
                                                                                                    "mass": "step"}},
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0},
                                    "htau": {"active": True, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_tanh_mass_step_htau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                 "host_colour": "tanh",
                                                                                                 "mass": "step"}},
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0},
                                    "htau": {"active": True, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_none_htau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                    "host_colour": "sigmoid",
                                                                                                    "mass": "none"}},
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0},
                                    "htau": {"active": True, "fixed": 0.2}, 
                                    "gamma": {"active": False, "fixed": 0}}),
            
            _build("host_col_model/host_colour_tanh_mass_none_htau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                 "host_colour": "tanh",
                                                                                                 "mass": "none"}},
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0},
                                    "htau": {"active": True, "fixed": 0.2}, 
                                    "gamma": {"active": False, "fixed": 0}}),
            
            _build("host_col_model/host_colour_asymm_mass_step_htau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                 "host_colour": "asymm",
                                                                                                 "mass": "step"}},
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0},
                                    "htau": {"active": True, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_asymm_mass_none_htau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                    "host_colour": "asymm",
                                                                                                    "mass": "none"}},
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "eta": {"active": True, "fixed": 0},
                                    "htau": {"active": True, "fixed": 0.2}, 
                                    "gamma": {"active": False, "fixed": 0}}),            
            
            # Interaction terms mass linear
            _build("host_col_model/host_colour_linear_mass_linear_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                      "host_colour": "linear", 
                                                                                                      "mass": "linear"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_quadratic_mass_linear_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "quadratic", 
                                                                                                         "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            
            _build("host_col_model/host_colour_sigmoid_mass_linear_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                      "host_colour": "sigmoid", 
                                                                                                      "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_tanh_mass_linear_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                    "host_colour": "tanh", 
                                                                                                    "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_broken_mass_linear_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                    "host_colour": "broken", 
                                                                                                    "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_asymm_mass_linear_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                     "host_colour": "asymm", 
                                                                                                     "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            # Interaction terms mass step
            _build("host_col_model/host_colour_linear_mass_step_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                    "host_colour": "linear", 
                                                                                                    "mass": "step"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_quadratic_mass_step_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                       "host_colour": "quadratic", 
                                                                                                       "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            
            _build("host_col_model/host_colour_sigmoid_mass_step_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                     "host_colour": "sigmoid", 
                                                                                                     "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_tanh_mass_step_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                  "host_colour": "tanh", 
                                                                                                  "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_broken_mass_step_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                     "host_colour": "broken", 
                                                                                                     "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_asymm_mass_step_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                  "host_colour": "asymm", 
                                                                                                  "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),            
            
            # Interaction terms mass sigmoid
            _build("host_col_model/host_colour_linear_mass_sigmoid_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                       "host_colour": "linear", 
                                                                                                       "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_linear_mass_sigmoid_inter_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "linear", 
                                                                                                          "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_quadratic_mass_sigmoid_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "quadratic", 
                                                                                                          "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_quadratic_mass_sigmoid_inter_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "quadratic", 
                                                                                                             "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_sigmoid_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                        "host_colour": "sigmoid", 
                                                                                                        "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_sigmoid_inter_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                           "host_colour": "sigmoid", 
                                                                                                           "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),

            _build("host_col_model/host_colour_tanh_mass_sigmoid_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                     "host_colour": "tanh", 
                                                                                                     "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_tanh_mass_sigmoid_inter_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                        "host_colour": "tanh", 
                                                                                                        "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_sigmoid_inter_tau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                            "host_colour": "sigmoid", 
                                                                                                            "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": True, "fixed": 0.2}}),

            _build("host_col_model/host_colour_tanh_mass_sigmoid_inter_tau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "tanh", 
                                                                                                         "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": True, "fixed": 0.2}}),
            
            # Interaction terms mass sigmoid with broken host colour law
            _build("host_col_model/host_colour_broken_mass_sigmoid_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "broken", 
                                                                                                         "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),                
            
            _build("host_col_model/host_colour_broken_mass_sigmoid_inter_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "broken", 
                                                                                                         "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),            

            _build("host_col_model/host_colour_broken_mass_sigmoid_inter_tau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "broken", 
                                                                                                         "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": True, "fixed": 0.2}}),
            
            # Interaction terms mass sigmoid with asymmetry in host colour (i.e. different slopes for red and blue hosts)
            _build("host_col_model/host_colour_asymm_mass_sigmoid_inter", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "asymm", 
                                                                                                         "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),                
            
            _build("host_col_model/host_colour_asymm_mass_sigmoid_inter_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "asymm", 
                                                                                                         "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}}),            

            _build("host_col_model/host_colour_asymm_mass_sigmoid_inter_tau", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "asymm", 
                                                                                                         "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "eta": {"active": True, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": True, "fixed": 0.2}}),            

            
            # Only interaction terms
            _build("host_col_model/host_colour_linear_mass_linear_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                           "host_colour": "linear", 
                                                                                                           "mass": "linear"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_linear_mass_step_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "linear", 
                                                                                                         "mass": "step"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_linear_mass_sigmoid_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                            "host_colour": "linear", 
                                                                                                            "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_linear_mass_tanh_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                         "host_colour": "linear", 
                                                                                                         "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": False, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_quadratic_mass_linear_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                              "host_colour": "quadratic", 
                                                                                                              "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_quadratic_mass_step_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                            "host_colour": "quadratic", 
                                                                                                            "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_quadratic_mass_sigmoid_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                               "host_colour": "quadratic", 
                                                                                                               "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_quadratic_mass_tanh_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                            "host_colour": "quadratic", 
                                                                                                            "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_linear_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                            "host_colour": "sigmoid", 
                                                                                                            "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_step_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "sigmoid", 
                                                                                                          "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_sigmoid_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "sigmoid", 
                                                                                                          "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_tanh_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "sigmoid", 
                                                                                                          "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_quadratic_mass_step_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                               "host_colour": "quadratic", 
                                                                                                               "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_quadratic_mass_sigmoid_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                                  "host_colour": "quadratic", 
                                                                                                                  "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_quadratic_mass_tanh_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                               "host_colour": "quadratic", 
                                                                                                               "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_step_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "sigmoid", 
                                                                                                             "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_sigmoid_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "sigmoid", 
                                                                                                             "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_sigmoid_mass_tanh_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "sigmoid", 
                                                                                                             "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            # Asymmetric host colour law variants (i.e. different slopes for red and blue hosts) with only interaction terms
            _build("host_col_model/host_colour_asymm_mass_linear_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                            "host_colour": "asymm", 
                                                                                                            "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_asymm_mass_step_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "asymm", 
                                                                                                          "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_asymm_mass_sigmoid_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "asymm", 
                                                                                                          "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_asymm_mass_tanh_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "asymm", 
                                                                                                          "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),            
            
            _build("host_col_model/host_colour_asymm_mass_step_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "asymm", 
                                                                                                             "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_asymm_mass_sigmoid_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "asymm", 
                                                                                                             "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_asymm_mass_tanh_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "asymm", 
                                                                                                             "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),            
            
            # Interaction terms host colour broken 
            _build("host_col_model/host_colour_broken_mass_linear_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                            "host_colour": "broken", 
                                                                                                            "mass": "linear"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_broken_mass_step_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "broken", 
                                                                                                          "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_broken_mass_sigmoid_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "broken", 
                                                                                                          "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_broken_mass_tanh_inter_only", config_overrides={"model": {**CONFIG["model"], 
                                                                                                          "host_colour": "broken", 
                                                                                                          "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": False, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),            
            
            _build("host_col_model/host_colour_broken_mass_step_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "broken", 
                                                                                                             "mass": "step"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_broken_mass_sigmoid_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "broken", 
                                                                                                             "mass": "sigmoid"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),
            
            _build("host_col_model/host_colour_broken_mass_tanh_inter_only_M0", config_overrides={"model": {**CONFIG["model"], 
                                                                                                             "host_colour": "broken", 
                                                                                                             "mass": "tanh"}}, 
                   param_overrides={"C0": {"active": True, "fixed": 0},
                                    "M0": {"active": True, "fixed": 10.0},
                                    "gamma": {"active": False, "fixed": 0}, 
                                    "eta": {"active": False, "fixed": 0}, 
                                    "xi_mass_col": {"active": True, "fixed": 0}, 
                                    "tau": {"active": False, "fixed": 0.2}}),                
            
            # -----------------------------------------------------------------------
            # INTRINSIC SCATTER VARIANTS
            # -----------------------------------------------------------------------

            # _build("scatter/sigma05", config_overrides={"sigma_int": 0.05}),

            # _build("scatter/sigma10", config_overrides={"sigma_int": 0.1}),
            
            # -----------------------------------------------------------------------
            # MASS STEP VARIANTS
            # -----------------------------------------------------------------------
            
            # Step masses
            _build("mass/mass_step_M0", config_overrides={"model": {**CONFIG["model"], "mass": "step"}}, 
                   param_overrides={"M0": {"active": True, "fixed": 10.0}}),
            
            # Linear masses
            _build("mass/mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear"}}, 
                   param_overrides={"M0": {"active": False, "fixed": 10.0}}),
            
            # Tanh masses
            _build("mass/mass_tanh_M0", config_overrides={"model": {**CONFIG["model"], "mass": "tanh"}}, 
                   param_overrides={"M0": {"active": True, "fixed": 10.0}}),
            
            _build("mass/mass_tanh", config_overrides={"model": {**CONFIG["model"], "mass": "tanh"}}, 
                   param_overrides={"M0": {"active": False, "fixed": 10.0}}),
            
            _build("mass/mass_tanh_M0_tau", config_overrides={"model": {**CONFIG["model"], "mass": "tanh"}},
                   param_overrides={"M0": {"active": True, "fixed": 10.0},
                                    "tau":  {"active": True, "fixed": 0.2}}),
            
            # Sigmoid masses
            _build("mass/mass_sigmoid_M0", config_overrides={"model": {**CONFIG["model"], "mass": "sigmoid"}}, 
                   param_overrides={"M0": {"active": True, "fixed": 10.0}}),
            
            _build("mass/mass_sigmoid", config_overrides={"model": {**CONFIG["model"], "mass": "sigmoid"}}, 
                   param_overrides={"M0": {"active": False, "fixed": 10.0}}),
            
            _build("mass/mass_sigmoid_M0_tau", config_overrides={"model": {**CONFIG["model"], "mass": "sigmoid"}},
                   param_overrides={"M0": {"active": True, "fixed": 10.0},
                                    "tau":  {"active": True, "fixed": 0.2}}),
            
            # Double Step masses
            _build("mass/mass_doublestep", config_overrides={"model": {**CONFIG["model"], "mass": "double_step"}}, 
                   param_overrides={"M0": {"active": False, "fixed": 9.5}, 
                                    "M1": {"active": False, "fixed": 10.5}}),
            
            _build("mass/mass_doublestep_M0_M1", config_overrides={"model": {**CONFIG["model"], "mass": "double_step"}}, 
                   param_overrides={"M0": {"active": True, "fixed": 9.5},
                                    "M1": {"active": True, "fixed": 10.5}}),
            
            # Gaussian Weight masses
            _build("mass/mass_gaussian_weight_M0", config_overrides={"model": {**CONFIG["model"], "mass": "gaussian_weight"}}, 
                   param_overrides={"M0": {"active": True, "fixed": 10.0}}),
            
            _build("mass/mass_gaussian_weight", config_overrides={"model": {**CONFIG["model"], "mass": "gaussian_weight"}}, 
                   param_overrides={"M0": {"active": False, "fixed": 10.0}}),
            
            _build("mass/mass_gaussian_weight_M0_tau", config_overrides={"model": {**CONFIG["model"], "mass": "gaussian_weight"}},
                   param_overrides={"M0": {"active": True, "fixed": 10.0},
                                    "tau":  {"active": True, "fixed": 0.2}}),
            
            # Spline masses
            _build("mass/mass_spline",
                   config_overrides={"model": {**CONFIG["model"], "mass": "spline"}},
                   param_overrides={"k1": {"active": True, "prior": "arcsinh",
                                           "range": [-3.0, 3.0], "scale": 0.5, "fixed": 0.0},
                                    "k2": {"active": True, "prior": "arcsinh",
                                           "range": [-3.0, 3.0], "scale": 0.5, "fixed": 0.0},
                                    "k3": {"active": True, "prior": "arcsinh",
                                           "range": [-3.0, 3.0], "scale": 0.5, "fixed": 0.0},}),

            # -----------------------------------------------------------------------
            # INTERACTION TERM VARIANTS
            # -----------------------------------------------------------------------

            _build("interaction/beta_alpha", param_overrides={"beta_alpha": {"active": True, "fixed": None}}),

            _build("interaction/gamma_alpha", param_overrides={"gamma_alpha": {"active": True, "fixed": None}}),

            _build("interaction/beta_gamma", param_overrides={"beta_gamma": {"active": True, "fixed": None}}),

            _build("interaction/all_interaction_terms", 
                   param_overrides={"beta_alpha": {"active": True, "fixed": None},
                                    "gamma_alpha": {"active": True, "fixed": None},
                                    "beta_gamma": {"active": True, "fixed": None}}),
            
            
       
              # =========================================================================
              # 1.  sSFR ALONE
              #     Test whether sSFR has any main effect in isolation, before coupling
              #     it to S or H.  All four sSFR functional forms are tested; the mass
              #     step (gamma/2 * S) and host-colour (eta * H) terms are retained at
              #     their baseline values so the evidence ratios reflect only the
              #     addition of the F term.
              #
              #     step  : hard passive/star-forming split at F0 (fixed at -10.5).
              #             The cleanest null test — one new parameter (zeta).
              #     tanh  : smooth step, F0 fixed; tests whether transition width matters.
              #     sigmoid: logistic step, same logic as tanh but different tail shape.
              #     linear: continuous sSFR trend, F0 fixed (degenerate with M — do not
              #             sample F0 here).
              #     step_F0: hard step with F0 free; tests whether the fiducial -10.5
              #              threshold is appropriate for this dataset.
              # =========================================================================

              # -- step (1 new param: zeta) --
              _build("ssfr/ssfr_step",
                     config_overrides={**_REG, "model": _M(ssfr="step")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5}}),

              _build("ssfr/ssfr_step_F0",
                     config_overrides={**_REG, "model": _M(ssfr="step")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5}}),

              # -- tanh (1 new param: zeta; F0 fixed, ftau fixed) --
              _build("ssfr/ssfr_tanh",
                     config_overrides={**_REG, "model": _M(ssfr="tanh")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5}}),

              # -- tanh with F0 free (2 new params: zeta, F0) --
              _build("ssfr/ssfr_tanh_F0",
                     config_overrides={**_REG, "model": _M(ssfr="tanh")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5}}),

              # -- tanh with ftau free (2 new params: zeta, ftau) --
              _build("ssfr/ssfr_tanh_ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": True,  "fixed": 0.5}}),

              # -- tanh fully free (3 new params: zeta, F0, ftau) --
              _build("ssfr/ssfr_tanh_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": True, "fixed": 0.5}}),

              # -- sigmoid (parallel set to tanh) --
              _build("ssfr/ssfr_sigmoid",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5}}),

              _build("ssfr/ssfr_sigmoid_F0",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5}}),

              _build("ssfr/ssfr_sigmoid_ftau",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": True,  "fixed": 0.5}}),

              _build("ssfr/ssfr_sigmoid_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": True, "fixed": 0.5}}),

              # -- linear (F0 fixed — degenerate with M if free) --
              _build("ssfr/ssfr_linear",
                     config_overrides={**_REG, "model": _M(ssfr="linear")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0}}),

              # =========================================================================
              # 2.  sSFR WITH MASS STEP  (S + F terms; no host-colour)
              #     Tests whether sSFR adds information beyond the mass step, and whether
              #     the sSFR × mass interaction (xi_sSFR_mass * F*S) is needed.
              #
              #     The host-colour term (eta * H) is kept at its baseline value.
              #     Mass step is fixed at M0=10.0 throughout this group; a separate
              #     sub-group adds M0 as a free parameter.
              # =========================================================================

              # -- sSFR step alongside mass step (zeta only) --
              _build("ssfr/ssfr_step_massstep",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_step_massstep_M0",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "M0":   {"active": True, "fixed": 10.0}}),

              # -- sSFR step + F*S interaction (zeta + theta) --
              _build("ssfr/ssfr_step_massstep_xi_sSFR_mass",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_mass": {"active": True,  "fixed": 0.0},
                                          "F0":    {"active": False, "fixed": -10.5},
                                          "M0":    {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_step_massstep_xi_sSFR_mass_M0F0",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                          "xi_sSFR_mass": {"active": True, "fixed": 0.0},
                                          "F0":    {"active": True, "fixed": -10.5},
                                          "M0":    {"active": True, "fixed": 10.0}}),

              # -- sSFR tanh alongside mass step --
              _build("ssfr/ssfr_tanh_massstep",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="step")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_tanh_massstep_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="step")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": True, "fixed": 0.5},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              # -- sSFR tanh + F*S interaction --
              _build("ssfr/ssfr_tanh_massstep_xi_sSFR_mass",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="step")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_mass": {"active": True,  "fixed": 0.0},
                                          "F0":    {"active": False, "fixed": -10.5},
                                          "ftau":  {"active": False, "fixed": 0.5},
                                          "M0":    {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_tanh_massstep_xi_sSFR_mass_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="step")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                          "xi_sSFR_mass": {"active": True, "fixed": 0.0},
                                          "F0":    {"active": True, "fixed": -10.5},
                                          "ftau":  {"active": True, "fixed": 0.5},
                                          "M0":    {"active": False, "fixed": 10.0}}),

              # -- sSFR alongside mass sigmoid (smooth mass transition) --
              _build("ssfr/ssfr_step_masssigmoid",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="sigmoid")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_step_masssigmoid_M0",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="sigmoid")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "M0":   {"active": True, "fixed": 10.0}}),

              _build("ssfr/ssfr_step_masssigmoid_xi_sSFR_mass",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="sigmoid")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_mass": {"active": True,  "fixed": 0.0},
                                          "F0":    {"active": False, "fixed": -10.5},
                                          "M0":    {"active": False, "fixed": 10.0}}),

              # -- sSFR alongside mass linear --
              _build("ssfr/ssfr_step_masslinear",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="linear")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5}}),

              _build("ssfr/ssfr_step_masslinear_xi_sSFR_mass",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="linear")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_mass": {"active": True,  "fixed": 0.0},
                                          "F0":    {"active": False, "fixed": -10.5}}),

              # =========================================================================
              # 3.  sSFR WITH HOST COLOUR  (H + F terms; default mass step retained)
              #     Tests whether sSFR adds information beyond host colour, and whether
              #     the sSFR × host-colour interaction (xi_sSFR_col * F*H) is needed.
              # =========================================================================

              # -- sSFR step alongside host-colour linear --
              _build("ssfr/ssfr_step_hcol_linear",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="linear")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_step_hcol_linear_xi_sSFR_col",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="linear")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": False, "fixed": 0.0}}),

              # -- sSFR step alongside host-colour tanh --
              _build("ssfr/ssfr_step_hcol_tanh",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="tanh")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": True,  "fixed": 0.0}}),

              _build("ssfr/ssfr_step_hcol_tanh_xi_sSFR_col",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="tanh")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": True,  "fixed": 0.0}}),

              # -- sSFR step alongside host-colour sigmoid --
              _build("ssfr/ssfr_step_hcol_sigmoid",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="sigmoid")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": True,  "fixed": 0.0}}),

              _build("ssfr/ssfr_step_hcol_sigmoid_xi_sSFR_col",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="sigmoid")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": True,  "fixed": 0.0}}),

              # -- sSFR step alongside host-colour quadratic --
              _build("ssfr/ssfr_step_hcol_quadratic",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="quadratic")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": True,  "fixed": 0.0}}),

              _build("ssfr/ssfr_step_hcol_quadratic_xi_sSFR_col",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="quadratic")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": True,  "fixed": 0.0}}),

              # -- sSFR step alongside host-colour asymm --
              _build("ssfr/ssfr_step_hcol_asymm",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="asymm")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": True,  "fixed": 0.0}}),

              _build("ssfr/ssfr_step_hcol_asymm_xi_sSFR_col",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="asymm")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": True,  "fixed": 0.0}}),

              # -- sSFR tanh alongside host-colour linear --
              _build("ssfr/ssfr_tanh_hcol_linear",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", host_colour="linear")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_tanh_hcol_linear_xi_sSFR_col",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", host_colour="linear")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "ftau":    {"active": False, "fixed": 0.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_tanh_hcol_tanh_xi_sSFR_col",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", host_colour="tanh")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "ftau":    {"active": False, "fixed": 0.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": True,  "fixed": 0.0}}),

              # =========================================================================
              # 4.  sSFR WITH FULL HOST TERM  (S + H + F simultaneously)
              #     The most physically complete models.  zeta, eta, and gamma/2 are all
              #     free; xi_mass_col (S*H interaction) carried from the baseline.  These are the
              #     models to focus on after identifying the best sSFR functional form.
              # =========================================================================

              # -- step F, step mass, linear host colour --
              _build("ssfr/ssfr_step_massstep_hcol_linear",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="linear")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": False, "fixed": 0.0},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              # -- step F, step mass, linear host colour, xi interaction retained --
              _build("ssfr/ssfr_step_massstep_hcol_linear_xi",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="linear")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "xi_mass_col":   {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": False, "fixed": 0.0},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              # -- step F, step mass, tanh host colour --
              _build("ssfr/ssfr_step_massstep_hcol_tanh",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="tanh")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": True,  "fixed": 0.0},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_step_massstep_hcol_tanh_F0",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="tanh")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "eta":  {"active": True, "fixed": 0.0},
                                          "C0":   {"active": True, "fixed": 0.0},
                                          "M0":   {"active": True, "fixed": 10.0}}),

              # -- tanh F, step mass, linear host colour --
              _build("ssfr/ssfr_tanh_massstep_hcol_linear",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="step", host_colour="linear")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": False, "fixed": 0.0},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_tanh_massstep_hcol_linear_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="step", host_colour="linear")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": True, "fixed": 0.5},
                                          "eta":  {"active": True, "fixed": 0.0},
                                          "C0":   {"active": False, "fixed": 0.0},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              # -- tanh F, step mass, tanh host colour --
              _build("ssfr/ssfr_tanh_massstep_hcol_tanh",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="step", host_colour="tanh")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": True,  "fixed": 0.0},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              # -- sigmoid F, step mass, linear host colour --
              _build("ssfr/ssfr_sigmoid_massstep_hcol_linear",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid", mass="step", host_colour="linear")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "ftau": {"active": False, "fixed": 0.5},
                                          "eta":  {"active": True,  "fixed": 0.0},
                                          "C0":   {"active": False, "fixed": 0.0},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_sigmoid_massstep_hcol_linear_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid", mass="step", host_colour="linear")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": True, "fixed": 0.5},
                                          "eta":  {"active": True, "fixed": 0.0},
                                          "C0":   {"active": False, "fixed": 0.0},
                                          "M0":   {"active": False, "fixed": 10.0}}),

              # =========================================================================
              # 5.  INTERACTION TERMS
              #     Tests each of the three higher-order terms that couple F to the
              #     other host observables, then their combinations.
              #     All built on top of the best-performing full_host foundation:
              #     sSFR step, mass step, host-colour linear (zeta + eta baseline).
              # =========================================================================

              # -- xi_sSFR_mass: F*S (sSFR × mass) --
              _build("ssfr/ssfr_xi_sSFR_mass_FSmassstep",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="linear")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_mass": {"active": True,  "fixed": 0.0},
                                          "F0":    {"active": False, "fixed": -10.5},
                                          "eta":   {"active": True,  "fixed": 0.0},
                                          "C0":    {"active": False, "fixed": 0.0},
                                          "M0":    {"active": False, "fixed": 10.0}}),

              # -- xi_sSFR_col: F*H (sSFR × host colour) --
              _build("ssfr/ssfr_xi_sSFR_col_FH_hcol_linear",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="linear")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": False, "fixed": 0.0},
                                          "M0":      {"active": False, "fixed": 10.0}}),

              _build("ssfr/ssfr_xi_sSFR_col_FH_hcol_tanh",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="tanh")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": True,  "fixed": 0.0},
                                          "M0":      {"active": False, "fixed": 10.0}}),

              # -- xi_sSFR_mass + xi_sSFR_col: F*S and F*H simultaneously --
              _build("ssfr/ssfr_xi_sSFR_mass_xi_sSFR_col",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="linear")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_mass":   {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": False, "fixed": 0.0},
                                          "M0":      {"active": False, "fixed": 10.0}}),

              # -- omega: F*S*H (three-way) alone --
              _build("ssfr/ssfr_omega_FSH",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="linear")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                          "omega": {"active": True,  "fixed": 0.0},
                                          "F0":    {"active": False, "fixed": -10.5},
                                          "eta":   {"active": True,  "fixed": 0.0},
                                          "C0":    {"active": False, "fixed": 0.0},
                                          "M0":    {"active": False, "fixed": 10.0}}),

              # -- Full four-term sSFR expansion: zeta + xi_sSFR_mass + xi_sSFR_col + omega --
              _build("ssfr/ssfr_full_expansion",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="linear")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_mass":   {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "omega":   {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "xi_mass_col":      {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": False, "fixed": 0.0},
                                          "M0":      {"active": False, "fixed": 10.0}}),

              # -- Full four-term with tanh host colour --
              _build("ssfr/ssfr_full_expansion_hcol_tanh",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="tanh")},
                     param_overrides={"zeta":    {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_mass":   {"active": True,  "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True,  "fixed": 0.0},
                                          "omega":   {"active": True,  "fixed": 0.0},
                                          "F0":      {"active": False, "fixed": -10.5},
                                          "eta":     {"active": True,  "fixed": 0.0},
                                          "xi_mass_col":      {"active": True,  "fixed": 0.0},
                                          "C0":      {"active": True,  "fixed": 0.0},
                                          "M0":      {"active": False, "fixed": 10.0}}),

              # -- Full four-term with F0 and M0 free --
              _build("ssfr/ssfr_full_expansion_F0M0",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="step", host_colour="linear")},
                     param_overrides={"zeta":    {"active": True, "fixed": 0.0},
                                          "xi_sSFR_mass":   {"active": True, "fixed": 0.0},
                                          "xi_sSFR_col": {"active": True, "fixed": 0.0},
                                          "omega":   {"active": True, "fixed": 0.0},
                                          "F0":      {"active": True, "fixed": -10.5},
                                          "eta":     {"active": True, "fixed": 0.0},
                                          "xi_mass_col":      {"active": True, "fixed": 0.0},
                                          "C0":      {"active": False, "fixed": 0.0},
                                          "M0":      {"active": True, "fixed": 10.0}}),

              # =========================================================================
              # 6.  sSFR REPLACES MASS STEP  (gamma fixed to 0)
              #     Tests whether sSFR is a *substitute* for the mass step rather than
              #     a complement.  If logZ(replace) > logZ(alone) then sSFR encodes
              #     a significant fraction of the information in S.
              # =========================================================================

              _build("ssfr/ssfr_step_nomass",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="none")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                          "F0":    {"active": False, "fixed": -10.5},
                                          "gamma": {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_step_nomass_F0",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="none")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                          "F0":    {"active": True, "fixed": -10.5},
                                          "gamma": {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_tanh_nomass_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="none")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                          "F0":    {"active": True, "fixed": -10.5},
                                          "ftau":  {"active": True, "fixed": 0.5},
                                          "gamma": {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_sigmoid_nomass_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid", mass="none")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                          "F0":    {"active": True, "fixed": -10.5},
                                          "ftau":  {"active": True, "fixed": 0.5},
                                          "gamma": {"active": False, "fixed": 0.0}}),

              # =========================================================================
              # 7.  sSFR REPLACES HOST COLOUR  (eta fixed to 0)
              #     Tests whether sSFR is a substitute for host colour.
              # =========================================================================

              _build("ssfr/ssfr_step_nohcol",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="none")},
                     param_overrides={"zeta": {"active": True,  "fixed": 0.0},
                                          "F0":   {"active": False, "fixed": -10.5},
                                          "eta":  {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_step_nohcol_F0",
                     config_overrides={**_REG, "model": _M(ssfr="step", host_colour="none")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "eta":  {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_tanh_nohcol_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", host_colour="none")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": True, "fixed": 0.5},
                                          "eta":  {"active": False, "fixed": 0.0}}),

              # =========================================================================
              # 8.  sSFR REPLACES BOTH MASS AND HOST COLOUR  (gamma=eta=0)
              #     The extreme hypothesis: sSFR alone captures all host-environment
              #     information.  Useful primarily for an evidence comparison.
              # =========================================================================

              _build("ssfr/ssfr_step_nomass_nohcol",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="none", host_colour="none")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                          "F0":    {"active": False, "fixed": -10.5},
                                          "gamma": {"active": False, "fixed": 0.0},
                                          "eta":   {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_step_nomass_nohcol_F0",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="none", host_colour="none")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                          "F0":    {"active": True, "fixed": -10.5},
                                          "gamma": {"active": False, "fixed": 0.0},
                                          "eta":   {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_tanh_nomass_nohcol_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="none", host_colour="none")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                          "F0":    {"active": True, "fixed": -10.5},
                                          "ftau":  {"active": True, "fixed": 0.5},
                                          "gamma": {"active": False, "fixed": 0.0},
                                          "eta":   {"active": False, "fixed": 0.0}}),

              # =========================================================================
              # 9.  SMOOTH F0 / SHAPE EXPLORATION
              #     Dedicated group for probing the sSFR threshold location and
              #     transition sharpness in a more systematic way, using the minimal
              #     zeta-only model to keep the parameter count low.
              # =========================================================================

              # Vary F0 prior — broader range to check whether -10.5 is correct.
              # NOTE: this used to share the tag "ssfr/ssfr_step_F0" with the
              # identical-looking entry in section 1 above, which meant this
              # entry was either silently skipped (already in the registry)
              # or silently overwrote section 1's fit -- and even when it did
              # run, "active": True alone does NOT broaden anything (fixed is
              # ignored once active=True), so it was never actually testing a
              # broader F0 prior. Fixed here: swap F0 to a uniform prior over
              # its existing hard range (DEFAULT_PARAM_SPECS["F0"]["range"])
              # instead of the informative gaussian, which is what "is -10.5
              # correct" actually requires testing, and give it its own tag.
              _build("ssfr/ssfr_step_F0_uniformF0",
                     config_overrides={**_REG, "model": _M(ssfr="step")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "prior": "uniform"}}),

              # tanh with ftau very free — distinguishes sharp vs smooth transition.
              # Same fix as above: previously identical to section 1's
              # "ssfr/ssfr_tanh_F0ftau" entry under the same tag (ftau's
              # missing "fixed" key has no effect while active=True). "Very
              # free" now means what it says: ftau gets a uniform prior over
              # its hard range rather than the informative log-normal.
              _build("ssfr/ssfr_tanh_F0ftau_uniformftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": True, "prior": "uniform"}}),

              # sigmoid with ftau very free — same fix as tanh above.
              _build("ssfr/ssfr_sigmoid_F0ftau_uniformftau",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                          "F0":   {"active": True, "fixed": -10.5},
                                          "ftau": {"active": True, "prior": "uniform"}}),

              # -----------------------------------------------------------------------
              # sSFR REPLACES BOTH MASS AND HOST COLOUR  (gamma=eta=0)
              #     The extreme hypothesis: sSFR alone captures all host-environment
              #     information.  Useful primarily for an evidence comparison.
              # -----------------------------------------------------------------------
              _build("ssfr/ssfr_replace_both_step_nomass_nohcol",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="none", host_colour="none")},
                     param_overrides={"zeta":  {"active": True,  "fixed": 0.0},
                                      "F0":    {"active": False, "fixed": -10.5},
                                      "gamma": {"active": False, "fixed": 0.0},
                                      "eta":   {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_replace_both_step_nomass_nohcol_F0",
                     config_overrides={**_REG, "model": _M(ssfr="step", mass="none", host_colour="none")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                      "F0":    {"active": True, "fixed": -10.5},
                                      "gamma": {"active": False, "fixed": 0.0},
                                      "eta":   {"active": False, "fixed": 0.0}}),

              _build("ssfr/ssfr_replace_both_tanh_nomass_nohcol_F0ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh", mass="none", host_colour="none")},
                     param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                      "F0":    {"active": True, "fixed": -10.5},
                                      "ftau":  {"active": True, "fixed": 0.5},
                                      "gamma": {"active": False, "fixed": 0.0},
                                      "eta":   {"active": False, "fixed": 0.0}}),

              # -----------------------------------------------------------------------
              # SMOOTH F0 / SHAPE EXPLORATION
              #     Probes the sSFR threshold location and transition sharpness with
              #     broadened priors, using the minimal zeta-only model to keep the
              #     parameter count low.  (Consolidated from extra_sSFR_runners.py.)
              # -----------------------------------------------------------------------
              _build("ssfr/ssfr_smooth_F0_step_broadF0",
                     config_overrides={**_REG, "model": _M(ssfr="step")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                      "F0":   {"active": True, "prior": "truncated_gaussian",
                                               "range": [-13.0, -8.0], "mu": -10.5, "sigma": 1.0,
                                               "fixed": -10.5}}),

              _build("ssfr/ssfr_smooth_F0_tanh_wide_ftau",
                     config_overrides={**_REG, "model": _M(ssfr="tanh")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                      "F0":   {"active": True, "fixed": -10.5},
                                      "ftau": {"active": True, "prior": "log_normal",
                                               "range": [0.05, 5.0], "mu": 0.0, "sigma": 1.0,
                                               "fixed": 0.5}}),

              _build("ssfr/ssfr_smooth_F0_sigmoid_wide_ftau",
                     config_overrides={**_REG, "model": _M(ssfr="sigmoid")},
                     param_overrides={"zeta": {"active": True, "fixed": 0.0},
                                      "F0":   {"active": True, "fixed": -10.5},
                                      "ftau": {"active": True, "prior": "log_normal",
                                               "range": [0.05, 5.0], "mu": 0.0, "sigma": 1.0,
                                               "fixed": 0.5}}),
    ]

# ===========================================================================
# RUNNER
# ===========================================================================

def _resolve_indices(index_str, n):
    """Parse '5' or '0-9' into a list of integer indices."""
    if "-" in index_str:
        lo, hi = index_str.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(index_str)]
 
def _run_one(args_tuple):
    import time, traceback, sys, os
 
    idx, cfg, log_dir = args_tuple
 
    # ── Belt-and-braces thread clamp ──────────────────────────────────────
    # With spawn mode the module-level env var block (top of file) already
    # runs in every worker before numpy loads, so this is truly redundant.
    # Kept only for safety if _run_one is ever called outside the pool.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = "1"
 
    # threadpoolctl guard — catches any BLAS libraries dlopen'd after env vars
    # were read.  Errors are silently swallowed; the env vars above suffice.
    import io
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_stderr_fd = os.dup(2)
    os.dup2(_devnull_fd, 2)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    finally:
        os.dup2(_saved_stderr_fd, 2)
        os.close(_saved_stderr_fd)
        os.close(_devnull_fd)
 
    tag = cfg["run_tag"]
    safe_tag = tag.replace("/", "_")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{safe_tag}.log")
 
    t0 = time.time()
    with open(log_path, "w", buffering=1) as log:   # buffering=1 → line-buffered
        log.write(f"=== [{idx}] {tag} ===\n")
        log.write(f"Started: {datetime.now().isoformat()}\n")
        log.write(f"PID: {os.getpid()}  CPU count: {os.cpu_count()}\n\n")
        log.flush()
 
        # Redirect both stdout and stderr to the log file for this process.
        # dynesty's progress bar and all print() calls from run.py go here.
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = log
        sys.stderr = log
 
        try:
            run_sampler(cfg)
            elapsed = time.time() - t0
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log.write(f"\n=== DONE in {elapsed:.1f}s ===\n")
            return (idx, tag, "ok", elapsed, "")
        except BaseException:
            # BaseException (not just Exception) catches MemoryError, KeyboardInterrupt,
            # and system signals that would otherwise silently kill the worker process
            # and surface only as BrokenProcessPool in the parent.
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            # Restore streams BEFORE writing so the log flush actually works
            # even if the log file handle itself is in a bad state.
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:
                log.write(f"\n=== FAILED after {elapsed:.1f}s ===\n{tb}\n")
                log.flush()
            except Exception:
                pass
            # Print to real stderr so the parent process sees it immediately
            # even if the log file write above failed.
            print(f"\n[worker {idx}] FAILED: {tb}", file=sys.stderr, flush=True)
            return (idx, tag, "failed", elapsed, tb)
 
def _parse_args():
    p = argparse.ArgumentParser(description="Run SNe Ia experiment suite")
    p.add_argument("--tag", default=None,
                   help="Only run experiments whose tag contains this string")
    p.add_argument("--index", default=None,
                   help="Run a single index or range e.g. 2 or 0-9")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without launching the sampler")
    p.add_argument("--list", action="store_true",
                   help="List all experiments with their index and tag, then exit")
    p.add_argument("--workers", type=int, default=None,
                   help="Max parallel processes (default: number of experiments, "
                        "capped at os.cpu_count())")
    p.add_argument("--log-dir", default="logs",
                   help="Directory for per-experiment log files (default: logs/)")
    p.add_argument("--progress-interval", type=float, default=None,
                   dest="progress_interval",
                   help="Seconds between dynesty progress lines in each log "
                        "(default: config's progress_interval, 1800 = every "
                        "30 min; 0 = dynesty's continuous progress bar)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress dynesty progress output in the logs "
                        "entirely (setup banners and summaries are kept)")
    p.add_argument("--sequential", action="store_true",
                   help="Disable parallelism — run one at a time (useful for debugging)")
    p.add_argument("--rerun", action="store_true",
                   help="Re-run experiments even if their tag already exists in the registry")
    # nlive mode — mutually exclusive; if neither is given the mode stored in
    # each experiment's config dict is used (default: "exploratory").
    nlive_group = p.add_mutually_exclusive_group()
    nlive_group.add_argument("--publication", action="store_true",
                             help="Override nlive_mode to 'publication' for all selected "
                                  "experiments (ndim x 500 live points)")
    nlive_group.add_argument("--explore", action="store_true",
                             help="Override nlive_mode to 'exploratory' for all selected "
                                  "experiments (ndim x 50 live points)")
    return p.parse_args()
 
def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime
 
    args = _parse_args()
 
    # ---- Resolve nlive mode from CLI flags ----
    # --publication / --explore override whatever nlive_mode is stored in each
    # experiment's config.  If neither flag is given, each experiment uses its
    # own stored mode (default: "exploratory").
    if args.publication:
        _cli_mode = "publication"
    elif args.explore:
        _cli_mode = "exploratory"
    else:
        _cli_mode = None   # use per-experiment setting
 
    def _nlive_display(cfg):
        """nlive that will be used, for display and summary purposes."""
        if cfg.get("nlive"):
            return int(cfg["nlive"])
        mode = _cli_mode or cfg.get("nlive_mode", "exploratory")
        n = sum(1 for s in cfg["param_specs"].values() if s["active"])
        return n * 500 if mode == "publication" else n * 50
 
    # ---- Filter experiments ----
    selected = list(enumerate(EXPERIMENTS))
 
    if args.list:
        mode_label = _cli_mode or "per-experiment"
        print(f"{'idx':>4}  {'tag':<45}  params  nlive  (mode: {mode_label})")
        print(f"{'---':>4}  {'---':<45}  ------  -----")
        for i, cfg in selected:
            n = sum(1 for s in cfg["param_specs"].values() if s["active"])
            print(f"{i:>4}  {cfg['run_tag']:<45}  {n:>6}  {_nlive_display(cfg)}")
        sys.exit(0)
 
    if args.index is not None:
        indices = _resolve_indices(args.index, len(EXPERIMENTS))
        selected = [(i, e) for i, e in selected if i in indices]
 
    if args.tag is not None:
        selected = [(i, e) for i, e in selected if args.tag in e["run_tag"]]
 
    if not selected:
        print("No experiments matched. Use --list to see all available.")
        sys.exit(1)
 
    # ---- Apply CLI nlive_mode override to every selected experiment ----
    # This must happen after filtering so we only mutate the configs that
    # will actually be run.  We deep-copy nothing extra — _build() already
    # deep-copied CONFIG when building each experiment, so mutating
    # cfg["nlive_mode"] here only affects this run's copy.
    if _cli_mode is not None:
        for _, cfg in selected:
            cfg["nlive_mode"] = _cli_mode

    # ---- Apply CLI progress-verbosity overrides ----
    # Each experiment's stdout is redirected to logs/<tag>.log, so these
    # control how much dynesty progress noise ends up in those files.
    for _, cfg in selected:
        if args.progress_interval is not None:
            cfg["progress_interval"] = args.progress_interval
        if args.quiet:
            cfg["verbose"] = False
 
    # ---- Skip experiments already in the registry (unless --rerun) ----
    # Read every run_tag recorded in any registry CSV referenced by the
    # selected experiments.  If the tag already appears, skip it.
    if not args.rerun:
        import csv as _csv
        _done_tags = set()
        _registry_paths = set(cfg.get("registry_file", CONFIG.get("registry_file", ""))
                               for _, cfg in selected)
        for _rpath in _registry_paths:
            if _rpath and os.path.isfile(_rpath):
                try:
                    with open(_rpath, newline="") as _f:
                        _reader = _csv.DictReader(_f)
                        for _row in _reader:
                            _tag = _row.get("run_tag") or _row.get("run_name") or ""
                            if _tag:
                                _done_tags.add(_tag.strip())
                except Exception:
                    pass  # unreadable registry — leave selected unchanged

        if _done_tags:
            _skipped = [(i, cfg) for i, cfg in selected if cfg["run_tag"] in _done_tags]
            selected  = [(i, cfg) for i, cfg in selected if cfg["run_tag"] not in _done_tags]
            if _skipped:
                print(f"\n[skip] {len(_skipped)} experiment(s) already in registry "
                      f"(use --rerun to force):")
                for _si, _scfg in _skipped:
                    print(f"  [{_si:>2}]  {_scfg['run_tag']}")

    if not selected:
        print("\nNo experiments to run (all already completed).  Use --rerun to force.")
        sys.exit(0)

    n_workers = min(
        args.workers or len(selected),
        os.cpu_count() or 1,
    )
    if args.sequential:
        n_workers = 1
 
    log_dir = args.log_dir
    mode_label = _cli_mode or "per-experiment"
 
    print(f"\n{'='*60}")
    print(f"Experiments : {len(selected)}")
    print(f"nlive mode  : {mode_label}")
    print(f"Workers     : {n_workers}  (cores available: {os.cpu_count()})")
    print(f"Log dir     : {os.path.abspath(log_dir)}/")
    print(f"{'='*60}")
    for i, cfg in selected:
        n = sum(1 for s in cfg["param_specs"].values() if s["active"])
        print(f"  [{i:>2}]  {cfg['run_tag']:<45}  {n} params  nlive={_nlive_display(cfg)}")
    print(f"{'='*60}\n")
 
    if args.dry_run:
        print("Dry run — exiting without sampling.")
        sys.exit(0)
 
    print(f"Logs are written to {os.path.abspath(log_dir)}/<tag>.log")
    print(f"Monitor a run with:  tail -f {log_dir}/<tag>.log\n")
 
    # ---- Master summary log ----
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "summary_kerr.log")
    summary = open(summary_path, "w", buffering=1)
    summary.write(f"Run started: {datetime.now().isoformat()}\n")
    summary.write(f"Experiments: {len(selected)}  Workers: {n_workers}  nlive mode: {mode_label}\n\n")
 
    # ---- Dispatch ----
    work = [(i, cfg, log_dir) for i, cfg in selected]
    results = []
 
    if n_workers == 1:
        # Sequential — useful for debugging or single-core servers
        for item in work:
            r = _run_one(item)
            results.append(r)
            idx, tag, status, elapsed, _ = r
            line = f"[{status.upper():>6}]  [{idx:>2}]  {tag:<45}  {elapsed:7.1f}s\n"
            print(line, end="")
            summary.write(line)
            summary.flush()
    else:
        # Use "spawn" instead of the default "fork" on Linux.
        #
        # Why: fork() copies the parent's entire memory space including any
        # already-initialised OpenBLAS/OMP thread pools.  When those pools
        # then try to synchronise across the fork boundary they can deadlock
        # or receive SIGKILL from the OS — which is the silent
        # "BrokenProcessPool" crash you see.  Spawn starts a clean Python
        # interpreter for each worker, imports from scratch, and avoids
        # all fork-safety issues with multi-threaded C libraries.
        #
        # Trade-off: spawn has ~1–2s startup overhead per worker (importing
        # numpy, scipy, dynesty).  For long-running nested sampling jobs
        # this is completely negligible.
        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")
 
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx) as pool:
            futures = {pool.submit(_run_one, item): item[0] for item in work}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception as exc:
                    # Worker process died with an unrecoverable error (e.g. OOM,
                    # signal).  Record it as failed rather than crashing the parent.
                    item_idx = futures[fut]
                    item_tag = next(cfg["run_tag"] for i, cfg in selected if i == item_idx)
                    elapsed  = 0.0
                    tb       = f"{type(exc).__name__}: {exc}"
                    print(f"\n[CRASH]  [{item_idx:>2}]  {item_tag}  —  {tb}", flush=True)
                    r = (item_idx, item_tag, "failed", elapsed, tb)
                results.append(r)
                idx, tag, status, elapsed, _ = r
                line = (f"[{status.upper():>6}]  [{idx:>2}]  "
                        f"{tag:<45}  {elapsed:7.1f}s\n")
                print(line, end="")
                summary.write(line)
                summary.flush()
 
    # ---- Final summary ----
    ok     = [r for r in results if r[2] == "ok"]
    failed = [r for r in results if r[2] == "failed"]
    footer = (f"\n{'='*60}\n"
              f"Finished {len(ok)}/{len(results)} experiments successfully.\n")
    if failed:
        footer += "Failed:\n"
        for idx, tag, _, elapsed, err in failed:
            first_line = err.strip().splitlines()[-1] if err else "unknown"
            footer += f"  [{idx}] {tag}: {first_line}\n"
    footer += f"{'='*60}\n"
 
    print(footer)
    summary.write(footer)
    summary.close()
    print(f"Full summary written to: {summary_path}")
 
 
if __name__ == "__main__":
    main()