"""
extra_runners_sSFR.py  —  SNe Ia Cosmology Pipeline
=====================================================
sSFR (specific star formation rate) experiment suite.

Mirrors the structure of the host_col_model section in experiment_runner.py,
but specifically for testing the sSFR host correction term and its combinations
with the existing mass (S) and host-colour (H) terms.

The full host environment correction is:
    G = gamma/2 * S  +  eta * H           +  xi_mass_col  * S*H
      + zeta   * F    +  xi_sSFR_col * F*H  +  xi_sSFR_mass * F*S  +  omega * F*S*H

where F = sSFR profile function (ssfr model), S = mass profile, H = host-colour
profile.  Each group below progressively adds complexity:

Sections
--------
  ssfr/alone/         — F term in isolation (no mass or host-colour coupling)
  ssfr/with_mass/     — F alongside mass step S (zeta only, then + xi_sSFR_mass)
  ssfr/with_hcol/     — F alongside host-colour H (zeta only, then + xi_sSFR_col)
  ssfr/full_host/     — F + S + H simultaneously (zeta, xi_sSFR_mass, xi_sSFR_col)
  ssfr/interactions/  — Three- and four-way interaction terms (xi_sSFR_mass, xi_sSFR_col, omega)
  ssfr/replace_mass/  — F replaces S entirely (gamma fixed 0, zeta free)
  ssfr/replace_hcol/  — F replaces H entirely (eta fixed 0, zeta free)
  ssfr/replace_both/  — F replaces both S and H (gamma=eta=0, zeta free)
  ssfr/smooth_F0/     — F0 and ftau free (smooth model shape exploration)

Usage
-----
  # Run everything
  python extra_runners_sSFR.py

  # Run only the 'alone' group
  python extra_runners_sSFR.py --tag ssfr/alone

  # Dry-run to inspect the experiment list
  python extra_runners_sSFR.py --dry-run --list

  # Run in parallel across N cores
  python extra_runners_sSFR.py --workers 8

  # Run in publication mode
  python extra_runners_sSFR.py --publication
"""

import copy
import argparse
import sys
import os
from datetime import datetime

# ===========================================================================
# THREAD CLAMPING  —  must happen BEFORE any numerical library is imported
# ===========================================================================
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_var] = "1"

try:
    from threadpoolctl import threadpool_limits as _tpl
    _tpl(1)
except Exception:
    pass
# ===========================================================================

from config import CONFIG, DEFAULT_PARAM_SPECS
from run import run_sampler

# ===========================================================================
# HELPERS  (identical to experiment_runner.py)
# ===========================================================================

def _override(base_specs, **param_overrides):
    specs = copy.deepcopy(base_specs)
    for name, updates in param_overrides.items():
        specs[name].update(updates)
    return specs


def _build(tag, param_overrides=None, config_overrides=None):
    cfg = copy.deepcopy(CONFIG)
    cfg["run_tag"]     = tag
    cfg["param_specs"] = _override(DEFAULT_PARAM_SPECS, **(param_overrides or {}))
    if config_overrides:
        cfg.update(config_overrides)
    return cfg


# ---------------------------------------------------------------------------
# Shorthand model-dict helpers
# ---------------------------------------------------------------------------
# These keep the _build() calls below compact.  Each returns a complete
# model dict by spreading CONFIG["model"] and overriding the relevant keys,
# exactly as done in experiment_runner.py.

def _M(ssfr="none", mass=None, host_colour=None, sn_colour=None):
    """Return a model dict with the given overrides on top of CONFIG['model']."""
    m = dict(CONFIG["model"])
    m["ssfr"] = ssfr
    if mass        is not None: m["mass"]        = mass
    if host_colour is not None: m["host_colour"] = host_colour
    if sn_colour   is not None: m["sn_colour"]   = sn_colour
    return m

# Registry CSV dedicated to sSFR runs — keeps results separate from the
# main experiment_runner registry so Bayes-factor comparisons are clean.
_REG = {"registry_file": "run_ssfr_registry.csv"}

# ===========================================================================
# EXPERIMENT DEFINITIONS
# ===========================================================================

EXPERIMENTS = [

    # =========================================================================
    # 0.  REFERENCE BASELINE
    #     The standard baseline (no sSFR term) re-run into the sSFR registry
    #     so every group has a common logZ anchor for evidence ratios.
    # =========================================================================

    _build("ssfr/reference/baseline",
           config_overrides=_REG),

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
    _build("ssfr/ssfr_tanh_F0_ftau",
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

    _build("ssfr/ssfr_sigmoid_F0_ftau",
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

    _build("ssfr/ssfr_tanh_massstep_F0_ftau",
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

    _build("ssfr/ssfr_tanh_massstep_xi_sSFR_mass_F0_ftau",
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

    _build("ssfr/ssfr_tanh_massstep_hcol_linear_F0_ftau",
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

    _build("ssfr/ssfr_sigmoid_massstep_hcol_linear_F0_ftau",
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
    _build("ssfr/ssfr_full_expansion_F0_M0",
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

    _build("ssfr/ssfr_tanh_nomass_F0_ftau",
           config_overrides={**_REG, "model": _M(ssfr="tanh", mass="none")},
           param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                            "F0":    {"active": True, "fixed": -10.5},
                            "ftau":  {"active": True, "fixed": 0.5},
                            "gamma": {"active": False, "fixed": 0.0}}),

    _build("ssfr/ssfr_sigmoid_nomass_F0_ftau",
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

    _build("ssfr/ssfr_tanh_nohcol_F0_ftau",
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

    _build("ssfr/ssfr_replace_both_tanh_nomass_nohcol_F0_ftau",
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

    # Vary F0 prior — broader range to check whether -10.5 is correct
    _build("ssfr/ssfr_smooth_F0_step_broadF0",
           config_overrides={**_REG, "model": _M(ssfr="step")},
           param_overrides={"zeta": {"active": True, "fixed": 0.0},
                            "F0":   {"active": True, "prior": "truncated_gaussian",
                                     "range": [-13.0, -8.0], "mu": -10.5, "sigma": 1.0,
                                     "fixed": -10.5}}),

    # tanh with ftau very free — distinguishes sharp vs smooth transition
    _build("ssfr/ssfr_smooth_F0_tanh_wide_ftau",
           config_overrides={**_REG, "model": _M(ssfr="tanh")},
           param_overrides={"zeta": {"active": True, "fixed": 0.0},
                            "F0":   {"active": True, "fixed": -10.5},
                            "ftau": {"active": True, "prior": "log_normal",
                                     "range": [0.05, 5.0], "mu": 0.0, "sigma": 1.0,
                                     "fixed": 0.5}}),

    # sigmoid with ftau very free
    _build("ssfr/ssfr_smooth_F0_sigmoid_wide_ftau",
           config_overrides={**_REG, "model": _M(ssfr="sigmoid")},
           param_overrides={"zeta": {"active": True, "fixed": 0.0},
                            "F0":   {"active": True, "fixed": -10.5},
                            "ftau": {"active": True, "prior": "log_normal",
                                     "range": [0.05, 5.0], "mu": 0.0, "sigma": 1.0,
                                     "fixed": 0.5}}),

]

# ===========================================================================
# RUNNER  (identical to experiment_runner.py)
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

    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[var] = "1"

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
    with open(log_path, "w", buffering=1) as log:
        log.write(f"=== [{idx}] {tag} ===\n")
        log.write(f"Started: {datetime.now().isoformat()}\n")
        log.write(f"PID: {os.getpid()}  CPU count: {os.cpu_count()}\n\n")
        log.flush()

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
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:
                log.write(f"\n=== FAILED after {elapsed:.1f}s ===\n{tb}\n")
                log.flush()
            except Exception:
                pass
            print(f"\n[worker {idx}] FAILED: {tb}", file=sys.stderr, flush=True)
            return (idx, tag, "failed", elapsed, tb)


def _parse_args():
    p = argparse.ArgumentParser(description="Run sSFR experiment suite")
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
    p.add_argument("--log-dir", default="logs_ssfr",
                   help="Directory for per-experiment log files (default: logs_ssfr/)")
    p.add_argument("--sequential", action="store_true",
                   help="Disable parallelism — run one at a time (useful for debugging)")
    nlive_group = p.add_mutually_exclusive_group()
    nlive_group.add_argument("--publication", action="store_true",
                             help="Override nlive_mode to 'publication' for all selected "
                                  "experiments (ndim x 300 live points)")
    nlive_group.add_argument("--explore", action="store_true",
                             help="Override nlive_mode to 'exploratory' for all selected "
                                  "experiments (ndim x 50 live points)")
    return p.parse_args()


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime

    args = _parse_args()

    if args.publication:
        _cli_mode = "publication"
    elif args.explore:
        _cli_mode = "exploratory"
    else:
        _cli_mode = None

    def _nlive_display(cfg):
        if cfg.get("nlive"):
            return int(cfg["nlive"])
        mode = _cli_mode or cfg.get("nlive_mode", "exploratory")
        n = sum(1 for s in cfg["param_specs"].values() if s["active"])
        return n * 300 if mode == "publication" else n * 50

    selected = list(enumerate(EXPERIMENTS))

    if args.list:
        mode_label = _cli_mode or "per-experiment"
        print(f"{'idx':>4}  {'tag':<55}  params  nlive  (mode: {mode_label})")
        print(f"{'---':>4}  {'---':<55}  ------  -----")
        for i, cfg in selected:
            n = sum(1 for s in cfg["param_specs"].values() if s["active"])
            print(f"{i:>4}  {cfg['run_tag']:<55}  {n:>6}  {_nlive_display(cfg)}")
        sys.exit(0)

    if args.index is not None:
        indices = _resolve_indices(args.index, len(EXPERIMENTS))
        selected = [(i, e) for i, e in selected if i in indices]

    if args.tag is not None:
        selected = [(i, e) for i, e in selected if args.tag in e["run_tag"]]

    if not selected:
        print("No experiments matched. Use --list to see all available.")
        sys.exit(1)

    if _cli_mode is not None:
        for _, cfg in selected:
            cfg["nlive_mode"] = _cli_mode

    n_workers = min(
        args.workers or len(selected),
        os.cpu_count() or 1,
    )
    if args.sequential:
        n_workers = 1

    log_dir    = args.log_dir
    mode_label = _cli_mode or "per-experiment"

    print(f"\n{'='*60}")
    print(f"sSFR Experiments : {len(selected)}")
    print(f"nlive mode       : {mode_label}")
    print(f"Workers          : {n_workers}  (cores available: {os.cpu_count()})")
    print(f"Log dir          : {os.path.abspath(log_dir)}/")
    print(f"Registry         : run_ssfr_registry.csv")
    print(f"{'='*60}")
    for i, cfg in selected:
        n = sum(1 for s in cfg["param_specs"].values() if s["active"])
        print(f"  [{i:>2}]  {cfg['run_tag']:<55}  {n} params  nlive={_nlive_display(cfg)}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry run — exiting without sampling.")
        sys.exit(0)

    print(f"Logs are written to {os.path.abspath(log_dir)}/<tag>.log")
    print(f"Monitor a run with:  tail -f {log_dir}/<tag>.log\n")

    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "summary_ssfr.log")
    summary = open(summary_path, "w", buffering=1)
    summary.write(f"Run started: {datetime.now().isoformat()}\n")
    summary.write(f"Experiments: {len(selected)}  Workers: {n_workers}  nlive mode: {mode_label}\n\n")

    work    = [(i, cfg, log_dir) for i, cfg in selected]
    results = []

    if n_workers == 1:
        for item in work:
            r = _run_one(item)
            results.append(r)
            idx, tag, status, elapsed, _ = r
            line = f"[{status.upper():>6}]  [{idx:>2}]  {tag:<55}  {elapsed:7.1f}s\n"
            print(line, end="")
            summary.write(line)
            summary.flush()
    else:
        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx) as pool:
            futures = {pool.submit(_run_one, item): item[0] for item in work}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception as exc:
                    item_idx = futures[fut]
                    item_tag = next(cfg["run_tag"] for i, cfg in selected if i == item_idx)
                    elapsed  = 0.0
                    tb       = f"{type(exc).__name__}: {exc}"
                    print(f"\n[CRASH]  [{item_idx:>2}]  {item_tag}  —  {tb}", flush=True)
                    r = (item_idx, item_tag, "failed", elapsed, tb)
                results.append(r)
                idx, tag, status, elapsed, _ = r
                line = (f"[{status.upper():>6}]  [{idx:>2}]  "
                        f"{tag:<55}  {elapsed:7.1f}s\n")
                print(line, end="")
                summary.write(line)
                summary.flush()

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