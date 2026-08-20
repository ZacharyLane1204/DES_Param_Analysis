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
from run    import run_sampler, pkl_path_for
from experiment_naming import ExperimentRegistry

# Post-processing checks (see --degeneracy-scan / --host-quality-check /
# --loo-zbins / --drilling-cones below) — imported lazily-looking but at
# module level like everything else here; they only get CALLED if the
# corresponding CLI flag is set, so importing them unconditionally costs
# nothing when those flags are off.
import degeneracy_scan
import host_match_quality
import loo_zbins
import drilling_cones

# ===========================================================================
# HELPERS
# ===========================================================================
# _override()/_build() now live in experiment_naming.py, shared with
# experiment_runner.py and combo_ablation_checks.py, instead of being
# copy-pasted per file (this copy and experiment_runner.py's were
# previously byte-identical). See that module's docstring for the
# duplicate-tag / duplicate-config / degenerate-parameter guards this
# gives every runner automatically.
_registry = ExperimentRegistry(CONFIG, DEFAULT_PARAM_SPECS)
_build    = _registry.build

# ===========================================================================
# EXPERIMENT DEFINITIONS
# ===========================================================================
# Each entry is a call to _build().  Add / remove entries freely.
# The tag becomes part of the run name and the registry CSV.
#
# Convention used here:
#   cosmology/    — cosmological model variants
#   nuisance/     — SALT2 nuisance parameter variants
#   sn_col_model/ — SN colour / mass / host-colour model variants
#   sampler/      — sampler setting variants
#   mass/         — mass step functional form variants
#   interaction/  — interaction term variants
#
# You can use any tag scheme you like; these are just strings.
# ===========================================================================

EXPERIMENTS = [
 
            # Standard
            _build("checks/std_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0},
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "w": {"active": False, "fixed": -1}}),
            
            _build("checks/std_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),            

            _build("checks/std_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),  

            _build("checks/std_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),  
            
            # standard mass
            
            _build("checks/std_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),            

            _build("checks/std_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),  

            _build("checks/std_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),              
            
            # standard mass and ssfr
            
            _build("checks/std_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),            

            _build("checks/std_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),  

            _build("checks/std_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": False, "fixed": -1}}),         
            
            # WCDM
            _build("checks/wcdm_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0},
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "w": {"active": True, "fixed": None}}),
            
            _build("checks/wcdm_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),            

            _build("checks/wcdm_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),  

            _build("checks/wcdm_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),  
            
            # WCDM mass
            
            _build("checks/wcdm_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),            

            _build("checks/wcdm_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),  

            _build("checks/wcdm_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),              
            
            # WCDM mass and ssfr
            
            _build("checks/wcdm_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),            

            _build("checks/wcdm_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),  

            _build("checks/wcdm_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "w": {"active": True, "fixed": None}}),     
            
            # LCDM
            _build("checks/lcdm_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0},
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),
            
            _build("checks/lcdm_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),            

            _build("checks/lcdm_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),  

            _build("checks/lcdm_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),  
            
            # LCDM mass
            
            _build("checks/lcdm_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),            

            _build("checks/lcdm_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),  

            _build("checks/lcdm_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),     
            
            # LCDM mass and ssfr
            
            _build("checks/lcdm_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),            

            _build("checks/lcdm_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),  

            _build("checks/lcdm_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "Ode0": {"active": True, "fixed": 0.6824}}),                
            
            # z Low
            _build("checks/zlow_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zlo": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "c0": {"active": False, "fixed": 0}}),
            
            _build("checks/zlow_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zlo": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/zlow_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zlo": 0.1},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/zlow_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zlo": 0.1},
                   param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),
            
            _build("checks/zlow_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "zlo": 0.1}),

            # z Low mass
            
            _build("checks/zlow_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zlo": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/zlow_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zlo": 0.1},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/zlow_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", }, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zlo": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None},
                                    "gamma": {"active": True, "fixed": 0.0}}),

            # z low mass and ssfr
            
            _build("checks/zlow_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "zlo": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),            

            _build("checks/zlow_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "zlo": 0.1},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),  

            _build("checks/zlow_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "zlo": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),    

            # z high
            _build("checks/zhi_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zhi": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "c0": {"active": False, "fixed": 0}}),
            
            _build("checks/zhi_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zhi": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/zhi_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zhi": 0.1},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/zhi_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zhi": 0.1},
                   param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),
            
            _build("checks/zhi_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "zhi": 0.1}),

            # z high mass
            
            _build("checks/zhi_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zhi": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/zhi_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zhi": 0.1},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/zhi_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", }, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "zhi": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None},
                                    "gamma": {"active": True, "fixed": 0.0}}),
            
            # z high mass and ssfr
            
            _build("checks/zhi_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "zhi": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),            

            _build("checks/zhi_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "zhi": 0.1},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),  

            _build("checks/zhi_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "zhi": 0.1},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),             
            
            # ID Survey
            _build("checks/id_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "idsurvey": True},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "c0": {"active": False, "fixed": 0}}),
            
            _build("checks/id_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "idsurvey": True},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/id_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "idsurvey": True},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/id_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "idsurvey": True},
                   param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),
            
            _build("checks/id_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "idsurvey": True}),

            # ID Survey mass
            
            _build("checks/id_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "idsurvey": True},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/id_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "idsurvey": True},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/id_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", }, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "idsurvey": True},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None},
                                    "gamma": {"active": True, "fixed": 0.0}}),
            
            # ID Survey mass and ssfr
            
            _build("checks/id_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "idsurvey": True},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),            

            _build("checks/id_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "idsurvey": True},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),  

            _build("checks/id_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "idsurvey": True},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),               

            # Mass low
            _build("checks/masscut_low_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "low"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "c0": {"active": False, "fixed": 0}}),
            
            _build("checks/masscut_low_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "low"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/masscut_low_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "low"},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/masscut_low_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "low"},
                   param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),
            
            _build("checks/masscut_low_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "mass_cut": "low"}),

            # Mass cut low mass linear
            
            _build("checks/masscut_low_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "low"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/masscut_low_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "low"},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/masscut_low_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", }, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "low"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None},
                                    "gamma": {"active": True, "fixed": 0.0}}), 
            
            # Mass low mass and ssfr
            
            _build("checks/masscut_low_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "mass_cut": "low"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),            

            _build("checks/masscut_low_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "mass_cut": "low"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),  

            _build("checks/masscut_low_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "mass_cut": "low"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),                   
            
            # Mass cut high
            _build("checks/masscut_high_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "high"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "c0": {"active": False, "fixed": 0}}),
            
            _build("checks/masscut_high_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "high"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/masscut_high_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "high"},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/masscut_high_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "high"},
                   param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),
            
            _build("checks/masscut_high_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "mass_cut": "high"}),

            # ---- Host redshift observation type (spec-z vs photo-z host) ----
            # NOTE: on the DES-Dovekie metadata used during development, the
            # photometric-only subsample is only ~8 SNe (1615 have host
            # spec-z, 197 have no valid host position at all) — far too few
            # to fit meaningfully on its own. checks/photz_baseline is left
            # in for completeness / future larger samples but expect it to
            # fail or return a garbage posterior on the current dataset;
            # check n_heldout-style SNe counts in the log before trusting it.
            _build("checks/specz_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "obs_z_type": "spec"}),
            _build("checks/photz_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "obs_z_type": "phot"}),

            # ---- Host-match quality (strict DDLR/CONFUSION/NMATCH cut) ----
            # See host_match_quality.py for the paired "all" vs "strict"
            # comparison via compare_runs — this EXPERIMENTS entry alone
            # only gives you the "strict" fit; run host_match_quality.py
            # (or `--host-quality-check` below) for the actual tension
            # comparison against your normal ("all") baseline.
            # host_ddlr_max is pinned explicitly to 2.0 here (the same value
            # config.py now carries and host_match_quality.DDLR_MAX uses) so
            # this row states the threshold it was run at rather than
            # inheriting whatever CONFIG happens to hold.
            _build("checks/hostquality_strict_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "host_quality_cut": "strict", "host_ddlr_max": 2.0}),

            # Mass cut high mass linear
            
            _build("checks/masscut_high_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "high"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/masscut_high_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "high"},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/masscut_high_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", }, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "mass_cut": "high"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None},
                                    "gamma": {"active": True, "fixed": 0.0}}),      
            
            # Mass high mass and ssfr
            
            _build("checks/masscut_high_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "mass_cut": "high"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),            

            _build("checks/masscut_high_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "mass_cut": "high"},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),  

            _build("checks/masscut_high_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "mass_cut": "high"},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),                  
            
            # x1 cuts
            _build("checks/x1cut_high_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "x1_range": [-2, 2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "c0": {"active": False, "fixed": 0}}),
            
            _build("checks/x1cut_high_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "x1_range": [-2, 2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/x1cut_high_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "x1_range": [-2, 2]},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/x1cut_high_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "x1_range": [-2, 2]},
                   param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),
            
            _build("checks/x1cut_high_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "x1_range": [-2, 2]}),

            # x1 cut high mass linear
            
            _build("checks/x1cut_high_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "x1_range": [-2, 2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/x1cut_high_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "x1_range": [-2, 2]},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/x1cut_high_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", }, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "x1_range": [-2, 2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None},
                                    "gamma": {"active": True, "fixed": 0.0}}),    
            
            # x1 cut low mass and ssfr
            
            _build("checks/x1cut_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "x1_range": [-2, 2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),            

            _build("checks/x1cut_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "x1_range": [-2, 2]},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),  

            _build("checks/x1cut_low_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "x1_range": [-2, 2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),                  
            
            # c cuts
            _build("checks/ccut_high_gamma_alpha_sncolour_softbrokensntau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "c_range": [-0.2, 0.2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "c0": {"active": False, "fixed": 0}}),
            
            _build("checks/ccut_high_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "c_range": [-0.2, 0.2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/ccut_high_sncolour_softbrokensntau_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "none", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "c_range": [-0.2, 0.2]},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/ccut_high_gamma_alpha_ssfr_tanhF0ftau", config_overrides={"model": {**CONFIG["model"], "mass": "none", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "c_range": [-0.2, 0.2]},
                   param_overrides={"zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": False, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),
            
            _build("checks/ccut_high_baseline", config_overrides={"registry_file": "run_checks_registry.csv", "c_range": [-0.2, 0.2]}),

            # c cut high mass linear
            
            _build("checks/ccut_high_gamma_alpha_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "c_range": [-0.2, 0.2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/ccut_high_sncolour_softbrokensntau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear"}, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "c_range": [-0.2, 0.2]},
                   param_overrides={"sn_tau": {"active": True, "fixed": 0.3},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "c0": {"active": False, "fixed": 0}}),

            _build("checks/ccut_high_gamma_alpha_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", }, 
                                                                                   "registry_file": "run_checks_registry.csv", 
                                                                                   "c_range": [-0.2, 0.2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None},
                                    "gamma": {"active": True, "fixed": 0.0}}),  

            # c cut low mass and ssfr
            
            _build("checks/ccut_gamma_alpha_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "c_range": [-0.2, 0.2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),            

            _build("checks/ccut_sncolour_softbrokensntau_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "sn_colour": "softbroken", 
                                                                                                      "mass": "linear", "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "c_range": [-0.2, 0.2]},
                   param_overrides={"c0": {"active": False, "fixed": 0}, 
                                    "sn_tau": {"active": True, "fixed": 0.3},
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),  

            _build("checks/ccut_low_gamma_alpha_ssfr_tanhF0ftau_mass_linear", config_overrides={"model": {**CONFIG["model"], "mass": "linear", 
                                                                                      "ssfr": "tanh", "host_colour": "none"}, 
                                                                                   "registry_file": "run_checks_registry.csv", "c_range": [-0.2, 0.2]},
                   param_overrides={"gamma_alpha": {"active": True, "fixed": None}, 
                                    "c0": {"active": False, "fixed": 0}, 
                                    "zeta":  {"active": True, "fixed": 0.0},
                                    "F0":    {"active": True, "fixed": -10.5},
                                    "ftau":  {"active": True, "fixed": 0.5},
                                    "gamma": {"active": True, "fixed": 0.0},
                                    "eta":   {"active": False, "fixed": 0.0}}),        

            # NOTE: uniform-prior / degeneracy-prescan checks (varying
            # alpha/beta/Om0/M0's prior shape rather than the best model's
            # own robustness under a fixed model) intentionally do NOT live
            # here. "checks/" in this file is reserved for post-hoc
            # robustness checks on the chosen best model (this section, the
            # c-cut variants above, etc.) -- see uniform_priors_check.py for
            # the prior-shrinkage/degeneracy-driven uniform-prior reruns.
    ] 

# ===========================================================================
# HOST MEASUREMENT ERROR SYSTEMATIC CHECK
# ===========================================================================
# Matched-pair checks on how the host mass / host colour / sSFR measurement
# errors are treated.  Every entry below fits the SAME model on the SAME
# 1820 SNe and changes only the error treatment, so the lnZ differences are
# attributable to that choice alone.
#
# The three switches under test (all defined in config.py):
#
#   host_colour_err_from_logmass
#       HOST_COLOR_ERR is -999 for every SN in the DES metadata.  Left alone,
#       the host colour would be the only host property treated as exactly
#       measured while mass and sSFR are smoothed by their errors -- an
#       asymmetry that flatters the host colour models.  With this on, the
#       colour error is derived as HOST_LOGMASS_ERR / slope, the slope coming
#       from the Taylor+2011 mass-to-light/colour relation.
#
#   ssfr_err_max
#       HOST_LOGsSFR_ERR is bimodal, with a failure-mode pileup near 10 dex
#       (larger than the entire ~2.4 dex population spread).  Above this
#       threshold the sSFR point estimate is masked to NaN.  The SN is kept
#       in the sample so evidences stay comparable.
#
#   host_var_penalty
#       Adds Var[f] to the covariance diagonal, i.e. accounts for the extra
#       SCATTER the host measurement error injects into mu, not just the bias
#       correction that the quadrature already applies.  This is the
#       expensive one: the covariance becomes parameter dependent and must be
#       refactorised on every likelihood call.  Expect these runs to take
#       substantially longer than their reference twin.
#
# ---------------------------------------------------------------------------
# EDIT THIS after model_comparison_suite.py has picked the best model.
# It must describe exactly one model -- the same combination you feed to
# combo_ablation_checks.py -- so the pairs below are true like-for-like.
# ---------------------------------------------------------------------------
HOSTERR_BEST = {
    "label": "best",
    # Model selection (merged on top of CONFIG["model"]).
    "model": {"sn_colour":   "softbroken",
              "mass":        "linear",
              "host_colour": "linear",
              "ssfr":        "tanh"},
    # Parameter activations for that model.
    "param_overrides": {
        "gamma_alpha": {"active": True,  "fixed": None},
        "c0":          {"active": False, "fixed": 0},
        "sn_tau":      {"active": True,  "fixed": 0.3},
        "gamma":       {"active": True,  "fixed": 0.0},
        "eta":         {"active": True,  "fixed": 0.0},
        "zeta":        {"active": True,  "fixed": 0.0},
        "F0":          {"active": True,  "fixed": -10.5},
        "ftau":        {"active": True,  "fixed": 0.5},
        "w":           {"active": False, "fixed": -1},
    },
}

_HOSTERR_REGISTRY = "run_checks_registry.csv"

# (tag suffix, config overrides on top of the reference treatment, comment)
# The reference treatment is whatever config.py defaults to, which is:
#   host_colour_err_from_logmass=True, ssfr_err_max=2.5, host_var_penalty=False
_HOSTERR_VARIANTS = [
    # --- reference -------------------------------------------------------
    ("ref",              {}),
    # --- the expensive variance term, on ---------------------------------
    ("varpen",           {"host_var_penalty": True}),
    # --- isolate the host colour asymmetry -------------------------------
    ("nocolourerr",      {"host_colour_err_from_logmass": False}),
    ("nocolourerr_varpen", {"host_colour_err_from_logmass": False,
                            "host_var_penalty": True}),
    # --- sensitivity to the assumed Taylor+2011 slope --------------------
    # The literature spans roughly 0.5-1.15; a smaller slope means a LARGER
    # derived colour error, so 0.50 is the pessimistic end.
    ("slope050",         {"host_colour_err_mass_slope": 0.50}),
    ("slope115",         {"host_colour_err_mass_slope": 1.15}),
    # --- sensitivity to the sSFR mask threshold --------------------------
    ("ssfrmask20",       {"ssfr_err_max": 2.0}),
    ("ssfrmask30",       {"ssfr_err_max": 3.0}),
    ("nossfrmask",       {"ssfr_err_max": None}),
    # --- all host measurement error switched off (pre-existing behaviour) -
    # Point estimates only: no smoothing, no derived colour error, no mask.
    ("noerrors",         {"col_logM_err": None,
                          "col_host_colour_err": None,
                          "col_logsSFR_err": None,
                          "host_colour_err_from_logmass": False,
                          "ssfr_err_max": None}),
    # --- quadrature convergence -------------------------------------------
    # Discontinuous profiles (mass/ssfr "step") converge slowly under
    # Gauss-Hermite quadrature, and the second moment converges more slowly
    # than the first.  If the best model uses a step, compare this against
    # "ref" (and against "varpen" for the variance term) before trusting the
    # host-error deltas at the 0.1 lnZ level.
    ("gh80",             {"n_gh_nodes": 80}),
    ("gh80_varpen",      {"n_gh_nodes": 80, "host_var_penalty": True}),
]


def _host_error_checks():
    """Build the matched-pair host measurement error checks for the best model."""
    best   = HOSTERR_BEST
    label  = best["label"]
    out    = []
    for suffix, overrides in _HOSTERR_VARIANTS:
        cfg_over = {"model": {**CONFIG["model"], **best["model"]},
                    "registry_file": _HOSTERR_REGISTRY}
        cfg_over.update(overrides)
        out.append(_build(f"hosterr/{label}_{suffix}",
                          config_overrides=cfg_over,
                          param_overrides=best["param_overrides"]))
    return out


EXPERIMENTS += _host_error_checks()

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
    p.add_argument("--sequential", action="store_true",
                   help="Disable parallelism — run one at a time (useful for debugging)")
    # nlive mode — mutually exclusive; if neither is given the mode stored in
    # each experiment's config dict is used (default: "exploratory").
    nlive_group = p.add_mutually_exclusive_group()
    nlive_group.add_argument("--publication", action="store_true",
                             help="Override nlive_mode to 'publication' for all selected "
                                  "experiments (ndim x 500 live points)")
    nlive_group.add_argument("--explore", action="store_true",
                             help="Override nlive_mode to 'exploratory' for all selected "
                                  "experiments (ndim x 50 live points)")

    # ---- Post-processing checks (run AFTER the normal fit(s) above) ----
    # These operate on the selected experiments' saved pickles (or, for
    # host-quality/loo-zbins/drilling-cones, run additional sub-fits of
    # their own) rather than being additional EXPERIMENTS grid entries —
    # see compare_runs.py / degeneracy_scan.py / host_match_quality.py /
    # loo_zbins.py / drilling_cones.py docstrings for what each does.
    # All are opt-in and off by default so a normal sweep is unaffected.
    p.add_argument("--degeneracy-scan", action="store_true",
                   help="After fitting, run a posterior correlation/"
                        "degeneracy scan on each selected experiment's "
                        "saved pickle (degeneracy_scan.py).")
    p.add_argument("--host-quality-check", action="store_true",
                   help="After fitting, additionally fit each selected "
                        "experiment with host_quality_cut='strict' and "
                        "report the tension against the normal fit "
                        "(host_match_quality.py). Adds a second fit per "
                        "selected experiment — expect roughly double the "
                        "runtime for whichever experiments you select.")
    p.add_argument("--loo-zbins", type=int, default=0, metavar="N_BINS",
                   help="After fitting, run leave-one-redshift-bin-out "
                        "cross-validation with N_BINS bins on each selected "
                        "experiment (loo_zbins.py). Adds N_BINS additional "
                        "fits per selected experiment.")
    p.add_argument("--drilling-cones", action="store_true",
                   help="After fitting, run the sky-position line-of-sight "
                        "systematic check on each selected experiment "
                        "(drilling_cones.py) — forces drilling_cones=True "
                        "for this invocation regardless of each "
                        "experiment's own config. Adds one fit per sky "
                        "cluster found, per selected experiment.")
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

    # ---- Post-processing checks ----
    # Only run these on experiments that actually finished successfully —
    # a failed fit has no saved pickle for degeneracy_scan to read, and no
    # sensible baseline for host_quality_check/loo_zbins/drilling_cones to
    # build on top of.
    any_postproc = (args.degeneracy_scan or args.host_quality_check
                    or args.loo_zbins or args.drilling_cones)
    if any_postproc:
        ok_tags = {tag for _, tag, status, _, _ in results if status == "ok"}
        ok_selected = [(i, cfg) for i, cfg in selected if cfg["run_tag"] in ok_tags]
        if not ok_selected:
            print("\nNo successful experiments to post-process — skipping "
                 "--degeneracy-scan/--host-quality-check/--loo-zbins/"
                 "--drilling-cones.")
        print(f"\n{'='*60}\nPost-processing checks on {len(ok_selected)} "
             f"successful experiment(s)\n{'='*60}")

    if args.degeneracy_scan:
        for i, cfg in ok_selected:
            pkl = pkl_path_for(cfg["run_tag"], cfg)
            print(f"\n--- degeneracy_scan: [{i}] {cfg['run_tag']} ---")
            try:
                degeneracy_scan.scan_degeneracies(pkl)
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")

    if args.host_quality_check:
        for i, cfg in ok_selected:
            print(f"\n--- host_match_quality: [{i}] {cfg['run_tag']} ---")
            try:
                host_match_quality.run_host_quality_check(
                    config_overrides=dict(cfg),
                    output_prefix=f"{cfg['run_tag'].replace('/', '_')}_host_quality")
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")

    if args.loo_zbins:
        for i, cfg in ok_selected:
            print(f"\n--- loo_zbins (n_bins={args.loo_zbins}): "
                 f"[{i}] {cfg['run_tag']} ---")
            try:
                loo_zbins.run_loo_zbins(
                    config_overrides=dict(cfg), n_bins=args.loo_zbins,
                    output_prefix=f"{cfg['run_tag'].replace('/', '_')}_loo")
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")

    if args.drilling_cones:
        for i, cfg in ok_selected:
            print(f"\n--- drilling_cones: [{i}] {cfg['run_tag']} ---")
            try:
                drilling_cones.run_drilling_cones(
                    config_overrides={**dict(cfg), "drilling_cones": True},
                    output_prefix=f"{cfg['run_tag'].replace('/', '_')}_cones")
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()