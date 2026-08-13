"""
config.py  —  SNe Ia Cosmology Pipeline
========================================
This is the ONLY file you should need to edit between runs.

Contents
--------
  DEFAULT_PARAM_SPECS   : prior definitions for every parameter
  PARAM_DISPLAY         : LaTeX labels and sig-fig counts for plots
  CONFIG                : data paths, model choices, sampler settings

How to use
----------
  # Run with defaults
  python run.py

  # Override a single parameter prior in-place
  from config import CONFIG, DEFAULT_PARAM_SPECS
  import copy
  cfg = copy.deepcopy(CONFIG)
  cfg["param_specs"]["a"]["active"] = True
  # then pass cfg to run_sampler(cfg)

Changes from previous version
------------------------------
  - sn_tau added: scale parameter for the tanh and softbroken SN colour
    models.  log_normal prior peaking around 0.3 mag (typical colour range)
    with long right tail so the linear limit (large sn_tau) is always
    accessible.

  - tau / htau priors changed from log_uniform to log_normal.
    Motivation: log_uniform gives equal weight per decade and does not
    disfavour either very small or very large values within the range.
    For tau and htau the physically motivated scale is ~0.2–0.5 dex
    (a gradual but not negligible transition).  A log_normal prior
    centred on log(0.3) ≈ -1.2 concentrates probability in that region
    while still giving non-negligible weight to the extremes of the range,
    avoiding artificial pressure toward either a hard step (tau → 0) or
    complete smearing (tau → large).  The range [0.02, 5.0] remains as a
    hard clip so the sampler never wastes time in the pathological tails.

  - z_evolve "linear" model added to CONFIG model options (see core.py).
    Simply a first-order Taylor expansion around z_pivot; the cleanest
    parametrisation for detecting a slow linear trend.

  - Z_PIVOT removed from CONFIG.  The pivot is now computed automatically
    from the loaded dataset (median redshift) in run.py and written into
    core.Z_PIVOT_RUNTIME.  Removing it from config prevents the common
    mistake of forgetting to update it when switching datasets.

  - x1_range / c_range added: optional [lo, hi] cuts on SALT2 stretch (x1)
    and colour (c) applied after all other data filters.  The covariance
    matrix is subsetted to exactly the surviving rows, so the full statistical
    covariance is preserved.  Set to None (default) to apply no cut.

  - x1_correction model added: a new model registry (X1_CORRECTION_MODELS)
    in core.py mirrors the SN_COLOUR_MODELS pattern, allowing nonlinear
    stretch corrections.  The centred-quadratic model is included as the
    natural counterpart to sn_colour_quadratic; additional models test for
    saturation and asymmetric response.  The associated parameters (x1_0,
    x1_tau) follow the same prior conventions as their colour analogues.

  - sSFR host term added: specific star formation rate (HOST_LOGsSFR) enters
    the host environment correction as F (the sSFR profile function).  The
    complete host term is:
        G = gamma/2 * S + eta * H + xi_mass_col * S*H
          + zeta * F + xi_sSFR_col * F*H + xi_sSFR_mass * F*S + omega * F*S*H
    New parameters: zeta, xi_sSFR_col, xi_sSFR_mass, omega (arcsinh priors),
    and the sSFR model shape parameters F0, ftau.  The sSFR column and
    error column are configured via col_logsSFR / col_logsSFR_err in CONFIG.

  - Host-property measurement-error marginalisation added.  Host mass,
    host colour, and sSFR were previously fed into the mass/host_colour/ssfr
    profile functions as bare point estimates, with zero measurement
    uncertainty — only the SN-side MUERR ever entered the covariance.
    Their point estimate is now replaced with its expectation under Gaussian
    measurement error via fixed Gauss-Hermite quadrature nodes, computed
    once at data-load time in run.py and consumed by core.compute_mu_corr.
    This does NOT touch the covariance matrix (inv_cov_mat / C_sum / the
    one-time Cholesky inversion are all unchanged), so it adds negligible
    runtime cost and is exact for a Gaussian error model, unlike inflating
    the diagonal with a fixed local-slope approximation.
    New CONFIG keys: col_logM_err, col_host_colour_err (col_logsSFR_err
    already existed but was previously unused), and n_gh_nodes.

  - muerr_cut added: optional QC cut on each SN's diagonal covariance
    entry (sqrt(diag(cov_mat) + muerr**2)). Off by default (None); set to
    a float threshold (mag) to drop SNe at or above it. See
    load_and_filter_data in run.py.
"""

# ---------------------------------------------------------------------------
# 1.  PARAMETER PRIOR SPECS
#     Each entry:
#       "active"  : bool   — include in the sampler (True) or use fixed/default
#       "prior"   : str    — "uniform" | "gaussian" | "truncated_gaussian"
#                            "log_uniform" | "log_normal" | "arcsinh"
#       "range"   : [lo,hi]— hard bounds (hard clip for gaussian types)
#       "mu","sigma"       — centre / width for gaussian / truncated_gaussian /
#                            log_normal (log-space mean and std for log_normal)
#       "fixed"   : float|None
#                           None   → if active: sample it
#                                    if inactive: fall back to "mu" (or 0)
#                           number → if inactive: pin to this value
#                                    (ignored when active=True; sampler wins)
# ---------------------------------------------------------------------------

DEFAULT_PARAM_SPECS = {

    # ---- SALT2 nuisance ----
    "alpha": {"active": True,  "prior": "truncated_gaussian",
              "range": [0.04, 0.26], "mu": 0.17, "sigma": 0.05,
              "fixed": 0.17},

    "beta":  {"active": True,  "prior": "truncated_gaussian",
              "range": [1.5, 6.5],  "mu": 3.12,  "sigma": 0.5,
              "fixed": 3.12},

    "gamma": {"active": True,  "prior": "uniform",
              "range": [-0.2, 0.4],
              "fixed": 0.038},

    # M is analytically marginalised; keep here as reference / optional fixed.
    # Set "active": False to always marginalise (recommended).
    # Set "active": True only if you want to sample M directly.
    "M":     {"active": False, "prior": "gaussian",
              "range": [-35.0, -25.0], "mu": -29.99, "sigma": 0.75,
              "fixed": None},

    # ---- Cosmology ----
    "Om0":   {"active": True,  "prior": "truncated_gaussian",
              "range": [0.2, 0.5], "mu": 0.3175, "sigma": 0.0275,
              "fixed": 0.3175},

    "w":     {"active": False, "prior": "uniform",
              "range": [-2.0, -1/3],
              "fixed": -1.0},

    "Ode0":  {"active": False, "prior": "truncated_gaussian",
              "range": [0.5, 0.9], "mu": 0.6824, "sigma": 0.0275,
              "fixed": 0.6824},

    # ---- Redshift-evolution exponents ----
    # a → alpha evolution, b → beta evolution, g → gamma evolution
    "a":     {"active": False, "prior": "arcsinh",
              "range": [-20, 20], "scale": 0.3,
              "fixed": 0.0},

    "b":     {"active": False, "prior": "arcsinh",
              "range": [-20, 20], "scale": 0.3,
              "fixed": 0.0},

    "g":     {"active": False, "prior": "arcsinh",
              "range": [-20, 20], "scale": 0.3,
              "fixed": 0.0},

    # ---- Tier-3 interaction terms ----
    "beta_alpha":  {"active": False, "prior": "arcsinh",
                    "range": [-20, 20], "scale": 1.2,
                    "fixed": 0.0},

    "gamma_alpha": {"active": False, "prior": "arcsinh",
                    "range": [-20, 20], "scale": 1.2,
                    "fixed": 0},

    "beta_gamma":  {"active": False, "prior": "arcsinh",
                    "range": [-20, 20], "scale": 1.2,
                    "fixed": 0.0},

    # ---- Host galaxy environment ----
    "eta":   {"active": False, "prior": "arcsinh",
              "range": [-5.0, 5.0], "scale": 0.2,
              "fixed": 0.035},

    "xi_mass_col":    {"active": False, "prior": "arcsinh",
              "range": [-5.0, 5.0], "scale": 0.3,
              "fixed": 0.0},

    # Mass-step location. Kept as an informative prior (unlike C0) -- M0~10
    # has genuine external anchoring from the broader host-mass-step
    # literature, not just an internal convention. prior_shrinkage.py does
    # flag M0 as prior-dominated in 26/937 rows, but degeneracy_scan.py's
    # prescan shows this is concentrated specifically in xi_mass_col-active
    # ("*_inter_only_M0") combos (M0<->xi_mass_col correlation ~-0.97) --
    # i.e. it's the interaction term trading off against M0, not a generic
    # problem with this prior. See extra_runners.py's checks/uniformpriors_*
    # block for a dedicated uniform-M0 robustness check on exactly those
    # interaction-active combos, rather than changing the default here.
    "M0":    {"active": False, "prior": "truncated_gaussian",
              "range": [9.2, 11.2], "mu": 10.0, "sigma": 0.4,
              "fixed": 10.0},

    # Width of the mass-step transition (tanh and sigmoid mass models).
    # Irrelevant for mass="step" (hard cutoff) or mass="none" — fix to default.
    # Prior: log_normal peaking near 0.3 dex (mu_ln = log(0.3) ≈ -1.20,
    # sigma_ln = 0.6), with hard clip [0.02, 5.0].
    # This concentrates probability around 0.2–0.5 while keeping the tails
    # accessible — see config.py docstring for full rationale.
    "tau":   {"active": False, "prior": "log_normal",
              "range": [0.02, 5.0], "mu": -1.20, "sigma": 0.60,
              "fixed": 0.2},

    # SN colour offset / quadratic coefficient (linear, quadratic, broken models)
    "c0":    {"active": False, "prior": "uniform",
              "range": [-2, 3],
              "fixed": 0.43},

    # SN colour tanh / softbroken transition width.
    # log_normal prior peaking near 0.3 mag (mu_ln = log(0.3) ≈ -1.20,
    # sigma_ln = 0.7).  Large sn_tau → linear limit, so the model can always
    # recover linearity if the data provide no evidence for saturation.
    "sn_tau": {"active": False, "prior": "log_normal",
               "range": [0.02, 10.0], "mu": 0, "sigma": 0.50,
               "fixed": 1.0},

    # Host colour centre / threshold.
    # Uniform over the existing hard range, not an informative gaussian.
    # Unlike alpha/beta/Om0 (which have external literature/CMB anchors for
    # their mu), there is no comparable physical prior on WHERE the host-
    # colour threshold sits -- prior_shrinkage.py's scan of
    # run_publication_registry.csv flagged C0 as prior-dominated
    # (shrinkage < 0.2) in 50/937 scored (run, param) rows, the single
    # largest flagged group, concentrated in host-colour x mass-colour
    # interaction combos where degeneracy_scan.py's prescan also shows
    # M0<->C0 / xi_mass_col<->C0 correlations >0.85 -- i.e. the previous
    # truncated_gaussian(mu=0, sigma=1) was doing real work pulling the
    # posterior toward 0 in exactly the runs where the data can least
    # independently constrain it. Was: truncated_gaussian(mu=0.0, sigma=1).
    "C0":    {"active": False, "prior": "uniform",
              "range": [-2, 3],
              "fixed": 0.0},

    # Width of the host-colour transition (sigmoid / tanh host_colour models).
    # Same log_normal rationale as tau above.
    "htau":  {"active": False, "prior": "log_normal",
              "range": [0.02, 5.0], "mu": 0, "sigma": 0.60,
              "fixed": 1},

    # ---- double_step upper threshold ----
    # Only used by mass="double_step". Prior centred higher than M0.
    "M1":    {"active": False, "prior": "truncated_gaussian",
              "range": [9.5, 11.5], "mu": 10.5, "sigma": 0.5,
              "fixed": 10.5},

    # ---- mass_spline knot coefficients ----
    # k1, k2, k3 are correction values at the 25th/50th/75th percentile of
    # logM, with linear interpolation between knots.  arcsinh prior gives
    # fine resolution near zero (no correction) and log-spaced resolution
    # at larger values.  Only active for mass="spline".
    "k1":    {"active": False, "prior": "arcsinh",
              "range": [-3.0, 3.0], "scale": 0.5, "fixed": 0.0},
    "k2":    {"active": False, "prior": "arcsinh",
              "range": [-3.0, 3.0], "scale": 0.5, "fixed": 0.0},
    "k3":    {"active": False, "prior": "arcsinh",
              "range": [-3.0, 3.0], "scale": 0.5, "fixed": 0.0},

    # =========================================================================
    # x1 (stretch) correction parameters
    # =========================================================================
    # x1_0: quadratic coefficient for x1_correction="quadratic", or break /
    #        pivot location for other nonlinear x1 models.
    #        Analogous to c0 in the SN colour models.
    #        For x1_correction="linear" x1_0 is DEGENERATE WITH M — fix to 0.
    #        Activate only when a genuine shape parameter is needed.
    "x1_0":  {"active": False, "prior": "uniform",
              "range": [-2.0, 2.0],
              "fixed": 0.0},

    # x1_tau: transition width for nonlinear x1 models (tanh, softbroken,
    #          stepbroken).  log_normal prior peaking near 0.5 (typical x1
    #          scatter), with hard clip [0.05, 10.0].
    #          Large x1_tau → linear limit (same as sn_tau for colour).
    "x1_tau": {"active": False, "prior": "log_normal",
               "range": [0.05, 10.0], "mu": -0.69, "sigma": 0.60,
               "fixed": 1.0},

    # =========================================================================
    # sSFR host environment parameters
    # =========================================================================
    # The complete host correction term is:
    #   G = gamma/2 * S + eta * H + xi_mass_col * S*H
    #     + zeta * F + xi_sSFR_col * F*H + xi_sSFR_mass * F*S + omega * F*S*H
    # where F is the sSFR model profile (analogous to S for mass, H for colour).
    #
    # zeta:    linear sSFR amplitude (main effect).
    #          arcsinh prior: fine resolution near zero, log-spaced at large |ζ|.
    "zeta":   {"active": False, "prior": "arcsinh",
               "range": [-5.0, 5.0], "scale": 0.2,
               "fixed": 0.0},

    # xi_sSFR_col: sSFR × host-colour interaction.
    "xi_sSFR_col": {"active": False, "prior": "arcsinh",
                    "range": [-5.0, 5.0], "scale": 0.3,
                    "fixed": 0.0},

    # xi_sSFR_mass: sSFR × mass interaction.
    "xi_sSFR_mass": {"active": False, "prior": "arcsinh",
                     "range": [-5.0, 5.0], "scale": 0.3,
                     "fixed": 0.0},

    # omega:   sSFR × mass × host-colour three-way interaction.
    "omega":  {"active": False, "prior": "arcsinh",
               "range": [-5.0, 5.0], "scale": 1.2,
               "fixed": 0.0},

    # F0:  sSFR step / threshold location (log sSFR units).
    #      The fiducial split in the literature is near log(sSFR) ≈ -10.5
    #      (yr⁻¹); DES data may prefer a slightly different value.
    "F0":    {"active": False, "prior": "truncated_gaussian",
              "range": [-13.0, -8.0], "mu": -10.5, "sigma": 0.5,
              "fixed": -10.5},

    # ftau:  transition width for smooth sSFR models (tanh, sigmoid).
    #        log_normal prior peaking near 0.5 dex; hard clip [0.05, 5.0].
    "ftau":  {"active": False, "prior": "log_normal",
              "range": [0.05, 5.0], "mu": -0.69, "sigma": 0.60,
              "fixed": 0.5},
}

# ---------------------------------------------------------------------------
# 2.  PARAMETER DISPLAY  (corner plot labels and decimal places)
# ---------------------------------------------------------------------------

PARAM_DISPLAY = {
    "alpha":       {"label": r"$\alpha$",                "sigfigs": 3},
    "beta":        {"label": r"$\beta$",                 "sigfigs": 3},
    "gamma":       {"label": r"$\gamma$",                "sigfigs": 3},
    "M":           {"label": r"$M_x^{}$",                "sigfigs": 3},
    "Om0":         {"label": r"$\Omega_{\rm M0}^{}$",    "sigfigs": 3},
    "w":           {"label": r"$w$",                     "sigfigs": 3},
    "Ode0":        {"label": r"$\Omega_{\Lambda 0}^{}$", "sigfigs": 3},
    "a":           {"label": r"$a$",                     "sigfigs": 3},
    "b":           {"label": r"$b$",                     "sigfigs": 3},
    "g":           {"label": r"$g$",                     "sigfigs": 3},
    "beta_alpha":  {"label": r"$\beta_\alpha^{}$",       "sigfigs": 3},
    "gamma_alpha": {"label": r"$\gamma_\alpha^{}$",      "sigfigs": 3},
    "beta_gamma":  {"label": r"$\beta_\gamma^{}$",       "sigfigs": 3},
    "eta":         {"label": r"$\eta$",                       "sigfigs": 3},
    "xi_mass_col": {"label": r"$\xi_{\rm MC}^{}$",            "sigfigs": 3},
    "M0":          {"label": r"$\log_{10}^{}{\rm M}^0$",             "sigfigs": 3},
    "tau":         {"label": r"$\tau_{\rm mass}^{}$",         "sigfigs": 3},
    "c0":          {"label": r"$c_0^{\rm SN}$",               "sigfigs": 3},
    "sn_tau":      {"label": r"$\tau_{\rm SN}^{}$",                      "sigfigs": 3},
    "C0":          {"label": r"$C_{\rm host}^{0}$",           "sigfigs": 3},
    "htau":        {"label": r"$\tau_{\rm host}^{}$",         "sigfigs": 3},
    "M1":          {"label": r"$\log_{10}^{}{\rm M}^1$",             "sigfigs": 3},
    "k1":          {"label": r"$k_1^{\rm spl}$",              "sigfigs": 3},
    "k2":          {"label": r"$k_2^{\rm spl}$",              "sigfigs": 3},
    "k3":          {"label": r"$k_3^{\rm spl}$",              "sigfigs": 3},
    # x1 correction
    "x1_0":        {"label": r"$x_{1,\,0}^{}$",                 "sigfigs": 3},
    "x1_tau":      {"label": r"$\tau_{x_1}^{}$",              "sigfigs": 3},
    # sSFR host term
    "zeta":         {"label": r"$\zeta$",                     "sigfigs": 3},
    "xi_sSFR_col":  {"label": r"$\xi_{S,C}^{}$",                  "sigfigs": 3},
    "xi_sSFR_mass": {"label": r"$\xi_{S,M}^{}$",                  "sigfigs": 3},
    "omega":        {"label": r"$\omega$",                         "sigfigs": 3},
    "F0":           {"label": r"$\log_{10}^{}S^0$",               "sigfigs": 3},
    "ftau":         {"label": r"$\tau_S^{}$",                      "sigfigs": 3},
}

# ---------------------------------------------------------------------------
# 3.  RUN CONFIG  —  edit paths, models, and sampler settings here
# ---------------------------------------------------------------------------

import os as _os

_REPO_DIR = _os.path.dirname(_os.path.abspath(__file__))

# Absolute paths to the two data products on the analysis server. Kept
# verbatim so a run there is byte-identical to the published runs.
_SERVER_DIR = "/home/users/zgl12/DES_Param_Analysis"


def _data_path(filename):
    """Resolve a data file to wherever it actually exists.

    Order: the server path first (so nothing changes on the machine the
    published runs came from), then alongside this config file, then the
    current working directory.

    Without this, data_file/cov_file were hard-coded to _SERVER_DIR and
    the pipeline could only ever run on that one machine -- which defeats
    the point of shipping environment.yml. Both files are also committed
    to the repository, so the second branch resolves for a fresh clone
    with no configuration at all. If none of the candidates exist the
    server path is returned unchanged, so the resulting error names the
    path that was expected rather than failing somewhere less obvious.
    """
    candidates = (_os.path.join(_SERVER_DIR, filename),
                  _os.path.join(_REPO_DIR,   filename),
                  _os.path.abspath(filename))
    for path in candidates:
        if _os.path.isfile(path):
            return path
    return candidates[0]


CONFIG = {
    # ---- File paths ----
    # Resolved at import time — see _data_path above. Replace either with a
    # literal path to point at a different dataset.
    "data_file":  _data_path("DES-Dovekie_Metadata.csv"),
    "cov_file":   _data_path("STAT+SYS.npz"),
    "output_dir": "Plots",

    # ---- Column names in the CSV ----
    "col_z":          "zHD",
    "col_x0":         "x0",
    "col_x1":         "x1",
    "col_c":          "c",
    "col_logM":       "HOST_LOGMASS",
    "col_muerr":      "MUERR",
    "col_delta_bias": "biasCor_mu",
    "col_host_colour":"HOST_COLOR",
    # ---- Host-property measurement errors ----
    # Used to marginalise mass/colour/sSFR measurement uncertainty out of the
    # host-environment correction via Gauss-Hermite quadrature (see
    # core.gauss_hermite_nodes and run.py's data-loading section).
    # Set to None (or point to a column that is absent) to treat that
    # property as measured exactly — every SN then collapses to a single
    # quadrature node at its point estimate, reproducing the old
    # point-estimate-only behaviour with zero extra cost.
    "col_logM_err":        "HOST_LOGMASS_ERR",
    "col_host_colour_err": "HOST_COLOR_ERR",
    # sSFR columns — set to None if the column is absent in the CSV and
    # ssfr model is "none" (which is the default).
    "col_logsSFR":     "HOST_LOGsSFR",
    "col_logsSFR_err": "HOST_LOGsSFR_ERR",

    # ---- Model selection ----
    # Each key selects from the registered model dicts in core.py.
    "model": {
        "sn_colour":    "linear",   # linear | quadratic | broken | softbroken | tanh
                                    # | dust | stepbroken | asymm_gauss_weight
        "x1_correction":"linear",   # linear | quadratic | tanh | softbroken | stepbroken
        "mass":         "step",     # none   | linear    | step | tanh | sigmoid
                                    # | double_step | gaussian_weight | spline
        "host_colour":  "linear",   # none   | linear    | quadratic | sigmoid | tanh
                                    # | broken | asymm
        "ssfr":         "none",     # none   | linear    | step | tanh | sigmoid
        "z_evolve":     "power",    # power  | log       | linear | zz | exp | step
    },

    # ---- Intrinsic scatter added to diagonal (mag) ----
    "sigma_int": 0.0,

    # ---- Host-property error marginalisation ----
    # Number of Gauss-Hermite quadrature nodes used to marginalise host
    # mass / host colour / sSFR measurement error out of the environment
    # correction (see core.gauss_hermite_nodes). 15-20 is effectively exact
    # for the smooth profile models (linear, tanh, sigmoid, spline).
    # mass="step" and mass="double_step" are true discontinuities, where
    # quadrature converges more slowly right at the threshold — use 20-30
    # there. Cost is trivial either way: this only widens the (N, K) host
    # arrays fed to the profile functions, not the O(N^2) likelihood matrix
    # product that dominates runtime.
    "n_gh_nodes": 40,

    # ---- Data filters (applied before any analysis) ----
    # Redshift cuts: set either or both to restrict the redshift range.
    #   "zlo": 0.1   -> keep only SNe with z >= 0.1   (discard low-z)
    #   "zhi": 0.1   -> keep only SNe with z <= 0.1   (discard high-z)
    #   Leave as None (default) to apply no cut on that side.
    "zlo": None,
    "zhi": None,

    # Stretch (x1) cut: keep only SNe whose SALT2 x1 lies within [lo, hi].
    #   "x1_range": [-1.5, 2.0]  -> discard SNe with x1 < -1.5 or x1 > 2.0
    #   Leave as None (default) to apply no x1 cut.
    # The covariance matrix is subsetted to the surviving rows automatically.
    "x1_range": None,

    # Colour (c) cut: keep only SNe whose SALT2 c lies within [lo, hi].
    #   "c_range": [-0.3, 0.3]  -> discard SNe with |c| > 0.3
    #   Leave as None (default) to apply no colour cut.
    # The covariance matrix is subsetted to the surviving rows automatically.
    "c_range": None,

    # Survey filter: when True, keep only DES (IDSURVEY == 10) and
    # Foundation (IDSURVEY == 150) supernovae.  Requires an "IDSURVEY"
    # column in the data CSV.  False (default) keeps all surveys.
    "idsurvey": False,

    # Mass sub-sample:
    #   "all"  -> use all SNe regardless of host mass   (default)
    #   "high" -> keep only SNe with HOST_LOGMASS >= 10
    #   "low"  -> keep only SNe with HOST_LOGMASS  < 10
    # NaN logM rows are dropped for "high" and "low", retained for "all".
    "mass_cut": "all",

    # Host-match quality cut: host mass/colour/sSFR corrections are only as
    # good as the SN-to-host association. A SN assigned to the wrong galaxy
    # poisons exactly the terms this pipeline is trying to measure, so this
    # is a distinct systematic from mass_cut above, not a duplicate of it.
    #   "all"    -> use all SNe regardless of host-match quality  (default)
    #   "strict" -> keep only SNe with an unambiguous, well-localised host:
    #                 HOST_DDLR   <= host_ddlr_max   (SN close to host centre,
    #                               relative to the host's light profile)
    #                 HOST_CONFUSION <= host_confusion_max (low chance the
    #                               wrong galaxy was picked as the host)
    #                 HOST_NMATCH == 1                (exactly one candidate
    #                               host within the search radius)
    # Thresholds below are conservative literature-typical defaults —
    # inspect the HOST_DDLR/HOST_CONFUSION distributions in your own sample
    # before trusting them for a final result. NOTE: HOST_DDLR == -9 is this
    # catalog's numeric sentinel for "no host match found at all" (not
    # NaN) — the strict-cut mask in run.py explicitly requires
    # HOST_DDLR >= 0 to exclude it; do not rely on host_ddlr_max alone,
    # since -9 <= any positive threshold. On the DES-Dovekie metadata
    # uploaded during development, real HOST_DDLR tops out around 3.9, so
    # the old host_ddlr_max=4.0 was effectively a no-op threshold by itself —
    # only the >= 0 sentinel exclusion did any work. host_ddlr_max is now
    # 2.0, which is genuinely discriminating: DDLR is the SN-host separation
    # in units of the host's directional light radius, so DDLR <= 2 is the
    # standard "the SN sits inside the host's light profile" association
    # criterion used in the DES host-matching literature, while DDLR ~ 4
    # admits associations that are as likely to be chance projections.
    # Raise it back toward 4.0 only if you deliberately want the strict cut
    # to be sentinel-exclusion-only.
    "host_quality_cut": "all",
    "host_ddlr_max":       2.0,
    "host_confusion_max":  0.1,

    # Host redshift observation type: "all" / "spec" / "phot" — see
    # run.py's load_and_filter_data for the exact HOST_ZSPEC/HOST_RA/
    # HOST_DEC logic. A distinct systematic axis from mass_cut/
    # host_quality_cut above (redshift accuracy, not host mass or match
    # quality). "col_host_ra"/"col_host_dec"/"col_host_zspec" let you point
    # at differently-named columns if needed.
    "obs_z_type": "all",
    "col_host_ra":    "HOST_RA",
    "col_host_dec":   "HOST_DEC",
    "col_host_zspec": "HOST_ZSPEC",

    # Optional per-SN diagonal-covariance QC cut. Not a normal analysis
    # choice like mass_cut/host_quality_cut/obs_z_type above -- this exists
    # because a handful of SNe can carry a pathological diagonal entry in
    # STAT+SYS.npz (a numerical instability in a systematic-derivative term,
    # uncorrelated with FITPROB/FITCHI2/SNR/MUERR -- see injection_test.py
    # seed=13 investigation). Off by default so a normal run/sweep is
    # unaffected; set explicitly per-experiment (e.g. in extra_runners.py)
    # to sweep the threshold and confirm results are stable against it.
    #   None       -> no cut applied (default)
    #   <float> N  -> drop any SN with sqrt(diag(cov_mat) + muerr**2) >= N mag
    "muerr_cut": None,

    # Line-of-sight / sky-position ("drilling cones") systematic check.
    # DES-SN (and similar surveys) observe a handful of fixed deep-field
    # pointings — clustering SNe by host sky position essentially recovers
    # those fields, so comparing each field/cone's posterior against the
    # full-sample posterior is a direct line-of-sight-systematic screen.
    # This is NOT a data cut applied during a normal fit (unlike mass_cut/
    # host_quality_cut/obs_z_type above) — it is a flag consumed by the
    # standalone drilling_cones.py script, which no-ops unless this is
    # explicitly set True, so a normal experiment_runner.py/extra_runners.py
    # sweep never triggers it by accident.
    "drilling_cones":      False,
    "cone_eps_deg":        0.7,     # DBSCAN angular linking length (degrees)
    "cone_min_samples":    20,      # DBSCAN min_samples (core-point threshold)
    "cone_min_fit_size":   50,      # skip fitting a cone with fewer SNe than this

    # ---- Parameter priors (override individual entries as needed) ----
    # Copy DEFAULT_PARAM_SPECS here and modify, or leave absent to use defaults.
    # Example: activate beta z-evolution using the log model:
    #   "param_specs": {
    #       **DEFAULT_PARAM_SPECS,
    #       "b": {**DEFAULT_PARAM_SPECS["b"], "active": True}
    #   }
    # "param_specs": DEFAULT_PARAM_SPECS,

    # ---- Sampler settings ----
    # nlive autoscaling (used when "nlive" is None):
    #   "exploratory"  → total_active_params * 50   (fast, for new model testing)
    #   "publication"  → total_active_params * 300  (your original formula)
    # e.g. baseline (alpha, beta, gamma, Om0) = 4 params:
    #   exploratory → 200 live points
    #   publication → 1200 live points
    # Set "nlive" to an explicit integer to override autoscaling entirely.
    "nlive":      None,
    "nlive_mode": "publication",  # "exploratory" | "publication"
    "dlogz":   1e-3,
    "bound":   "multi",
    "sample":  "rslice",
    "verbose": True,
    "sampler_mode": "dynamic",  # "dynamic" | "static"

    # ---- Run registry ----
    # Path to the CSV file that logs every run's settings and evidence.
    # Created automatically on first run; appended to on subsequent runs.
    "registry_file": "run_publication_registry.csv",

    # Optional human-readable tag appended to the auto-generated run name
    # e.g. "wCDM_test" → run name becomes "20240601_143022_wCDM_test"
    # Leave as "" or None for a pure timestamp name.
    "run_tag": "",
}