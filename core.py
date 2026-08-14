"""
core.py  —  SNe Ia Cosmology Pipeline
=======================================
Pure physics and inference machinery.  Nothing here should need editing
between runs; all user-facing choices live in config.py.

Contents
--------
  Model registries   : SN_COLOUR_MODELS, X1_CORRECTION_MODELS,
                       MASS_MODELS, SSFR_MODELS,
                       HOST_COLOUR_MODELS, Z_EVOLVE_MODELS
  mu_theory          : cosmological distance modulus
  compute_mu_corr    : SALT2 distance modulus residual (M excluded)
  cov_log_likelihood : marginalised Gaussian log-likelihood (cho_solve path)
  build_param_getter : maps theta array → full parameter dict
  make_prior_transform : dynesty prior_transform factory
  make_loglike       : dynesty loglike factory
  build_covariance   : load Dovekie .npz covariance

Likelihood performance note (N=1820)
--------------------------------------
  The hot path at every likelihood call is:
      chit2 = delta^T C^{-1} delta
      B     = 1^T C^{-1} delta
      C_sum = 1^T C^{-1} 1          ← scalar, precomputed once
  For N=1820 a dense matrix–vector multiply is O(N^2) = 3.3M flops.
  We precompute inv_cov_mat once in run.py (via Cholesky inversion) and store
  it.  Inside the likelihood we do two matrix–vector products:
      v     = inv_cov_mat @ delta    (reused for both chit2 and B)
      chit2 = delta @ v
      B     = ones @ v
  This is faster than two separate cho_solve calls per likelihood evaluation
  because cho_solve has a Python-level dispatch overhead that accumulates over
  millions of calls.  Storing the explicit inverse is the right trade-off for
  N=1820 (< 26 MB float64) and a fixed covariance matrix.  For N > ~5000 or a
  parameter-dependent covariance, switch to cho_solve.

  The Cholesky factor is still used for the log-determinant (numerically
  superior to np.linalg.slogdet for near-singular matrices) and for the
  one-time inversion via cho_solve in build_covariance and run.py.
"""

import numpy as np

from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats  import truncnorm, loguniform
from scipy.special import ndtri

from astropy.cosmology import FlatLambdaCDM, LambdaCDM, wCDM

import warnings

# ===========================================================================
# 0.  RUNTIME STATE  (written by run.py before building the likelihood)
# ===========================================================================

# The per-dataset redshift pivot.  run.py overwrites this once after loading
# data so that all z-evolution functions see the same, data-derived value.
# Do NOT set this manually here; it is updated automatically.
Z_PIVOT_RUNTIME: float = 0.5   # safe fallback; run.py always overrides

# ===========================================================================
# 0a.  HOST-PROPERTY ERROR MARGINALISATION  (Gauss-Hermite quadrature)
# ===========================================================================
#
# Host mass, host colour, and sSFR all carry measurement error
# (HOST_LOGMASS_ERR, HOST_COLOR_ERR, HOST_LOGsSFR_ERR).  Feeding only the
# point estimate into a nonlinear profile function (step, tanh, sigmoid, ...)
# implicitly assumes zero measurement error on that axis, while the SN-side
# distance-modulus error (MUERR) is fully accounted for on the covariance
# diagonal.  To fix this without breaking the "factorise the covariance once,
# reuse inv_cov_mat for every likelihood call" optimisation described at the
# top of this file, we do NOT touch the covariance at all.  Instead, each
# profile function is evaluated at a small, fixed set of quadrature points
# around each SN's observed host property, and the outputs are combined with
# quadrature weights to give (an excellent numerical approximation to) the
# expectation of the profile function under Gaussian measurement error:
#
#   E[f(x_true)] = integral  f(x_obs + sigma_x * eps) * phi(eps) d(eps)
#                ~ sum_k  w_k * f(x_obs + sigma_x * eps_k)
#
# where eps ~ N(0,1) and (eps_k, w_k) are Gauss-Hermite nodes/weights
# rescaled for a standard-normal weight function (w_k sum to 1).  This is
# exact for polynomials up to degree 2K-1 in f, and is deterministic and
# reproducible (no RNG / seed to manage), unlike plain Monte Carlo sampling.
# For SNe with zero (or missing) measurement error, sigma_x = 0 collapses
# every node to the point estimate, exactly reproducing the old
# point-estimate-only behaviour with zero extra cost.
#
# Convergence is fast (K~10-15 is essentially exact) for the smooth models
# (linear, tanh, sigmoid, gaussian_weight, spline).  mass="step" and
# mass="double_step" are true discontinuities, where quadrature (like plain
# Monte Carlo) loses its fast convergence rate right at the threshold —
# use a larger K (20-30) for those.  See CONFIG["n_gh_nodes"] in config.py.
#
# Mean vs variance
# ----------------
# The above gives E[f], which removes the BIAS from evaluating a nonlinear f
# at a noisy x.  Note that for a LINEAR profile E[f(x)] = f(E[x]) exactly, so
# smoothing a linear model changes nothing — that is correct behaviour, not a
# bug.  What E[f] does not capture is the extra SCATTER the same measurement
# error injects into mu_corr, which requires adding Var[f] to the covariance
# diagonal.  That is implemented (compute_mu_corr(..., return_var=True) and
# cov_log_likelihood_hetero) but is OFF by default, because it makes the
# covariance parameter dependent and so forfeits the "factorise once" design
# above: every likelihood call then costs O(N^3) instead of O(N^2).
#
# Var[f] is computed exactly rather than by linearisation.  G is multilinear
# in the three host profiles and their measurement errors are independent, so
# E[G^2] factorises into univariate moments that this same 1-D quadrature
# supplies — no K^3 tensor grid is required.  Be aware that the second moment
# converges more slowly than the first for the discontinuous profiles, so
# raise K further if the best model uses a step.

def gauss_hermite_nodes(K=20):
    """
    Return (eps, weights) for Gauss-Hermite quadrature against a standard
    normal distribution.

    For X ~ N(mu, sigma^2):
        E[f(X)] ~= sum_k weights[k] * f(mu + sigma * eps[k])

    weights sum to 1 (they are a proper probability weighting, not the raw
    physicists'-convention Hermite weights).

    Parameters
    ----------
    K : int  number of quadrature nodes.

    Returns
    -------
    eps     : ndarray (K,)  standard-normal quadrature abscissas
    weights : ndarray (K,)  quadrature weights, sum to 1
    """
    nodes, herm_weights = np.polynomial.hermite.hermgauss(K)
    eps     = nodes * np.sqrt(2.0)
    weights = herm_weights / np.sqrt(np.pi)
    return eps, weights

# ===========================================================================
# 1.  MODEL REGISTRIES
# ===========================================================================

# ---------------------------------------------------------------------------
# SN colour correction
# ---------------------------------------------------------------------------
#
# Every model receives c (SALT2 colour) plus named kwargs extracted from the
# full params dict.  Unknown kwargs are silently swallowed via **_.
#
# Parameter conventions
# ---------------------
#   c      : SALT2 colour parameter
#   c0     : location shift (linear, broken, softbroken, tanh) OR quadratic
#            coefficient (quadratic model — see note there).
#            For purely linear models c0 is degenerate with M and should be
#            FIXED to zero or left inactive.  Only activate it when the model
#            genuinely uses it as a shape parameter.
#   sn_tau : positive transition width for tanh / softbroken / dust models.
#            log_normal prior; large sn_tau → linear limit.
#
# On c0 degeneracy in linear models
# -----------------------------------
# In sn_colour_linear, c_eff = c - c0.  Since mu_corr enters the likelihood
# only as (mu_corr - mu_cosmo - M), and M is analytically marginalised, a
# constant shift in c_eff is absorbed into M_hat.  Therefore c0 is EXACTLY
# degenerate with M in the linear model and must not be sampled alongside it.
# The same reasoning applies to hcol_linear and C0, and to mass_linear and M0
# (which shifts the zero-point of the linear mass correction).  The experiment
# runner enforces this by never activating these shift parameters for their
# respective linear models.

def sn_colour_linear(c, c0=0.0, **_):
    """
    Standard linear colour correction.
    c0 is a location shift — DEGENERATE WITH M in the linear model.
    Keep c0 fixed to 0 (or inactive) when using this model.
    """
    return c - c0

def sn_colour_quadratic(c, c0=0.0, c_centre=0.0, **_):
    """
    Linear + quadratic colour correction.

    Internally centres c by subtracting c_centre (the sample mean of the raw
    SALT2 colour, computed once in run.py and passed via the data dict).
    data["c"] is NEVER modified — the centring happens here, at call time,
    so it has no effect on any other colour model.

    c0 is the *quadratic coefficient*, not a location shift.  With centred c,
    E[c · c²] ≈ 0 (near-symmetric distribution), so beta (linear slope) and
    c0 (quadratic curvature) are approximately orthogonal, breaking the
    beta ↔ c0 ridge that would otherwise dominate the posterior.

    c0 > 0 → correction steeper at red end (|c| > 0 penalised more)
    c0 < 0 → correction steeper at blue end
    c0 = 0 → reduces to standard linear correction (beta · c_centred)

    Note: beta is still the slope of the *centred* colour correction.  If you
    compare beta to results from analyses that do not centre c, add
    beta_reported = beta_this_fit * (1 + 2*c0*c_centre) to first order.
    """
    cc = c - c_centre          # centred colour; c_centre≈0 → no effect
    return cc + c0 * cc**2

def sn_colour_broken(c, c0=0.0, **_):
    """
    Two-slope (broken-linear) colour correction pivoted at c0.
    Below c0: slope 1 (shallower).  Above c0: slope 2 (steeper).
    c0 here is a genuine break location, NOT degenerate with M because it
    changes the shape of the correction, not just its mean.
    """
    x = c - c0
    return np.where(x < 0, x, 2.0 * x)

def sn_colour_softbroken(c, c0=0.0, sn_tau=1, **_):
    """
    Soft broken-linear colour correction, normalised so beta retains its
    linear-model meaning.

    The unnormalised form x*(1 + weight) has a mean correction factor of 1.5
    (blend of slopes 1 and 2 at equal weight), which would rescale beta down
    by 1/1.5 relative to the linear model.  We divide by 1.5 so that:
      - beta has the same interpretation as in sn_colour_linear
      - sn_tau controls only the *sharpness* of the transition, not the scale
      - large sn_tau (smooth blend): c_eff → (2/3)*x + (2/3)*x = (4/3)*x... 
        Actually at sn_tau → ∞, weight → 0.5 everywhere, so raw → 1.5*x,
        normalised → x.  This recovers the linear model exactly.
      - sn_tau → 0 (hard break): one side → 0, other → 2x, 
        normalised → 0 or (4/3)*x.

    c0: break location (shape parameter, not degenerate with M).
    sn_tau: transition width (log_normal prior; large → linear recovery).
    """
    x      = c - c0
    weight = 0.5 * (1.0 + np.tanh(x / sn_tau))
    raw    = x * (1.0 + weight)   # spans slopes 1 (blue) to 2 (red)
    return raw / 1.5               # normalise: mean factor = 1.5

def sn_colour_tanh(c, c0=0.0, sn_tau=1, **_):
    """
    Saturating (tanh) colour correction.

      c_eff = sn_tau * tanh((c - c0) / sn_tau)

    Linear limit (sn_tau → ∞): c_eff → c - c0  (recovers linear model).
    Saturation (small sn_tau): correction plateaus at ±sn_tau for extreme c.

    c0 is a location shift.  Because the model is nonlinear in c0, it is NOT
    purely degenerate with M — activate it when exploring asymmetric colour
    populations.  For a symmetric null test, fix c0=0 and only sample sn_tau.
    """
    return sn_tau * np.tanh((c - c0) / sn_tau)

def sn_colour_dust(c, c0=0.0, sn_tau=1.0, c_ref=0.1, **_):
    """
    Power-law (dust-motivated) colour correction.

      x     = c - c0                            (pivot at c0)
      c_eff = sign(x) * |x|^sn_tau / |c_ref|^(sn_tau - 1)

    c0:     pivot location — the colour at which the correction changes sign.
            Prior: truncated_gaussian centred on 0, sigma=0.1, range [-0.5, 0.5].
            NOT degenerate with M due to the nonlinearity.

    sn_tau: power-law exponent (repurposed — log_normal prior centred on 0
            i.e. exp=1 → linear). sn_tau=1 → c_eff = x (linear, no normalisation
            effect). sn_tau < 1 → sub-linear (dust-like flattening at extremes).
            sn_tau > 1 → super-linear.

    c_ref:  fixed normalisation reference (data median |c|, set in run.py).
            Pins beta to "slope at c_ref", comparable across models.
    """
    c_ref_abs = max(abs(float(c_ref)), 1e-4)
    exponent  = np.clip(sn_tau, 0.1, 5.0)   # sn_tau carries the exponent
    x         = c - c0                       # pivot at c0
    norm      = c_ref_abs ** (exponent - 1.0)
    return np.sign(x) * np.abs(x) ** exponent / norm

def sn_colour_stepbroken(c, c0=0.0, sn_tau=1.0, **_):
    """
    Asymmetric two-slope correction with beta_mean reparametrisation.

    Instead of (beta_blue, slope_ratio), the model uses:
      beta_mean  = 0.5 * (slope_blue + slope_red)   [this is the beta parameter]
      sn_tau     = slope_red / slope_blue             [the ratio]

    Solving: slope_blue = 2 / (1 + sn_tau),  slope_red = 2*sn_tau / (1 + sn_tau)

    So:
      c_eff = (c - c0) * 2 / (1 + sn_tau)            if c < c0
      c_eff = (c - c0) * 2*sn_tau / (1 + sn_tau)     if c >= c0

    Key properties
    --------------
    sn_tau = 1 → slope_blue = slope_red = 1 → c_eff = c - c0  (linear, comparable)
    sn_tau > 1 → steeper correction for red SNe  (c >= c0)
    sn_tau < 1 → shallower correction for red SNe

    This reparametrisation makes beta directly comparable to the linear model:
    both represent the *average* colour slope, so posterior beta values can be
    compared across model variants without rescaling.  The prior on beta needs
    no adjustment for this model.

    c0 prior:     gaussian centred on 0, sigma ~ 0.1  (break near c=0)
    sn_tau prior: log_normal centred on log(1)=0, sigma ~ 0.4  (sn_tau=1 is linear)
    """
    f_blue = 2.0 / (1.0 + sn_tau)
    f_red  = 2.0 * sn_tau / (1.0 + sn_tau)
    return np.where(c < c0, (c - c0) * f_blue, (c - c0) * f_red)

def sn_colour_asymm_gauss_weight(c, c0=0.0, sn_tau=0.3, **_):
    """
    Gaussian-weighted linear colour correction.

    The contribution of each SN's colour correction is weighted by a
    Gaussian centred at c0 with width sn_tau:

      w(c)  = exp(-0.5 * ((c - c0) / sn_tau)^2)
      c_eff = c * w(c)

    This down-weights SNe with colours far from c0 (both very red and very
    blue), reducing their leverage on beta without a hard cut.

    Limit: as sn_tau → ∞, w → 1 and c_eff → c (recovers unweighted linear).

    Physical motivation: the SALT2 training set is densest near c ~ 0.
    Outlier colours may have poorly calibrated corrections; down-weighting
    them by a smooth Gaussian is a principled alternative to sigma-clipping.

    c0 prior: gaussian centred on 0, sigma ~ 0.1 (near sample centre).
    sn_tau prior: log_normal centred on log(0.3), sigma ~ 0.5.
    """
    w = np.exp(-0.5 * ((c - c0) / sn_tau) ** 2)
    return c * w

SN_COLOUR_MODELS = {"linear":             sn_colour_linear,
                    "quadratic":          sn_colour_quadratic,
                    "broken":             sn_colour_broken,
                    "softbroken":         sn_colour_softbroken,
                    "tanh":               sn_colour_tanh,
                    "dust":               sn_colour_dust,
                    "stepbroken":         sn_colour_stepbroken,
                    "asymm_gauss_weight": sn_colour_asymm_gauss_weight}

# ---------------------------------------------------------------------------
# Stretch (x1) correction models
# ---------------------------------------------------------------------------
#
# The standard SALT2 distance modulus uses a purely linear x1 correction:
#   mu += alpha * x1
# The models below replace that linear term with a (possibly nonlinear)
# effective stretch correction x1_eff, so that:
#   mu += alpha * x1_eff(x1, ...)
# This preserves alpha's role as the overall stretch-luminosity slope.
#
# Parameter conventions (mirrors SN_COLOUR_MODELS)
# -------------------------------------------------
#   x1       : SALT2 stretch parameter
#   x1_0     : quadratic coefficient (quadratic model) OR break / pivot
#              location (other models).  Degenerate with M for "linear"
#              if used as a location shift — fix to 0 in that case.
#   x1_tau   : positive transition width for tanh / softbroken / stepbroken.
#              log_normal prior, mu=-0.69 (peak near 0.5); large → linear.
#   x1_centre: sample mean of x1, precomputed in run.py.  Stored in
#              data["x1_centre"].  Used only by x1_correction_quadratic
#              to centre the quadratic term (same rationale as c_centre).
#
# Model summary
# -------------
#   linear      : x1_eff = x1                    (standard, no new params)
#   quadratic   : x1_eff = (x1-μ) + x1_0*(x1-μ)²  (centred quadratic)
#   tanh        : saturating correction (plateaus for large |x1|)
#   softbroken  : asymmetric two-slope with smooth transition at x1_0
#   stepbroken  : asymmetric beta_mean reparametrisation (mirrors sn_colour)

def x1_correction_linear(x1, x1_0=0.0, x1_tau=1.0, **_):
    """
    Standard linear stretch correction.  x1_eff = x1.
    x1_0 is NOT used — the parameter is accepted and silently ignored via **_
    to allow uniform call signatures across all x1 models.
    """
    return x1

def x1_correction_quadratic(x1, x1_0=0.0, x1_tau=1.0, x1_centre=0.0, **_):
    """
    Centred quadratic stretch correction.

    Analogous to sn_colour_quadratic: centres x1 at its sample mean before
    applying the quadratic term, making alpha (linear slope) and x1_0
    (quadratic curvature) approximately orthogonal in the posterior.

      x1c = x1 - x1_centre          (centred stretch)
      x1_eff = x1c + x1_0 * x1c²

    x1_0 > 0 → correction steeper for high-|x1| SNe (both ends penalised more)
    x1_0 < 0 → correction flatter at extremes (saturation-like)
    x1_0 = 0 → reduces to linear in centred x1 ≡ standard correction

    Note: comparing alpha to linear-model results requires
      alpha_linear ≈ alpha * (1 + 2*x1_0*x1_centre) to first order.
    """
    x1c = x1 - x1_centre
    return x1c + x1_0 * x1c**2

def x1_correction_tanh(x1, x1_0=0.0, x1_tau=1.0, **_):
    """
    Saturating (tanh) stretch correction.

      x1_eff = x1_tau * tanh((x1 - x1_0) / x1_tau)

    Physical motivation: very high-|x1| SNe may have poorly calibrated SALT2
    fits.  The tanh correction down-weights their leverage smoothly without
    a hard cut, with saturation scale x1_tau.

    Linear limit (x1_tau → ∞): x1_eff → x1 - x1_0.
    Large x1: correction plateaus at ±x1_tau.

    x1_0 is a location shift (not purely degenerate with M due to nonlinearity).
    x1_tau prior: log_normal centred on log(0.5), sigma=0.6.
    """
    return x1_tau * np.tanh((x1 - x1_0) / x1_tau)

def x1_correction_sigmoid(x1, x1_0=0.0, x1_tau=1.0, **_):
    """
    Logistic (sigmoid) stretch correction.  x1_eff ∈ (-2*x1_tau, +2*x1_tau).

      x1_eff = x1_tau * (2 / (1 + exp(-x1 / x1_tau)) - 1) * (1 / tanh(1))

    The factor 1/tanh(1) ≈ 1.313 normalises so that at x1 = ±x1_tau the
    correction equals ±x1_tau (matching the tanh model at that scale),
    making alpha comparable between the two models.

    x1_tau → ∞: x1_eff → x1  (recovers linear)
    x1_tau → 0: hard saturation at ±x1_tau
    """
    norm   = 1.0 / np.tanh(1.0)          # ≈ 1.3130
    x1_eff = x1_tau * (2.0 / (1.0 + np.exp(-(x1 - x1_0) / x1_tau)) - 1.0) * norm
    return x1_eff

def x1_correction_doublebroken(x1, x1_0=0.0, x1_tau=1.0, x1_ref=0.5, **_):
    """
    Three-slope stretch correction with two break points at ±x1_0.

    Slopes: (1/r) below -x1_0, 1 between ±x1_0, r above +x1_0
    where r = x1_tau (the slope ratio for the outer segments).

    Normalised so alpha = slope in the central region [-x1_0, +x1_0].

      x1_eff = (x1 - (-x1_0)) / x1_tau   if x1 < -x1_0  (sub-linear blue end)
             = x1                          if -x1_0 ≤ x1 ≤ x1_0
             = (x1 - x1_0) * x1_tau + x1_0  if x1 > x1_0  (super-linear red end)

    Wait — actually simpler as:
      centre region:  x1_eff = x1                         (slope 1)
      outer regions:  x1_eff = x1_0 * sign(x1) + (x1 - x1_0*sign(x1)) * x1_tau

    x1_0  : half-width of the linear central region.
             prior: log_normal mu=log(0.8), sigma=0.4, range=[0.2, 3.0]
             (physically: the linear regime spans typical |x1| < 1)
    x1_tau: slope ratio for outer segments (x1_tau > 1 → steeper wings).
             prior: log_normal mu=0, sigma=0.4, range=[0.2, 5.0]

    x1_tau = 1 → reduces to linear everywhere.
    """
    inner = np.abs(x1) <= x1_0
    sign  = np.sign(x1)
    x1_eff = np.where(inner, x1, x1_0 * sign + (x1 - x1_0 * sign) * x1_tau)
    return x1_eff

def x1_correction_powerlaw(x1, x1_0=0.0, x1_tau=1.0, x1_ref=0.5, **_):
    """
    Power-law stretch correction.

      x1_eff = sign(x1 - x1_0) * |x1 - x1_0|^x1_tau / |x1_ref|^(x1_tau - 1)

    x1_tau = 1  → linear (recovers standard correction)
    x1_tau < 1  → sub-linear, flattens at large |x1| (saturation-like)
    x1_tau > 1  → super-linear, penalises outlier stretch more heavily

    x1_ref: fixed normalisation point (sample median |x1|, set in run.py),
            so alpha retains its meaning as the slope at the reference stretch.

    x1_0  prior: uniform [-1.5, 1.5]
    x1_tau prior: log_normal mu=0 (peak=1=linear), sigma=0.5, range=[0.1, 5.0]
    """
    x1_ref_abs = max(abs(float(x1_ref)), 1e-4)
    exponent   = np.clip(x1_tau, 0.1, 5.0)
    x          = x1 - x1_0
    norm       = x1_ref_abs ** (exponent - 1.0)
    return np.sign(x) * np.abs(x) ** exponent / norm

def x1_correction_asymm_gauss_weight(x1, x1_0=0.0, x1_tau=1.0, **_):
    """
    Gaussian-weighted stretch correction.

      w(x1)  = exp(-0.5 * ((x1 - x1_0) / x1_tau)^2)
      x1_eff = x1 * w(x1)

    Down-weights SNe with |x1| far from x1_0, reducing their leverage on
    alpha without a hard cut.  x1_tau → ∞ recovers unweighted linear.

    x1_0  prior: uniform [-1, 1]  (near sample centre)
    x1_tau prior: log_normal mu=0, sigma=0.5  (peak near 1.0, typical x1 scatter)
    """
    w = np.exp(-0.5 * ((x1 - x1_0) / x1_tau) ** 2)
    return x1 * w

def x1_correction_softbroken(x1, x1_0=0.0, x1_tau=1.0, **_):
    """
    Soft broken-linear stretch correction with normalised alpha.

    Mirrors sn_colour_softbroken exactly, replacing c/c0/sn_tau with
    x1/x1_0/x1_tau.  The /1.5 normalisation preserves alpha's interpretation
    as the *mean* stretch slope (same as in the linear model).

    x1_tau → 0  : hard asymmetric break at x1_0
    x1_tau → ∞  : smooth blend → recovers linear
    """
    x      = x1 - x1_0
    weight = 0.5 * (1.0 + np.tanh(x / x1_tau))
    raw    = x * (1.0 + weight)
    return raw / 1.5

def x1_correction_stepbroken(x1, x1_0=0.0, x1_tau=1.0, **_):
    """
    Asymmetric two-slope stretch correction with alpha_mean reparametrisation.

    Mirrors sn_colour_stepbroken: x1_tau is the slope ratio (high/low x1),
    and alpha represents the *average* correction slope.

      x1_eff = (x1 - x1_0) * 2 / (1 + x1_tau)           if x1 < x1_0
      x1_eff = (x1 - x1_0) * 2*x1_tau / (1 + x1_tau)    if x1 >= x1_0

    x1_tau = 1 → symmetric linear (alpha comparable to standard model)
    x1_tau > 1 → steeper correction for high-stretch SNe (x1 >= x1_0)
    x1_tau < 1 → shallower correction for high-stretch SNe

    Physical motivation: the Tripp formula assumes a universal alpha; the
    step-broken model tests whether over-luminous (high-x1) and faint
    (low-x1) SNe need different stretch slopes.

    x1_0 prior: uniform(range=[-2,2], default 0).
    x1_tau prior: log_normal centred on log(1)=0, sigma=0.4 (x1_tau=1 is linear).
    """
    f_lo = 2.0 / (1.0 + x1_tau)
    f_hi = 2.0 * x1_tau / (1.0 + x1_tau)
    return np.where(x1 < x1_0, (x1 - x1_0) * f_lo, (x1 - x1_0) * f_hi)

X1_CORRECTION_MODELS = {"linear":             x1_correction_linear,
                        "quadratic":          x1_correction_quadratic,
                        "tanh":               x1_correction_tanh,
                        "stepbroken":         x1_correction_stepbroken,
                        "softbroken":         x1_correction_softbroken,
                        "asymm_gauss_weight": x1_correction_asymm_gauss_weight,
                        "powerlaw":           x1_correction_powerlaw,
                        "doublebroken":       x1_correction_doublebroken,
                        "sigmoid":            x1_correction_sigmoid}


# ---------------------------------------------------------------------------
# Host mass step / profile
# ---------------------------------------------------------------------------
#
# On M0 degeneracy in mass_linear
# ---------------------------------
# mass_linear returns (logM - M0).  A constant shift -M0 in S enters G as
# gamma/2 * (-M0), which is a constant absorbed into M_hat.  Therefore M0
# is degenerate with M in the linear mass model — do not activate it there.
# The step-function models genuinely use M0 as a threshold, so it is not
# degenerate.

def mass_none(logM, M0=10.0, tau=0.2, **_):
    """No host-mass correction.  S = 0 everywhere."""
    return np.zeros_like(logM)

def mass_linear(logM, M0=10.0, tau=0.2, **_):
    """
    Linear mass correction.
    M0 is a zero-point shift — DEGENERATE WITH M.  Fix M0=0 or leave inactive.
    """
    return logM - M0

def mass_step(logM, M0=10.0, tau=0.2, **_):
    """Hard step at M0.  M0 is the threshold mass — sample it."""
    return np.where(logM > M0, 1.0, -1.0)

def mass_tanh(logM, M0=10.0, tau=0.2, **_):
    """
    Smooth tanh step centred at M0 with width tau.
    tau * tanh((logM-M0)/tau) recovers the linear correction (logM-M0) as tau→∞,
    consistent with sn_colour_tanh's linear recovery limit.
    gamma/2 is then the slope of the mass correction at M0, not the step amplitude.
    """
    return tau * np.tanh((logM - M0) / tau)

def mass_sigmoid(logM, M0=10.0, tau=0.2, **_):
    """Logistic step centred at M0 with width tau; returns ∈ (-1, 1)."""
    return 2.0 / (1.0 + np.exp(-(logM - M0) / tau)) - 1.0

def mass_double_step(logM, M0=9.5, tau=0.2, M1=10.5, **_):
    """
    Two-threshold hard step: three mass bins (low, mid, high).

      S = -1   if logM < M0
          0    if M0 <= logM < M1
         +1   if logM >= M1

    Physical motivation: some analyses find that the mass-luminosity
    relation is non-monotonic or has two transitions — one at the
    low-mass dwarf/spiral boundary and one at the high-mass elliptical
    boundary.  This model tests whether both thresholds are needed.

    Active parameters: M0, M1 (both sample).  tau unused (passed through).
    Prior for M0: truncated_gaussian centred on 9.5, sigma 0.5, range [8.5, 10.5].
    Prior for M1: truncated_gaussian centred on 10.5, sigma 0.5, range [9.5, 11.5].
    Constraint: enforced in compute_mu_corr (M1 > M0 check).
    """
    S = np.zeros_like(logM, dtype=float)
    S[logM >= M1] =  1.0
    S[logM <  M0] = -1.0
    return S

def mass_gaussian_weight(logM, M0=10.0, tau=0.5, **_):
    """
    Gaussian-weighted signed mass correction.

      S(logM) = sign(logM - M0) * exp(-0.5 * ((logM - M0) / tau)^2)

    Returns ∈ (−1, +1]; antisymmetric about M0.
    Peaks at +1 just above M0 and −1 just below M0, fading to zero far from M0.

    Physical motivation: the mass-luminosity correction may be strongest near
    the transition region and weaker for galaxies far from the threshold in
    either direction.  The antisymmetric form keeps γ identifiable (γ > 0 →
    high-mass hosts appear brighter, consistent with the step convention) and
    prevents the bimodality that arises from the previously unsigned form where
    the sign of γ was unconstrained when M0 rails to a mass-sparse region.

    M0: transition mass (sample; prior centred on 10.0, tightened to σ=0.4).
    tau: width of the sensitive region (sample; log_normal prior).
    tau → 0: approaches the hard step function.
    tau → large: approaches a linear-in-logM correction (fades to zero
                 at both extremes, peaks near M0).
    """
    dm = logM - M0
    return np.sign(dm) * np.exp(-0.5 * (dm / tau) ** 2)

# Spline knot positions: fixed at 25th, 50th, 75th percentile of logM
# (computed from data in run.py and stored in data["logM_knots"]).
# Spline coefficients k1, k2, k3 are the free parameters.
# The spline is a piecewise linear interpolation through the knots —
# this is the simplest non-parametric form and does not require scipy.
# For a cubic spline replace np.interp with scipy.interpolate.CubicSpline.

def mass_spline(logM, M0=10.0, tau=0.2, k1=0.0, k2=0.0, k3=0.0,
                logM_knots=None, **_):
    """
    Three-knot piecewise-linear (spline) mass correction.

    The correction is defined by three values (k1, k2, k3) at fixed knot
    positions (logM_knots), with linear interpolation between knots and
    linear extrapolation beyond the endpoints.

    logM_knots is passed via data["logM_knots"] (set in run.py from
    np.percentile(data["logM"], [25, 50, 75])).

    k1, k2, k3: correction values at the 25th, 50th, 75th percentile knots.
    Prior: arcsinh, scale=0.5 for each (centred on zero = no correction).

    Physical motivation: allows the data to find any monotonic or
    non-monotonic mass-luminosity relation without imposing a parametric
    form.  The 3-knot version adds only 3 parameters; extend to 5 knots
    for more flexibility (at a higher evidence penalty).
    """
    if logM_knots is None:
        # Fallback: evenly-spaced knots if not set by run.py
        logM_knots = np.array([9.0, 10.0, 11.0])
    knot_vals = np.array([k1, k2, k3])
    return np.interp(logM, logM_knots, knot_vals)

MASS_MODELS = {"none":            mass_none,
               "linear":          mass_linear,
               "step":            mass_step,
               "tanh":            mass_tanh,
               "sigmoid":         mass_sigmoid,
               "double_step":     mass_double_step,
               "gaussian_weight": mass_gaussian_weight,
               "spline":          mass_spline}

# ---------------------------------------------------------------------------
# Specific star formation rate (sSFR) correction models
# ---------------------------------------------------------------------------
#
# F is a profile function of log(sSFR), entirely analogous to S (mass) and
# H (host colour).  It enters the full host environment term:
#
#   G = gamma/2 * S  +  eta * H       +  xi * S*H
#     + zeta  * F    +  epsilon * F*H  +  theta * F*S  +  omega * F*S*H
#
# Every model receives logsSFR plus named kwargs (F0, ftau); unknown kwargs
# are silently swallowed via **_.
#
# Parameter conventions
# ----------------------
#   logsSFR : log10 specific star formation rate (yr⁻¹), from HOST_LOGsSFR
#   F0      : step threshold / centre of transition.
#             Literature: log(sSFR) ≈ -10.5 yr⁻¹ separates passive from
#             star-forming galaxies.
#             Prior: truncated_gaussian(mu=-10.5, sigma=0.5, range=[-13,-8])
#   ftau    : transition width for smooth models (tanh, sigmoid).
#             Prior: log_normal peaking near 0.5 dex, range=[0.05,5].
#             Large ftau → linear limit; small ftau → hard step.
#
# On degeneracy
# -------------
# ssfr_linear returns (logsSFR - F0).  A constant offset -F0 enters G as
# zeta * (-F0), which is absorbed into M_hat.  Therefore F0 is DEGENERATE
# WITH M in the linear sSFR model — fix F0 to a constant or leave inactive.
# The step, tanh, and sigmoid models use F0 as a genuine threshold / centre,
# so it is NOT degenerate with M in those cases.
#
# Handling NaN
# ------------
# SNe without a host sSFR measurement (logsSFR = NaN) should return F = 0.0
# so they contribute nothing to the sSFR term.  All models below implement
# this behaviour.  run.py loads logsSFR as NaN when the column is missing or
# blank, so F naturally collapses to zero for those SNe regardless of model.

def ssfr_none(logsSFR, F0=-10.5, ftau=0.5, **_):
    """No sSFR correction.  F = 0 everywhere."""
    return np.zeros_like(logsSFR)

def ssfr_linear(logsSFR, F0=-10.5, ftau=0.5, **_):
    """
    Linear sSFR correction.  F = logsSFR - F0.
    F0 is a zero-point shift — DEGENERATE WITH M in this model.
    Fix F0 or leave inactive when using ssfr_linear.
    NaN → 0.
    """
    F = logsSFR
    return np.where(np.isfinite(F), F, 0.0)

def ssfr_step(logsSFR, F0=-10.5, ftau=0.5, **_):
    """
    Hard step at F0.  F = +1 (star-forming) / -1 (passive).
    NaN → 0.

    Convention: star-forming galaxies (logsSFR > F0) → F = +1.
    This means zeta > 0 → star-forming hosts appear *brighter*, consistent
    with the mass step sign convention (gamma > 0 → high-mass brighter).
    """
    F = np.where(logsSFR > F0, 1.0, -1.0)
    return np.where(np.isfinite(logsSFR), F, 0.0)

def ssfr_tanh(logsSFR, F0=-10.5, ftau=0.5, **_):
    """
    Smooth tanh step centred at F0 with width ftau.

      F = ftau * tanh((logsSFR - F0) / ftau)

    Linear limit (ftau → ∞): F → logsSFR - F0.
    Hard step (ftau → 0):    F → ±ftau (constant amplitude, variable sign).

    ftau prior: log_normal centred on log(0.5), sigma=0.6.
    F0 prior: truncated_gaussian(mu=-10.5, sigma=0.5).
    NaN → 0.
    """
    F = ftau * np.tanh((logsSFR - F0) / ftau)
    return np.where(np.isfinite(logsSFR), F, 0.0)

def ssfr_sigmoid(logsSFR, F0=-10.5, ftau=0.5, **_):
    """
    Logistic step centred at F0 with width ftau.  F ∈ (-1, +1).

      F = 2 / (1 + exp(-(logsSFR - F0) / ftau)) - 1

    The factor of 2 and -1 rescale the logistic to match the step convention
    (F = ±1 at the extremes), keeping zeta comparable to the step amplitude.

    ftau → 0: approaches ssfr_step (hard step at F0).
    ftau → ∞: F flattens to 0 (no correction); use ssfr_linear instead.
    NaN → 0.
    """
    F = 2.0 / (1.0 + np.exp(-(logsSFR - F0) / ftau)) - 1.0
    return np.where(np.isfinite(logsSFR), F, 0.0)

SSFR_MODELS = {
    "none":    ssfr_none,
    "linear":  ssfr_linear,
    "step":    ssfr_step,
    "tanh":    ssfr_tanh,
    "sigmoid": ssfr_sigmoid,
}

# ---------------------------------------------------------------------------
# Host colour correction
# ---------------------------------------------------------------------------
#
# On C0 degeneracy in hcol_linear
# ---------------------------------
# hcol_linear returns (C - C0).  A constant -C0 in H enters G as eta*(-C0),
# which is absorbed into M_hat.  C0 is therefore degenerate with M in the
# linear host_colour model — do not activate it there.

def hcol_none(C, C0=0.0, htau=0.2, **_):
    """No host-colour correction.  H = 0 everywhere."""
    return np.zeros_like(C)

def hcol_linear(C, C0=0.0, htau=0.2, **_):
    """
    Linear host-colour correction.
    C0 is a location shift — DEGENERATE WITH M.  Fix C0=0 or leave inactive.
    """
    return C - C0

def hcol_quadratic(C, C0=0.0, htau=0.2, **_):
    """
    Quadratic host-colour correction.
    C0 is the quadratic coefficient (not a location shift).
    """
    x = C - np.nanmedian(C)
    return x + C0 * x**2

def hcol_sigmoid(C, C0=0.0, htau=0.2, **_):
    return 2.0 / (1.0 + np.exp(-(C - C0) / htau)) - 1.0

def hcol_tanh(C, C0=0.0, htau=1, **_):
    """
    Smooth tanh host-colour correction centred at C0 with width htau.
    htau * tanh((C-C0)/htau) recovers linear (C-C0) as htau→∞.
    """
    return htau * np.tanh((C - C0) / htau)

def hcol_broken(C, C0=0.0, htau=0.2, **_):
    x = C - C0
    return np.where(x < 0, x, 2.0 * x)

def hcol_asymm(C, C0=0.0, htau=1.0, **_):
    """
    Asymmetric two-slope host-colour correction with eta_mean reparametrisation.

    Mirrors sn_colour_stepbroken: htau is the slope ratio (red/blue), and eta
    (the overall amplitude) represents the *average* correction slope, making
    it directly comparable to the linear host-colour model's eta.

      H_blue = (C - C0) * 2 / (1 + htau)            if C < C0
      H_red  = (C - C0) * 2*htau / (1 + htau)        if C >= C0

    htau = 1 → H = C - C0 everywhere (linear, comparable to hcol_linear)
    htau > 1 → steeper correction for red host colours  (C >= C0)
    htau < 1 → shallower correction for red host colours

    Physical motivation: red host galaxies (older stellar populations) and
    blue hosts (actively star-forming) may have different environmental
    corrections.  The mean parametrisation makes eta interpretable regardless
    of htau, and the Bayes evidence correctly penalises htau ≠ 1 if the data
    do not support asymmetry.

    C0:   break location in host colour space.
          Prior: truncated_gaussian(mu=0.0, sigma=1.0, range=[-2.0, 3.0])
          Note: posterior often prefers C0 < 0 when the bulk of hosts are
          at C > 0, meaning the asymmetry applies to nearly all hosts —
          interpret this as evidence for the model degenerating toward linear.
    htau: slope ratio (log_normal centred on log(1)=0, sigma=0.4).
          htau=1 is the linear limit; Bayes evidence will penalise htau≠1
          if the asymmetry is not supported by the data.
    """
    f_blue = 2.0 / (1.0 + htau)
    f_red  = 2.0 * htau / (1.0 + htau)
    return np.where(C < C0, (C - C0) * f_blue, (C - C0) * f_red)

HOST_COLOUR_MODELS = {
    "none":      hcol_none,
    "linear":    hcol_linear,
    "quadratic": hcol_quadratic,
    "sigmoid":   hcol_sigmoid,
    "tanh":      hcol_tanh,
    "broken":    hcol_broken,
    "asymm":     hcol_asymm,
}

# ---------------------------------------------------------------------------
# Redshift evolution
# ---------------------------------------------------------------------------
#
# All four functions return (f_alpha, f_beta, f_gamma), multiplicative
# factors applied to alpha, beta, gamma respectively.
#
# Pivot strategy: see Z_PIVOT_RUNTIME note at top of file.
# All functions accept z_pivot=None (falls back to Z_PIVOT_RUNTIME).

def z_evolve_power(z, a=0.0, b=0.0, g=0.0, z_pivot=None):
    """
    Power-law: f_x(z) = [(1+z)/(1+z_pivot)]^x.
    x=0 → no evolution.  Best for multiplicative scalings.
    """
    zp = Z_PIVOT_RUNTIME if z_pivot is None else z_pivot
    r  = (1.0 + z) / (1.0 + zp)
    return r**a, r**b, r**g

def z_evolve_log(z, a=0.0, b=0.0, g=0.0, z_pivot=None):
    """
    Logarithmic: f_x(z) = 1 + x * log10[(1+z)/(1+z_pivot)].
    x=0 → no evolution.  Sub-power-law growth.
    """
    zp = Z_PIVOT_RUNTIME if z_pivot is None else z_pivot
    lz = np.log10((1.0 + z) / (1.0 + zp))
    return 1.0 + a * lz, 1.0 + b * lz, 1.0 + g * lz

def z_evolve_linear(z, a=0.0, b=0.0, g=0.0, z_pivot=None):
    """
    Linear-in-z: f_x(z) = 1 + x * (z - z_pivot).
    x=0 → no evolution.  First-order Taylor; cleanest interpretation.
    """
    zp = Z_PIVOT_RUNTIME if z_pivot is None else z_pivot
    dz = z - zp
    return 1.0 + a * dz, 1.0 + b * dz, 1.0 + g * dz

def z_evolve_zz(z, a=0.0, b=0.0, g=0.0, z_pivot=None):
    """
    z/(1+z): f_x(z) = 1 + x * [z/(1+z) - z_p/(1+z_p)].
    x=0 → no evolution.  Bounded at high-z; safer with broad priors on x.
    """
    zp  = Z_PIVOT_RUNTIME if z_pivot is None else z_pivot
    fz  = z  / (1.0 + z)
    fp  = zp / (1.0 + zp)
    return 1.0 + a * (fz - fp), 1.0 + b * (fz - fp), 1.0 + g * (fz - fp)

def z_evolve_exp(z, a=0.0, b=0.0, g=0.0, z_pivot=None):
    """
    Exponential: f_x(z) = exp(x * (z - z_pivot)).

    x=0 → no evolution (f=1).  x is the fractional change per unit Δz.
    Small |x|: equivalent to linear (exp(xΔz) ≈ 1 + xΔz).
    Large |x|: allows rapid evolution without the power-law's (1+z) weighting.

    Comparison with power-law
    -------------------------
    z_evolve_power uses (1+z) as the base, which downweights low-z evolution
    and upweights high-z.  z_evolve_exp uses z directly, giving equal weight
    to equal Δz intervals throughout the survey.  For DES (0.1 < z < 1.3)
    the two are nearly indistinguishable; the difference matters mainly if
    you have low-z (<0.1) anchors from a separate sample.
    """
    zp = Z_PIVOT_RUNTIME if z_pivot is None else z_pivot
    dz = z - zp
    return np.exp(a * dz), np.exp(b * dz), np.exp(g * dz)

def z_evolve_step(z, a=0.0, b=0.0, g=0.0, z_pivot=None):
    """
    Redshift step: independent nuisance values above and below z_pivot.

      f_x(z) = 1 + x * sign(z - z_pivot)
             = 1 - x   for z <  z_pivot
             = 1 + x   for z >= z_pivot

    x=0 → no evolution.  a, b, g represent the half-amplitude of the step
    in alpha, beta, gamma respectively.

    Physical motivation
    -------------------
    The step model makes no assumption about the *form* of redshift evolution.
    It simply asks: are the nuisance parameters different at low-z vs high-z?
    This is the maximum-flexibility test.  If the step model is strongly
    preferred over the baseline, it tells you evolution is present but does
    NOT tell you the functional form — run z_evolve_power/log/linear after.
    If the step is NOT preferred, it is strong evidence against any smooth
    evolution model.

    Note: z_pivot is the sample median (set by run.py).  With the step model,
    the pivot is genuinely important — it determines the split.  The median
    ensures equal sample sizes in both bins, maximising sensitivity.
    """
    zp   = Z_PIVOT_RUNTIME if z_pivot is None else z_pivot
    sign = np.where(z >= zp, 1.0, -1.0)
    return 1.0 + a * sign, 1.0 + b * sign, 1.0 + g * sign

Z_EVOLVE_MODELS = {
    "power":  z_evolve_power,
    "log":    z_evolve_log,
    "linear": z_evolve_linear,
    "zz":     z_evolve_zz,
    "exp":    z_evolve_exp,
    "step":   z_evolve_step,
}

# ===========================================================================
# 2.  COSMOLOGICAL DISTANCE MODULUS
# ===========================================================================

def mu_theory(z, Om0, w=-1.0, Ode0=0.7, cosmo_type="FlatLambdaCDM"):
    """
    Return the theoretical distance modulus mu(z) for the given cosmology.

    Note: H0 is degenerate with the absolute magnitude M and is therefore
    fixed at 70 km/s/Mpc.  The M degeneracy is handled analytically in
    cov_log_likelihood via the Betoule/March marginalisation formula.
    """
    if cosmo_type == "FlatLambdaCDM":
        cosmo = FlatLambdaCDM(H0=70, Om0=Om0)
    elif cosmo_type == "wCDM":
        cosmo = wCDM(H0=70, Om0=Om0, Ode0=1 - Om0, w0=w)
    elif cosmo_type == "LambdaCDM":
        cosmo = LambdaCDM(H0=70, Om0=Om0, Ode0=Ode0)
    else:
        raise ValueError(f"Unknown cosmo_type '{cosmo_type}'")
    return cosmo.distmod(z).value

# ===========================================================================
# 3.  DISTANCE MODULUS RESIDUAL  (M excluded — marginalised separately)
# ===========================================================================

def compute_mu_corr(data, params, model_cfg, return_var=False):
    """
    Return the corrected distance modulus array with M excluded.

    Parameters
    ----------
    data      : dict  {z, x0, x1, c, logM, delta_bias, host_colour,
                       logsSFR,             # sSFR; NaN where unavailable
                       logM_knots,          # optional, for mass_spline
                       c_centre, c_ref,     # precomputed colour stats
                       x1_centre,           # precomputed stretch mean
                       logM_draws, host_colour_draws, logsSFR_draws,
                                            # (N, K) Gauss-Hermite quadrature
                                            # draws around each point estimate,
                                            # built once in run.py; falls back
                                            # to the point estimate (K=1) if
                                            # absent, so this function also
                                            # works on a plain data dict that
                                            # has no error columns.
                       gh_weights}          # (K,) quadrature weights, sum 1
    params    : dict  full parameter set (active + fixed combined)
    model_cfg : dict  model-selection strings
    return_var: bool  if True, also return Var[mu_corr] induced by the host
                      measurement errors (see below).

    Returns
    -------
    mu_corr : ndarray, shape (N,)
        or, when return_var=True, the tuple (mu_corr, var_mu).

    Host measurement error: mean vs variance
    ----------------------------------------
    The quadrature below gives E[f(x_true)] for each host profile, which
    corrects the *bias* from evaluating a nonlinear f at a noisy x.  It does
    not, on its own, account for the extra *scatter* that the same noise
    injects into mu_corr.  Doing that properly requires adding Var[mu_corr]
    to the covariance diagonal, which makes the covariance parameter
    dependent and so forces a refactorisation on every likelihood call (see
    cov_log_likelihood_hetero).  That is why it is opt-in via return_var
    rather than always on.

    Host environment term
    ---------------------
    G = gamma/2 * L  +  eta * H           +  xi_mass_col  * L*H
      + zeta   * S   +  xi_sSFR_col * S*H  +  xi_sSFR_mass * S*L  +  omega * S*L*H

    where S = mass profile, H = host-colour profile, F = sSFR profile.
    """
    z           = data["z"]
    x0          = data["x0"]
    x1          = data["x1"]
    c           = data["c"]
    logM        = data["logM"]
    delta_bias  = data["delta_bias"]
    host_colour = data["host_colour"]
    logsSFR     = data.get("logsSFR", np.zeros_like(z))  # zeros if absent

    alpha       = params["alpha"]
    beta        = params["beta"]
    gamma       = params["gamma"]
    a           = params["a"]
    b           = params["b"]
    g           = params["g"]
    beta_alpha  = params["beta_alpha"]
    gamma_alpha = params["gamma_alpha"]
    beta_gamma  = params["beta_gamma"]
    eta         = params["eta"]
    xi_mass_col = params["xi_mass_col"]
    M0          = params["M0"]
    M1          = params.get("M1",    11.0)   # double_step upper threshold
    c0          = params["c0"]
    C0          = params["C0"]
    tau         = params.get("tau",    0.2)
    htau        = params.get("htau",   0.2)
    sn_tau      = params.get("sn_tau", 0.3)
    k1          = params.get("k1",     0.0)   # spline knot coefficients
    k2          = params.get("k2",     0.0)
    k3          = params.get("k3",     0.0)
    # x1 correction
    x1_0        = params.get("x1_0",   0.0)
    x1_tau      = params.get("x1_tau", 1.0)
    # sSFR host term
    zeta        = params.get("zeta",         0.0)
    xi_sSFR_col  = params.get("xi_sSFR_col",  0.0)
    xi_sSFR_mass = params.get("xi_sSFR_mass", 0.0)
    omega       = params.get("omega",        0.0)
    F0          = params.get("F0",          -10.5)
    ftau        = params.get("ftau",         0.5)

    # Guard: x0 must be positive for log10
    if np.any(x0 <= 0):
        bad = np.full(len(x0), np.nan)
        return (bad, bad) if return_var else bad

    # Guard: double_step requires M1 > M0
    if model_cfg["mass"] == "double_step" and M1 <= M0:
        bad = np.full(len(x0), np.nan)
        return (bad, bad) if return_var else bad

    # SN colour correction
    # c_centre is only consumed by sn_colour_quadratic; all other models
    # receive it via **_ and ignore it harmlessly.
    # c_ref is only consumed by sn_colour_dust for its normalisation;
    # all other models receive it via **_ and ignore it.
    c_eff = SN_COLOUR_MODELS[model_cfg["sn_colour"]](
        c, c0=c0, sn_tau=sn_tau,
        c_centre=data.get("c_centre", 0.0),
        c_ref=data.get("c_ref", 0.1))

    # x1 (stretch) correction
    # x1_centre is only consumed by x1_correction_quadratic; others ignore.
    x1_eff = X1_CORRECTION_MODELS[model_cfg.get("x1_correction", "linear")](
        x1, x1_0=x1_0, x1_tau=x1_tau,
        x1_centre=data.get("x1_centre", 0.0), 
        x1_ref=data.get("x1_ref", 0.5))
    
    # print()

    # ---- Host mass / host colour / sSFR profiles, marginalised over each
    #      SN's measurement error via Gauss-Hermite quadrature -------------
    # *_draws has shape (N, K): K quadrature abscissas per SN, built once in
    # run.py from that SN's point estimate and its measurement error.  If
    # run.py (or a caller) hasn't supplied draws — e.g. plain point-estimate
    # data dicts used by older code/tests — fall back to a single "draw" at
    # the point estimate itself (K=1, weight=1), which reproduces the exact
    # old point-estimate behaviour at zero extra cost.
    logM_draws = data.get("logM_draws", logM[:, None])
    host_draws = data.get("host_colour_draws", host_colour[:, None])
    sfr_draws  = data.get("logsSFR_draws", logsSFR[:, None])
    gh_weights = data.get("gh_weights", np.array([1.0]))

    # Host mass profile — every MASS_MODELS function is elementwise, so it
    # returns an (N, K) array unchanged in shape; the quadrature weights then
    # collapse the K axis to the expectation E[L] under logM's measurement
    # error (mass_spline's np.interp also broadcasts correctly over 2-D x).
    L_k = MASS_MODELS[model_cfg["mass"]](
        logM_draws, M0=M0, tau=tau, M1=M1,
        k1=k1, k2=k2, k3=k3,
        logM_knots=data.get("logM_knots"))
    L = L_k @ gh_weights

    # Host colour correction
    H_k = HOST_COLOUR_MODELS[model_cfg["host_colour"]](
        host_draws, C0=C0, htau=htau)
    H = H_k @ gh_weights

    # sSFR profile.  ssfr_* models already zero out NaN (missing sSFR) rows;
    # run.py preserves per-row NaN in sfr_draws so that masking still applies
    # correctly under quadrature.
    S_k = SSFR_MODELS[model_cfg.get("ssfr", "none")](
        sfr_draws, F0=F0, ftau=ftau)
    S = S_k @ gh_weights

    # Combined environment correction (full 8-term host expression)
    G = (gamma / 2.0     * L
         + eta           * H
         + xi_mass_col   * L * H
         + zeta          * S
         + xi_sSFR_col   * S * H
         + xi_sSFR_mass  * S * L
         + omega         * S * L * H)

    # Redshift evolution factors
    f_alpha, f_beta, f_gamma = Z_EVOLVE_MODELS[model_cfg["z_evolve"]](z, a, b, g)

    # Full distance modulus corrected for SALT2 nuisance parameters
    mu_corr = (-2.5 * np.log10(x0)
               + alpha * x1_eff * f_alpha
               - beta  * c_eff  * f_beta
               + G * f_gamma
               + beta_alpha  * c_eff  * x1_eff * f_beta
               + gamma_alpha * x1_eff * G       * f_gamma
               + beta_gamma  * c_eff  * G       * f_beta
               - delta_bias)

    if not return_var:
        return mu_corr

    # ---- Var[mu_corr] from host-property measurement error ----------------
    # G is multilinear in (L, H, S), whose measurement errors are mutually
    # independent, so E[G^2] factorises into univariate moments and needs no
    # K^3 tensor grid: for monomials M_k, M_l with exponent vectors
    # (p,q,r) in {0,1},
    #     E[M_k M_l] = E[L^(p_k+p_l)] E[H^(q_k+q_l)] E[S^(r_k+r_l)]
    # and every required moment (order 0, 1 or 2) comes from the same 1-D
    # quadrature already used above.  Var[G] = E[G^2] - E[G]^2 is therefore
    # exact to the accuracy of the quadrature, with no linearisation.
    one = np.ones_like(L)
    mL = (one, L, L_k * L_k @ gh_weights)     # E[L^0], E[L^1], E[L^2]
    mH = (one, H, H_k * H_k @ gh_weights)
    mS = (one, S, S_k * S_k @ gh_weights)

    #        coefficient      (p, q, r)  exponents of (L, H, S)
    monomials = ((gamma / 2.0,  (1, 0, 0)),
                 (eta,          (0, 1, 0)),
                 (xi_mass_col,  (1, 1, 0)),
                 (zeta,         (0, 0, 1)),
                 (xi_sSFR_col,  (0, 1, 1)),
                 (xi_sSFR_mass, (1, 0, 1)),
                 (omega,        (1, 1, 1)))

    EG2 = np.zeros_like(G)
    for ck, (pk, qk, rk) in monomials:
        if ck == 0.0:
            continue
        for cl, (pl, ql, rl) in monomials:
            if cl == 0.0:
                continue
            EG2 = EG2 + (ck * cl) * mL[pk + pl] * mH[qk + ql] * mS[rk + rl]

    var_G = np.maximum(EG2 - G * G, 0.0)      # clip quadrature round-off

    # G enters mu_corr with this total multiplier, so Var[mu] = w^2 Var[G].
    w = (f_gamma
         + gamma_alpha * x1_eff * f_gamma
         + beta_gamma  * c_eff  * f_beta)

    return mu_corr, w * w * var_G

# ===========================================================================
# 4.  LIKELIHOOD  (analytic M-marginalisation via Betoule/March formula)
# ===========================================================================

def cov_log_likelihood(mu_corr, mu_cosmo, inv_cov, log_det_const, C_sum):
    """
    Gaussian log-likelihood with M analytically marginalised.

      chi2_marg = chi2_tilde - B^2/C + ln(C / 2pi)

    where
      delta      = mu_corr - mu_cosmo  (residual, M-free)
      chi2_tilde = delta^T C^{-1} delta
      B          = 1^T C^{-1} delta
      C_sum      = 1^T C^{-1} 1        (scalar, precomputed — passed as inv_cov)

    Implementation note
    -------------------
    inv_cov is the explicit dense inverse covariance matrix (precomputed once
    in run.py via Cholesky solve).  For N=1820 this is the fastest path:
    a single matrix–vector product v = inv_cov @ delta (BLAS dgemv, ~3.3M
    flops) gives both chit2 = delta@v and B = ones@v in one pass.

    Parameters
    ----------
    mu_corr      : ndarray (N,)
    mu_cosmo     : ndarray (N,)
    inv_cov      : ndarray (N,N)  precomputed inverse covariance
    log_det_const: float          ln|C| + N*ln(2pi), precomputed once

    Returns
    -------
    float : log-likelihood
    """
    delta     = mu_corr - mu_cosmo
    if np.any(~np.isfinite(delta)):
        return -1e30
    v         = inv_cov @ delta            # single BLAS dgemv; reused below
    chit2     = float(delta @ v)
    B         = float(np.sum(v))
    chi2_marg = chit2 - (B**2 / C_sum) + np.log(C_sum / (2.0 * np.pi))
    return -0.5 * (chi2_marg + log_det_const)

def cov_log_likelihood_hetero(mu_corr, var_mu, mu_cosmo, cov_base, n_ln2pi):
    """
    As cov_log_likelihood, but with a parameter-dependent diagonal added to
    the covariance, so the factorisation cannot be cached.

    Used when host-property measurement error is propagated as a variance
    (not just as a bias correction on the mean).  The extra per-SN variance
    var_mu depends on the sampled parameters, so C = cov_base + diag(var_mu)
    changes on every call and must be re-factorised.  That is O(N^3) per
    likelihood evaluation, versus the O(N^2) matrix-vector product of the
    cached-inverse path, so this is reserved for targeted systematic checks
    rather than production sweeps.

    A perturbative (Neumann series) update of the cached inverse was tested
    and rejected: for realistic sSFR errors the correction diagonal reaches
    ~25x the covariance diagonal, where the series diverges outright.

    Parameters
    ----------
    mu_corr  : ndarray (N,)
    var_mu   : ndarray (N,)  extra variance from host measurement error
    mu_cosmo : ndarray (N,)
    cov_base : ndarray (N,N) covariance INCLUDING muerr^2/sigma_int but
                             EXCLUDING var_mu
    n_ln2pi  : float         N * ln(2 pi), precomputed once

    Returns
    -------
    float : log-likelihood
    """
    delta = mu_corr - mu_cosmo
    if np.any(~np.isfinite(delta)) or np.any(~np.isfinite(var_mu)):
        return -1e30

    n = delta.shape[0]
    C = cov_base.copy()
    C.reshape(-1)[::n + 1] += var_mu          # in-place diagonal update
    try:
        chol = cho_factor(C, lower=True, overwrite_a=True)
    except LinAlgError:
        return -1e30

    log_det = 2.0 * np.sum(np.log(np.diag(chol[0]))) + n_ln2pi
    # Solve for delta and 1 in a single call (two right-hand sides).
    rhs   = np.column_stack((delta, np.ones(n)))
    sol   = cho_solve(chol, rhs)
    chit2 = float(delta @ sol[:, 0])
    B     = float(np.sum(sol[:, 0]))
    C_sum = float(np.sum(sol[:, 1]))
    if not np.isfinite(C_sum) or C_sum <= 0:
        return -1e30

    chi2_marg = chit2 - (B**2 / C_sum) + np.log(C_sum / (2.0 * np.pi))
    return -0.5 * (chi2_marg + log_det)

# ===========================================================================
# 5.  PARAMETER HELPER
# ===========================================================================

def build_param_getter(param_specs, active_names):
    """
    Return a closure  get_params(theta) -> dict  that merges a sampled theta
    vector with the fixed/default values of inactive parameters.

    Resolution order (highest priority first)
    -----------------------------------------
    1. Parameter is active  → use sampled theta value.
    2. Inactive with non-None "fixed"  → use that value.
    3. Inactive with fixed=None  → use spec["mu"] (or 0).
    """
    active_set = set(active_names)
    fixed_vals = {}
    for name, spec in param_specs.items():
        if name in active_set:
            continue
        fixed_vals[name] = spec["fixed"] if spec["fixed"] is not None else spec.get("mu", 0.0)

    def get_params(theta):
        params = dict(fixed_vals)
        for i, name in enumerate(active_names):
            params[name] = theta[i]
        return params

    return get_params

# ===========================================================================
# 6.  PRIOR TRANSFORM  (built dynamically from param_specs)
# ===========================================================================

def make_prior_transform(param_specs, active_names):
    """
    Return a prior_transform(u) -> x mapping the unit hypercube to physical
    parameter space, using only active parameters.
    """
    def prior_transform(u):
        x = np.empty_like(u)
        for i, name in enumerate(active_names):
            spec   = param_specs[name]
            lo, hi = spec["range"]
            ptype  = spec["prior"]

            if ptype == "uniform":
                x[i] = lo + (hi - lo) * u[i]

            elif ptype == "gaussian":
                mu, sigma = spec["mu"], spec["sigma"]
                x[i] = np.clip(mu + sigma * ndtri(np.clip(u[i], 1e-6, 1 - 1e-6)), lo, hi)

            elif ptype == "truncated_gaussian":
                mu, sigma = spec["mu"], spec["sigma"]
                a_std = (lo - mu) / sigma
                b_std = (hi - mu) / sigma
                x[i]  = truncnorm.ppf(np.clip(u[i], 1e-6, 1 - 1e-6),
                                      a_std, b_std, loc=mu, scale=sigma)

            elif ptype == "log_uniform":
                if lo <= 0:
                    raise ValueError(f"log_uniform prior for '{name}' requires range[0] > 0")
                x[i] = loguniform.ppf(np.clip(u[i], 1e-6, 1 - 1e-6), lo, hi)

            elif ptype == "log_normal":
                mu_ln, sigma_ln = spec["mu"], spec["sigma"]
                raw  = np.exp(mu_ln + sigma_ln * ndtri(np.clip(u[i], 1e-6, 1 - 1e-6)))
                x[i] = np.clip(raw, lo, hi)

            elif ptype == "arcsinh":
                scale = spec.get("scale", 1.0)
                lo_t  = np.arcsinh(lo / scale)
                hi_t  = np.arcsinh(hi / scale)
                x[i]  = scale * np.sinh(lo_t + (hi_t - lo_t) * u[i])

            else:
                raise ValueError(f"Unknown prior type '{ptype}' for '{name}'")

        return x

    return prior_transform

# ===========================================================================
# 7.  LOGLIKE FACTORY
# ===========================================================================

def infer_cosmo_type(param_specs):
    """
    Infer cosmology type from active parameters.
      Ode0 active, w inactive  →  LambdaCDM   (non-flat, w=-1)
      w active, Ode0 inactive  →  wCDM        (flat, free w)
      neither                  →  FlatLambdaCDM
      both                     →  ValueError
    """
    w_active    = param_specs.get("w",    {}).get("active", False)
    Ode0_active = param_specs.get("Ode0", {}).get("active", False)
    if w_active and Ode0_active:
        raise ValueError("Both 'w' and 'Ode0' active — choose one.")
    if Ode0_active: return "LambdaCDM"
    if w_active:    return "wCDM"
    return "FlatLambdaCDM"

def _cosmo_kwargs(params, cosmo_type):
    Om0 = params["Om0"]
    if cosmo_type == "FlatLambdaCDM":
        return {"Om0": Om0, "cosmo_type": cosmo_type}
    elif cosmo_type == "wCDM":
        return {"Om0": Om0, "w": params["w"], "cosmo_type": cosmo_type}
    elif cosmo_type == "LambdaCDM":
        return {"Om0": Om0, "Ode0": params["Ode0"], "cosmo_type": cosmo_type}
    else:
        raise ValueError(f"Unknown cosmo_type '{cosmo_type}'")

def make_loglike(data, inv_cov_mat, log_det_const, C_sum,
                 param_specs, active_names, model_cfg, cosmo_type,
                 cov_base=None):
    """
    Return a dynesty-compatible loglike(theta) -> float closure.
    inv_cov_mat is the precomputed dense inverse (passed from run.py).

    If cov_base is not None, the host-property measurement error is
    propagated as a variance as well as a bias: the covariance becomes
    cov_base + diag(Var[mu_corr]) and is re-factorised on every call.  This
    is much slower (O(N^3) per likelihood) and is intended for systematic
    checks; see cov_log_likelihood_hetero.
    """
    get_params = build_param_getter(param_specs, active_names)
    hetero     = cov_base is not None
    n_ln2pi    = len(data["z"]) * np.log(2.0 * np.pi) if hetero else 0.0

    def loglike(theta):
        params = get_params(theta)
        try:
            mu_cosmo = mu_theory(data["z"], **_cosmo_kwargs(params, cosmo_type))
        except Exception:
            return -1e30
        if hetero:
            mu_corr, var_mu = compute_mu_corr(data, params, model_cfg,
                                              return_var=True)
            return cov_log_likelihood_hetero(mu_corr, var_mu, mu_cosmo,
                                             cov_base, n_ln2pi)
        mu_corr = compute_mu_corr(data, params, model_cfg)
        return cov_log_likelihood(mu_corr, mu_cosmo, inv_cov_mat, log_det_const, C_sum)

    return loglike

def get_best_fit(results, active_names):
    """Return the maximum-likelihood sample as a dict."""
    idx = np.argmax(results.logl)
    return dict(zip(active_names, results.samples[idx]))

# ===========================================================================
# 8.  COVARIANCE LOADER
# ===========================================================================

def build_covariance(filename):
    """
    Load the Dovekie SNe Ia covariance matrix from a .npz file.

    The file stores the *inverse* covariance in upper-triangular format.
    This function reconstructs the full symmetric inv_cov matrix, then
    solves for the covariance via Cholesky factorisation (numerically
    superior to np.linalg.inv for near-singular matrices).

    Returns: cov_mat  (N x N float64)
    The caller (run.py) then:
      1. Adds diagonal noise terms (muerr^2, sigma_int^2).
      2. Cholesky-factors the result for log-det.
      3. Computes inv_cov_mat via cho_solve for the likelihood hot path.
    """
    print(f"Loading Dovekie SN covariance from {filename}")
    d = np.load(filename)
    n = int(d[d.files[0]][0])
    inv_cov = np.zeros((n, n))
    inv_cov[np.triu_indices(n)] = d[d.files[1]]
    i_lower = np.tril_indices(n, -1)
    inv_cov[i_lower] = inv_cov.T[i_lower]
    # Cholesky-solve for the covariance (more stable than np.linalg.inv)
    try:
        chol_fac = cho_factor(inv_cov, lower=True)
        cov_mat  = cho_solve(chol_fac, np.eye(n))
    except LinAlgError:
        print("  Warning: Cholesky failed for stored inv_cov; falling back to np.linalg.inv")
        cov_mat = np.linalg.inv(inv_cov)
    return cov_mat