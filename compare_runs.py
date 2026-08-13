"""
compare_runs.py  —  SNe Ia Cosmology Pipeline
================================================
Quantitative, publication-grade comparison of two (or more) saved runs:
different correction models, different mass_cut / redshift-bin subsamples,
or different survey/dataset runs fit with this same pipeline. Intended to be
reused throughout — model-selection checks, cross-survey consistency,
systematics-missing diagnostics (see run.py's residual-trend plot) — not a
one-off script.

Two independent questions, two independent tools
--------------------------------------------------
  Bayesian evidence   (bayes_factor)
      "which model does the DATA prefer, penalising complexity" — you
      already have this via results.logz; this module just wraps it with
      the standard Jeffreys'-scale label for convenience.

  Parameter-space tension   (this module's main contribution)
      "given two independently-inferred posteriors on (a subset of) the
       same parameters, how far apart are they, expressed as an
       equivalent Gaussian n-sigma" — this is the tool for "are these two
       model fits/subsamples/surveys actually consistent", which the
       evidence alone does NOT answer (a model can have a much better
       evidence while still being parameter-consistent with a simpler one,
       or a small evidence difference can still hide a real multi-sigma
       parameter shift).

Two complementary tension estimators are provided, both reported on the
SAME universal probability<->sigma scale (n_sigma = sqrt(2)*erfinv(P), the
1D-Gaussian-equivalent sigma for a region enclosing probability P — see
Raveri & Hu 2018, Lemos+ 2021), so a "Gaussian tension 2.1 sigma" and a
"KDE tension 2.1 sigma" mean exactly the same thing and can be quoted side
by side:

  gaussian_tension()
      Mahalanobis/chi^2 distance between the two posteriors' weighted mean
      and covariance:  Q = dM^T (C1+C2)^-1 dM ~ chi2_k.
      Exact for Gaussian posteriors, fast, always available, the standard
      choice in the literature (e.g. Planck-style internal consistency
      tests). Can be MISLEADING for multimodal / strongly non-Gaussian
      posteriors (e.g. mass="step" runs near a poorly-constrained M0, or
      any run flagged by run.py's diagnose_modes as multimodal) — always
      cross-check against the overlay corner plot this module also
      produces.

  kde_shift_probability()
      Nonparametric "parameter-difference" statistic (Raveri & Hu 2018):
      Monte-Carlo draws of the parameter DIFFERENCE Delta = theta1 - theta2
      are formed by randomly pairing equal-weight posterior samples, a
      Gaussian KDE is fit to the resulting difference cloud, and the
      statistic asks what fraction of that cloud is MORE probable than the
      "zero shift" point. Handles non-Gaussian / mildly multimodal
      posteriors correctly, at the cost of the curse of dimensionality —
      restrict to <= ~5 shared parameters (kde_max_dims); for higher-
      dimensional comparisons, pass a smaller, physically-motivated subset
      of shared_params rather than every active nuisance parameter.

Validated against synthetic Gaussian/non-Gaussian test cases with known
analytic separations before being wired to real dynesty output — see the
worked examples at the bottom of this docstring's companion chat message.

Usage
-----
  from compare_runs import compare_two_runs, tension_matrix

  # Single pairwise comparison (e.g. mass="step" vs mass="tanh")
  summary = compare_two_runs("stepmodel_results.pkl", "tanhmodel_results.pkl",
                             output_prefix="mass_step_vs_tanh")

  # Cross-subsample / cross-survey consistency grid
  df = tension_matrix(
      ["all_results.pkl", "masscut_low_results.pkl", "masscut_high_results.pkl"],
      labels=["All", "Low mass", "High mass"],
      output_prefix="mass_subsample_consistency")
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.stats import chi2, gaussian_kde
from scipy.special import erfinv

from dynesty import utils as dyfunc
import dynesty.plotting as dyplot

from run    import load_results      # reuse the existing pickle format exactly
from config import PARAM_DISPLAY

# ===========================================================================
# 1.  CORE STATISTICS  (pure numpy/scipy — independent of dynesty objects,
#     unit-testable on synthetic arrays)
# ===========================================================================

def enclosed_prob_to_nsigma(p):
    """
    Convert an "enclosed probability" P (the credible level of the smallest
    HPD region containing a given point) into a number-of-sigma on the
    universal scale used throughout the tension-metric literature:

        n_sigma = sqrt(2) * erfinv(P)

    This is the 1D-Gaussian-equivalent sigma for a region enclosing
    probability P, and is dimension-agnostic — a Gaussian-tension result
    and a KDE-tension result are therefore always directly comparable.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0 - 1e-12)
    return np.sqrt(2.0) * erfinv(p)


def gaussian_tension(mean1, cov1, mean2, cov2):
    """
    Mahalanobis/chi^2 "difference of Gaussians" tension between two
    posteriors, assumed independent, over the same (shared) parameter
    vector.

        Q = (mean1 - mean2)^T (cov1 + cov2)^-1 (mean1 - mean2)  ~ chi2_k

    Returns
    -------
    dict: Q, dof, pvalue (P(chi2_k > Q) — the usual "probability of a
    shift this large or larger by chance"), nsigma.
    """
    delta = np.atleast_1d(mean1) - np.atleast_1d(mean2)
    k     = len(delta)
    C     = np.atleast_2d(cov1) + np.atleast_2d(cov2)
    Cinv  = np.linalg.inv(C)
    Q     = float(delta @ Cinv @ delta)
    pval  = float(chi2.sf(Q, df=k))
    return {"Q": Q, "dof": k, "pvalue": pval,
            "nsigma": float(enclosed_prob_to_nsigma(1.0 - pval))}


def kde_shift_probability(samples1, samples2, rng=None, n_diff=20000,
                          n_eval=4000):
    """
    Nonparametric parameter-difference tension (Raveri & Hu 2018).

    Forms Monte Carlo draws of the parameter DIFFERENCE Delta = theta1 -
    theta2 by randomly pairing (with replacement) equal-weight samples
    from each posterior, fits a Gaussian KDE to the resulting difference
    cloud, and computes what fraction of that cloud is MORE probable
    (under the KDE) than the "zero shift" point Delta = 0. That fraction
    IS the credible level of the smallest HPD region containing zero
    shift, converted to a universal n-sigma via enclosed_prob_to_nsigma.

    Parameters
    ----------
    samples1, samples2 : (N1, k), (N2, k)  EQUAL-WEIGHT samples (already
        resampled from the nested-sampling importance weights — see
        dyfunc.resample_equal / _equal_weight_samples below) restricted to
        the same k shared parameters, in the same column order.
    n_diff : number of difference-cloud draws used to FIT the KDE.
    n_eval : number of INDEPENDENTLY drawn difference samples used to
        estimate the enclosed-probability fraction — kept separate from
        the KDE fit set so the estimate doesn't "grade its own homework"
        (Delta=0 would otherwise look artificially probable simply because
        it sits amid its own fitting data).

    Returns
    -------
    dict: p_enclosed, nsigma, n_shared_dims, n_diff, n_eval.
    """
    rng = np.random.default_rng(rng)
    k   = samples1.shape[1]

    def _diff_draws(n):
        i = rng.integers(0, len(samples1), size=n)
        j = rng.integers(0, len(samples2), size=n)
        return samples1[i] - samples2[j]

    diff_fit  = _diff_draws(n_diff)
    diff_eval = _diff_draws(n_eval)

    kde       = gaussian_kde(diff_fit.T)          # scipy expects shape (k, N)
    dens_zero = kde(np.zeros((k, 1)))[0]
    dens_eval = kde(diff_eval.T)

    p_enclosed = float(np.mean(dens_eval > dens_zero))
    return {"p_enclosed": p_enclosed,
            "nsigma": float(enclosed_prob_to_nsigma(p_enclosed)),
            "n_shared_dims": k, "n_diff": n_diff, "n_eval": n_eval}


def bayes_factor(results1, results2):
    """
    ln(Bayes factor) = logZ1 - logZ2, with the standard Jeffreys'-scale
    interpretation (Kass & Raftery 1995) as a human-readable label.
    Positive => data favour model/run 1; negative => favour model/run 2.
    Error propagated in quadrature from dynesty's logzerr.
    """
    lnB     = float(results1.logz[-1] - results2.logz[-1])
    lnB_err = float(np.hypot(results1.logzerr[-1], results2.logzerr[-1]))
    a = abs(lnB)
    if   a < 1.0: label = "inconclusive"
    elif a < 2.5: label = "weak"
    elif a < 5.0: label = "moderate"
    else:         label = "strong"
    return {"lnB": lnB, "lnB_err": lnB_err, "label": label,
            "favours": "run_1" if lnB > 0 else "run_2"}

# ===========================================================================
# 2.  DYNESTY GLUE  (loading, resampling, shared-parameter alignment)
# ===========================================================================

def _equal_weight_samples(results):
    """Resample a dynesty Results object to equal-weight posterior samples."""
    weights = np.exp(results.logwt - results.logz[-1])
    return dyfunc.resample_equal(results.samples, weights)


def _weighted_mean_cov(results, active_names, shared):
    idx = [active_names.index(p) for p in shared]
    weights = np.exp(results.logwt - results.logz[-1])
    return dyfunc.mean_and_cov(results.samples[:, idx], weights)


def shared_parameters(active_names_1, active_names_2):
    """
    Parameters that were SAMPLED (active) in both runs, in the order they
    appear in active_names_1. Fixed/inactive parameters are excluded even
    if present in both param_specs — comparing a sampled posterior to a
    fixed constant is not a meaningful tension statistic.
    """
    return [p for p in active_names_1 if p in active_names_2]


def _restrict(samples, active_names, shared):
    idx = [active_names.index(p) for p in shared]
    return samples[:, idx]

# ===========================================================================
# 3.  OVERLAY CORNER PLOT
# ===========================================================================

def plot_comparison_corner(results1, active1, results2, active2, shared,
                           labels, output_prefix, annotation=None):
    """
    Overlay corner plot of two runs, restricted to their shared active
    parameters, reusing dynesty's own cornerplot for both — draw run 1's
    cornerplot, then draw run 2's onto the SAME figure/axes with a
    different colour, which is the standard dynesty pattern for overlays.
    """
    dims1 = [active1.index(p) for p in shared]
    dims2 = [active2.index(p) for p in shared]
    display_labels = [PARAM_DISPLAY.get(p, {}).get("label", p) for p in shared]

    fig, axes = dyplot.cornerplot(
        results1, dims=dims1, labels=display_labels,
        color="steelblue", show_titles=False, max_n_ticks=3)

    dyplot.cornerplot(
        results2, dims=dims2, labels=display_labels,
        color="crimson", show_titles=False, max_n_ticks=3,
        fig=(fig, axes))

    proxies = [Line2D([0], [0], color="steelblue", lw=3, label=labels[0]),
               Line2D([0], [0], color="crimson",  lw=3, label=labels[1])]
    fig.legend(handles=proxies, loc="upper right", fontsize=11,
              bbox_to_anchor=(0.98, 0.98))

    if annotation:
        fig.suptitle(annotation, fontsize=13, y=1.01)

    path = f"{output_prefix}_overlay_corner.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Overlay corner plot saved: {path}")
    return path

# ===========================================================================
# 4.  TOP-LEVEL PAIRWISE COMPARISON
# ===========================================================================

def compare_two_runs(pkl_path_1, pkl_path_2, output_prefix,
                     kde_max_dims=5, rng=0, make_plot=True):
    """
    Full publication-grade comparison of two saved runs — different
    correction models, different mass_cut/z-bin subsamples, or different
    survey/dataset runs fit with this pipeline.

    Writes
    ------
    <output_prefix>_tension_registry.csv
        One row per call, appended if the file already exists (same
        append-or-create pattern as run.py's update_registry).
    <output_prefix>_overlay_corner.pdf
        Shared-parameter corner overlay (only if make_plot and there are
        >= 2 shared dimensions to plot).

    Returns
    -------
    dict summary (identical to the row written to the tension registry).
    """
    results1, active1, specs1, cfg1 = load_results(pkl_path_1)
    results2, active2, specs2, cfg2 = load_results(pkl_path_2)

    shared = shared_parameters(active1, active2)
    if len(shared) == 0:
        raise ValueError(
            f"No shared active parameters between {pkl_path_1} ({active1}) "
            f"and {pkl_path_2} ({active2}) — nothing to compare.")

    mean1, cov1 = _weighted_mean_cov(results1, active1, shared)
    mean2, cov2 = _weighted_mean_cov(results2, active2, shared)
    gauss = gaussian_tension(mean1, cov1, mean2, cov2)

    if len(shared) <= kde_max_dims:
        eq1 = _restrict(_equal_weight_samples(results1), active1, shared)
        eq2 = _restrict(_equal_weight_samples(results2), active2, shared)
        kde = kde_shift_probability(eq1, eq2, rng=rng)
    else:
        print(f"  Skipping KDE tension: {len(shared)} shared dims > "
              f"kde_max_dims={kde_max_dims} (curse of dimensionality). "
              f"Consider passing a smaller, physically-motivated subset of "
              f"shared params if you need the nonparametric statistic here.")
        kde = {"p_enclosed": np.nan, "nsigma": np.nan,
              "n_shared_dims": len(shared), "n_diff": 0, "n_eval": 0}

    bf = bayes_factor(results1, results2)

    summary = {
        "run_1":            pkl_path_1,
        "run_2":            pkl_path_2,
        "n_shared_params":  len(shared),
        "shared_params":    "|".join(shared),
        "gaussian_nsigma":  round(gauss["nsigma"], 3),
        "gaussian_pvalue":  gauss["pvalue"],
        "gaussian_Q":       round(gauss["Q"], 3),
        "gaussian_dof":     gauss["dof"],
        "kde_nsigma":       (round(kde["nsigma"], 3)
                             if np.isfinite(kde["nsigma"]) else ""),
        "kde_p_enclosed":   kde["p_enclosed"],
        "lnB":              round(bf["lnB"], 3),
        "lnB_err":          round(bf["lnB_err"], 3),
        "lnB_label":        bf["label"],
        "lnB_favours":      bf["favours"],
    }

    print(f"\n{'='*60}")
    print(f"Comparing:\n  1) {pkl_path_1}\n  2) {pkl_path_2}")
    print(f"Shared active parameters ({len(shared)}): {shared}")
    print(f"Gaussian tension  : {gauss['nsigma']:.2f} sigma "
          f"(p={gauss['pvalue']:.4g}, chi2/dof={gauss['Q']:.2f}/{gauss['dof']})")
    if np.isfinite(kde["nsigma"]):
        print(f"KDE tension       : {kde['nsigma']:.2f} sigma "
              f"(p_enclosed={kde['p_enclosed']:.4g})")
    print(f"Bayes factor      : ln B = {bf['lnB']:+.2f} +/- {bf['lnB_err']:.2f}  "
          f"({bf['label']}, favours {bf['favours']})")
    print(f"{'='*60}\n")

    registry_path = f"{output_prefix}_tension_registry.csv"
    df_new = pd.DataFrame([summary])
    if os.path.isfile(registry_path):
        existing = pd.read_csv(registry_path)
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new
    combined.to_csv(registry_path, index=False)
    print(f"Tension registry updated: {registry_path}")

    if make_plot and len(shared) >= 2:
        kde_str = (f"   KDE: {kde['nsigma']:.2f}$\\sigma$"
                  if np.isfinite(kde["nsigma"]) else "")
        plot_comparison_corner(
            results1, active1, results2, active2, shared,
            labels=(os.path.basename(pkl_path_1), os.path.basename(pkl_path_2)),
            output_prefix=output_prefix,
            annotation=f"Gaussian: {gauss['nsigma']:.2f}$\\sigma${kde_str}")

    return summary

# ===========================================================================
# 5.  MANY-WAY CONSISTENCY GRID  (subsamples / surveys / systematics variants)
# ===========================================================================

def tension_matrix(pkl_paths, labels=None, output_prefix="tension_matrix"):
    """
    Pairwise Gaussian-tension matrix across N saved runs — the natural tool
    for "how consistent are these fits across subsamples/surveys" questions
    (mass_cut splits, redshift bins, a different survey run through this
    same pipeline, systematics variants, ...).

    KDE tension is intentionally NOT computed for the full N-way grid — it's
    the expensive, dimension-limited statistic. Run compare_two_runs() on
    whichever pair(s) the Gaussian matrix flags as interesting for the
    nonparametric cross-check.

    Writes <output_prefix>.csv (the matrix) and <output_prefix>.pdf
    (a heatmap). Returns the matrix as a pandas.DataFrame (nsigma values,
    symmetric, diagonal = 0; NaN where two runs share no active parameters).
    """
    n = len(pkl_paths)
    labels = labels or [os.path.basename(p) for p in pkl_paths]
    loaded = [load_results(p) for p in pkl_paths]

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r1, a1, _, _ = loaded[i]
            r2, a2, _, _ = loaded[j]
            shared = shared_parameters(a1, a2)
            if not shared:
                mat[i, j] = mat[j, i] = np.nan
                continue
            m1, c1 = _weighted_mean_cov(r1, a1, shared)
            m2, c2 = _weighted_mean_cov(r2, a2, shared)
            g = gaussian_tension(m1, c1, m2, c2)
            mat[i, j] = mat[j, i] = g["nsigma"]

    df = pd.DataFrame(mat, index=labels, columns=labels)
    df.to_csv(f"{output_prefix}.csv")

    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.0 * n + 2))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=3)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}$\\sigma$", ha="center",
                        va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label=r"Gaussian tension ($n\sigma$)")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.pdf", dpi=150)
    plt.close(fig)
    print(f"Tension matrix saved: {output_prefix}.csv / {output_prefix}.pdf")
    return df