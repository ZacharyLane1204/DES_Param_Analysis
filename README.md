# DES_Param_Analysis

Nested-sampling model comparison for DES SNe Ia standardisation and host
correlations.

## Environment

The pipeline pins an exact, reproducible Python 3.9.7 environment. Build it
once with:

```bash
./setup_env.sh                 # add --recreate to rebuild from scratch
conda activate des-param-analysis
```

`setup_env.sh` creates the conda environment from `environment.yml`, installs
the pip-only pins from `requirements.txt`, then verifies every package version
and that the data files resolve. It exits non-zero if anything drifts.

The split between the two files is not arbitrary: `numpy`, `pandas` and
`scikit-learn` must come from conda (no osx-arm64 wheels exist at the pinned
versions), while `matplotlib==3.7.5` and `astropy==5.2.2` must come from pip
(conda-forge has no py39 build of the former, and its build of the latter
forces a numpy upgrade). `contourpy` is pinned to `1.1.1` because newer
releases require `numpy>=1.23` and would silently break the pinned numpy.

## Running the sweeps

```bash
python experiment_runner.py --publication --workers N   # main 268-run sweep
python uniform_priors_check.py                          # broad-uniform checks
python model_comparison_suite.py                        # posterior comparison
python combo_ablation_checks.py                         # term ablations
python extra_runners.py                                 # systematics checks
```

Use `--rerun` to overwrite existing registry rows (`experiment_runner.py`
otherwise skips any tag already present in the registry), and `--tag <prefix>`
to restrict a run to one section, e.g.
`python experiment_runner.py --tag evolution/ --rerun --publication`.

## Prior sensitivity

The redshift-evolution runs (`evolution/*`) are fitted with **broad uniform**
priors on `alpha` and `beta` so that a low `ln Z` cannot be blamed on the
informative priors on the standardisation coefficients.

`Om0` deliberately keeps its informative CMB-level prior in that section. It is
near-degenerate with the evolution exponents over the DES redshift range -- both
change the shape of the distance-redshift relation -- so a free `Om0` would
absorb the redshift dependence the sweep exists to measure, and the exponents
would return consistent with zero for reasons unrelated to the data. Holding the
background cosmology fixed with an external constraint is the point. The
`prior_shrinkage.py` flags on `Om0` in this section are therefore expected and
should not be "fixed".

Because widening `alpha` and `beta` still lowers `ln Z` through the Occam factor,
those runs are **not** comparable with the informative-prior `baseline`. They are
differenced against `evolution/baseline_broaduniform`, the no-evolution model
fitted under the same priors.

`uniform_priors_check.py` extends the same idea to any model or combination of
terms. It writes to `uniform_checks/` and
`run_publication_registry_uniform.csv`, with `uniformpriors/baseline` as the
matched-prior reference. Edit its `TERMS` and `COMBOS` blocks to test
combinations of your chosen best models:

```bash
python uniform_priors_check.py --list          # show all runs
python uniform_priors_check.py --dry-run       # show priors without sampling
python uniform_priors_check.py --combos-only
```

Every registry row records a `prior_overrides` column, and `prior_shrinkage.py`
uses it to skip parameters whose priors were deliberately widened, so uniform
runs are never re-flagged as prior-dominated.

## LaTeX tables

```bash
python latex_tables.py --preamble --priors --iterations --evidence --checks \
                       --uniform-priors --additional-checks --host-error \
                       --out tables.tex
```

`--additional-checks` reads the uniform registry and takes all deltas against
`uniformpriors/baseline`; `--uniform-priors` documents the widened ranges.
`--host-error` builds the host measurement error systematic table from the
`hosterr/` runs in the checks registry (see "Host measurement errors" below).

## Log verbosity

Every runner redirects its run's stdout to `logs/<tag>.log`. dynesty's default
progress writer assumes a terminal and repaints its status line several times a
second, which in a file means hundreds of thousands of near-identical lines and
a log big enough to bury the actual output.

Progress is therefore throttled to one compact heartbeat line every 30 minutes
by default (`progress_interval` in `config.py`), showing elapsed time, iteration,
call count, `logz`, `dlogz` against its target, and sampling efficiency:

```
  [2:30:00]  iter=184203  ncall=1043118  logz=-454.671 +/- 0.212  dlogz=0.0031 (target 0.001)  eff=17.6%
```

Both `run.py` and `experiment_runner.py` accept:

```bash
--progress-interval 3600   # hourly instead
--progress-interval 0      # dynesty's continuous progress bar
--quiet                    # no progress output at all
```

Setup banners, parameter summaries, warnings and the final evidence summary are
never suppressed.

## Host matching

Host-galaxy quality cuts use a DDLR threshold of 2.0 (`host_ddlr_max` in
`config.py`), i.e. the SN must lie within the host light profile.

## Host measurement errors

The host mass, host colour and sSFR all carry measurement error, and the
environment correction is a nonlinear function of them. The likelihood
marginalises each host property over its own error by Gauss-Hermite
quadrature (`n_gh_nodes`), which removes the bias from evaluating a nonlinear
profile at a noisy point estimate. For a *linear* profile this changes
nothing, which is the expected behaviour, not a bug.

Two data problems need handling on top of that.

**Host colour has no error column.** `HOST_COLOR_ERR` is `-999` for all 1820
SNe. Left alone, host colour would be the only host property treated as
exactly measured while mass and sSFR are smoothed, which quietly flatters the
host colour models. Since `HOST_COLOR` and `HOST_LOGMASS` come from the same
SED fit to the same host photometry, the colour error is derived from the mass
error using the Taylor et al. (2011) mass-to-light/colour slope:

```
sigma_colour = HOST_LOGMASS_ERR / host_colour_err_mass_slope   # slope 0.70
```

This gives a median 0.037 mag against a host colour spread of 0.50 mag, and is
conservative because `sigma_logM` also absorbs distance and luminosity terms.
`HOST_LOGMASS_ERR` correlates with host apparent magnitude at +0.61 (fainter
host, larger error - the correct sign) and only weakly with `MUERR` (-0.25).

SN-side proxies were tested and rejected. `mBERR` correlates with `MUERR` at
+0.55 because it is a *component* of `MUERR`, so using it would couple the
x- and y-errors by construction; it also correlates with host magnitude at
-0.24, i.e. the wrong sign, and with redshift at +0.44, which would contaminate
the evolution tests. `SNRMAX1` fails the same way.

This is a **derived** error, never a measured one. Set
`host_colour_err_from_logmass = False` to disable it; if the column is ever
properly populated, the real values take precedence.

**sSFR errors are bimodal.** `HOST_LOGsSFR_ERR` has a well-measured population
plus a failure-mode pileup near 10 dex, larger than the entire ~2.4 dex
population spread, with a clean valley at 2-3 dex between them. Above
`ssfr_err_max` (2.5 dex, masking 457 SNe) the sSFR point estimate is set to
`NaN` so it contributes nothing. Masking is preferred to capping, which would
keep a meaningless value and give it artificial weight. **The SNe stay in the
sample** - dropping them would make evidences incomparable across models.

### Measurement error as a variance

The quadrature above corrects the *bias* (it computes `E[f]`). Accounting for
the extra *scatter* the same error injects into mu requires adding `Var[f]` to
the covariance diagonal. `Var[f]` is computed exactly, with no linearisation:
`G` is multilinear in the three host profiles and their errors are independent,
so `E[G^2]` factorises into univariate moments that the existing 1-D quadrature
already provides - no `K^3` grid is needed.

This is off by default (`host_var_penalty`) because it is expensive: the
covariance becomes parameter dependent, so it must be refactorised on every
likelihood call (O(N^3) instead of O(N^2), roughly 10x slower in practice). A
perturbative update of the cached inverse was tested and rejected - for
realistic sSFR errors the correction diagonal reaches ~25x the covariance
diagonal, where the series diverges.

The term is always a **net penalty** on ln Z: because chi2/dof sits slightly
below 1 for this sample, the log-determinant cost always outweighs the chi2
gain. It cannot be used to flatter a fit.

### Running the systematic check

Edit `HOSTERR_BEST` in `extra_runners.py` to your chosen best model - the same
combination you give `combo_ablation_checks.py` - then:

```bash
python extra_runners.py --tag hosterr/ --publication
python latex_tables.py --host-error --out host_error.tex
```

That fits one model twelve ways, varying only the error treatment: the
reference, `Var[f]` on, the derived colour error off (with and without
`Var[f]`, which isolates the asymmetry), two alternative Taylor slopes, three
sSFR mask thresholds, all host errors off, and a `K=80` quadrature convergence
pair. Every run keeps the same 1820 SNe, so the deltas are like-for-like.

Note that discontinuous profiles (`mass`/`ssfr` = `step`) converge slowly under
Gauss-Hermite quadrature, and the second moment converges more slowly than the
first. If your best model uses a step, check the `K=80` rows before trusting
these deltas at the 0.1 ln Z level.
