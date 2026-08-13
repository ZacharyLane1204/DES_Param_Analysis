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

The redshift-evolution runs (`evolution/*`) are fitted under **broad uniform**
priors on `alpha`, `beta` and `Om0` so that a low `ln Z` cannot be blamed on
the informative priors. Because widening a prior lowers `ln Z` through its
Occam factor regardless of fit quality, those runs are **not** comparable with
the informative-prior `baseline`. They are instead differenced against
`evolution/baseline_broaduniform`, the no-evolution model fitted under the same
broad priors.

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
                       --uniform-priors --additional-checks --out tables.tex
```

`--additional-checks` reads the uniform registry and takes all deltas against
`uniformpriors/baseline`; `--uniform-priors` documents the widened ranges.

## Host matching

Host-galaxy quality cuts use a DDLR threshold of 2.0 (`host_ddlr_max` in
`config.py`), i.e. the SN must lie within the host light profile.
