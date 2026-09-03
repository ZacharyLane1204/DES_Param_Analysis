"""
uniform_priors_check.py  —  SNe Ia Cosmology Pipeline
=========================================================
Broad-uniform-prior reruns: "is this model's posterior (and its ln Z)
driven by the data, or by the informative priors it was fitted under?"

This is deliberately its own script, not part of experiment_runner.py or
extra_runners.py:

  - experiment_runner.py defines the informative-prior DEFAULT_PARAM_SPECS
    sweep. Swapping a parameter's prior shape there changes every entry
    that activates it, which is what we want for a genuinely un-anchored
    parameter (config.py's C0) or for a whole section the prior-shrinkage
    scan condemned (experiment_runner.py's "evolution/" block, now broad
    uniform throughout), but NOT for a targeted "does this specific
    combo's posterior survive without the informative prior" question.

  - extra_runners.py's "checks/" tag prefix is reserved for post-hoc
    robustness checks on the chosen BEST model (host-match quality,
    LOO-z, c-cuts, ...). A prior-shape comparison is a different kind of
    question, asked of specific combos, so it does not belong under that
    prefix. This file's tags use "uniformpriors/" and never "checks/".

Outputs
-------
Everything from this script lands in its own places, kept separate from
the publication/checks outputs on purpose so these can never be mixed
into, skipped against, or deduped with them:

  output_dir     uniform_checks/                       (corner plots, pkls)
  registry_file  run_publication_registry_uniform.csv
  summary        uniform_priors_check_summary.csv
  logs           logs/<tag>.log  (per-entry, same convention as
                 extra_runners.py) + logs/uniformpriors_summary.log
                 (one line per entry, kept separate from extra_runners.py's
                 own logs/summary_kerr.log so a parallel run of both
                 scripts never clobbers the other's master log)

CRITICAL -- ln Z from this registry is NOT comparable to ln Z in
run_publication_registry.csv. Widening a prior always costs evidence
through the Occam factor, so an entry here will generally have a lower
ln Z than its informative-prior counterpart purely because of the prior
volume, with no change in fit quality whatsoever. Compare WITHIN this
registry (each entry against "uniformpriors/baseline", which is fitted
under exactly the same broad uniform priors) and use the informative-
prior registry only for parameter ESTIMATES: a posterior mean that
shifts meaningfully once the informative prior is removed means the
estimate was prior-dominated rather than data-dominated.

Two ways to define what gets run
--------------------------------
1. ENTRIES -- explicit one-off entries, written out in full. Use for
   single models you want tested as-is (the stretch model checks below).

2. TERMS + COMBOS -- the flexible route, same idea as
   combo_ablation_checks.py. TERMS names each reusable correction block
   (a model-dict fragment plus param_overrides); COMBOS lists which named
   terms to merge into each run. So with TERMS "stretch" and "sn_colour"
   defined, COMBOS of

       [],                          -> base model alone
       ["stretch"],                 -> base + stretch
       ["sn_colour"],               -> base + sn colour
       ["stretch", "sn_colour"],    -> base + stretch + sn colour

   gives you the full incremental ladder without hand-writing four
   near-identical _build() calls. This is the intended route once you
   have picked your best models: fill in TERMS with your winners, list
   the combinations in COMBOS, and every one of them is fitted under the
   same broad uniform priors so their ln Z values ARE mutually
   comparable.

Which priors get widened
------------------------
UNIFORM_PRIORS below: alpha and beta get uniform priors over ranges far
wider than both their informative sigma and their old hard clips. Om0
deliberately KEEPS its informative CMB prior -- see the comment on
UNIFORM_PRIORS for why freeing it would defeat the purpose of the check.
Shape parameters that are active in a given entry (x1_tau, sn_tau, tau,
htau, ftau, M0, F0) are widened too, but only where the entry actually
samples them -- see _uniformise(). The same applies to the term-AMPLITUDE
coefficients (gamma_alpha, zeta, eta), which are the parameters that
answer "does this term exist at all" for interaction/ssfr/host_colour
respectively, as opposed to the shape parameters above which only shape
an already-active term. Parameters that are already uniform by default
(gamma, c0, C0, M0, F0, x1_0) need no override; entries for them are kept
anyway as a complete, self-documenting record.

Running in parallel
--------------------
Same model as extra_runners.py: a ProcessPoolExecutor of worker
processes, each running exactly one entry through run_sampler with its
own thread pool clamped to 1 (so `--workers K` consumes exactly K cores
total, not K x whatever BLAS would otherwise grab per process). Every
entry's full run_sampler output -- setup banners, dynesty progress,
warnings, evidence summary -- goes to its own logs/<tag>.log; only a
one-line completion status prints to the console per entry, plus a
master logs/uniformpriors_summary.log. This applies whether you pass
--workers or not (the default runs one worker per selected entry, capped
at os.cpu_count()) and even under --sequential -- every real (non-dry-run)
invocation always logs to files, matching extra_runners.py's convention.

Usage
-----
  python uniform_priors_check.py
  python uniform_priors_check.py --list
  python uniform_priors_check.py --only baseline,stretch_powerlaw
  python uniform_priors_check.py --entries-only
  python uniform_priors_check.py --combos-only
  python uniform_priors_check.py --dry-run

  # Parallel, capped at 4 workers, deprioritised:
  nice -n 19 python uniform_priors_check.py --combos-only --workers 4

  # Force one at a time (debugging) -- still logs to logs/<tag>.log:
  python uniform_priors_check.py --combos-only --sequential

or:
  from uniform_priors_check import run_uniform_priors_check
  report = run_uniform_priors_check(workers=4)
"""

import argparse
import copy
import sys
import os
import time
import traceback
from datetime import datetime

# ===========================================================================
# THREAD CLAMPING  —  must happen BEFORE any numerical library is imported
# ===========================================================================
# Same rationale as extra_runners.py's identical block: NumPy / OpenBLAS /
# MKL / OMP read their thread-count env vars at import time, not at call
# time, and pandas (imported just below) already pulls numpy in -- so this
# has to run before even that import, not just before `from config import
# ...`. With this in place each worker process (spawned, not forked -- see
# the ProcessPoolExecutor call below) uses exactly 1 CPU thread, so
# `--workers K` consumes exactly K cores total.
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
    pass  # threadpoolctl not installed or broken — env vars above are sufficient
# ===========================================================================

import pandas as pd

from config import CONFIG, DEFAULT_PARAM_SPECS
from run    import run_sampler, pkl_path_for

REGISTRY_FILE = "run_publication_registry_uniform.csv"
OUTPUT_DIR    = "uniform_checks"
SUMMARY_FILE  = "uniform_priors_check_summary.csv"
TAG_PREFIX    = "uniformpriors"


# ===========================================================================
# THE UNIFORM PRIORS
# ===========================================================================
# Ranges are deliberately over-wide: the point is to give the data room to
# move the posterior anywhere it likes, so that if it does NOT move, that
# is a genuine statement about the data rather than about the prior.
#
# alpha/beta mirror experiment_runner.py's _ZEVO_BROAD_UNIFORM exactly, so
# an "evolution/" run and a "uniformpriors/" run of the same model are
# fitted under identical nuisance priors and their parameter estimates are
# directly comparable.
#
# Om0 IS NOT WIDENED, HERE OR IN THE EVOLUTION SWEEP.
# ---------------------------------------------------
# Its truncated_gaussian(0.3175, 0.0275) is an external CMB constraint,
# not a guess that needs testing, and it is far tighter than this sample
# can deliver on its own. Freeing it does not test prior sensitivity of
# the standardisation model; it just lets the background cosmology slide
# to soak up whatever the standardisation terms fail to absorb, so every
# model comes out looking equally adequate and the comparison loses its
# power to discriminate. Holding Om0 fixed at CMB precision is what makes
# the residual differences between these models attributable to the
# models.
#
# The consequence is that prior_shrinkage.py will keep flagging Om0 as
# prior-dominated in these runs. That is the intended state, not a defect
# -- Om0 is not in the prior_overrides column precisely because it was
# never overridden.
UNIFORM_PRIORS = {
    # SALT3 standardisation coefficients (truncated_gaussian by default).
    # Om0 is intentionally absent -- see the block comment above.
    "alpha":  {"prior": "uniform", "range": [0.0, 0.5]},
    "beta":   {"prior": "uniform", "range": [0.0, 8.0]},
    # Shape / width parameters (log_uniform by default -- already fairly
    # uninformative on SUPPORT, spanning orders of magnitude, but their
    # SHAPE (log density) is still informative; these swap that shape to
    # flat uniform over the same existing hard range).
    "x1_tau": {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["x1_tau"]["range"]},
    "sn_tau": {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["sn_tau"]["range"]},
    "tau":    {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["tau"]["range"]},
    "htau":   {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["htau"]["range"]},
    "ftau":   {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["ftau"]["range"]},
    # M0/F0 are already "uniform" by default in DEFAULT_PARAM_SPECS -- these
    # two entries are a no-op restating the existing prior. Left in
    # deliberately (rather than dropped) so this dict is a complete,
    # self-documenting record of every shape/width parameter the check
    # considers, not because they change anything.
    "M0":     {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["M0"]["range"]},
    "F0":     {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["F0"]["range"]},
    # ---- Term-AMPLITUDE coefficients (arcsinh by default) --------------
    # gamma_alpha/zeta/eta are the parameters that answer "does this term
    # exist at all" for the interaction, ssfr, and host_colour terms
    # respectively -- as opposed to tau/ftau/M0/F0 above, which only shape
    # an already-active term. All three default to an informative
    # arcsinh(scale=...) prior in DEFAULT_PARAM_SPECS. Only add
    # xi_mass_col/omega/beta_alpha/beta_gamma here too if a TERMS entry
    # actually activates them.
    "gamma_alpha": {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["gamma_alpha"]["range"]},
    "zeta":        {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["zeta"]["range"]},
    "eta":         {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["eta"]["range"]},
}

# Always uniformised, whether active or not: these are the parameters the
# whole exercise is about, and they are active in every run anyway.
_ALWAYS_UNIFORM = ("alpha", "beta")


def _uniformise(specs):
    """Apply UNIFORM_PRIORS in place to a param_specs dict.

    _ALWAYS_UNIFORM parameters are always overridden. Everything else in
    UNIFORM_PRIORS is overridden only if that parameter is active in this
    particular entry -- so an entry that doesn't sample x1_tau doesn't
    carry a confusing uniform-x1_tau spec it never used, and the
    "prior_overrides" column run.py writes to the registry stays an honest
    record of what was actually sampled.
    """
    for name, updates in UNIFORM_PRIORS.items():
        if name not in specs:
            continue
        if name in _ALWAYS_UNIFORM or specs[name].get("active"):
            specs[name].update(copy.deepcopy(updates))
    return specs


def _build(tag, param_overrides=None, config_overrides=None, model=None):
    """Build a complete config dict for one entry.

    The tag is prefixed with "uniformpriors/" automatically -- pass the
    bare descriptive name. param_overrides are applied FIRST and the
    uniform priors SECOND, so an entry activating a shape parameter gets
    that parameter uniformised automatically; pass the prior explicitly in
    param_overrides only if you want something other than UNIFORM_PRIORS.
    """
    cfg = copy.deepcopy(CONFIG)
    cfg["run_tag"]       = f"{TAG_PREFIX}/{tag}"
    cfg["registry_file"] = REGISTRY_FILE
    cfg["output_dir"]    = OUTPUT_DIR
    if model:
        cfg["model"] = {**CONFIG["model"], **model}
    if config_overrides:
        cfg.update(copy.deepcopy(config_overrides))

    specs = copy.deepcopy(DEFAULT_PARAM_SPECS)
    for name, updates in (param_overrides or {}).items():
        specs[name].update(updates)
    cfg["param_specs"] = _uniformise(specs)
    return cfg


# ===========================================================================
# 1. ENTRIES  —  explicit one-off runs
# ===========================================================================
# The stretch checks requested: stretch_powerlaw and stretch_doublebroken,
# each in all three of experiment_runner.py's variants (plain, _x1tau,
# _x10x1tau), fitted under the broad uniform priors above. x1_0 is already
# uniform in DEFAULT_PARAM_SPECS, so activating it is enough; x1_tau is
# log_normal by default and _uniformise() swaps it to uniform over its own
# hard range in the entries that sample it.
#
# "baseline" is the matched-prior reference for everything in this file --
# the ordinary baseline model under these same broad uniform priors. Delta
# ln Z against this row is the meaningful comparison; Delta ln Z against
# run_publication_registry.csv's "baseline" is not (see module docstring).
ENTRIES = [

    _build("baseline"),

    # ---- Stretch: power-law ----
    _build("stretch_powerlaw",
           model={"x1_correction": "powerlaw"},
           param_overrides={"x1_0":   {"active": False},
                            "x1_tau": {"active": False}}),

    _build("stretch_powerlaw_x1tau",
           model={"x1_correction": "powerlaw"},
           param_overrides={"x1_0":   {"active": False},
                            "x1_tau": {"active": True}}),

    _build("stretch_powerlaw_x10x1tau",
           model={"x1_correction": "powerlaw"},
           param_overrides={"x1_0":   {"active": True},
                            "x1_tau": {"active": True}}),

    # ---- Stretch: double-broken ----
    _build("stretch_doublebroken",
           model={"x1_correction": "doublebroken"},
           param_overrides={"x1_0":   {"active": False},
                            "x1_tau": {"active": False}}),

    _build("stretch_doublebroken_x1tau",
           model={"x1_correction": "doublebroken"},
           param_overrides={"x1_0":   {"active": False},
                            "x1_tau": {"active": True}}),

    _build("stretch_doublebroken_x10x1tau",
           model={"x1_correction": "doublebroken"},
           param_overrides={"x1_0":   {"active": True},
                            "x1_tau": {"active": True}}),
]


# ===========================================================================
# 2 & 3. TERMS + COMBOS  —  now shared with combo_ablation_checks.py,
# extra_runners.py (HOSTERR_BEST), and z_uncertainty_check.py via
# best_model.py, so a winning term/combo only needs to be edited in ONE
# place. Edit TERMS/COMBOS/BEST_COMBO in best_model.py, not here -- a term
# can still be copy-pasted from/to combo_ablation_checks.py unchanged since
# both files now read from the exact same TERMS dict.
#
# [] (the base model with no terms added, i.e. the same fit as ENTRIES'
# "baseline") is deliberately never a COMBOS entry here, to avoid
# duplicating that tag -- see best_model.py if you want it as its own row.
# ===========================================================================
from best_model import TERMS, COMBOS, merge_terms as _merge_terms


def _combo_tag(term_names):
    return "combo_" + "_".join(term_names) if term_names else "combo_base"


def build_combo_entries(combos=None):
    """Turn COMBOS into the same kind of config dicts ENTRIES holds."""
    combos = COMBOS if combos is None else combos
    built = []
    for names in combos:
        model_overrides, param_overrides = _merge_terms(names)
        built.append(_build(_combo_tag(names),
                            param_overrides=param_overrides,
                            model=model_overrides))
    return built


def all_entries(include_entries=True, include_combos=True):
    """Every config this script would run, ENTRIES then COMBOS.

    Duplicate tags are rejected: a repeated run_tag means one run silently
    overwrites the other's registry row and pkl, which is exactly the
    failure mode experiment_runner.py's own header warns about.
    """
    entries = []
    if include_entries:
        entries += ENTRIES
    if include_combos:
        entries += build_combo_entries()

    seen = {}
    for cfg in entries:
        tag = cfg["run_tag"]
        if tag in seen:
            raise ValueError(
                f"Duplicate run_tag {tag!r} between ENTRIES and COMBOS. "
                f"Rename one of them -- otherwise the second run overwrites "
                f"the first's registry row and output files.")
        seen[tag] = True
    return entries


# ===========================================================================
# PARALLEL WORKER  —  same pattern as extra_runners.py's _run_one, so a
# parallel uniform_priors_check.py run behaves and logs identically to a
# parallel extra_runners.py run: one entry per worker process, its own
# thread pool clamped to 1, full run_sampler output redirected to its own
# logs/<tag>.log, BaseException caught so one failed entry never takes
# down the batch or the parent process.
# ===========================================================================

def _run_one(args_tuple):
    """Run a single entry's fit in an isolated (spawned) worker process.

    Returns (idx, tag, status, elapsed, pkl_path, error_traceback).
    status is "ok" or "failed"; pkl_path is "" on failure.
    """
    idx, cfg, log_dir = args_tuple

    # Belt-and-braces thread clamp — with spawn mode the module-level env
    # var block (top of file) already runs in every worker before numpy
    # loads, so this is truly redundant. Kept only for safety if _run_one
    # is ever called outside the pool.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = "1"

    # threadpoolctl guard — catches any BLAS libraries dlopen'd after env
    # vars were read. Errors are silently swallowed; the env vars above
    # suffice.
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

        # Redirect both stdout and stderr to the log file for this entry —
        # dynesty's progress bar and all print() calls from run.py go here.
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = log
        sys.stderr = log

        try:
            results, sampler, active_names, data, run_name = run_sampler(cfg)
            pkl_path = pkl_path_for(run_name, cfg)
            elapsed = time.time() - t0
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log.write(f"\n=== DONE in {elapsed:.1f}s ===\n")
            return (idx, tag, "ok", elapsed, pkl_path, "")
        except BaseException:
            # BaseException (not just Exception) catches MemoryError,
            # KeyboardInterrupt, and system signals that would otherwise
            # silently kill the worker and surface only as
            # BrokenProcessPool in the parent.
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
            return (idx, tag, "failed", elapsed, "", tb)


def run_uniform_priors_check(only=None, dry_run=False, include_entries=True,
                             include_combos=True, workers=None,
                             log_dir="logs", sequential=False):
    """
    Parameters
    ----------
    only    : optional iterable of bare tags (without the
        "uniformpriors/" prefix, e.g. "baseline,stretch_powerlaw") to
        restrict to.
    dry_run : print what would run without sampling. Ignores workers/
        log_dir/sequential entirely -- nothing is dispatched or logged.
    include_entries / include_combos : run only one of the two sources.
    workers : max parallel worker processes (default: number of selected
        entries, capped at os.cpu_count()). Each worker uses exactly 1
        CPU thread (see the module-level thread-clamping block), so
        `workers=K` consumes exactly K cores total.
    log_dir : directory for per-entry log files (default "logs/", same
        convention as extra_runners.py). Every entry's full run_sampler
        output goes to "<log_dir>/<tag with / -> _>.log"; only a one-line
        completion status prints to the console. Applies whenever
        dry_run=False, regardless of workers/sequential.
    sequential : force workers=1 regardless of `workers` (useful for
        debugging -- still logs to log_dir/<tag>.log, just one at a time).

    Returns
    -------
    pandas.DataFrame, one row per entry: run_tag, status, elapsed_s,
    pkl_path, active_params, uniform_params. Also saved to SUMMARY_FILE.
    (dry_run=True omits status/elapsed_s and sets pkl_path="(dry-run)",
    matching the previous dry-run report shape.)

    Cross-reference each row's parameter estimates against its
    informative-prior counterpart in run_publication_registry.csv with
    compare_runs.compare_two_runs. Do NOT compare ln Z across the two
    registries -- see the module docstring.
    """
    entries = all_entries(include_entries=include_entries,
                          include_combos=include_combos)
    if only is not None:
        only = {t if t.startswith(f"{TAG_PREFIX}/") else f"{TAG_PREFIX}/{t}"
                for t in only}
        entries = [cfg for cfg in entries if cfg["run_tag"] in only]
        if not entries:
            raise SystemExit("No entries matched --only. "
                             "Use --list to see the available tags.")

    # ---- Manifest: active/uniformised params for every selected entry,
    # printed up front regardless of dry_run (cheap -- no sampling). This
    # is exactly what --dry-run has always shown; for a real run it also
    # doubles as your pre-flight check that the right parameters get
    # uniformised before any compute is spent (see UNIFORM_PRIORS'
    # module-docstring note on gamma_alpha/zeta/eta). ----
    manifest = {}
    for cfg in entries:
        active  = [n for n, s in cfg["param_specs"].items() if s["active"]]
        uniform = [n for n in active
                   if cfg["param_specs"][n].get("prior")
                   != DEFAULT_PARAM_SPECS[n].get("prior")]
        manifest[cfg["run_tag"]] = (active, uniform)
        print(f"\n{'='*60}\n{cfg['run_tag']}\n{'='*60}")
        print(f"  model          : {cfg['model']}")
        print(f"  active params  : {active}")
        print(f"  uniformised    : {uniform}")

    if dry_run:
        rows = [{"run_tag": tag, "pkl_path": "(dry-run)",
                 "active_params": "|".join(active),
                 "uniform_params": "|".join(uniform)}
                for tag, (active, uniform) in manifest.items()]
        return pd.DataFrame(rows)

    # ---- Dispatch the actual fits ----
    n_workers = min(workers or len(entries), os.cpu_count() or 1)
    if sequential:
        n_workers = 1

    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "uniformpriors_summary.log")
    summary = open(summary_path, "w", buffering=1)
    summary.write(f"Run started: {datetime.now().isoformat()}\n")
    summary.write(f"Entries: {len(entries)}  Workers: {n_workers}\n\n")

    print(f"\n{'='*60}")
    print(f"Entries     : {len(entries)}")
    print(f"Workers     : {n_workers}  (cores available: {os.cpu_count()})")
    print(f"Log dir     : {os.path.abspath(log_dir)}/")
    print(f"{'='*60}")
    print(f"Logs are written to {os.path.abspath(log_dir)}/<tag>.log")
    print(f"Monitor a run with:  tail -f {log_dir}/<tag>.log\n")

    work = [(i, cfg, log_dir) for i, cfg in enumerate(entries)]
    results = []

    if n_workers == 1:
        # Sequential — useful for debugging. Still routes through
        # _run_one, so this still logs to log_dir/<tag>.log exactly like
        # the parallel path.
        for item in work:
            r = _run_one(item)
            results.append(r)
            idx, tag, status, elapsed, _, _ = r
            line = f"[{status.upper():>6}]  [{idx:>2}]  {tag:<50}  {elapsed:7.1f}s\n"
            print(line, end="")
            summary.write(line)
            summary.flush()
    else:
        # spawn, not fork — see extra_runners.py's identical comment: fork
        # would copy the parent's already-initialised BLAS thread pool
        # across the fork boundary, risking deadlocks/SIGKILL
        # ("BrokenProcessPool"). Spawn starts a clean interpreter per
        # worker instead, at the cost of ~1-2s import overhead per worker,
        # negligible for a nested-sampling run.
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        _ctx = _mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx) as pool:
            futures = {pool.submit(_run_one, item): item[0] for item in work}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception as exc:
                    # Worker process died with an unrecoverable error
                    # (e.g. OOM, signal) -- record as failed rather than
                    # crashing the parent.
                    item_idx = futures[fut]
                    item_tag = entries[item_idx]["run_tag"]
                    tb = f"{type(exc).__name__}: {exc}"
                    print(f"\n[CRASH]  [{item_idx:>2}]  {item_tag}  —  {tb}",
                         flush=True)
                    r = (item_idx, item_tag, "failed", 0.0, "", tb)
                results.append(r)
                idx, tag, status, elapsed, _, _ = r
                line = (f"[{status.upper():>6}]  [{idx:>2}]  "
                        f"{tag:<50}  {elapsed:7.1f}s\n")
                print(line, end="")
                summary.write(line)
                summary.flush()

    ok     = [r for r in results if r[2] == "ok"]
    failed = [r for r in results if r[2] == "failed"]
    footer = (f"\n{'='*60}\n"
              f"Finished {len(ok)}/{len(results)} entries successfully.\n")
    if failed:
        footer += "Failed:\n"
        for idx, tag, _, elapsed, _, err in failed:
            first_line = err.strip().splitlines()[-1] if err else "unknown"
            footer += f"  [{idx}] {tag}: {first_line}\n"
    footer += f"{'='*60}\n"

    print(footer)
    summary.write(footer)
    summary.close()
    print(f"Full summary written to: {summary_path}")

    # ---- Build the final per-entry report CSV, joining the manifest
    # (active/uniformised params, known before any fit ran) with the
    # execution result (status/elapsed/pkl_path, known after) by index. ----
    by_idx = {r[0]: r for r in results}
    rows = []
    for i, cfg in enumerate(entries):
        tag = cfg["run_tag"]
        active, uniform = manifest[tag]
        r = by_idx.get(i)
        status, elapsed, pkl_path = (r[2], r[3], r[4]) if r else ("not_run", "", "")
        rows.append({"run_tag": tag, "status": status, "elapsed_s": elapsed,
                     "pkl_path": pkl_path,
                     "active_params": "|".join(active),
                     "uniform_params": "|".join(uniform)})

    report = pd.DataFrame(rows)
    report.to_csv(SUMMARY_FILE, index=False)
    print(f"\nUniform-priors check summary saved: {SUMMARY_FILE}")
    print(f"  outputs  : {OUTPUT_DIR}/")
    print(f"  registry : {REGISTRY_FILE}")
    print(f"  NOTE: compare ln Z only WITHIN {REGISTRY_FILE} "
          f"(against '{TAG_PREFIX}/baseline'), never against "
          f"run_publication_registry.csv -- the prior volumes differ.")
    return report


def _parse_args():
    p = argparse.ArgumentParser(
        description=f"Broad-uniform-prior reruns. Outputs go to "
                    f"{OUTPUT_DIR}/ with their own registry "
                    f"({REGISTRY_FILE}); tags use '{TAG_PREFIX}/'. Define "
                    f"runs either as explicit ENTRIES or as TERMS+COMBOS "
                    f"for your chosen best models. Runs in parallel via "
                    f"--workers, same model as extra_runners.py.")
    p.add_argument("--only", default=None,
                   help="Comma-separated bare tags (e.g. "
                        "'baseline,stretch_powerlaw') to run. Default: "
                        "every entry.")
    p.add_argument("--list", action="store_true",
                   help="List every tag that would run, then exit.")
    p.add_argument("--entries-only", action="store_true",
                   help="Run only the explicit ENTRIES, skipping COMBOS.")
    p.add_argument("--combos-only", action="store_true",
                   help="Run only the TERMS/COMBOS entries, skipping ENTRIES.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=None,
                   help="Max parallel worker processes (default: number "
                        "of selected entries, capped at os.cpu_count()). "
                        "Each worker uses exactly 1 CPU thread.")
    p.add_argument("--log-dir", default="logs",
                   help="Directory for per-entry log files (default: logs/).")
    p.add_argument("--sequential", action="store_true",
                   help="Disable parallelism — run one entry at a time "
                        "(useful for debugging). Still logs to "
                        "--log-dir/<tag>.log, same as the parallel path.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.entries_only and args.combos_only:
        raise SystemExit("--entries-only and --combos-only are mutually "
                         "exclusive.")

    include_entries = not args.combos_only
    include_combos  = not args.entries_only

    if args.list:
        for cfg in all_entries(include_entries, include_combos):
            n = sum(1 for s in cfg["param_specs"].values() if s["active"])
            print(f"  {cfg['run_tag']:<50}  {n} params")
        raise SystemExit(0)

    only = args.only.split(",") if args.only else None
    run_uniform_priors_check(only=only, dry_run=args.dry_run,
                             include_entries=include_entries,
                             include_combos=include_combos,
                             workers=args.workers, log_dir=args.log_dir,
                             sequential=args.sequential)