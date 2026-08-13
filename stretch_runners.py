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
            
            _build("stretch/stretch_tanh_x10_x1tau",
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
            
            _build("stretch/stretch_softbroken_x10_x1tau",
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
            
            _build("stretch/stretch_stepbroken_x10_x1tau",
                   config_overrides={"model": {**CONFIG["model"], "x1_correction": "stepbroken"}},
                   param_overrides={"x1_0": {"active": True, "fixed": 0.0},
                                    "x1_tau": {"active": True, "fixed": 0.3}}),      
       
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
    p.add_argument("--sequential", action="store_true",
                   help="Disable parallelism — run one at a time (useful for debugging)")
    # nlive mode — mutually exclusive; if neither is given the mode stored in
    # each experiment's config dict is used (default: "exploratory").
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
        return n * 300 if mode == "publication" else n * 50
 
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
 
 
if __name__ == "__main__":
    main()