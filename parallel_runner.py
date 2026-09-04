"""
parallel_runner.py  —  SNe Ia Cosmology Pipeline
====================================================
Shared parallel job pool for every driver script that needs to launch many
independent nested-sampling jobs (combo_ablation_checks.py,
z_uncertainty_check.py, extra_runners.py, ...).

This is the same battle-tested recipe experiment_runner.py's main()/_run_one()
already used -- spawn-mode ProcessPoolExecutor, per-job log files, a thread
clamp so N workers don't each spin up M BLAS threads -- lifted into one module
so the other drivers stop re-implementing it (previously they were all strictly
serial) and so a fix to the pool benefits all of them at once.

WHY SPAWN, NOT FORK
-------------------
fork() copies the parent's entire memory space, including any already-
initialised OpenBLAS/OMP thread pools. When those pools try to synchronise
across the fork boundary they deadlock or get SIGKILLed, which surfaces in the
parent as an unexplained BrokenProcessPool. Spawn starts a clean interpreter
per worker. The ~1-2 s startup cost is irrelevant next to a nested-sampling
run.

WHY THE THREAD CLAMP
--------------------
dynesty/numpy will happily use every core for BLAS. Running W workers each
using C threads oversubscribes the box by W*C and is typically SLOWER than
serial. The env vars must be set BEFORE numpy is imported, which is why they
are set at module scope here (spawn re-imports this module in every worker,
so the clamp lands before the worker's numpy import) as well as defensively
inside the worker.

USAGE
-----
    from parallel_runner import Job, run_jobs

    jobs = [Job(label="combo/fit_mass_linear", func=my_module.fit_one,
                kwargs={"cfg": cfg}, group="fit")]
    results = run_jobs(jobs, n_workers=8, log_dir="logs/combo")

`func` MUST be a module-level function (spawn pickles it by qualified name, so
a lambda, closure or locally-defined function will fail), and both `kwargs`
and whatever `func` returns must be picklable.
"""

import os

# --- Thread clamp: MUST run before numpy/scipy/dynesty are imported ---------
# setdefault (not hard assignment) so a user who deliberately exports
# OMP_NUM_THREADS=4 to run a single big job keeps their setting.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
             "BLIS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional


# ===========================================================================
# 1.  JOB / RESULT CONTAINERS
# ===========================================================================

@dataclass
class Job:
    """One unit of parallel work.

    label  : unique human-readable identifier. Also becomes the per-job log
             filename (with "/" -> "_"), so it must be unique across the whole
             job list or two jobs will fight over the same log file.
    func   : module-level callable, invoked as func(**kwargs) in the worker.
    kwargs : picklable keyword arguments.
    group  : optional grouping label used only for the terminal summary
             (e.g. "host_qual", "loo_zbin") so a 200-job run prints a readable
             per-section breakdown instead of 200 undifferentiated lines.
    cost   : optional relative cost hint. Jobs are dispatched heaviest-first so
             one very long job doesn't start last and leave workers idle at the
             end. Pure scheduling hint; it never changes what is run.
    """
    label: str
    func: Callable
    kwargs: Dict[str, Any] = field(default_factory=dict)
    group: str = ""
    cost: float = 1.0


@dataclass
class JobResult:
    index: int
    label: str
    group: str
    status: str            # "ok" | "failed"
    elapsed: float
    error: str = ""
    value: Any = None      # whatever func returned (None if it failed)


def safe_label(label):
    """Filesystem-safe form of a job label ("combo/fit_x" -> "combo_fit_x")."""
    return label.replace("/", "_").replace(" ", "_")


# ===========================================================================
# 2.  WORKER
# ===========================================================================

def _worker(payload):
    """Run one job in this (spawned) process, logging to its own file.

    Returns a plain tuple rather than a JobResult so the return value stays
    trivially picklable even if this module is reloaded oddly in the worker.
    """
    index, label, group, func, kwargs, log_dir, capture = payload

    # Belt-and-braces: the module-scope clamp above already ran on import in
    # this worker, so this is redundant under spawn. Kept for the case where
    # _worker is called directly (sequential mode, tests).
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS"):
        os.environ[var] = "1"

    # threadpoolctl catches BLAS libraries dlopen'd after the env vars were
    # read. Its own import chatter goes to /dev/null; failure is harmless
    # because the env vars above already do the real work.
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

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{safe_label(label)}.log")

    t0 = time.time()
    if not capture:
        # Debug mode: let everything go to the parent's terminal.
        try:
            value = func(**kwargs)
            return (index, label, group, "ok", time.time() - t0, "", value)
        except BaseException:
            return (index, label, group, "failed", time.time() - t0,
                    traceback.format_exc(), None)

    with open(log_path, "w", buffering=1) as log:   # line-buffered for tail -f
        log.write(f"=== [{index}] {label} ===\n")
        log.write(f"Group   : {group or '-'}\n")
        log.write(f"Started : {datetime.now().isoformat()}\n")
        log.write(f"PID     : {os.getpid()}   CPU count: {os.cpu_count()}\n\n")
        log.flush()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = log
        sys.stderr = log
        try:
            value = func(**kwargs)
            elapsed = time.time() - t0
            sys.stdout, sys.stderr = old_stdout, old_stderr
            try:
                log.write(f"\n=== DONE in {elapsed:.1f}s ===\n")
            except Exception:
                pass
            return (index, label, group, "ok", elapsed, "", value)
        except BaseException:
            # BaseException, not Exception: MemoryError / KeyboardInterrupt /
            # SystemExit would otherwise kill the worker silently and surface
            # in the parent only as an opaque BrokenProcessPool.
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            try:
                log.write(f"\n=== FAILED after {elapsed:.1f}s ===\n{tb}\n")
                log.flush()
            except Exception:
                pass
            print(f"[worker {index}] FAILED {label}: "
                  f"{tb.strip().splitlines()[-1]}", file=sys.stderr, flush=True)
            return (index, label, group, "failed", elapsed, tb, None)


# ===========================================================================
# 3.  TERMINAL PRESENTATION
# ===========================================================================

def _fmt_hms(seconds):
    if seconds is None or seconds != seconds or seconds < 0:      # NaN guard
        return "--:--:--"
    return str(timedelta(seconds=int(seconds)))


def _print_plan(jobs, n_workers, log_dir, title, extra_lines=None):
    """Compact, grouped pre-flight summary.

    Deliberately prints ONE line per group rather than one per job: these
    drivers routinely launch 200+ jobs and a 200-line wall of text before
    anything starts is noise, not information. Use --list/--dry-run on the
    calling script for the full enumeration.
    """
    width = 74
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
    for line in (extra_lines or []):
        print(f"  {line}")
    print(f"  Jobs        : {len(jobs)}")
    print(f"  Workers     : {n_workers}   (cores available: {os.cpu_count()})")
    print(f"  Log dir     : {os.path.abspath(log_dir)}/")

    groups = {}
    for j in jobs:
        groups.setdefault(j.group or "-", 0)
        groups[j.group or "-"] += 1
    if len(groups) > 1:
        print(f"  {'-' * (width - 4)}")
        for g, n in groups.items():
            print(f"  {g:<28s} {n:>4d} job(s)")
    print(f"{'=' * width}")
    print(f"  Monitor with:  tail -f {log_dir}/<label>.log")
    print(f"{'=' * width}\n")


def _print_progress(done, total, t_start, res):
    """One tidy line per completed job: status, counter, %, ETA, label, time."""
    pct = 100.0 * done / total if total else 100.0
    elapsed_wall = time.time() - t_start
    # Throughput-based ETA. Meaningful because jobs are broadly homogeneous
    # (one nested-sampling fit each); still only an estimate.
    eta = (elapsed_wall / done) * (total - done) if done else float("nan")
    mark = " ok " if res.status == "ok" else "FAIL"
    label = res.label if len(res.label) <= 46 else res.label[:43] + "..."
    print(f"[{mark}] {done:>4}/{total:<4} {pct:5.1f}%  "
          f"ETA {_fmt_hms(eta)}  {label:<46s} {res.elapsed:8.1f}s",
          flush=True)


# ===========================================================================
# 4.  DRIVER
# ===========================================================================

def run_jobs(jobs, n_workers=None, log_dir="logs", sequential=False,
             title="Parallel jobs", summary_name="summary.log",
             extra_lines=None, capture_output=True, dry_run=False):
    """Run `jobs` in parallel, returning a list of JobResult in completion order.

    Parameters
    ----------
    jobs          : list of Job.
    n_workers     : max parallel processes. Default min(len(jobs), cpu_count).
                    Capped at cpu_count() -- oversubscribing nested sampling
                    with BLAS underneath makes the whole run slower.
    log_dir       : directory for per-job "<label>.log" files plus the master
                    summary log.
    sequential    : force one-at-a-time (equivalent to n_workers=1). Useful for
                    debugging, since tracebacks then surface in order.
    capture_output: redirect each job's stdout/stderr into its log file. Set
                    False to let output stream to the terminal (only sensible
                    with sequential=True).
    dry_run       : print the plan and return [] without running anything.

    Failure policy: a failing job is recorded and the rest continue. Nothing is
    retried and nothing is aborted early -- with jobs this expensive, losing 90
    good results because number 91 crashed would be far worse than reporting a
    partial set the caller can inspect and re-run selectively.
    """
    jobs = list(jobs)
    if not jobs:
        print("No jobs to run.")
        return []

    dupes = {j.label for j in jobs if sum(1 for k in jobs if k.label == j.label) > 1}
    if dupes:
        raise ValueError(
            f"Duplicate job labels would collide in the log directory and, if "
            f"they are run tags, in the registry: {sorted(dupes)}")

    n_workers = min(n_workers or len(jobs), os.cpu_count() or 1)
    if sequential:
        n_workers = 1

    _print_plan(jobs, n_workers, log_dir, title, extra_lines)
    if dry_run:
        print("Dry run -- exiting without running anything.\n")
        return []

    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, summary_name)

    # Heaviest-first so a single long job can't be the last one dispatched.
    order = sorted(range(len(jobs)), key=lambda i: -jobs[i].cost)
    work = [(i, jobs[i].label, jobs[i].group, jobs[i].func, jobs[i].kwargs,
             log_dir, capture_output) for i in order]

    results = []
    t_start = time.time()

    with open(summary_path, "w", buffering=1) as summary:
        summary.write(f"{title}\n")
        summary.write(f"Started : {datetime.now().isoformat()}\n")
        summary.write(f"Jobs    : {len(jobs)}   Workers: {n_workers}\n\n")

        def _record(raw):
            res = JobResult(*raw)
            results.append(res)
            _print_progress(len(results), len(jobs), t_start, res)
            summary.write(f"[{res.status.upper():>6}] {res.label:<52s} "
                          f"{res.elapsed:8.1f}s\n")
            return res

        if n_workers == 1:
            for item in work:
                _record(_worker(item))
        else:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            ctx = _mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {pool.submit(_worker, item): item for item in work}
                for fut in as_completed(futures):
                    item = futures[fut]
                    try:
                        raw = fut.result()
                    except BaseException as exc:
                        # The worker process itself died (OOM, signal) so it
                        # never got to return its own failure tuple.
                        raw = (item[0], item[1], item[2], "failed", 0.0,
                               f"{type(exc).__name__}: {exc}", None)
                    _record(raw)

        ok = [r for r in results if r.status == "ok"]
        failed = [r for r in results if r.status == "failed"]
        width = 74
        footer = [f"\n{'=' * width}",
                  f"  Finished {len(ok)}/{len(results)} job(s) successfully "
                  f"in {_fmt_hms(time.time() - t_start)}."]
        if failed:
            footer.append(f"  {len(failed)} FAILED:")
            for r in failed:
                first = r.error.strip().splitlines()[-1] if r.error else "unknown"
                footer.append(f"    {r.label}: {first}")
                footer.append(f"      log: {os.path.join(log_dir, safe_label(r.label))}.log")
        footer.append(f"{'=' * width}\n")
        text = "\n".join(footer)
        print(text)
        summary.write(text)

    print(f"Summary written to: {summary_path}\n")
    # Restore submission order so callers can zip results back to their inputs.
    results.sort(key=lambda r: r.index)
    return results


def add_parallel_args(parser, default_log_dir="logs"):
    """Attach the standard --workers/--sequential/--log-dir/--dry-run flags.

    Shared so every driver script exposes the SAME flag names and semantics
    rather than each inventing its own.
    """
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel processes (default: one per job, "
                             "capped at the number of CPU cores).")
    parser.add_argument("--sequential", action="store_true",
                        help="Run one job at a time (debugging; tracebacks "
                             "then appear in order).")
    parser.add_argument("--log-dir", default=default_log_dir,
                        help=f"Per-job log directory (default: {default_log_dir}/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the jobs that would run, then exit.")
    parser.add_argument("--no-capture", action="store_true",
                        help="Stream job output to the terminal instead of "
                             "per-job log files (use with --sequential).")
    return parser
