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
UNIFORM_PRIORS below: alpha, beta and Om0 get uniform priors over ranges
far wider than both their informative sigma and their old hard clips.
Shape parameters that are active in a given entry (x1_tau, sn_tau, tau,
htau, ftau, M0, F0) are widened too, but only where the entry actually
samples them -- see _uniformise(). Parameters that are already uniform by
default (gamma, c0, C0, x1_0) need no override.

Usage
-----
  python uniform_priors_check.py
  python uniform_priors_check.py --list
  python uniform_priors_check.py --only baseline,stretch_powerlaw
  python uniform_priors_check.py --entries-only
  python uniform_priors_check.py --combos-only
  python uniform_priors_check.py --dry-run

or:
  from uniform_priors_check import run_uniform_priors_check
  report = run_uniform_priors_check()
"""

import argparse
import copy

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
# alpha/beta/Om0 mirror experiment_runner.py's _ZEVO_BROAD_UNIFORM exactly,
# so an "evolution/" run and a "uniformpriors/" run of the same model are
# fitted under identical nuisance priors and their parameter estimates are
# directly comparable.
#
# The remaining entries are shape parameters whose defaults are log_normal
# or truncated_gaussian -- informative by construction. They are applied
# ONLY when the entry being built actually samples them (see _uniformise),
# because overriding the prior of an inactive parameter is a no-op that
# just makes the resulting spec harder to read.
UNIFORM_PRIORS = {
    # SALT2 nuisance + cosmology (truncated_gaussian by default)
    "alpha":  {"prior": "uniform", "range": [0.0,  0.5]},
    "beta":   {"prior": "uniform", "range": [0.0,  8.0]},
    "Om0":    {"prior": "uniform", "range": [0.05, 0.95]},
    # Shape / width parameters (log_normal or truncated_gaussian by default).
    # Ranges are each parameter's own existing hard clip -- these already
    # span orders of magnitude, so the informative part of these priors is
    # the SHAPE, not the support, and widening the support further would
    # only add Occam penalty without adding reachable models.
    "x1_tau": {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["x1_tau"]["range"]},
    "sn_tau": {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["sn_tau"]["range"]},
    "tau":    {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["tau"]["range"]},
    "htau":   {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["htau"]["range"]},
    "ftau":   {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["ftau"]["range"]},
    "M0":     {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["M0"]["range"]},
    "F0":     {"prior": "uniform", "range": DEFAULT_PARAM_SPECS["F0"]["range"]},
}

# Always uniformised, whether active or not: these are the parameters the
# whole exercise is about, and they are active in every run anyway.
_ALWAYS_UNIFORM = ("alpha", "beta", "Om0")


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
# 2. TERMS  —  named reusable blocks for your chosen best models
# ===========================================================================
# EDIT THIS once you have picked your winners from
# model_comparison_suite.py. Each term is a model-dict fragment plus
# param_overrides, exactly the format combo_ablation_checks.py's TERMS
# uses, so a term can be copy-pasted between the two files unchanged.
#
# The examples below are the two families this file already tests
# explicitly; replace or extend them freely. A term whose "model" is {}
# (pure parameter activation, e.g. an interaction term) is fine.
TERMS = {
    "stretch": {
        "model": {"x1_correction": "doublebroken"},
        "param_overrides": {"x1_0":   {"active": True},
                            "x1_tau": {"active": True}},
    },
    "sn_colour": {
        "model": {"sn_colour": "softbroken"},
        "param_overrides": {"sn_tau": {"active": True, "fixed": 0.3}},
    },
}

# ===========================================================================
# 3. COMBOS  —  which TERMS to merge for each run
# ===========================================================================
# One entry per run. [] is the base model with no terms added, i.e. the
# same fit as ENTRIES' "baseline" -- it is NOT included by default to avoid
# duplicating that tag. Membership is what matters, not order.
#
# The default ladder below is the incremental comparison described for the
# best models: base + stretch, base + sn colour, base + stretch + sn
# colour, all under identical broad uniform priors so their ln Z values
# are directly comparable to each other AND to "uniformpriors/baseline".
COMBOS = [
    ["stretch"],
    ["sn_colour"],
    ["stretch", "sn_colour"],
]


def _combo_tag(term_names):
    return "combo_" + "_".join(term_names) if term_names else "combo_base"


def _merge_terms(term_names):
    """Union the model-dict and param_overrides of every named term.

    Raises on conflict rather than letting one term silently overwrite
    another: two terms in the same combo setting the same model key or the
    same param_specs field to DIFFERENT values is almost certainly a
    mistake in TERMS/COMBOS, and silently keeping the last one would
    produce a run whose tag does not describe what it fitted.
    """
    model_overrides, param_overrides = {}, {}
    for t in term_names:
        if t not in TERMS:
            raise KeyError(f"Unknown term {t!r} in COMBOS; "
                           f"known terms: {sorted(TERMS)}")
        term = TERMS[t]
        for k, v in term.get("model", {}).items():
            if k in model_overrides and model_overrides[k] != v:
                raise ValueError(f"Conflicting model['{k}'] between terms in "
                                 f"combo {term_names}: "
                                 f"{model_overrides[k]!r} vs {v!r}")
            model_overrides[k] = v
        for name, updates in term.get("param_overrides", {}).items():
            if name in param_overrides and param_overrides[name] != updates:
                raise ValueError(f"Conflicting param_overrides[{name!r}] "
                                 f"between terms in combo {term_names}: "
                                 f"{param_overrides[name]!r} vs {updates!r}")
            param_overrides[name] = updates
    return model_overrides, param_overrides


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


def run_uniform_priors_check(only=None, dry_run=False, include_entries=True,
                             include_combos=True):
    """
    Parameters
    ----------
    only    : optional iterable of bare tags (without the
        "uniformpriors/" prefix, e.g. "baseline,stretch_powerlaw") to
        restrict to.
    dry_run : print what would run without sampling.
    include_entries / include_combos : run only one of the two sources.

    Returns
    -------
    pandas.DataFrame, one row per entry: run_tag, pkl_path, active_params,
    uniform_params. Also saved to SUMMARY_FILE.

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

    rows = []
    for cfg in entries:
        active  = [n for n, s in cfg["param_specs"].items() if s["active"]]
        uniform = [n for n in active
                   if cfg["param_specs"][n].get("prior")
                   != DEFAULT_PARAM_SPECS[n].get("prior")]
        print(f"\n{'='*60}\n{cfg['run_tag']}\n{'='*60}")
        print(f"  model          : {cfg['model']}")
        print(f"  active params  : {active}")
        print(f"  uniformised    : {uniform}")
        if dry_run:
            rows.append({"run_tag": cfg["run_tag"], "pkl_path": "(dry-run)",
                         "active_params": "|".join(active),
                         "uniform_params": "|".join(uniform)})
            continue

        results, sampler, active_names, data, run_name = run_sampler(cfg)
        rows.append({"run_tag": cfg["run_tag"],
                     "pkl_path": pkl_path_for(run_name, cfg),
                     "active_params": "|".join(active_names),
                     "uniform_params": "|".join(uniform)})

    report = pd.DataFrame(rows)
    if not dry_run:
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
                    f"for your chosen best models.")
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
                             include_combos=include_combos)
