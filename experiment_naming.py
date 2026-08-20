"""
experiment_naming.py  —  SNe Ia Cosmology Pipeline
====================================================
Single source of truth for how every runner script (experiment_runner.py,
extra_runners.py, combo_ablation_checks.py) turns a set of overrides into
a registered run config.

Why this file exists
---------------------
Before this pass, `_override()` and `_build()` were copy-pasted, near-
identically, into all three runner scripts. A fix made in one copy (e.g.
"don't sample gamma when mass='none'") had no way of propagating to the
other two, and nothing stopped two different tag strings from describing
the exact same fit (found: 3 such cases in experiment_runner.py) or the
same literal tag being reused (found: 2 cases, which -- per this
project's own convention -- are silently skipped or silently overwritten
depending on --rerun, not flagged). Centralising the construction here
means every runner gets these guarantees automatically, and a future fix
only has to be made once.

Naming convention
------------------
Every tag should be "<category>/<description>", where <category> is one
of CATEGORY_PREFIXES below and <description> lists only the pieces of
the model that are genuinely free to vary in that experiment.

Two redundancy rules are enforced in code, not left to each call site to
remember correctly:

1. mass='none' implies gamma is unconstrained.
   core.mass_none() returns S == 0 for every SN, so gamma/2 * S is zero
   regardless of gamma's value: gamma cannot be constrained by the data
   under mass='none', and leaving it active just burns evidence on an
   unconstrained parameter. Because this is a property of the model
   choice alone, `build()` below fixes gamma off automatically whenever
   mass='none', and a tag no longer needs (and should not carry) a
   separate "nogamma" fragment to say so -- "mass_none" already says it.
   ("nogamma" still means something real when it's paired with a mass
   model that ISN'T 'none', e.g. isolating an interaction term from the
   direct mass-step effect -- that case is untouched here.)

2. A "linear" functional form's own zero-point parameter is degenerate
   with M (the analytically-marginalised absolute magnitude). This is
   already documented per-model in core.py's docstrings (mass_linear,
   hcol_linear, ssfr_linear, sn_colour_linear, x1_correction_linear);
   `build()` below turns those docstrings into an enforced check so the
   mistake can't quietly reappear in a future edit.

Neither rule invents new physics -- both are already documented in
core.py. This file just makes them impossible to get wrong by accident.
"""

import copy
import hashlib
import json

# ---------------------------------------------------------------------------
# Category prefixes used by experiment_runner.py's EXPERIMENTS list.
# extra_runners.py uses "checks/" exclusively; combo_ablation_checks.py
# uses "combo/" exclusively; rerun_prior_dominated.py / uniform_priors_check.py
# use "uniformcheck/" for their own re-fit registries. Keeping the set
# closed here means a typo'd prefix (e.g. "chekcs/") is caught immediately
# instead of silently splitting one category across two spellings.
# ---------------------------------------------------------------------------
CATEGORY_PREFIXES = {
    "baseline", "cosmo", "nuisance", "sn_col_model", "stretch",
    "host_col_model", "mass", "ssfr", "evolution", "interaction",
    "checks", "combo", "uniformcheck", "hosterr",
}

# model_key -> (value that makes the shape parameter below a pure
# zero-point shift, parameter name). See core.py's per-model docstrings
# for the derivation of each of these.
DEGENERATE_WITH_M = {
    "mass":          ("linear", "M0"),
    "host_colour":   ("linear", "C0"),
    "ssfr":          ("linear", "F0"),
    "sn_colour":     ("linear", "c0"),
    "x1_correction": ("linear", "x1_0"),
}


def _override(base_specs, **param_overrides):
    """
    Return a deep copy of base_specs with per-parameter overrides applied.

    Each key in param_overrides is a parameter name; the value is a dict of
    fields to update, e.g.:

        _override(base, Om0={"active": False}, w={"active": True})

    Only the listed fields are changed -- all other fields for that
    parameter are inherited from base_specs unchanged.
    """
    specs = copy.deepcopy(base_specs)
    for name, updates in param_overrides.items():
        specs[name].update(updates)
    return specs


def model_dict(base_model, sn_colour=None, x1_correction=None, mass=None,
              host_colour=None, ssfr=None, z_evolve=None):
    """
    Return a model dict with the given overrides on top of `base_model`
    (normally CONFIG["model"]). Only the explicitly-passed keys change;
    everything else is inherited. Equivalent to (and replaces) the
    `_M()` helper that used to live in experiment_runner.py alone --
    extra_runners.py and combo_ablation_checks.py can now build model
    dicts the same way instead of hand-writing `{**CONFIG["model"], ...}`
    dicts inline, which is what let the same combination get typo'd into
    two different-looking dict literals in different places.
    """
    m = dict(base_model)
    for key, val in (("sn_colour", sn_colour), ("x1_correction", x1_correction),
                     ("mass", mass), ("host_colour", host_colour),
                     ("ssfr", ssfr), ("z_evolve", z_evolve)):
        if val is not None:
            m[key] = val
    return m


def _fingerprint(cfg):
    """
    Canonical hash of a resolved experiment config, used to catch the case
    where two DIFFERENT tag strings describe the exact same fit. Compares
    everything except `run_tag` itself: model dict, full param_specs
    (active params compared on every field except "fixed", which is
    irrelevant once a parameter is active; inactive params compared on
    "fixed" only), and every other top-level CONFIG override (registry
    file, data cuts, host-error settings, sampler settings, ...), so
    e.g. the twelve `hosterr/*` entries in extra_runners.py -- which
    deliberately share the same model and active parameters but differ
    in `host_var_penalty` / `ssfr_err_max` / etc. -- are correctly seen
    as distinct, while two entries that differ only in tag spelling are
    correctly seen as duplicates.
    """
    specs = cfg["param_specs"]
    norm_specs = {}
    for k, v in sorted(specs.items()):
        if v["active"]:
            norm_specs[k] = {kk: vv for kk, vv in sorted(v.items()) if kk != "fixed"}
        else:
            norm_specs[k] = {"active": False, "fixed": v.get("fixed")}

    rest = {k: v for k, v in cfg.items()
           if k not in ("run_tag", "param_specs")}

    payload = json.dumps({"rest": rest, "param_specs": norm_specs},
                         sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


class ExperimentRegistry:
    """
    Accumulates _build()-style configs while guarding against:

      * a repeated literal tag string (silently skipped or silently
        overwritten depending on --rerun -- see experiment_runner.py's
        module docstring);
      * a *different* tag string whose resolved config is functionally
        identical to an earlier one (same model, same active parameters
        and priors, same data cuts / sampler / registry settings);
      * activating gamma alongside mass="none" (see module docstring,
        rule 1);
      * activating a shape parameter the model itself makes degenerate
        with M (see module docstring, rule 2).

    Each runner script owns one registry instance built from its own
    base CONFIG/DEFAULT_PARAM_SPECS, and calls `.build(...)` exactly
    where it used to call the old local `_build(...)`.
    """

    def __init__(self, base_config, base_param_specs):
        self.base_config = base_config
        self.base_param_specs = base_param_specs
        self.experiments = []
        self._tags_seen = set()
        self._fingerprints_seen = {}

    def build(self, tag, param_overrides=None, config_overrides=None,
             allow_duplicate_fingerprint=False):
        """
        Build and register one experiment config.

        Parameters
        ----------
        tag             : str -- unique human-readable label (must be
            unique across every call to this registry).
        param_overrides : dict -- {param_name: {field: value, ...}, ...}
        config_overrides: dict -- top-level CONFIG fields to override,
            e.g. {"sigma_int": 0.1, "nlive": 2000}. If it contains
            "model", that's merged onto CONFIG["model"] (or pass a
            complete dict from `model_dict()`).
        allow_duplicate_fingerprint : bool -- set True only for the rare,
            deliberate case where two tags are SUPPOSED to describe an
            identical fit (e.g. a cross-check re-run under a different
            registry file). Defaults to False so an accidental duplicate
            fails loudly instead of silently doubling your compute.
        """
        model = dict(self.base_config["model"])
        if config_overrides and "model" in config_overrides:
            model.update(config_overrides["model"])

        param_overrides = dict(param_overrides or {})

        # ---- rule 1: mass="none" => gamma cannot be constrained -------
        if model.get("mass") == "none":
            requested = param_overrides.get("gamma", {})
            if requested.get("active"):
                raise ValueError(
                    f"'{tag}': mass='none' means the mass profile S is "
                    f"zero for every SN (core.mass_none), so gamma cannot "
                    f"be constrained by the data -- activating it only "
                    f"burns evidence on an unconstrained parameter. This "
                    f"combination is never intentional; either drop the "
                    f"gamma override or use a mass model other than "
                    f"'none'.")
            param_overrides["gamma"] = {"active": False, "fixed": 0.0}

        specs = _override(self.base_param_specs, **param_overrides)

        # ---- rule 2: shape parameter degenerate with M -----------------
        for model_key, (trigger_val, param_name) in DEGENERATE_WITH_M.items():
            if model.get(model_key) == trigger_val and specs[param_name]["active"]:
                raise ValueError(
                    f"'{tag}': {model_key}='{trigger_val}' makes "
                    f"'{param_name}' a pure zero-point shift that is "
                    f"exactly degenerate with the analytically-"
                    f"marginalised M (see core.py's docstring for this "
                    f"model) -- sampling it adds a direction the data "
                    f"cannot constrain, which only costs evidence. Fix "
                    f"'{param_name}' inactive, or use the model variant "
                    f"where it is a genuine shape parameter (e.g. "
                    f"'quadratic', 'tanh', 'step', ...).")

        cfg = copy.deepcopy(self.base_config)
        cfg["run_tag"] = tag
        cfg["param_specs"] = specs
        if config_overrides:
            cfg.update({k: v for k, v in config_overrides.items() if k != "model"})
        cfg["model"] = model

        # ---- duplicate guards ------------------------------------------
        if tag in self._tags_seen:
            raise ValueError(
                f"Duplicate run tag '{tag}' -- every tag registered here "
                f"must be unique (a repeated tag is silently skipped or "
                f"silently overwrites the first entry depending on "
                f"--rerun).")
        self._tags_seen.add(tag)

        fp = _fingerprint(cfg)
        if fp in self._fingerprints_seen and not allow_duplicate_fingerprint:
            raise ValueError(
                f"'{tag}' has an identical resolved config (model + "
                f"active parameters + data/sampler settings) to "
                f"'{self._fingerprints_seen[fp]}' -- same fit registered "
                f"under two different names. If this is deliberate, call "
                f"build(..., allow_duplicate_fingerprint=True).")
        self._fingerprints_seen[fp] = tag

        self.experiments.append(cfg)
        return cfg

    def validate_category_prefixes(self, exempt_tags=()):
        """
        Optional stricter check: every registered tag (except those in
        `exempt_tags`, e.g. a literal "baseline" with no slash) must
        start with "<category>/" for a category in CATEGORY_PREFIXES.
        Not called automatically -- existing sections predate the
        prefix convention being enforced and a full rename would orphan
        already-completed sampler runs in run_publication_registry.csv
        (see the "naming" section of the accompanying audit notes).
        Call this explicitly from a NEW section to keep new entries
        honest without touching old ones.
        """
        bad = []
        for tag in self._tags_seen:
            if tag in exempt_tags:
                continue
            prefix = tag.split("/", 1)[0] if "/" in tag else None
            if prefix not in CATEGORY_PREFIXES:
                bad.append(tag)
        if bad:
            raise ValueError(
                f"{len(bad)} tag(s) don't start with a recognised "
                f"category/ prefix ({sorted(CATEGORY_PREFIXES)}): {bad}")