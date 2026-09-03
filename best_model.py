"""
best_model.py  —  SNe Ia Cosmology Pipeline
===============================================
Single source of truth for "your current best/candidate model", shared by
combo_ablation_checks.py, uniform_priors_check.py, extra_runners.py
(HOSTERR_BEST), and z_uncertainty_check.py -- so a winning term is written
down ONCE and every downstream check picks it up automatically instead of
requiring the same edit in four different files.

Format
------
TERMS follows exactly the format combo_ablation_checks.py and
uniform_priors_check.py already used locally before this file existed (a
term is a model-dict fragment plus param_overrides), so a term can still
be copy-pasted from/to either file's old inline TERMS block unchanged.

COMBOS is the ablation ladder consumed by combo_ablation_checks.py and
uniform_priors_check.py -- one entry per run, each a list of TERMS keys.

BEST_COMBO is the single chosen FINAL model -- the one combination you've
settled on. extra_runners.py's HOSTERR_BEST and z_uncertainty_check.py's
default model are both built from merge_terms(BEST_COMBO), so moving the
pipeline on to a new best model is a one-line change to BEST_COMBO here
(plus, if needed, a new TERMS entry) rather than four hand-edits.

Edit TERMS / COMBOS / BEST_COMBO below; nothing else in this file needs
touching. The four consumer scripts import TERMS/COMBOS/BEST_COMBO/
merge_terms/combo_tag from here rather than defining their own copies.
"""

import copy

# ===========================================================================
# 1. TERMS  —  named reusable correction blocks
# ===========================================================================
# Each term is a model-dict fragment (merged on top of CONFIG["model"]) plus
# param_overrides (merged on top of DEFAULT_PARAM_SPECS). A term whose
# "model" is {} is fine (e.g. a pure parameter activation like an
# interaction term).
TERMS = {
    "interaction": {
        "model": {},
        "param_overrides": {"gamma_alpha": {"active": True, "fixed": None}},
    },
    "sn_colour": {
        "model": {"sn_colour": "softbroken"},
        "param_overrides": {"sn_tau": {"active": True, "fixed": 0.3}},
    },
    "host": {
        "model": {"mass": "linear"},
        "param_overrides": {},
    },
    "host_colour": {
        "model": {"host_colour": "tanh"},
        # eta is host_colour's own amplitude coefficient (arcsinh prior,
        # DEFAULT_PARAM_SPECS default active=False, fixed=0.035). Left
        # inactive here previously, so turning on model["host_colour"]
        # applied the tanh correction at that FIXED 0.035 value without
        # ever sampling or fitting it -- the ablation wasn't actually
        # testing this term's amplitude at all. Activate it, same pattern
        # as "interaction" activating gamma_alpha above.
        "param_overrides": {"eta": {"active": True, "fixed": None}},
    },
    # Add new terms here as they're picked, e.g.:
    # "mass_sigmoid_M0tau": {
    #     "model": {"mass": "sigmoid"},
    #     "param_overrides": {"M0": {"active": True, "fixed": 10.0},
    #                         "tau": {"active": True, "fixed": 0.2}},
    # },
    # "ssfr_tanh_nominal": {
    #     "model": {"ssfr": "tanh"},
    #     "param_overrides": {"zeta": {"active": True, "fixed": 0.0},
    #                         "F0":   {"active": False, "fixed": -10.5},
    #                         "ftau": {"active": False, "fixed": 0.5}},
    # },
}

# ===========================================================================
# 2. COMBOS  —  ablation ladder for combo_ablation_checks.py /
#    uniform_priors_check.py
# ===========================================================================
# Any subset of TERMS.keys() is valid; list order doesn't matter, only
# membership.
COMBOS = [
    ["interaction"],
    ["interaction", "sn_colour"],
    ["interaction", "sn_colour", "host"],
    ["sn_colour", "host"],
    ["sn_colour"],
    ["host_colour"],
]

# ===========================================================================
# 3. BEST_COMBO  —  the single chosen FINAL model
# ===========================================================================
# EDIT THIS as your best model changes. extra_runners.py's HOSTERR_BEST and
# z_uncertainty_check.py's default model are both derived from this list
# via merge_terms() below -- updating it here updates both automatically.
BEST_COMBO = ["interaction", "sn_colour", "host"]


def merge_terms(term_names, terms=None):
    """Union the model-dict and param_overrides of every named term.

    Raises on conflict rather than letting one term silently overwrite
    another -- two terms in the same combo both setting the same
    config["model"] key, or the same param_specs field, to DIFFERENT
    values is almost certainly a mistake in TERMS/COMBOS/BEST_COMBO worth
    catching immediately. Shared by every consumer of TERMS/COMBOS so this
    conflict-checking behaviour is identical everywhere instead of
    (previously) being copy-pasted near-verbatim into combo_ablation_
    checks.py and uniform_priors_check.py separately.

    Parameters
    ----------
    term_names : iterable of TERMS keys (or `terms` keys, if given).
    terms      : dict to look names up in; defaults to TERMS.

    Returns
    -------
    (model_overrides, param_overrides) : two dicts, ready to be merged on
        top of CONFIG["model"] / DEFAULT_PARAM_SPECS respectively.
    """
    terms = TERMS if terms is None else terms
    model_overrides, param_overrides = {}, {}
    for t in term_names:
        if t not in terms:
            raise KeyError(f"Unknown term {t!r}; known terms: {sorted(terms)}")
        term = terms[t]
        for k, v in term.get("model", {}).items():
            if k in model_overrides and model_overrides[k] != v:
                raise ValueError(f"Conflicting model['{k}'] between terms "
                                 f"in combo {term_names}: "
                                 f"{model_overrides[k]!r} vs {v!r}")
            model_overrides[k] = v
        for name, updates in term.get("param_overrides", {}).items():
            if name in param_overrides and param_overrides[name] != updates:
                raise ValueError(f"Conflicting param_overrides[{name!r}] "
                                 f"between terms in combo {term_names}: "
                                 f"{param_overrides[name]!r} vs {updates!r}")
            param_overrides[name] = updates
    return model_overrides, param_overrides


def combo_tag(term_names):
    """Underscore-joined tag fragment for a combo, e.g. ["a","b"] ->
    'a_b'. Empty combo -> 'base'. Each consumer script still prefixes
    this with its own namespace (e.g. combo_ablation_checks.py uses
    "combo/" + combo_tag(...), uniform_priors_check.py uses "combo_" +
    combo_tag(...)) so tags stay distinguishable across files/registries."""
    return "_".join(term_names) if term_names else "base"


def best_model_overrides():
    """(model_overrides, param_overrides) for BEST_COMBO -- the two dicts
    every downstream script merges on top of CONFIG["model"] /
    DEFAULT_PARAM_SPECS. This is the one call extra_runners.py and
    z_uncertainty_check.py both make to stay in sync with BEST_COMBO."""
    return merge_terms(BEST_COMBO)


def resolved_best_param_specs():
    """Full param_specs dict for BEST_COMBO: DEFAULT_PARAM_SPECS with
    best_model_overrides()'s param_overrides applied on top -- the exact
    shape config["param_specs"] / run_sampler() expects. For scripts (like
    z_uncertainty_check.py) that build config_overrides by hand instead of
    going through ExperimentRegistry.build()/_build()."""
    from config import DEFAULT_PARAM_SPECS
    _, param_overrides = best_model_overrides()
    specs = copy.deepcopy(DEFAULT_PARAM_SPECS)
    for name, updates in param_overrides.items():
        specs[name].update(updates)
    return specs