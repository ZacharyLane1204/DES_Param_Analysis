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
    # -- sSFR environment profile -------------------------------------------
    # ssfr="tanh" with zeta (the sSFR amplitude) sampled and F0/ftau held at
    # their defaults. This is the single largest evidence gain in the whole
    # publication sweep. Freeing F0/ftau as well is WORSE, not better:
    #   ssfr_tanh_hcol_none_mass_linear         lnZ = -439.837  (zeta only)
    #   ssfr_tanh_F0ftau_hcol_none_mass_linear  lnZ = -440.884  (+F0,+ftau)
    # i.e. the two extra shape parameters cost ~1.05 in Occam factor and buy
    # nothing, so they stay fixed here.
    "ssfr_tanh": {
        "model": {"ssfr": "tanh"},
        "param_overrides": {"zeta": {"active": True, "fixed": None}},
    },

    # -- Host stellar mass profile ------------------------------------------
    # mass="linear" replaces the classic hard mass step. gamma is already
    # active=True in DEFAULT_PARAM_SPECS and serves as this term's amplitude,
    # so there is nothing to activate here.
    #   mass/mass_linear   lnZ = -447.335   (dlnZ = +5.44 vs baseline)
    #   mass/mass_step     lnZ = -456.520   (dlnZ = -3.74)
    #   mass/mass_none     lnZ = -465.534   (dlnZ = -12.76)
    # mass_sigmoid_M0tau scores marginally higher alone (-446.952) but needs
    # two extra parameters for +0.38, which is inside the logZ_err budget --
    # prefer the cheaper linear form.
    "mass_linear": {
        "model": {"mass": "linear"},
        "param_overrides": {},
    },

    # -- No host stellar mass profile ---------------------------------------
    # MUST be named explicitly in a combo to take effect: CONFIG["model"]
    # defaults to mass="step", so a combo that simply omits any mass term
    # silently inherits the hard step -- the form the sweep disfavours by
    # ~9 lnZ. Naming "mass_none" makes "there is deliberately no mass term"
    # explicit in both the config and the run tag.
    #
    # experiment_naming.ExperimentRegistry auto-deactivates gamma when
    # mass="none" (gamma is mass's amplitude and would otherwise be an
    # unconstrained direction), so no param_override is needed here.
    #
    # NOTE the knock-on effect on gamma_alpha, which core.py applies as
    # gamma_alpha * x1_eff * G (core.py:1185) where
    #     G = gamma/2*L + eta*H + xi_mass_col*L*H + zeta*S + ...
    # With mass="none" the mass profile L is identically zero, so G reduces
    # to eta*H + zeta*S. A combo that pairs "interaction_gammaalpha" with
    # "mass_none" and activates NEITHER zeta (ssfr) NOR eta (host_colour)
    # therefore has G == 0 identically, which makes gamma_alpha a perfectly
    # flat, entirely unconstrained direction: it cannot change the
    # likelihood at all and the run reports an lnZ numerically identical to
    # the same model without it. That is why the two gamma_alpha combos
    # below that have no other host term use "mass_linear" instead.
    "mass_none": {
        "model": {"mass": "none"},
        "param_overrides": {},
    },

    # -- SN colour law ------------------------------------------------------
    # Best sn_colour form in the sweep, measured on the OLD baseline
    # (mass=step, host_colour=linear): dlnZ = +3.54. Whether it survives
    # alongside the other terms is exactly what the ladder below tests.
    # sn_tau is the break width; active=True means the sampler chooses it and
    # "fixed" is ignored (see config.py's spec docstring). Named with the
    # "_sntau" suffix to match the registry run name
    # sncolour/sncolour_softbroken_sntau it was measured under.
    "sncolour_softbroken_sntau": {
        "model": {"sn_colour": "softbroken"},
        "param_overrides": {"sn_tau": {"active": True, "fixed": None}},
    },

    # -- gamma x alpha interaction ------------------------------------------
    # Best interaction term in the sweep: dlnZ = +2.02 on the old baseline.
    # Every other interaction (beta_gamma, beta_alpha, the three-way) is
    # DISFAVOURED. Again, measured pre-ssfr -- the ladder re-tests it.
    #
    # REQUIRES a non-zero host term to be identifiable at all: core.py:1185
    # applies it as gamma_alpha * x1_eff * G, so any combo containing this
    # term must also contain something that makes G non-zero -- "mass_linear"
    # (via gamma/2*L), "ssfr_tanh" (via zeta*S) or a host_colour term with
    # eta activated (via eta*H). See the "mass_none" note above.
    "interaction_gammaalpha": {
        "model": {},
        "param_overrides": {"gamma_alpha": {"active": True, "fixed": None}},
    },

    # -- Host colour profile (falsification control) ------------------------
    # Kept defined but NOT used in the current COMBOS ladder (that ladder is
    # a factorial over the other three terms). Add it to a combo when you
    # want to DISPROVE host colour rather than silently omit it.
    # eta is host_colour's amplitude and C0 its centre; both default to
    # active=False with a non-zero "fixed", so turning on model["host_colour"]
    # without activating them applies an UNFITTED correction the sampler never
    # sees. Activate both, the same way "interaction_gammaalpha" activates
    # gamma_alpha.
    # Expectation from the sweep: this term is not wanted --
    #   ssfr_tanh_hcol_none_mass_linear  lnZ = -439.837
    #   ssfr_tanh_hcol_tanh_mass_linear  lnZ = -439.978  (2 more params, worse)
    "host_colour_tanh": {
        "model": {"host_colour": "tanh"},
        "param_overrides": {"eta": {"active": True, "fixed": None},
                            "C0":  {"active": True, "fixed": None}},
    },
}

# ===========================================================================
# 2. COMBOS  —  ablation ladder for combo_ablation_checks.py /
#    uniform_priors_check.py
# ===========================================================================
# Any subset of TERMS.keys() is valid; list order doesn't matter, only
# membership.
#
# DESIGN: this is a 3-term factorial over {ssfr_tanh,
# sncolour_softbroken_sntau, interaction_gammaalpha} -- the three requested
# pairs, PLUS the singletons and empty reference each pair needs to be
# interpretable. A pair's lnZ on its own says nothing; only
#     dlnZ(pair) - dlnZ(single A) - dlnZ(single B)
# tells you whether two terms are synergistic, redundant, or independent,
# and that requires the singletons and the reference to be fitted under the
# SAME background model.
#
# Because gamma_alpha needs a non-zero host term to be identifiable (see the
# "mass_none" and "interaction_gammaalpha" notes above), the factorial splits
# into two self-contained blocks, each internally matched:
#
#   BLOCK A (mass="none") -- the sSFR profile is the only host term, so
#   G = zeta*S and gamma_alpha is identifiable whenever ssfr_tanh is present.
#   Covers the ssfr_tanh+sncolour and ssfr_tanh+gamma_alpha pairs.
#
#   BLOCK B (mass="linear") -- supplies G = gamma/2*L so gamma_alpha is
#   identifiable without an sSFR term. Covers the gamma_alpha+sncolour pair,
#   which would be degenerate under mass="none".
#
# Deltas are only meaningful WITHIN a block (blocks differ in background
# model, so their absolute lnZ are not comparable term-for-term).
#
# Nine entries is deliberate but not cheap: combo_ablation_checks.py runs a
# fit PLUS a LOO z-bin CV (n_bins refits), a strict-host-match refit and a
# refit per drilling cone for EVERY entry, so each line here is of order ten
# nested sampling runs, not one. Use --only to run a block at a time.
COMBOS = [
    # ---- BLOCK A: mass="none" -------------------------------------------
    ["mass_none"],                                                   # A ref
    ["mass_none", "ssfr_tanh"],                                      # A single
    ["mass_none", "sncolour_softbroken_sntau"],                      # A single
    ["mass_none", "ssfr_tanh", "sncolour_softbroken_sntau"],         # A PAIR 1
    ["mass_none", "ssfr_tanh", "interaction_gammaalpha"],            # A PAIR 2

    # ---- BLOCK B: mass="linear" (gamma_alpha needs G != 0) --------------
    ["mass_linear"],                                                 # B ref
    ["mass_linear", "ssfr_tanh"],                                    # B single
    ["mass_linear", "interaction_gammaalpha"],                       # B single
    ["mass_linear", "sncolour_softbroken_sntau"],                    # B single
    ["mass_linear", "interaction_gammaalpha",
     "sncolour_softbroken_sntau"],                                   # B PAIR 3
]

# BEST_COMBO below MUST be one of the entries above. Every "all competing
# models" driver (combo_ablation_checks.py, z_uncertainty_check.py,
# extra_runners.py) iterates COMBOS, so a BEST_COMBO that is not in COMBOS
# would be the one model the paper actually reports and the ONLY model that
# never gets a host-quality check, a leave-one-z-bin-out validation, a
# drilling-cone check, a redshift-uncertainty check or a wCDM/subsample
# refit. That was the case before ["mass_linear", "ssfr_tanh"] was added to
# Block B above. The assertion at the bottom of this module enforces it.

# ===========================================================================
# 3. BEST_COMBO  —  the single chosen FINAL model
# ===========================================================================
# EDIT THIS as your best model changes. extra_runners.py's HOSTERR_BEST and
# z_uncertainty_check.py's default model are both derived from this list
# via merge_terms() below -- updating it here updates both automatically.
#
# Set to the publication sweep's outright winner
# (ssfr/ssfr_tanh_hcol_none_mass_linear, lnZ = -439.837, dlnZ = +12.94 over
# baseline, and the best of all 485 runs -- still the champion after the
# x1_tau fix, which changed only the stretch/* family).
#
# NOTE this is now ALSO one of the COMBOS entries above (Block B "single").
# It has to be: every driver that runs "all competing models" iterates COMBOS,
# so a BEST_COMBO outside that list would be the single model the paper
# reports and the only one never checked for host-match quality, z-bin
# stability, line-of-sight systematics, redshift uncertainty or subsample
# robustness. Promote a different ladder entry here only if it beats
# mass_linear+ssfr_tanh by more than the combined logZ_err (~0.1, so require
# dlnZ > ~1 to justify an extra parameter), and remember Block A's entries are
# fitted on mass="none" so their lnZ is not directly comparable to this
# model's -- refit the winner with mass="linear" before promoting it.
BEST_COMBO = ["mass_linear", "ssfr_tanh"]


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

# ===========================================================================
# 5. IMPORT-TIME CONSISTENCY CHECKS
# ===========================================================================
# These run on `import best_model`, so a bad edit above fails immediately in
# every driver rather than silently producing a systematically incomplete set
# of checks that only becomes obvious at write-up.

# Every combo must be uniquely named, or the drivers' run tags collide and the
# second fit either overwrites the first or is skipped.
_seen_tags = {}
for _c in COMBOS:
    _t = combo_tag(_c)
    if _t in _seen_tags:
        raise ValueError(
            f"Duplicate combo tag {_t!r} in COMBOS: {_seen_tags[_t]} and {_c} "
            f"produce the same tag, so their run tags would collide in every "
            f"driver.")
    _seen_tags[_t] = _c
    merge_terms(_c)          # surfaces unknown/conflicting terms at import
del _seen_tags, _c, _t

# BEST_COMBO must be a member of COMBOS. See the comment above BEST_COMBO for
# why: everything that runs "all competing models" iterates COMBOS.
if list(BEST_COMBO) not in [list(c) for c in COMBOS]:
    raise ValueError(
        f"BEST_COMBO {BEST_COMBO} is not in COMBOS. Every driver that checks "
        f"'all competing models' (combo_ablation_checks.py, "
        f"z_uncertainty_check.py, extra_runners.py) iterates COMBOS, so the "
        f"reported best model would be the only one never checked. Add it to "
        f"COMBOS, or point BEST_COMBO at a combo that is already there.")
