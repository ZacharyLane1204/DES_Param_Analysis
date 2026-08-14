r"""
latex_tables.py  —  SNe Ia Cosmology Pipeline
===============================================
Generates four LaTeX tables from config.py and run registry CSVs.

  Table 1 (priors)      : parameter prior specifications.  Uses table*.
  Table 2 (iterations)  : one row per run.  Uses longtable (full page
                          width, automatic multi-page continuation).
  Table 3 (evidence)    : chi2, chi2/dof, Delta AIC, Delta BIC, Delta ln B.
                          Uses longtable.
  Table 4 (checks)      : same columns, grouped by section, from
                          run_checks_registry.csv.  Uses longtable.
  Table 5 (drilling cones): line-of-sight/sky-position systematic check,
                          from a CSV written by drilling_cones.py /
                          drilling_cones_checks.py.  Uses longtable.

longtable notes
---------------
  Requires in preamble:
    \usepackage{longtable}
    \usepackage{booktabs}        % for \toprule etc (optional but nice)
    \setlength{\LTleft}{0pt}    % flush left so table spans full width
    \setlength{\LTright}{0pt}

  longtable automatically repeats the header on every new page and prints
  "Table N continued" via \endhead and \endfirsthead.
  \endfoot / \endlastfoot control the footer on mid-page / final page.

  cellcolor (xcolor) works normally inside longtable.

Usage
-----
  python latex_tables.py --priors
  python latex_tables.py --iterations
  python latex_tables.py --evidence
  python latex_tables.py --checks
  python latex_tables.py --preamble
  python latex_tables.py --evidence --out evidence_table.tex
"""

import argparse
import sys
import math

import pandas as pd

from config import DEFAULT_PARAM_SPECS, PARAM_DISPLAY, CONFIG

# ===========================================================================
# Shared helpers
# ===========================================================================

def _fmt_range(lo, hi):
    def _fmt(v):
        if v == int(v):
            return str(int(v))
        return f"{v:g}"
    return rf"$[{_fmt(lo)}, {_fmt(hi)}]$"


def _prior_type_latex(ptype):
    mapping = {
        "uniform":            "Uniform",
        "gaussian":           "Gaussian",
        "truncated_gaussian": "Truncated Gaussian",
        "log_uniform":        "Log-Uniform",
        "log_normal":         "Log-Normal",
        "arcsinh":            r"$\mathrm{arcsinh}$",
    }
    return mapping.get(ptype, ptype)


def _dist_params_latex(name, spec):
    ptype = spec["prior"]
    if ptype in ("uniform", "log_uniform"):
        return ""
    elif ptype in ("gaussian", "truncated_gaussian"):
        mu    = spec.get("mu",    "")
        sigma = spec.get("sigma", "")
        return rf"$\mu = {mu}$, $\sigma = {sigma}$"
    elif ptype == "log_normal":
        mu_ln    = spec.get("mu",    "")
        sigma_ln = spec.get("sigma", "")
        return (rf"$\mu_{{\ln}}^{{}} = {mu_ln}$, "
                rf"$\sigma_{{\ln}}^{{}} = {sigma_ln}$")
    elif ptype == "arcsinh":
        scale = spec.get("scale", "")
        return rf"$s = {scale}$"
    return ""


def _label(name):
    # Local aliases handle parameter names that may differ between registry
    # entries (logged under an older name) and the current PARAM_DISPLAY keys.
    _ALIASES = {
        "x1_0":  "x1_0",
        "x1_0":  "x1_0",
    }
    resolved = _ALIASES.get(name, name)
    return PARAM_DISPLAY.get(resolved, {}).get("label", resolved)


def _esc(s):
    return str(s).replace("_", r"\_")


def _read_float(row, col):
    raw = row.get(col, None)
    try:
        return float(raw) if pd.notna(raw) else None
    except (TypeError, ValueError):
        return None


# ===========================================================================
# Colour schemes
# ===========================================================================

_AIC_COLORS = [
    (-math.inf, -10,  "aicgreen"),
    (-10,        -7,  "aiclightgreen"),
    (-7,         -4,  "aicyellow"),
    (-4,         -2,  "aiclightyellow"),
    # neutral: (-2, +2)
    (2,           4,  "aiclightorange"),
    (4,           7,  "aicorange"),
    (7,          10,  "aiclightred"),
    (10,   math.inf,  "aicred"),
]

_BIC_COLORS = [
    (-math.inf, -10,  "aicgreen"),
    (-10,        -6,  "aiclightgreen"),
    (-6,         -4,  "aicyellow"),
    (-4,         -2,  "aiclightyellow"),
    # neutral: (-2, +2)
    (2,           4,  "aiclightorange"),
    (4,           6,  "aicorange"),
    (6,          10,  "aiclightred"),
    (10,   math.inf,  "aicred"),
]

# Reversed: negative delta = model disfavoured (red), positive = preferred (green)
_LOGZ_COLORS = [
    (-math.inf, -5,  "aicred"),
    (-5,        -3,  "aiclightred"),
    (-3,        -1,  "aicorange"),
    # neutral: (-1, +1)
    (1,          3,  "aicyellow"),
    (3,          5,  "aiclightgreen"),
    (5,   math.inf,  "aicgreen"),
]


def _color_cell(value, scheme):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    fmt = f"{value:.2f}"
    for lo, hi, color in scheme:
        if lo <= value < hi:
            return rf"\cellcolor{{{color}}}{fmt}"
    return fmt


def _delta_cell(raw_val, base_val, scheme):
    if raw_val is None or base_val is None:
        return "---"
    return _color_cell(raw_val - base_val, scheme)


def _delta_logz_cell(raw_val, raw_err, base_val, base_err, scheme):
    """Delta ln B with combined uncertainty sqrt(err_run^2 + err_base^2)."""
    if raw_val is None or base_val is None:
        return "---"
    delta = raw_val - base_val
    if raw_err is not None and base_err is not None:
        err_comb  = math.sqrt(raw_err**2 + base_err**2)
        cell_body = rf"{delta:.2f} $\pm$ {err_comb:.2f}"
    else:
        cell_body = f"{delta:.2f}"
    for lo, hi, color in scheme:
        if lo <= delta < hi:
            return rf"\cellcolor{{{color}}}{cell_body}"
    return cell_body


# ===========================================================================
# longtable builder helpers
# ===========================================================================

def _lt_header_block(col_header_line, ncols, caption, label, is_first):
    """
    Return lines for the header block that appears at the top of each page.
    is_first=True  → used before \endfirsthead (includes caption + label).
    is_first=False → used before \endhead (includes "Table N continued").
    """
    lines = []
    lines
    if is_first:
        lines.append(rf"    \caption{{{caption}}} \label{{{label}}} \\")
    else:
        # \tablename is AASTeX-only; use \thetable directly for standard LaTeX
        lines.append(
            rf"    \multicolumn{{{ncols}}}{{c}}"
            rf"{{Table~\thetable{{}} \emph{{(continued)}}}} \\"
        )
    lines += [
        r"    \hline",
        r"    \hline",
        f"    {col_header_line} \\\\",
        r"    \hline",
        r"    \hline",
    ]
    return lines


def _lt_wrap(col_spec, caption, label, col_header_line, data_rows,
             ncols, notes=None):
    """
    Build a complete longtable.

    data_rows : list of strings.
      - Regular data rows: a complete &-separated row, no trailing \\.
        _lt_wrap appends \\.
      - Section separators: must be exactly r"\hline" or start with
        r"\multicolumn" — these are emitted as-is (\hline no \\,
        \multicolumn gets \\ appended).
    The very last data/notes row must NOT have \\ — longtable forbids it.

    A \clearpage is emitted before \onecolumngrid.  longtable is not a
    LaTeX float — it typesets exactly where it sits in the source — but
    table* environments (e.g. the Mass Options table) ARE floats, and
    LaTeX is free to delay placing them until it finds room, which is
    often AFTER a later longtable that forces a column/page break via
    \onecolumngrid.  This is why a table* defined earlier in the source
    can visually appear after a longtable defined later.  \clearpage
    forces every float queued so far to flush before the longtable
    begins, restoring the order tables actually appear in the .tex
    source.  If you don't want page breaks between every single table,
    see the comment in PREAMBLE_SNIPPET for the alternative \FloatBarrier
    approach.
    """
    lines = []
    lines.append(r"\clearpage")
    lines.append(r"\onecolumngrid")
    lines.append(rf"\begin{{longtable}}{{{col_spec}}}")

    # --- first-page header ---
    lines += _lt_header_block(col_header_line, ncols, caption, label,
                               is_first=True)
    lines.append(r"    \endfirsthead")

    # --- continuation header ---
    lines += _lt_header_block(col_header_line, ncols, caption, label,
                               is_first=False)
    lines.append(r"    \endhead")

    # --- mid-page footer ---
    lines.append(r"    \hline")
    lines.append(
        rf"    \multicolumn{{{ncols}}}{{r}}{{\emph{{Continued on next page}}}} \\"
    )
    lines.append(r"    \endfoot")

    # --- last-page footer ---
    lines.append(r"    \hline")
    lines.append(r"    \endlastfoot")

    # Combine data rows + optional notes into one list so we can withhold
    # the trailing \\ from the very last item.
    # notes may be a single string or a list of strings; each becomes its
    # own \multicolumn note line.
    all_body = list(data_rows)
    if notes:
        note_list = [notes] if isinstance(notes, str) else list(notes)
        for note in note_list:
            all_body.append(
                rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{Note.}}~{note}}}"
            )

    for idx, row in enumerate(all_body):
        stripped = row.strip()
        is_last  = (idx == len(all_body) - 1)
        if stripped == r"\hline":
            # bare \hline — no \\ needed or wanted
            lines.append(r"    \hline")
        elif stripped.startswith(r"\multicolumn"):
            # section title or notes — needs \\ unless it's the very last line
            suffix = "" if is_last else r" \\"
            lines.append(f"    {stripped}{suffix}")
        else:
            # normal data row — needs \\ unless last
            suffix = "" if is_last else r" \\"
            lines.append(f"    {row}{suffix}")

    lines.append(r"\end{longtable}")
    lines.append(r"\twocolumngrid")
    return "\n".join(lines)


# ===========================================================================
# TABLE 1 — Priors  (table*, single-page)
# ===========================================================================

_PRIOR_ROW_ORDER = [
    "Om0", "fv0", "alpha", "beta", "gamma",
    "Ode0", "w",
    "a", "b", "g",
    "M0", "M1", "c0", "C0", "eta", "xi",
    "tau", "htau", "sn_tau",
    "k1", "k2", "k3",
    "beta_alpha", "beta_gamma", "gamma_alpha",
]

_PRIOR_GROUPS = [
    {"Om0", "fv0", "alpha", "beta", "gamma"},
    {"Ode0", "w"},
    {"a", "b", "g"},
    {"M0", "M1", "c0", "C0", "eta", "xi"},
    {"tau", "htau", "sn_tau"},
    {"k1", "k2", "k3"},
    {"beta_alpha", "beta_gamma", "gamma_alpha"},
]


def _group_index(name):
    for i, g in enumerate(_PRIOR_GROUPS):
        if name in g:
            return i
    return len(_PRIOR_GROUPS)


def generate_priors_table(param_specs=None):
    if param_specs is None:
        param_specs = DEFAULT_PARAM_SPECS

    ordered, seen = [], set()
    for name in _PRIOR_ROW_ORDER:
        if name in param_specs:
            ordered.append(name); seen.add(name)
    for name in param_specs:
        if name not in seen and name != "M":
            ordered.append(name)

    lines = [
        r"\begin{table*}",
        r"    \caption{Priors}",
        r"    \label{tab:priors}",
        r"    \centering",
        r"    \begin{tabular}{cccc}",
        r"        \hline",
        r"        \hline",
        r"        Parameters & Prior Range & Prior Distribution & Distribution Parameters \\",
        r"        \hline",
        r"        \hline",
    ]
    prev_group = _group_index(ordered[0]) if ordered else -1
    for name in ordered:
        spec = param_specs[name]
        grp  = _group_index(name)
        if grp != prev_group:
            lines.append(r"        \hline")
        prev_group = grp
        lo, hi = spec["range"]
        lines.append(
            f"        {_label(name)} & {_fmt_range(lo, hi)}"
            f" & {_prior_type_latex(spec['prior'])}"
            f" & {_dist_params_latex(name, spec)} \\\\"
        )
    lines += [r"        \hline", r"    \end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ===========================================================================
# TABLE 2 — Iterations  (longtable, full width, multi-page)
# ===========================================================================

# ---------------------------------------------------------------------------
# Section ordering shared by the Iterations and Evidence tables
# ---------------------------------------------------------------------------
# Runs are grouped by the prefix of their run_name tag.  The order here is
# the order sections appear in BOTH tables, with a \hline + bold header
# between each section.  Runs whose tag doesn't match any prefix land in
# "Other" at the end.  Rows keep their original CSV order within a section.

_REGISTRY_SECTIONS = [
    # (display_label,        tag_prefix)
    ("Baseline",             "baseline"),
    ("Cosmology",            "cosmo/"),
    ("Nuisance Parameters",  "nuisance/"),
    # The evolution runs are deliberately fitted under broad uniform
    # alpha/beta priors (see experiment_runner._ZEVO_BROAD_UNIFORM), so the
    # label says so rather than letting a reader assume they share the
    # baseline's priors.  Om0 keeps its informative CMB prior in this
    # section on purpose -- a free Om0 is near-degenerate with the
    # evolution exponents and would absorb the signal being measured.
    (r"Redshift Evolution (uniform $\alpha$, $\beta$)", "evolution/"),
    ("Interaction Terms",    "interaction/"),
    ("Stretch Correction",   "stretch/"),
    ("SN Colour Model",      "sn_col_model/"),
    ("Mass Step",            "mass/"),
    ("Host Colour Model",    "host_col_model/"),
    ("sSFR Host Environment","ssfr/"),
]


def _section_for_tag(tag):
    """Return the section index for a run_name tag, or len(...) for 'Other'."""
    tag = str(tag)
    for i, (_, prefix) in enumerate(_REGISTRY_SECTIONS):
        if prefix == "baseline":
            if tag == "baseline":
                return i
        elif tag.startswith(prefix):
            return i
    return len(_REGISTRY_SECTIONS)


def _group_by_section(df):
    """
    Return a list of (section_label, sub_df) pairs in _REGISTRY_SECTIONS
    order.  Rows keep their original CSV order within each section.
    """
    df = df.copy()
    df["_sec"] = df["run_name"].astype(str).apply(_section_for_tag)
    df["_pos"] = range(len(df))
    df = df.sort_values(["_sec", "_pos"], kind="stable")

    groups = []
    for sec_idx, sub in df.groupby("_sec", sort=True):
        label = (_REGISTRY_SECTIONS[sec_idx][0]
                 if sec_idx < len(_REGISTRY_SECTIONS) else "Other")
        groups.append((label, sub.drop(columns=["_sec", "_pos"])))
    return groups


def generate_iterations_table(registry_path=None):
    """
    Columns: Run | N | Parameter List | Stretch | SN Colour | Mass
             | Host Colour | sSFR

    Runs are grouped into sections (Baseline, Cosmology, Nuisance, ...,
    sSFR) in the fixed order given by _REGISTRY_SECTIONS, each separated by
    a \\hline + bold section header.  Within a section, rows keep their
    original CSV order.

    If a run has more than 7 active parameters, the parameter list is split
    roughly in half and the second half is placed on its own continuation
    row directly beneath the main row (a \\multicolumn spanning just the
    Parameter List column, left-blank in the other columns) — this keeps
    very long parameter lists from forcing the whole row onto a single
    unreadably-wide line.
    """
    if registry_path is None:
        registry_path = CONFIG.get("registry_file", "run_publication_registry.csv")
    try:
        df = pd.read_csv(registry_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    NCOLS = 8
    col_header = (
        r"Run & $N$ & Parameter List"
        r" & Stretch & SN Colour & Mass & Host Colour & sSFR"
    )

    # Parameter-list column occupies position 3 of 8 (1-indexed) in the
    # row; continuation lines need to pad columns 1-2 (blank) then
    # \multicolumn the remaining text, then leave columns 4-8 blank so the
    # & count still matches col_spec.
    N_WRAP_THRESHOLD = 7

    section_groups = _group_by_section(df)

    data_rows = []
    row_num = 1   # counter for non-baseline rows, continues across sections
    first_section = True
    for section_label, sub_df in section_groups:
        if not first_section:
            data_rows.append(r"\hline")
        first_section = False
        data_rows.append(
            rf"\multicolumn{{{NCOLS}}}{{c}}{{\textbf{{{section_label}}}}}"
        )
        data_rows.append(r"\hline")

        for _, row in sub_df.iterrows():
            _rn = str(row.get("run_name", ""))
            if _rn == "baseline":
                run_name = "baseline"
            else:
                run_name = str(row_num)
                row_num += 1
            n_params     = int(row.get("ndim", 0))
            active_raw   = str(row.get("active_params", ""))
            active_list  = [p.strip() for p in active_raw.split("|") if p.strip()]

            x1_mod   = _esc(str(row.get("x1_correction_model", "") or "---"))
            sn_col   = _esc(str(row.get("sn_colour_model",     "") or "---"))
            mass_mod = _esc(str(row.get("mass_model",           "") or "---"))
            host_col = _esc(str(row.get("host_colour_model",    "") or "---"))
            ssfr_mod = _esc(str(row.get("ssfr_model",           "") or "---"))

            if len(active_list) > N_WRAP_THRESHOLD:
                # Split roughly in half; first half stays on the main row,
                # second half drops to a continuation row underneath.
                split_at     = (len(active_list) + 1) // 2
                first_half   = ", ".join(_label(p) for p in active_list[:split_at])
                second_half  = ", ".join(_label(p) for p in active_list[split_at:])

                data_rows.append(
                    f"{run_name} & {n_params} & {first_half}"
                    f" & {x1_mod} & {sn_col} & {mass_mod} & {host_col} & {ssfr_mod}"
                )
                # Continuation row: blank Run/N columns, second half of the
                # parameter list spans just that column, remaining columns blank.
                data_rows.append(
                    rf"\multicolumn{{1}}{{c}}{{}} & \multicolumn{{1}}{{c}}{{}}"
                    rf" & {second_half}"
                    rf" & \multicolumn{{1}}{{c}}{{}} & \multicolumn{{1}}{{c}}{{}}"
                    rf" & \multicolumn{{1}}{{c}}{{}} & \multicolumn{{1}}{{c}}{{}}"
                    rf" & \multicolumn{{1}}{{c}}{{}}"
                )
            else:
                param_labels = ", ".join(_label(p) for p in active_list)
                data_rows.append(
                    f"{run_name} & {n_params} & {param_labels}"
                    f" & {x1_mod} & {sn_col} & {mass_mod} & {host_col} & {ssfr_mod}"
                )

    col_spec = r"c c c c c c c c"

    return _lt_wrap(
        col_spec        = col_spec,
        caption         = "Iterations",
        label           = "tab:iterations",
        col_header_line = col_header,
        data_rows       = data_rows,
        ncols           = NCOLS,
        notes           = (
            r"Runs grouped by category.  Parameter lists over 7 entries "
            r"continue below."
        ),
    )

# ===========================================================================
# Shared evidence row builder
# ===========================================================================

def _build_evidence_rows(df, base_aic, base_bic, base_logz, base_logz_err,
                         baseline_run_name="baseline",
                         first_col_fn=None):
    """
    Build data-row strings for an evidence table.

    first_col_fn(row) → string for the leftmost cell.
    Defaults to the 1-based row index if not supplied.
    """
    rows = []
    for i, row in df.iterrows():
        run_name     = str(row.get("run_name", ""))
        n_params     = int(row.get("ndim", 0))
        is_base      = (run_name == baseline_run_name)

        raw_chi2     = _read_float(row, "chi2")
        raw_chi2_dof = _read_float(row, "chi2_dof")
        raw_aic      = _read_float(row, "AIC")
        raw_bic      = _read_float(row, "BIC")
        raw_logz     = _read_float(row, "logZ")
        raw_logz_err = _read_float(row, "logZ_err")

        chi2_str     = f"{raw_chi2:.2f}"     if raw_chi2     is not None else "---"
        chi2_dof_str = f"{raw_chi2_dof:.3f}" if raw_chi2_dof is not None else "---"

        if is_base:
            aic_str  = "---"
            bic_str  = "---"
            logz_str = "---"
        else:
            aic_str  = _delta_cell(raw_aic, base_aic, _AIC_COLORS)
            bic_str  = _delta_cell(raw_bic, base_bic, _BIC_COLORS)
            logz_str = _delta_logz_cell(raw_logz, raw_logz_err,
                                        base_logz, base_logz_err,
                                        _LOGZ_COLORS)

        first = first_col_fn(row) if first_col_fn else str(i + 1)

        rows.append(
            f"{first} & {n_params}"
            f" & {chi2_str} & {chi2_dof_str}"
            f" & {aic_str} & {bic_str} & {logz_str}"
        )
    return rows


# ===========================================================================
# TABLE 3 — Evidence  (longtable, full width, multi-page)
# ===========================================================================

def generate_evidence_table(registry_path=None):
    """
    Columns: Run | N | chi2 | chi2/dof | Delta AIC | Delta BIC | Delta ln B

    Runs are grouped into the SAME sections, in the SAME order, as the
    Iterations table (_REGISTRY_SECTIONS), each separated by a \\hline +
    bold section header — so a given run number refers to the same model
    combination in both tables.  Within a section, rows keep their original
    CSV order.

    All deltas relative to the row with run_name == "baseline".
    """
    if registry_path is None:
        registry_path = CONFIG.get("registry_file", "run_publication_registry.csv")
    try:
        df = pd.read_csv(registry_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    baseline_mask = df["run_name"].astype(str) == "baseline"
    if not baseline_mask.any():
        raise RuntimeError("No row with run_name == 'baseline' found.")
    if baseline_mask.sum() > 1:
        import warnings; warnings.warn("Multiple baseline rows; using first.")

    b             = df[baseline_mask].iloc[0]
    base_aic      = _read_float(b, "AIC")
    base_bic      = _read_float(b, "BIC")
    base_logz     = _read_float(b, "logZ")
    base_logz_err = _read_float(b, "logZ_err")

    # ---- Matched-prior reference rows ------------------------------------
    # Some sections are fitted under deliberately different priors from the
    # global "baseline" row -- most importantly the whole "evolution/"
    # section, which uses broad uniform alpha/beta/Om0 (see
    # experiment_runner.py's _ZEVO_BROAD_UNIFORM). Widening a prior costs
    # evidence through the Occam factor regardless of fit quality, so
    # differencing those rows against the informative-prior baseline
    # reports a penalty for the prior volume and calls it evidence against
    # the model, which is precisely the misreading this whole exercise
    # exists to avoid.
    #
    # Where such a section ships its own matched-prior reference (a row
    # fitted under the section's priors but without the feature under
    # test), deltas within that section are taken against it instead. The
    # reference row itself is shown with "---" deltas, exactly like the
    # global baseline.
    section_baselines = {
        # section prefix -> matched-prior reference run_name
        "evolution/": "evolution/baseline_broaduniform",
    }
    section_bases = {}
    for prefix, ref_name in section_baselines.items():
        ref_mask = df["run_name"].astype(str) == ref_name
        if ref_mask.any():
            r = df[ref_mask].iloc[0]
            section_bases[prefix] = (ref_name,
                                     _read_float(r, "AIC"),
                                     _read_float(r, "BIC"),
                                     _read_float(r, "logZ"),
                                     _read_float(r, "logZ_err"))

    NCOLS = 7
    col_header = (
        r"Run & $N$ & $\chi^2$ & $\chi^2/\mathrm{d.o.f}$"
        r" & $\Delta$AIC & $\Delta$BIC & $\Delta\ln B$"
    )
    col_spec = r"c c c c c c c"

    section_groups = _group_by_section(df)

    data_rows = []
    row_num = 1   # counter for non-baseline rows, continues across sections
    first_section = True
    used_section_baseline = False
    for section_label, sub_df in section_groups:
        if not first_section:
            data_rows.append(r"\hline")
        first_section = False
        data_rows.append(
            rf"\multicolumn{{{NCOLS}}}{{c}}{{\textbf{{{section_label}}}}}"
        )
        data_rows.append(r"\hline")

        # Does this section have its own matched-prior reference row?
        sec_ref = None
        for prefix, entry in section_bases.items():
            if sub_df["run_name"].astype(str).str.startswith(prefix).any():
                sec_ref = entry
                break

        if sec_ref is not None:
            ref_name, sec_aic, sec_bic, sec_logz, sec_logz_err = sec_ref
            used_section_baseline = True
        else:
            ref_name = "baseline"
            sec_aic, sec_bic = base_aic, base_bic
            sec_logz, sec_logz_err = base_logz, base_logz_err

        def _first_col(row, _ref=ref_name):
            nonlocal row_num
            name = str(row.get("run_name", ""))
            if name == "baseline":
                return "baseline"
            if name == _ref:
                # Section's matched-prior reference: label it rather than
                # numbering it, so the table shows at a glance what the
                # section's deltas are measured against.
                return "ref"
            label = str(row_num)
            row_num += 1
            return label

        data_rows.extend(_build_evidence_rows(
            sub_df, sec_aic, sec_bic, sec_logz, sec_logz_err,
            baseline_run_name=ref_name,
            first_col_fn=_first_col,
        ))

    notes = (
        r"Runs grouped by category (same order as Iterations table).  "
        r"$\Delta\ln B$ errors are $\sqrt{\sigma_\mathrm{run}^2 "
        r"+ \sigma_\mathrm{base}^2}$."
    )
    if used_section_baseline:
        notes += (
            r"  Sections fitted under different priors from the baseline "
            r"are differenced against their own matched-prior reference "
            r"row, marked \textit{ref}, rather than against the baseline: "
            r"widening a prior lowers $\ln Z$ through the Occam factor "
            r"alone, so a cross-prior $\Delta\ln B$ would not measure "
            r"model preference.  The redshift-evolution runs use uniform "
            r"$\alpha$ and $\beta$ but retain the CMB prior on "
            r"$\Omega_{\rm M0}^{}$, which is near-degenerate with the "
            r"evolution exponents."
        )

    return _lt_wrap(
        col_spec        = col_spec,
        caption         = "Model Comparison Statistics",
        label           = "tab:bic",
        col_header_line = col_header,
        data_rows       = data_rows,
        ncols           = NCOLS,
        notes           = notes,
    )


# ===========================================================================
# TABLE 4 — Checks  (longtable, full width, multi-page, section \hlines)
# ===========================================================================

_CHECKS_SECTIONS = [
    # (display_label,  baseline_run_name,         prefixes,           source)
    (r"$w$CDM",
     "baseline",           ["checks/wcdm_"],         "pub"),
    (r"Non-flat $\Lambda$CDM",
     "baseline",           ["checks/lcdm_"],          "pub"),
    (r"$z \geq 0.1$",
     "checks/zlow_baseline", ["checks/zlow_"],         "checks"),
    (r"$z \leq 0.1$",
     "checks/zhi_baseline",  ["checks/zhi_"],          "checks"),
    (r"DES + Foundation",
     "checks/id_baseline",   ["checks/id_"],           "checks"),
    (r"$\log_{10}^{} \mathrm{M}^* < 10$",
     "checks/masscut_low_baseline",  ["checks/masscut_low_"],  "checks"),
    (r"$\log_{10}^{} \mathrm{M}^* \geq 10$",
     "checks/masscut_high_baseline", ["checks/masscut_high_"], "checks"),
    (r"Spectroscopic host-$z$ only",
     "baseline",              ["checks/specz_"],        "pub"),
    (r"Photometric host-$z$ only",
     "baseline",              ["checks/photz_"],        "pub"),
    (r"Strict host-match quality",
     "baseline",              ["checks/hostquality_strict_"], "pub"),
    (r"$x_1 \in [-2, 2]$",
     "checks/x1cut_high_baseline", ["checks/x1cut_"],    "checks"),
    (r"$|c| \leq 0.2$",
     "checks/ccut_high_baseline",  ["checks/ccut_"],     "checks"),
    (r"Uniform priors ($\alpha,\beta,\Omega_{\rm M0}$)",
     "checks/uniformpriors_baseline", ["checks/uniformpriors_"], "checks"),
]

_VARIANT_LABELS = {
    # Exact-match overrides take priority over the token parser below.
    # Mostly useful for legacy / hand-tuned phrasing on common variants.
    "baseline": r"Baseline",
}

# ---------------------------------------------------------------------------
# Token-based variant-label parser
# ---------------------------------------------------------------------------
# Run-name suffixes are built by concatenating short keyword fragments with
# underscores, e.g. "sncolour_softbrokensntau_mass_linear" or
# "gamma_alpha_ssfr_tanhF0ftau_mass_linear".  Rather than hand-listing every
# combination in _VARIANT_LABELS, _TOKEN_PHRASES maps each recognised
# fragment to a human-readable LaTeX phrase, and _variant_label stitches
# matched fragments together in encounter order, comma-separated.
#
# Longer / more specific keys are tried first so e.g. "softbrokensntau"
# matches before a bare "softbroken" would.  Add new fragments here as new
# model options are introduced — no need to enumerate every combination.
_TOKEN_PHRASES = [
    # --- parameter / interaction toggles ---
    ("gamma_alpha",        r"$\gamma_\alpha^{}$"),
    ("beta_alpha",         r"$\beta_\alpha^{}$"),
    ("beta_gamma",         r"$\beta_\gamma^{}$"),

    # --- SN colour models (sncolour_<model>) ---
    ("sncolour_softbrokensntau", r"softbroken $\tau_S^{}$"),
    ("sncolour_asymm_gauss_weight", r"gauss.\ weight colour"),
    ("sncolour_softbroken",      r"softbroken colour"),
    ("sncolour_quadratic",       r"quad.\ colour"),
    ("sncolour_broken",          r"broken colour"),
    ("sncolour_tanh",            r"$\tanh$ colour"),
    ("sncolour_dust",            r"dust colour"),
    ("sncolour_stepbroken",      r"step-broken colour"),
    ("sncolour_linear",          r"lin.\ colour"),

    # --- mass models (mass_<model>) ---
    ("mass_linear",         r"lin.\ mass"),
    ("mass_step",           r"step mass"),
    ("mass_tanh",           r"$\tanh$ mass"),
    ("mass_sigmoid",        r"sigmoid mass"),
    ("mass_double_step",    r"double-step mass"),
    ("mass_gaussian_weight",r"Gaussian-weight mass"),
    ("mass_spline",         r"spline mass"),
    ("mass_none",           r"no mass term"),

    # --- host colour models (hcol_<model> / host_colour_<model>) ---
    ("hcol_linear",         r"lin.\ host colour"),
    ("hcol_quadratic",      r"quad.\ host colour"),
    ("hcol_sigmoid",        r"sigmoid host colour"),
    ("hcol_tanh",           r"$\tanh$ host colour"),
    ("hcol_broken",         r"broken host colour"),
    ("hcol_asymm",          r"asymm.\ host colour"),
    ("hcol_none",           r"no host-colour term"),

    # --- sSFR models (ssfr_<model><shape params>) ---
    ("ssfr_tanhF0ftau",     r"$\tanh$ sSFR ($F_0^{}$, $\tau_S^{}$ free)"),
    ("ssfr_tanhftau",       r"$\tanh$ sSFR ($\tau_S^{}$ free)"),
    ("ssfr_tanhF0",         r"$\tanh$ sSFR ($F_0^{}$ free)"),
    ("ssfr_tanh",           r"$\tanh$ sSFR"),
    ("ssfr_sigmoidF0ftau",  r"sigmoid sSFR ($F_0^{}$, $\tau_S^{}$ free)"),
    ("ssfr_sigmoidftau",    r"sigmoid sSFR ($\tau_S^{}$ free)"),
    ("ssfr_sigmoidF0",      r"sigmoid sSFR ($F_0^{}$ free)"),
    ("ssfr_sigmoid",        r"sigmoid sSFR"),
    ("ssfr_step",           r"step sSFR"),
    ("ssfr_linear",         r"lin.\ sSFR"),
    ("ssfr_none",           r"no sSFR term"),

    # --- x1 / stretch correction (x1_<model>) ---
    ("x1_quadratic",        r"quad.\ stretch"),
    ("x1_tanh",             r"$\tanh$ stretch"),
    ("x1_softbroken",       r"softbroken stretch"),
    ("x1_stepbroken",       r"step-broken stretch"),
    ("x1_linear",           r"lin.\ stretch"),
]


def _tokenize_variant_suffix(suffix):
    """
    Greedily match known fragments against the underscore-joined suffix,
    longest fragment first, then return their LaTeX phrases ordered by
    where each fragment originally appeared in the suffix (left to right) —
    not by the order in which they happened to be matched.

    Any leftover, unmatched text is appended at the end (escaped), so
    nothing is silently dropped — an unrecognised fragment is visible in
    the output as a prompt to add it to _TOKEN_PHRASES.
    """
    # Try longest fragments first so e.g. "sncolour_softbrokensntau" is
    # matched before the shorter "sncolour_softbroken" would steal part of it.
    candidates = sorted(_TOKEN_PHRASES, key=lambda kv: -len(kv[0]))

    # mask[i] = True once character i of the *original* suffix has been
    # claimed by some matched fragment.  This lets every match record its
    # position relative to the one fixed coordinate system (the original
    # string), so final ordering is correct no matter which fragment was
    # found first.
    mask = [False] * len(suffix)
    found = []  # list of (start_index, phrase)

    for frag, phrase in candidates:
        start = 0
        while True:
            idx = suffix.find(frag, start)
            if idx == -1:
                break
            # Only accept this match if none of its characters were
            # already claimed by an earlier (longer/higher-priority) match.
            span = range(idx, idx + len(frag))
            if not any(mask[i] for i in span):
                for i in span:
                    mask[i] = True
                found.append((idx, phrase))
            start = idx + 1

    found.sort(key=lambda t: t[0])
    ordered = [phrase for _, phrase in found]

    # Anything not claimed by a fragment match, stripped of stray
    # underscores left behind by the claimed spans, is appended verbatim.
    leftover_chars = [c if not m else "_" for c, m in zip(suffix, mask)]
    leftover = "_".join(p for p in "".join(leftover_chars).split("_") if p)
    if leftover:
        ordered.append(_esc(leftover))

    return ordered


def _variant_label(row):
    """Accept either a DataFrame row (Series) or a bare run_name string."""
    if hasattr(row, "get"):
        run_name = str(row.get("run_name", ""))
    else:
        run_name = str(row)
    parts  = run_name.split("/")
    suffix = parts[-1] if parts else run_name
    for _, _, prefixes, _ in _CHECKS_SECTIONS:
        for p in prefixes:
            key = p.replace("checks/", "")
            if suffix.startswith(key):
                suffix = suffix[len(key):]
                break

    if suffix in _VARIANT_LABELS:
        return _VARIANT_LABELS[suffix]

    phrases = _tokenize_variant_suffix(suffix)
    if not phrases:
        return _esc(suffix) if suffix else "Baseline"
    return ", ".join(phrases)


def generate_checks_table(checks_registry_path=None, pub_registry_path=None):
    """
    Robustness-checks evidence table, grouped into labelled sections with
    \hline separators.  wCDM and non-flat LCDM sections use the publication
    registry baseline; all other sections use their own within-section baseline.
    """
    if checks_registry_path is None:
        checks_registry_path = "run_checks_registry.csv"
    if pub_registry_path is None:
        pub_registry_path = CONFIG.get("registry_file", "run_publication_registry.csv")

    try:
        df_checks = pd.read_csv(checks_registry_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checks registry not found: {checks_registry_path}")

    try:
        df_pub = pd.read_csv(pub_registry_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Publication registry not found: {pub_registry_path}")

    pub_mask = df_pub["run_name"].astype(str) == "baseline"
    if not pub_mask.any():
        raise RuntimeError("No 'baseline' row in publication registry.")

    def _model_sig(row):
        """
        Canonical model signature used for matching a checks row to its
        flat-LambdaCDM ("std") equivalent.  Built from the model columns so
        that order and naming differences in run_name strings don't matter.

        Returns a tuple of (sn_colour, mass, host_colour, z_evolve,
        has_gamma_alpha).  cosmo_type is intentionally excluded — it is the
        axis that differs between the wCDM / non-flat-LCDM checks run and
        its flat-LCDM ("std") counterpart.
        """
        sn   = str(row.get("sn_colour_model",   "") or "").strip()
        mass = str(row.get("mass_model",         "") or "").strip()
        hcol = str(row.get("host_colour_model",  "") or "").strip()
        zev  = str(row.get("z_evolve_model",     "") or "").strip()
        apar = str(row.get("active_params",      "") or "").strip()
        has_gamma_alpha = "gamma_alpha" in apar
        return (sn, mass, hcol, zev, has_gamma_alpha)

    # Build signature → flat-LCDM "std" row lookup.
    # The flat-LambdaCDM counterpart of every wCDM / non-flat-LCDM checks
    # variant lives in the SAME checks registry, tagged "checks/std_<variant>"
    # — NOT in the publication registry under a "best_fit/" prefix.  Looking
    # for best_fit/ in df_pub was the bug: this pipeline's checks workflow
    # never writes that prefix, so the lookup was always empty and every
    # wCDM/non-flat-LCDM row fell back to '---'.
    pub_lookup = {}
    for _, r in df_checks.iterrows():
        rn = str(r.get("run_name", ""))
        if rn.startswith("checks/std_") or rn in ("checks/baseline", "baseline"):
            sig = _model_sig(r)
            pub_lookup[sig] = r
    # Fall back to the publication registry's own baseline row too, in case
    # the no-variant flat-LCDM case was logged there instead of in checks/.
    for _, r in df_pub[pub_mask].iterrows():
        sig = _model_sig(r)
        pub_lookup.setdefault(sig, r)

    col_header = (
        r"Variant & $N$ & $\chi^2$ & $\chi^2/\mathrm{d.o.f}$"
        r" & $\Delta$AIC & $\Delta$BIC & $\Delta\ln B$"
    )
    # Left-aligned first column for the variant label; rest centred except last
    col_spec = r"c c c c c c c"

    all_data_rows = []
    first_section = True

    for section_label, baseline_name, prefixes, baseline_source in _CHECKS_SECTIONS:
        def _in_section(rn, _pfx=prefixes):
            return any(str(rn).startswith(p) for p in _pfx)
        mask       = df_checks["run_name"].astype(str).apply(_in_section)
        section_df = df_checks[mask].copy()

        if section_df.empty:
            continue

        # Section separator
        if not first_section:
            all_data_rows.append(r"\hline")
        first_section = False

        # Section title spanning all 7 columns
        all_data_rows.append(
            rf"\multicolumn{{7}}{{c}}{{\textbf{{{section_label}}}}}"
        )
        all_data_rows.append(r"\hline")

        if baseline_source == "pub":
            # Per-row matching using model signature (sn_colour_model, mass_model,
            # host_colour_model, z_evolve_model, gamma_alpha active flag).
            # cosmo_type is excluded so a wCDM/non-flat-LCDM run matches its
            # flat-LCDM ("std") equivalent, which lives in the checks
            # registry under "checks/std_<same suffix>".
            for _, row in section_df.iterrows():
                run_name = str(row.get("run_name", ""))
                n_params = int(row.get("ndim", 0))
                sig      = _model_sig(row)

                pub_row = pub_lookup.get(sig)
                if pub_row is None:
                    import warnings
                    warnings.warn(
                        f"No 'checks/std_*' row matching model signature {sig} "
                        f"for checks run '{run_name}'.  Expected a counterpart "
                        f"named e.g. 'checks/std_{run_name.split('checks/', 1)[-1].split('_', 1)[-1]}' "
                        f"in run_checks_registry.csv; deltas will be '---'."
                    )

                b_aic      = _read_float(pub_row, "AIC")      if pub_row is not None else None
                b_bic      = _read_float(pub_row, "BIC")      if pub_row is not None else None
                b_logz     = _read_float(pub_row, "logZ")     if pub_row is not None else None
                b_logz_err = _read_float(pub_row, "logZ_err") if pub_row is not None else None

                raw_chi2     = _read_float(row, "chi2")
                raw_chi2_dof = _read_float(row, "chi2_dof")
                raw_aic      = _read_float(row, "AIC")
                raw_bic      = _read_float(row, "BIC")
                raw_logz     = _read_float(row, "logZ")
                raw_logz_err = _read_float(row, "logZ_err")

                chi2_str     = f"{raw_chi2:.2f}"     if raw_chi2     is not None else "---"
                chi2_dof_str = f"{raw_chi2_dof:.3f}" if raw_chi2_dof is not None else "---"
                aic_str      = _delta_cell(raw_aic,  b_aic,  _AIC_COLORS)
                bic_str      = _delta_cell(raw_bic,  b_bic,  _BIC_COLORS)
                logz_str     = _delta_logz_cell(raw_logz, raw_logz_err,
                                                b_logz, b_logz_err, _LOGZ_COLORS)

                all_data_rows.append(
                    f"{_variant_label(row)} & {n_params}"
                    f" & {chi2_str} & {chi2_dof_str}"
                    f" & {aic_str} & {bic_str} & {logz_str}"
                )

        else:
            bmask = section_df["run_name"].astype(str) == baseline_name
            if not bmask.any():
                import warnings
                warnings.warn(
                    f"No baseline '{baseline_name}' in section "
                    f"'{section_label}'; skipping section."
                )
                continue
            b_row      = section_df[bmask].iloc[0]
            b_aic      = _read_float(b_row, "AIC")
            b_bic      = _read_float(b_row, "BIC")
            b_logz     = _read_float(b_row, "logZ")
            b_logz_err = _read_float(b_row, "logZ_err")

            section_rows = _build_evidence_rows(
                section_df,
                b_aic, b_bic, b_logz, b_logz_err,
                baseline_run_name = baseline_name,
                first_col_fn      = _variant_label,
            )
            all_data_rows.extend(section_rows)

    return _lt_wrap(
        col_spec        = col_spec,
        caption         = "Robustness Checks",
        label           = "tab:checks",
        col_header_line = col_header,
        data_rows       = all_data_rows,
        ncols           = 7,
        notes           = [
            r"Each section is compared to its own baseline run.",
            r"$w$CDM and non-flat $\Lambda$CDM rows use the matched publication run (same nuisance variant).",
            r"$\Delta\ln B$ errors are $\sqrt{\sigma_\mathrm{run}^2 + \sigma_\mathrm{base}^2}$.",
        ],
    )


# ===========================================================================
# TABLE 5 — Drilling cones  (longtable, full width)
# ===========================================================================
# Reads the summary CSV written by drilling_cones.run_drilling_cones (see
# drilling_cones.py / drilling_cones_checks.py) — one row per sky cluster,
# each already compared against the full-sample FlatLambdaCDM baseline via
# compare_runs.compare_two_runs. This table does NOT read a run registry;
# it reads that CSV directly, so it is independent of the checks/
# publication tables above (a deliberately separate systematic).

_TENSION_COLORS = [
    # neutral: [0, 1)
    (1.0, 2.0, "aiclightyellow"),
    (2.0, 3.0, "aicorange"),
    (3.0, math.inf, "aicred"),
]


def _tension_cell(nsigma):
    if nsigma is None or (isinstance(nsigma, float) and math.isnan(nsigma)):
        return "---"
    fmt = f"{nsigma:.2f}$\\sigma$"
    for lo, hi, color in _TENSION_COLORS:
        if lo <= nsigma < hi:
            return rf"\cellcolor{{{color}}}{fmt}"
    return fmt


def generate_uniform_checks_table(registry_path=None):
    r"""
    TABLE 6 — Additional (broad-uniform-prior) checks.

    Columns: Run | N | chi2 | chi2/dof | Delta AIC | Delta BIC | Delta ln B,
    i.e. the same columns as the Evidence and Checks tables, read from
    uniform_priors_check.py's own registry
    (run_publication_registry_uniform.csv).

    Every delta is taken against "uniformpriors/baseline" -- the baseline
    model fitted under the SAME broad uniform priors as every other row in
    that registry -- and never against run_publication_registry.csv's
    "baseline". That is not a stylistic choice: a uniform prior spans far
    more prior volume than the informative one it replaces, so its Occam
    factor alone lowers ln Z by an amount that has nothing to do with how
    well the model fits. Differencing across the two registries would
    report that penalty as evidence against the model.

    Within this table the comparison is sound, because every row pays the
    same penalty: a positive Delta ln B here means the extra freedom earns
    its keep even under priors chosen to be maximally unhelpful to it.

    Parameters
    ----------
    registry_path : defaults to "run_publication_registry_uniform.csv".
    """
    if registry_path is None:
        registry_path = "run_publication_registry_uniform.csv"
    try:
        df = pd.read_csv(registry_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Uniform-prior registry not found: {registry_path}. "
            f"Run uniform_priors_check.py first.")

    base_name = "uniformpriors/baseline"
    base_mask = df["run_name"].astype(str) == base_name
    if not base_mask.any():
        raise RuntimeError(
            f"No '{base_name}' row in {registry_path}. That run is the "
            f"matched-prior reference every delta in this table is measured "
            f"against, so the table cannot be built without it -- run "
            f"`python uniform_priors_check.py --only baseline` first.")

    b             = df[base_mask].iloc[0]
    base_aic      = _read_float(b, "AIC")
    base_bic      = _read_float(b, "BIC")
    base_logz     = _read_float(b, "logZ")
    base_logz_err = _read_float(b, "logZ_err")

    NCOLS = 7
    col_header = (
        r"Run & $N$ & $\chi^2$ & $\chi^2/\mathrm{d.o.f}$"
        r" & $\Delta$AIC & $\Delta$BIC & $\Delta\ln B$"
    )
    col_spec = r"c c c c c c c"

    # Reference row first, then everything else in registry order, so the
    # row the deltas are relative to is at the top rather than buried
    # wherever it happened to be run.
    ordered = pd.concat([df[base_mask], df[~base_mask]])

    def _first_col(row):
        name = str(row.get("run_name", ""))
        if name == base_name:
            return "ref"
        return _esc(name.split("/", 1)[-1].replace("_", " "))

    data_rows = _build_evidence_rows(
        ordered, base_aic, base_bic, base_logz, base_logz_err,
        baseline_run_name=base_name,
        first_col_fn=_first_col,
    )

    return _lt_wrap(
        col_spec        = col_spec,
        caption         = "Additional checks: broad uniform priors",
        label           = "tab:uniform_checks",
        col_header_line = col_header,
        data_rows       = data_rows,
        ncols           = NCOLS,
        notes           = (
            r"All runs use uniform priors on $\alpha$, $\beta$ and "
            r"$\Omega_{\rm M0}^{}$ (and on any active shape parameter); "
            r"see Table~\ref{tab:priors_uniform}.  Deltas are relative to "
            r"the reference row, marked \textit{ref}, which is the "
            r"baseline model fitted under those same priors.  They are "
            r"\textit{not} comparable with "
            r"Table~\ref{tab:bic}: a wider prior lowers $\ln Z$ through "
            r"its Occam factor irrespective of fit quality."
        ),
    )


def generate_host_error_table(registry_path=None, label="best"):
    r"""
    TABLE 8 — host measurement error treatment (matched-pair systematic).

    Reads the "hosterr/" rows written by extra_runners.py and differences
    them against "hosterr/<label>_ref", the same model under the default
    error treatment.  Every row fits the same model on the same SNe and
    changes only how the host mass / colour / sSFR measurement errors are
    handled, so the deltas isolate that choice.

    Two distinct effects are being separated here:

      * the quadrature already in the likelihood corrects the BIAS from
        evaluating a nonlinear host profile at a noisy host property
        (it computes E[f]).  For a *linear* profile this changes nothing,
        so linear-model rows will match the reference exactly;

      * "var" rows additionally add Var[f] to the covariance diagonal,
        i.e. the extra scatter that the same measurement error injects
        into mu.  This is always a net penalty on ln Z for this sample,
        because chi2/dof sits slightly below 1 and the log-determinant
        cost outweighs the chi2 gain.  It therefore cannot flatter a fit.

    The "no colour err" rows exist because HOST_COLOR_ERR is unpopulated
    in the DES metadata: without a derived error the host colour would be
    the only host property treated as exactly measured, which would hand
    the host colour models an unearned advantage.

    Parameters
    ----------
    registry_path : defaults to "run_checks_registry.csv".
    label         : the HOSTERR_BEST label used in the run tags.
    """
    if registry_path is None:
        registry_path = "run_checks_registry.csv"
    try:
        df = pd.read_csv(registry_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checks registry not found: {registry_path}. "
            f"Run extra_runners.py --tag hosterr/ first.")

    df = df[df["run_name"].astype(str).str.contains("hosterr/", na=False)]
    if df.empty:
        raise RuntimeError(
            f"No 'hosterr/' rows in {registry_path}. Run "
            f"`python extra_runners.py --tag hosterr/ --publication` first.")

    base_name = f"hosterr/{label}_ref"
    base_mask = df["run_name"].astype(str) == base_name
    if not base_mask.any():
        raise RuntimeError(
            f"No '{base_name}' row in {registry_path}. That run is the "
            f"matched reference every delta in this table is measured "
            f"against, so the table cannot be built without it.")

    b             = df[base_mask].iloc[0]
    base_aic      = _read_float(b, "AIC")
    base_bic      = _read_float(b, "BIC")
    base_logz     = _read_float(b, "logZ")
    base_logz_err = _read_float(b, "logZ_err")

    # Human-readable descriptions, keyed by the tag suffix used in
    # extra_runners._HOSTERR_VARIANTS.
    _DESC = {
        "ref":                 "ref",
        "varpen":              r"$+\,\mathrm{Var}[f]$",
        "nocolourerr":         "no colour err",
        "nocolourerr_varpen":  r"no colour err $+\,\mathrm{Var}[f]$",
        "slope050":            r"colour err slope $0.50$",
        "slope115":            r"colour err slope $1.15$",
        "ssfrmask20":          r"sSFR mask $2.0$ dex",
        "ssfrmask30":          r"sSFR mask $3.0$ dex",
        "nossfrmask":          "no sSFR mask",
        "noerrors":            "no host errors at all",
        "gh80":                r"$K=80$ nodes",
        "gh80_varpen":         r"$K=80$, $+\,\mathrm{Var}[f]$",
    }

    NCOLS = 7
    col_header = (
        r"Treatment & $N$ & $\chi^2$ & $\chi^2/\mathrm{d.o.f}$"
        r" & $\Delta$AIC & $\Delta$BIC & $\Delta\ln B$"
    )
    col_spec = r"l c c c c c c"

    ordered = pd.concat([df[base_mask], df[~base_mask]])

    def _first_col(row):
        name   = str(row.get("run_name", ""))
        suffix = name.split("/", 1)[-1]
        if suffix.startswith(label + "_"):
            suffix = suffix[len(label) + 1:]
        return _DESC.get(suffix, _esc(suffix.replace("_", " ")))

    data_rows = _build_evidence_rows(
        ordered, base_aic, base_bic, base_logz, base_logz_err,
        baseline_run_name=base_name,
        first_col_fn=_first_col,
    )

    return _lt_wrap(
        col_spec        = col_spec,
        caption         = "Systematic check: host measurement error treatment",
        label           = "tab:host_error",
        col_header_line = col_header,
        data_rows       = data_rows,
        ncols           = NCOLS,
        notes           = (
            r"Every row fits the same model on the same supernovae and "
            r"varies only the treatment of the host mass, host colour and "
            r"sSFR measurement errors; deltas are against the reference "
            r"row, marked \textit{ref}.  The default treatment marginalises "
            r"each host property over its measurement error by "
            r"Gauss--Hermite quadrature, derives the host colour error as "
            r"$\sigma_{\log M}/0.70$ (Taylor et al. 2011) because "
            r"\texttt{HOST\_COLOR\_ERR} is unpopulated, and masks sSFR "
            r"measurements quoted with $\sigma>2.5$ dex.  Masked "
            r"supernovae are retained in the sample so that all evidences "
            r"remain comparable.  Rows marked $+\,\mathrm{Var}[f]$ "
            r"additionally propagate the measurement error as a variance "
            r"on the covariance diagonal, which is always a net penalty "
            r"here and so cannot flatter a fit."
        ),
    )


def generate_uniform_priors_table():
    r"""
    TABLE 7 — the broad uniform priors used by the additional checks.

    Same layout as the ordinary priors table, but built from
    uniform_priors_check.UNIFORM_PRIORS layered on top of
    DEFAULT_PARAM_SPECS, so the paper can state exactly what "broad
    uniform priors" meant rather than leaving the reader to infer it.

    Only the parameters that actually change are listed -- a table
    repeating the ~30 unchanged rows of Table~\ref{tab:priors} would bury
    the four or five that matter.
    """
    # Imported lazily: uniform_priors_check imports run.py, which pulls in
    # dynesty and the data files. latex_tables.py is otherwise a pure
    # config+CSV reader that must stay runnable without them.
    from uniform_priors_check import UNIFORM_PRIORS

    specs = {}
    for name, updates in UNIFORM_PRIORS.items():
        if name not in DEFAULT_PARAM_SPECS:
            continue
        merged = dict(DEFAULT_PARAM_SPECS[name])
        merged.update(updates)
        specs[name] = merged

    lines = [
        r"\begin{table*}",
        r"    \caption{Broad uniform priors used for the additional "
        r"prior-sensitivity checks (Table~\ref{tab:uniform_checks}).  "
        r"Every other parameter keeps the prior given in "
        r"Table~\ref{tab:priors}.}",
        r"    \label{tab:priors_uniform}",
        r"    \centering",
        r"    \begin{tabular}{cccc}",
        r"        \hline",
        r"        \hline",
        r"        Parameters & Uniform Range & Default Prior "
        r"& Default Distribution Parameters \\",
        r"        \hline",
        r"        \hline",
    ]
    for name in _PRIOR_ROW_ORDER:
        if name not in specs:
            continue
        default = DEFAULT_PARAM_SPECS[name]
        lo, hi  = specs[name]["range"]
        lines.append(
            f"        {_label(name)} & {_fmt_range(lo, hi)}"
            f" & {_prior_type_latex(default['prior'])}"
            f" & {_dist_params_latex(name, default)} \\\\"
        )
    # Anything in UNIFORM_PRIORS that _PRIOR_ROW_ORDER doesn't mention.
    for name in specs:
        if name in _PRIOR_ROW_ORDER:
            continue
        default = DEFAULT_PARAM_SPECS[name]
        lo, hi  = specs[name]["range"]
        lines.append(
            f"        {_label(name)} & {_fmt_range(lo, hi)}"
            f" & {_prior_type_latex(default['prior'])}"
            f" & {_dist_params_latex(name, default)} \\\\"
        )
    lines += [r"        \hline", r"    \end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def generate_drilling_cones_table(csv_path):
    """
    Columns: Cone | N_SNe | RA | Dec | Gaussian tension (vs. full-sample
    baseline) | ln B (cone vs. baseline).

    Parameters
    ----------
    csv_path : path to a "<output_prefix>_drilling_cones.csv" written by
        drilling_cones.run_drilling_cones / drilling_cones_checks.py.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Drilling-cones CSV not found: {csv_path}")

    if "skipped_too_few" in df.columns:
        df = df.sort_values("cluster_id")

    NCOLS = 6
    col_header = (
        r"Cone & $N_{\rm SNe}$ & R.A.\ (deg) & Dec.\ (deg)"
        r" & Tension vs.\ baseline & $\ln B$"
    )
    col_spec = r"c c c c c c"

    data_rows = []
    n_flagged = 0
    for _, row in df.iterrows():
        skipped = bool(row.get("skipped_too_few", False))
        cid     = row.get("cluster_id", "")
        n_sne   = int(row.get("n_sne", 0))
        ra      = _read_float(row, "ra_centre")
        dec     = _read_float(row, "dec_centre")
        ra_str  = f"{ra:.1f}" if ra is not None else "---"
        dec_str = f"{dec:.1f}" if dec is not None else "---"

        if skipped:
            data_rows.append(
                f"{cid} & {n_sne} & {ra_str} & {dec_str} & (too few SNe) & ---"
            )
            continue

        nsigma = _read_float(row, "gaussian_nsigma")
        lnB    = _read_float(row, "lnB")
        if nsigma is not None and nsigma >= 2.0:
            n_flagged += 1
        tension_str = _tension_cell(nsigma)
        lnB_str     = f"{lnB:+.2f}" if lnB is not None else "---"

        data_rows.append(
            f"{cid} & {n_sne} & {ra_str} & {dec_str} & {tension_str} & {lnB_str}"
        )

    if "skipped_too_few" in df.columns:
        n_fitted = int((~df["skipped_too_few"].astype(bool)).sum())
    else:
        n_fitted = len(df)

    notes = [
        r"Each cone is a DBSCAN-clustered sky pointing, refit independently and "
        r"compared against the full-sample FlatLambdaCDM baseline (broad uniform "
        r"prior on $\Omega_{\rm M0}$) via a weighted-posterior Gaussian tension.",
        rf"Cones at or above $2\sigma$ tension are highlighted; "
        rf"{n_flagged}/{n_fitted} fitted cone(s) meet that threshold.",
    ]

    return _lt_wrap(
        col_spec        = col_spec,
        caption         = "Line-of-Sight / Sky-Position (Drilling Cones) Systematic Check",
        label           = "tab:drilling_cones",
        col_header_line = col_header,
        data_rows       = data_rows,
        ncols           = NCOLS,
        notes           = notes,
    )


# ===========================================================================
# Preamble snippet
# ===========================================================================

PREAMBLE_SNIPPET = r"""% ---- paste into your LaTeX preamble (before \begin{document}) ----
\usepackage{longtable}
\usepackage[table]{xcolor}
% Make longtable span the full text width
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
% Colour definitions
\definecolor{aiclightyellow}{RGB}{255, 255, 180}
\definecolor{aicyellow}{RGB}{255, 230,   0}
\definecolor{aiclightgreen}{RGB}{180, 230, 180}
\definecolor{aicgreen}{RGB}{ 80, 175,  80}
\definecolor{aiclightorange}{RGB}{255, 210, 150}
\definecolor{aicorange}{RGB}{255, 160,   0}
\definecolor{aiclightred}{RGB}{255, 180, 180}
\definecolor{aicred}{RGB}{210,  80,  80}
% -----------------------------------------

% ---- table* / longtable ordering fix ----
% table* (e.g. "Mass Options") is a genuine LaTeX float: LaTeX may delay
% placing it until it finds room, which can be AFTER a later longtable —
% producing the "Table 6 appears before Table 1" symptom.  longtable is
% NOT a float, so it always typesets exactly where it sits in the source.
%
% The generated tables already emit \clearpage before every longtable to
% force pending floats (like table*) to flush first, restoring source
% order.  If \clearpage's forced page break is too aggressive (leaves
% ugly whitespace on small tables), swap it for the gentler alternative
% below: \FloatBarrier flushes pending floats WITHOUT forcing a new page.
%
%   \usepackage{placeins}
%   % then replace each \clearpage this script emits with \FloatBarrier
%   % (search/replace in the generated .tex, or post-process the string
%   % returned by _lt_wrap before writing it to disk)
% ------------------------------------------"""


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Auto-generate LaTeX tables from config.py / registry CSVs"
    )
    p.add_argument("--priors",     action="store_true")
    p.add_argument("--iterations", action="store_true")
    p.add_argument("--evidence",   action="store_true")
    p.add_argument("--checks",     action="store_true")
    p.add_argument("--additional-checks", action="store_true",
                   dest="additional_checks",
                   help="Generate the broad-uniform-prior additional-checks "
                        "table from uniform_priors_check.py's registry "
                        "(see --uniform-registry).")
    p.add_argument("--uniform-priors", action="store_true",
                   dest="uniform_priors",
                   help="Generate the prior table for those uniform checks.")
    p.add_argument("--host-error", action="store_true",
                   dest="host_error",
                   help="Generate the host measurement error systematic-check "
                        "table from extra_runners.py's 'hosterr/' runs "
                        "(see --checks-registry and --host-error-label).")
    p.add_argument("--host-error-label", default="best",
                   dest="host_error_label",
                   help="HOSTERR_BEST label used in the hosterr/ run tags "
                        "(default: best).")
    p.add_argument("--drilling-cones", action="store_true",
                   help="Generate the drilling-cones systematic-check table "
                        "from a CSV written by drilling_cones.py / "
                        "drilling_cones_checks.py (see --drilling-cones-csv).")
    p.add_argument("--preamble",   action="store_true")
    p.add_argument("--registry",   default=None,
                   help="Publication registry CSV (default: CONFIG['registry_file'])")
    p.add_argument("--checks-registry", default="run_checks_registry.csv",
                   dest="checks_registry")
    p.add_argument("--uniform-registry",
                   default="run_publication_registry_uniform.csv",
                   dest="uniform_registry",
                   help="Registry written by uniform_priors_check.py "
                        "(default: run_publication_registry_uniform.csv)")
    p.add_argument("--drilling-cones-csv", default=None,
                   dest="drilling_cones_csv",
                   help="Path to a '<output_prefix>_drilling_cones.csv' "
                        "(required with --drilling-cones).")
    p.add_argument("--out", default=None)
    return p.parse_args()

def main():
    args = _parse_args()
    if not any([args.priors, args.iterations, args.evidence,
                args.checks, args.additional_checks, args.uniform_priors,
                args.host_error, args.drilling_cones, args.preamble]):
        print("Specify: --priors  --iterations  --evidence  --checks  "
              "--additional-checks  --uniform-priors  --host-error  "
              "--drilling-cones  --preamble")
        sys.exit(1)
    if args.drilling_cones and not args.drilling_cones_csv:
        print("--drilling-cones requires --drilling-cones-csv <path>")
        sys.exit(1)

    parts = []
    if args.preamble:
        parts.append(PREAMBLE_SNIPPET)
    if args.priors:
        parts.append(generate_priors_table())
    if args.iterations:
        parts.append(generate_iterations_table(registry_path=args.registry))
    if args.evidence:
        parts.append(generate_evidence_table(registry_path=args.registry))
    if args.checks:
        parts.append(generate_checks_table(
            checks_registry_path=args.checks_registry,
            pub_registry_path=args.registry,
        ))
    if args.uniform_priors:
        parts.append(generate_uniform_priors_table())
    if args.additional_checks:
        parts.append(generate_uniform_checks_table(
            registry_path=args.uniform_registry))
    if args.host_error:
        parts.append(generate_host_error_table(
            registry_path=args.checks_registry,
            label=args.host_error_label))
    if args.drilling_cones:
        parts.append(generate_drilling_cones_table(args.drilling_cones_csv))

    output = "\n\n".join(parts)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
        print(f"Written to {args.out}")
    else:
        print(output)

if __name__ == "__main__":
    main()