#!/usr/bin/env bash
#
# setup_env.sh  —  DES_Param_Analysis
# ===================================
# Create (or recreate) the conda environment this pipeline runs in, then
# verify it. See environment.yml / requirements.txt for what gets
# installed and why each package sits on the conda or the pip side.
#
#   ./setup_env.sh              create the environment, skip if it exists
#   ./setup_env.sh --recreate   delete and rebuild it from scratch
#   ./setup_env.sh --name foo   use an environment name other than the
#                               one in environment.yml
#
# Afterwards:
#
#   conda activate des-param-analysis
#   python experiment_runner.py --list
#
# The verification step at the end is the point of this script existing
# rather than just documenting `conda env create`. Two failure modes here
# are silent rather than loud:
#
#   1. pip can satisfy a transitive version floor by upgrading a package
#      conda installed, leaving an environment that imports fine but
#      whose pandas/scikit-learn were compiled against a different numpy
#      ABI. (contourpy pulling numpy 1.20 -> 1.26 does exactly this if
#      left unpinned; see requirements.txt.)
#   2. A missing optional-looking package such as threadpoolctl is caught
#      by a bare `except Exception: pass` in the runners, so an
#      oversubscribed 80-worker job looks like it is merely slow.
#
# So this script asserts the resolved versions match the pins and imports
# every module the pipeline actually uses, and exits non-zero otherwise.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${REPO_DIR}/environment.yml"
ENV_NAME=""
RECREATE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recreate) RECREATE=1; shift ;;
        --name)     ENV_NAME="${2:-}"; shift 2 ;;
        -h|--help)  sed -n '2,36p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi

# Environment name comes from environment.yml unless --name overrides it,
# so the two can't disagree about what to activate.
if [[ -z "${ENV_NAME}" ]]; then
    ENV_NAME="$(awk '/^name:/ {print $2; exit}' "${ENV_FILE}")"
fi
if [[ -z "${ENV_NAME}" ]]; then
    echo "ERROR: no 'name:' in ${ENV_FILE} and --name not given." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    echo "Install Miniforge or Miniconda first: https://conda-forge.org/download/" >&2
    exit 1
fi

# `conda activate` is a shell function, not an executable, so it is not
# available in a non-interactive script until conda.sh is sourced.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

if env_exists; then
    if [[ "${RECREATE}" -eq 1 ]]; then
        echo "==> Removing existing environment '${ENV_NAME}'"
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "==> Environment '${ENV_NAME}' already exists; skipping creation."
        echo "    (re-run with --recreate to rebuild it from scratch)"
    fi
fi

if ! env_exists; then
    echo "==> Creating environment '${ENV_NAME}' from ${ENV_FILE}"
    echo "    This resolves conda and then pip; expect a few minutes."
    conda env create -f "${ENV_FILE}" -n "${ENV_NAME}"
fi

echo "==> Verifying environment '${ENV_NAME}'"
conda activate "${ENV_NAME}"

# Run the check from the repo directory so `import config` (and the data
# files it resolves) behave the same way they will during a real run.
cd "${REPO_DIR}"
python - <<'PYEOF'
import sys

# (module to import, distribution name, expected version)
# Expected versions are the pins in environment.yml / requirements.txt.
# Keep this list in step with them -- a mismatch here means the resolver
# gave you something other than what those files asked for, which is
# exactly the situation this check exists to surface.
EXPECTED = [
    ("numpy",         "numpy",        "1.20.3"),
    ("scipy",         "scipy",        "1.10.1"),
    ("pandas",        "pandas",       "1.3.4"),
    ("matplotlib",    "matplotlib",   "3.7.5"),
    ("astropy",       "astropy",      "5.2.2"),
    ("dynesty",       "dynesty",      "2.1.4"),
    ("sklearn",       "scikit-learn", "0.24.2"),
    ("threadpoolctl", "threadpoolctl", "2.2.0"),
]

EXPECTED_PYTHON = "3.9.7"

failures = []

actual_python = ".".join(str(v) for v in sys.version_info[:3])
if actual_python != EXPECTED_PYTHON:
    failures.append(f"python: expected {EXPECTED_PYTHON}, got {actual_python}")
print(f"  python          {actual_python}")

for module_name, dist_name, expected in EXPECTED:
    try:
        module = __import__(module_name)
    except Exception as exc:
        failures.append(f"{dist_name}: import failed ({exc})")
        print(f"  {dist_name:<15} MISSING")
        continue
    actual = getattr(module, "__version__", "unknown")
    flag = "" if actual == expected else f"  <-- expected {expected}"
    if flag:
        failures.append(f"{dist_name}: expected {expected}, got {actual}")
    print(f"  {dist_name:<15} {actual}{flag}")

# Import the pipeline itself, not just its dependencies: config.py
# resolves the data files at import time, so this also confirms
# DES-Dovekie_Metadata.csv and STAT+SYS.npz were found.
try:
    import config
    import core   # noqa: F401  (imports astropy cosmology + scipy linalg)
    print(f"\n  data_file  {config.CONFIG['data_file']}")
    print(f"  cov_file   {config.CONFIG['cov_file']}")
    import os
    for key in ("data_file", "cov_file"):
        path = config.CONFIG[key]
        if not os.path.isfile(path):
            failures.append(f"{key} does not exist: {path}")
except Exception as exc:
    failures.append(f"importing the pipeline failed: {exc}")

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("\nEnvironment OK.")
PYEOF

echo
echo "==> Done. Activate it with:"
echo
echo "      conda activate ${ENV_NAME}"
echo
