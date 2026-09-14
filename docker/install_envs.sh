#!/usr/bin/env bash
set -euo pipefail
cd /opt/venvs

export PYENV_ROOT="/opt/pyenv"
export PATH="/usr/local/bin:/usr/bin:/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin"
eval "$(pyenv init -)"

req_dir="/tmp/requirements"
# Put kernelspecs somewhere writable/persistent in the image
KERNEL_PREFIX="/opt/venvs/_kernels"
mkdir -p "$KERNEL_PREFIX"

resolve_pyenv_version() {
  local req="$1"
  # 1. Exact match
  if pyenv versions --bare | grep -qx "$req"; then
    printf '%s\n' "$req"
    return 0
  fi
  # 2. Find latest patch for major.minor (e.g., 3.9 -> 3.9.25)
  local req_esc
  req_esc="$(printf '%s' "$req" | sed 's/[.[\*^$()+?{|]/\\&/g')"
  local found
  found="$(pyenv versions --bare | grep -E "^${req_esc}\.[0-9]+$" | sort -V | tail -n 1)"
  if [[ -n "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi
  # 3. Not found
  return 1
}

escape_ere() {
  # escape for ERE (sed -E)
  printf '%s' "$1" | sed 's/[][\/.^$*+?|(){}]/\\&/g'
}

# Normalize a parsed "failed package" token from pip output
# - strips surrounding quotes/backticks
# - strips trailing punctuation like "." ":" "," ";" (but keeps "-" "_" etc)
normalize_pkg_name() {
  local s="${1:-}"
  s="$(printf '%s' "$s" | sed -E \
    -e "s/^[[:space:]]+//; s/[[:space:]]+\$//" \
    -e "s/^[\"'\\\`]+//; s/[\"'\\\`]+\$//" \
    -e "s/[,:;]+$//; s/\\.+$//" \
  )"
  printf '%s' "$s"
}

# Fetch candidate versions from PyPI for pkg, optionally filtering by the original requirement line
pypi_versions_for_reqline() {
  local pkg="$1"
  local reqline="$2"
  python3 - <<'PY' "$pkg" "$reqline" 2>/dev/null || true
import json, sys, urllib.request, re, subprocess, ssl

# Fix SSL context for legacy/proxy envs
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

pkg = sys.argv[1]
reqline = (sys.argv[2] or "").strip()

# Ensure packaging exists
Version = None
InvalidVersion = Exception
SpecifierSet = None
try:
    from packaging.version import Version, InvalidVersion
    from packaging.specifiers import SpecifierSet
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q","packaging"])
        from packaging.version import Version, InvalidVersion
        from packaging.specifiers import SpecifierSet
    except Exception:
        Version = None
        SpecifierSet = None

url = "https://pypi.org/pypi/{}/json".format(pkg)
data = json.loads(urllib.request.urlopen(url, timeout=20).read().decode("utf-8"))
raw_versions = list((data.get("releases") or {}).keys())

# Extract spec part from the original requirement line (e.g., "<=3.3", "==1.2.0")
spec = ""
m = re.match(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)\s*$", reqline)
if m:
    spec = (m.group(2) or "").strip()
else:
    # allow passing just the spec like "<=3.5"
    if reqline.startswith(("<", ">", "=", "!", "~")):
        spec = reqline.strip()

if Version:
    # Drop invalid/non-PEP440 versions (e.g., "2.0.1rc2-git")
    parsed = []
    for v in raw_versions:
        try:
            pv = Version(v)
        except InvalidVersion:
            continue
        parsed.append((pv, v))

    # Newest -> oldest
    parsed.sort(key=lambda t: t[0], reverse=False)

    # Apply spec filter if present
    if spec:
        ss = SpecifierSet(spec)
        parsed = [v for (pv, v) in parsed if ss.contains(pv, prereleases=True)]

    for v in parsed:
        print(v)
else:
    # Fallback: no packaging; emit raw versions (unsorted) so caller can still attempt something
    for v in raw_versions:
        print(v)
PY
}

IFS=',' read -ra tasks <<< "$tasks"
total=${#tasks[@]}
count=0

for line in "${tasks[@]}"; do
    count=$((count + 1))
    # if [[ $count -lt 165 ]]; then
    #     continue
    # fi
    env_name=$(echo "$line" | awk '{print $1}')
    req_name=$(echo "$line" | awk '{print $2}')
    py_ver=$(echo "$line" | awk '{print $3}')
    req_file="$req_dir/$req_name.txt"

    echo "[$count/$total] Building $env_name (Python $py_ver)"

    cached_version="$(resolve_pyenv_version "$py_ver" || true)"
    if [[ -z "$cached_version" ]]; then
        echo "--> Installing Python $py_ver via pyenv..."
        pyenv install -s "$py_ver" >/dev/null 2>&1 || true
        py_ver=$(resolve_pyenv_version "$py_ver")
        echo ">>> Using Python version: $py_ver"
    else
        py_ver="$cached_version"
    fi

    pyenv shell "$py_ver"  
    export PYENV_VERSION="$py_ver"

    if [[ "$py_ver" == 2.7* ]]; then
        python -m pip install --upgrade 'virtualenv<20.16' >/dev/null 2>&1 || true
    else
        python -m pip install --upgrade virtualenv >/dev/null 2>&1 || true
    fi

    python -m virtualenv "$env_name" >/dev/null 2>&1
    VENV_PY="/opt/venvs/${env_name}/bin/python"
    # set +u
    # . "$env_name/bin/activate"
    # set -u

    # For very EOL Pythons
    if [[ "$py_ver" == 2.7* ]]; then
        "$VENV_PY" -m pip install -U 'pip<21' 'setuptools<45' 'wheel<0.38' 'build<0.6' 'ipykernel<=4.10.1' 'nbformat<=4.4.0' 'ipython<=5.10.0' 'beautifulsoup4<=4.9.3' 'jupyter_client<6' 'jupyter_core<4.7' 'traitlets<5' 'pyzmq<=19.0.2' 'tornado<6' 'h5py<=2.10.0' >/dev/null 2>&1 || true
    elif [[ "$py_ver" == 3.5* ]]; then
        "$VENV_PY" -m pip install -U 'pip<21' 'setuptools<51' 'wheel<0.38' 'build<0.6' 'ipykernel<6' 'nbformat<=5.1.3' 'ipython>=7.0,<8.0' 'beautifulsoup4<4.11.0' 'h5py<=2.10.0' >/dev/null 2>&1 || true
    elif [[ "$py_ver" == 3.6* ]]; then
        "$VENV_PY" -m pip install -U 'pip<23' 'setuptools<59.7' 'wheel<0.38' 'build<1.0' 'ipykernel<5' 'nbformat<=5.1.3' 'ipython>=7.0,<8.0' 'beautifulsoup4<4.13.1' 'h5py<=3.1.0' >/dev/null 2>&1 || true
    elif [[ "$py_ver" == 3.7* ]]; then
        "$VENV_PY" -m pip install -U 'pip<24.1' 'setuptools<68.1' 'wheel<0.43' 'build<1.2' 'ipykernel<6.17' 'nbformat<=5.8.0' 'ipython>=7.0,<8.0' 'beautifulsoup4<=4.14.3' 'h5py<=3.8.0' >/dev/null 2>&1 || true
    elif [[ "$py_ver" == 3.8* ]]; then
        "$VENV_PY" -m pip install -U 'pip<25.1' 'setuptools<75.4' 'wheel<0.46' 'build<1.3' 'ipykernel<6.30' 'nbformat<=5.10.4' 'ipython>=8.0,<8.18.0' 'beautifulsoup4<=4.10' 'h5py<=3.11.0' >/dev/null 2>&1 || true
    else
        "$VENV_PY" -m pip install -U pip setuptools wheel build ipykernel nbformat typing_extensions ipython 'beautifulsoup4<=4.14.3' 'h5py<=3.15.1' >/dev/null 2>&1 || true
    fi

    "$VENV_PY" -m ipykernel install \
        --prefix="/opt/venvs/_kernels" \
        --name "$env_name" \
        --display-name "Python ($env_name)" >/dev/null 2>&1 || true

    echo ">>> Finishing setup of virtualenv $env_name"
    if [[ "$req_name" == "_skip_" ]]; then
        # set +u
        # deactivate
        # set -u

        unset PYENV_VERSION
        pyenv shell system
        continue
    fi

    # Determine pip install flags based on Python version
    pip_flags="--upgrade-strategy eager --no-cache-dir"
    # Check if pip supports --prefer-binary
    if pip install --help 2>&1 | grep -q -- '--prefer-binary'; then
        # For very old/EOL Pythons, avoid source builds (they cause long discard loops)
        if [[  "$py_ver" == 2.7* || "$py_ver" == 3.5* || "$py_ver" == 3.6* ]]; then
            pip_flags="$pip_flags --only-binary=:all:"
        else
            pip_flags="$pip_flags --prefer-binary"
        fi
    fi

    # If pip supports legacy resolver, use it on EOL Pythons to avoid long backtracking
    if "$VENV_PY" -m pip install --help 2>&1 | grep -q -- '--use-deprecated'; then
        if [[ "$py_ver" == 2.7* || "$py_ver" == 3.5* || "$py_ver" == 3.6* ]]; then
            pip_flags="$pip_flags --use-deprecated=legacy-resolver"
        fi
    fi

    constraints="/tmp/constraints.${env_name}.txt"
    : > "$constraints"

    # Copy requirements
    # Prefer a fully pinned requirements file mounted by docker to save time.
    pinned_req="/kaggle/working/pinned_requirements.txt"
    pinned_mode=0
    if [[ "${USE_PINNED_REQ:-0}" == "1" && -s "$pinned_req" ]]; then
        echo ">>> Using pre-pinned requirements from: $pinned_req"
        cp "$pinned_req" /tmp/requirements.work
        pinned_mode=1

        # Remove already-installed bootstrap packages *only when the pinned requirement is the same exact version*.
        # (This avoids reinstalling pip/setuptools/wheel/build/ipykernel/nbformat/typing_extensions/ipython/bs4/h5py, etc.)
        python3 - <<'PY'
import re, sys, subprocess

def canon(name: str) -> str:
    # simple PEP503-ish normalization
    return re.sub(r"[-_.]+", "-", name.strip().lower())

# Get currently installed exact versions in this venv
freeze_b = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=freeze"])
freeze = freeze_b.decode("utf-8", "replace")

print(f">>> requirements.work: found {len(freeze.splitlines())} installed packages", flush=True)

installed = {}
for line in freeze.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    m = re.match(r"^([A-Za-z0-9_.-]+)==(.+)$", line)
    if m:
        installed[canon(m.group(1))] = m.group(2)

print(f">>> requirements.work: found {len(installed)} installed packages", flush=True)

print(f"{installed}", flush=True)

req_path = "/tmp/requirements.work"
out_lines = []
removed = 0

with open(req_path, "r", encoding="utf-8", errors="replace") as f:
    for raw in f:
        line = raw.rstrip("\n")
        s = line.strip()

        # keep blanks/comments/option lines/URLs/editables untouched
        if (not s) or s.startswith("#") or s.startswith("-") or "://" in s or s.startswith(("-e ", "--editable")):
            out_lines.append(line)
            continue

        m = re.match(r"^([A-Za-z0-9_.-]+)\s*==\s*([^#;]+)", s)
        if not m:
            out_lines.append(line)
            continue

        name = canon(m.group(1))
        ver = m.group(2).strip()

        if installed.get(name) == ver:
            removed += 1
            continue

        out_lines.append(line)

with open(req_path, "w", encoding="utf-8") as f:
    for l in out_lines:
        f.write(l + "\n")

print(f">>> requirements.work: removed {removed} already-installed exact pins, {len(out_lines)} remaining", flush=True)
PY
    else
        cp "$req_file" /tmp/requirements.work

        # ---- Pre-seed known compatibility pins (critical) ----
        # TensorFlow 1.x or TensorFlow 2.0–2.11, force protobuf to 3.20.3
        if grep -qiE '^[[:space:]]*(tensorflow|tensorflow-gpu)([[:space:]]|[<>=!~]).*(1\.|2\.(0|1|2|3|4|5|6|7|8|9|10|11)([[:space:]]|\.|$))' /tmp/requirements.work; then
            echo "protobuf>=3.9.2,<3.20" >> "$constraints"
        # TensorFlow 2.12–2.19, allow protobuf 3.20.3 up to (but not including) 5, and block the bad 4.21.0–4.21.5 releases
        elif grep -qiE '^[[:space:]]*(tensorflow|tensorflow-gpu)([[:space:]]|[<>=!~]).*(2\.1[2-9]([[:space:]]|\.|$))' /tmp/requirements.work; then
            echo "protobuf>=3.20.3,<5.0.0dev,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5" >> "$constraints"
        # TensorFlow 2.20+, protobuf 5.28+
        elif grep -qiE '^[[:space:]]*(tensorflow|tensorflow-gpu)([[:space:]]|[<>=!~]).*(2\.(2[0-9]|[3-9][0-9])([[:space:]]|\.|$))' /tmp/requirements.work; then
            echo "protobuf>=5.28.0,<6.0.0dev" >> "$constraints"
        fi

        # ---- Torch 1.3.1 on modern images: prefer CPU wheel to avoid CUDA/runtime issues ----
        if grep -qiE '^[[:space:]]*torch[[:space:]]*<=[[:space:]]*1\.3\.1([[:space:]]|$)' /tmp/requirements.work; then
            # Use PyTorch CPU index (contains torch==1.3.1+cpu wheels)
            pip_flags="$pip_flags --extra-index-url https://download.pytorch.org/whl/cpu"
            # Pin to CPU build explicitly (still satisfies <=1.3.1)
            echo "torch==1.3.1+cpu" >> "$constraints"
        fi
    fi


    removed_file="/tmp/removed.${env_name}.txt"
    : > "$removed_file"

    # Track TOP-LEVEL packages we already "moved to top" once (per env)
    moved_first_file="/tmp/moved_first.${env_name}.txt"
    : > "$moved_first_file"

    pip_log="/tmp/pip.${env_name}.log"

    last_sig=""
    install_ok=0

    # Iteratively remove failing deps
    while true; do    
        echo ">>> Installing APIs..."
        : > "$pip_log"
        set +e

        if [[ "$pinned_mode" -eq 1 ]]; then
            # Pinned list already contains all deps with exact versions; avoid dependency resolution.
            "$VENV_PY" -m pip install --no-deps -r /tmp/requirements.work -c "$constraints" $pip_flags 2>&1 | tee "$pip_log"
        else
            "$VENV_PY" -m pip install -r /tmp/requirements.work -c "$constraints" $pip_flags 2>&1 | tee "$pip_log"
        fi

        status=${PIPESTATUS[0]}
        set -e

        out="$(cat "$pip_log")"
        echo ">>> pip exit code: $status"

        if [[ $status -eq 0 ]]; then
            install_ok=1
            "$VENV_PY" -m pip install -U 'ipython<8.37' >/dev/null 2>&1 || true
            break
        fi

        # If pinned install failed, don’t waste time trying to "infer" constraints again.
        if [[ "$pinned_mode" -eq 1 ]]; then
            echo "✗ Pinned requirements install failed; aborting to avoid slow resolver/retry loop."
            echo ">>> pip log tail:"
            tail -200 "$pip_log" || true
            exit 1
        fi

        failed=$(
            {   
                # Case 0: "requires a different Python" (e.g., protobuf requires >=3.7)
                echo "$out" | sed -nE "s/.*Package '([^']+)'.*requires a different Python.*/\1/p" || true

                # Case 1: No matching versions / no matching dist - "Could not find a version that satisfies the requirement X"
                echo "$out" | sed -nE 's/.*(Could not find a version that satisfies the requirement|No matching distribution found for)[[:space:]]+[^ ]+[[:space:]]+\(from[[:space:]]+([A-Za-z0-9_.-]+)\).*/\2/p' || true

                echo "$out" | sed -nE 's/.*Could not find a version that satisfies the requirement[[:space:]]+([A-Za-z0-9_.-]+).*/\1/p' || true
                echo "$out" | sed -nE 's/.*No matching distribution found for[[:space:]]+([A-Za-z0-9_.-]+).*/\1/p' || true

                # Case 2: "The user requested X"
                echo "$out" | grep -Eo 'The user requested [^ ]+' \
                | sed 's/The user requested //' || true

                # Case 3: Dependency conflict - ResolutionImpossible conflict X depends on Y
                echo "$out" | awk '
                    /The conflict is caused by:/ { inblock=1; next }
                    inblock && /depends on/ {
                        # If the first token is a version-like number (e.g., 2.1.6), fall back to $2.
                        # Otherwise, use $1 as the package name.
                        if ($1 ~ /^[0-9]+(\.[0-9]+)*([a-z].*)?$/) {
                            if (NF >= 2) print $2
                            else print $1
                        } else {
                            print $1
                        }
                        exit
                    }
                ' || true

                # Case 4: Source dist missing setup.py/pyproject.toml + setup.py install/build errors
                echo "$out" | awk '
                    # Pattern: "ERROR: <pkg>==<ver> from ... does not appear to be a Python project"
                    /does not appear to be a Python project/ {
                        for (i = 1; i <= NF; i++) {
                            if ($i == "ERROR:") {
                                pkg = $(i+1)
                                sub(/[<>=!~].*$/, "", pkg)
                                sub(/from.*$/, "", pkg)
                                print pkg
                                exit
                            }
                        }
                    }
                ' || true

                # extract from setup.py path if present
                # "pip-install-UsXl3K/<pkg>/setup.py"
                printf '%s\n' "$out" | sed -nE 's#.*pip-install-[^/]+/([^/]+)/setup\.py.*#\1#p' || true

                # Case 5: subprocess-exited-with-error
                echo "$out" | awk '
                    function norm(pkg) {
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", pkg)
                        sub(/\(.*/, "", pkg)
                        sub(/\[.*/, "", pkg)
                        sub(/[<>=!~].*$/, "", pkg)
                        gsub(/[,;:.]+$/, "", pkg)
                        return pkg
                    }
                    /^Collecting[[:space:]]+/ {
                        pkg = $2
                        last = norm(pkg)
                        next
                    }
                    /subprocess-exited-with-error/ || /metadata-generation-failed/ || /Encountered error while generating package metadata/ {
                        if (last != "") { print last; exit }
                    }
                ' || true

                # Case 6: wheel/build/setup.py failures
                echo "$out" | sed -nE 's/^Failed to build ([^ ]+).*/\1/p' || true
                echo "$out" | sed -nE "s/.*Building wheel for ([^ ]+).*finished with status 'error'.*/\1/p" || true
                echo "$out" | sed -nE "s/.*Running setup\.py install for ([^: ]+): finished with status 'error'.*/\1/p" || true
                echo "$out" | sed -nE 's/.*Failed building wheel for ([^ ]+).*/\1/p' || true
                echo "$out" | awk '
                    function strip_prefix(line) {
                        sub(/^[0-9]+(\.[0-9]+)?[[:space:]]+/, "", line)  # drop "206.6 "
                        sub(/^ERROR:[[:space:]]+/, "", line)             # drop "ERROR: "
                        return line
                    }
                    function norm(pkg) {
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", pkg)
                        sub(/\(.*/, "", pkg)           # drop "(pyproject.toml):..."
                        sub(/\[.*/, "", pkg)           # drop extras like "foo[extra]"
                        sub(/[<>=!~].*$/, "", pkg)     # drop pins like "<=3.0.0"
                        gsub(/[,;]$/, "", pkg)         # drop trailing punctuation
                        return pkg
                    }

                    # strip common ANSI sequences just in case
                    function deansi(s) { gsub(/\033\[[0-9;]*[[:alpha:]]/, "", s); return strip_prefix(s) }

                    # block:
                    # Failed to build installable wheels ...
                    # ╰─> X"   (or any arrow containing >)
                    /Failed to build installable wheels/ {
                        if (getline > 0) {
                            line = deansi($0)
                            sub(/^.*>[[:space:]]*/, "", line)   # drop up to last >
                            split(line, a, /[[:space:]]+/)
                            if (a[1] != "") { print norm(a[1]); exit }
                        }
                    }

                    # Failed to build X
                    /Failed to build[[:space:]]/ {
                        line = deansi($0)
                        sub(/^.*Failed to build[[:space:]]+/, "", line)
                        split(line, a, /[[:space:]]+/)
                        if (a[1] != "") { print norm(a[1]); exit }
                    }

                    # ERROR: Failed building wheel for X
                    /Failed building wheel for/ {
                        line = deansi($0)
                        sub(/^.*Failed building wheel for[[:space:]]*/, "", line)
                        split(line, a, /[[:space:]]+/)
                        if (a[1] != "") { print norm(a[1]); exit }
                    }
                ' || true

                # Case 7: "Command errored out ... setup.py egg_info" (use last Collecting ...)
                echo "$out" | awk '
                    /Collecting[[:space:]]+/ {
                        pkg=$2
                        sub(/\(.*/, "", pkg)
                        sub(/\[.*/, "", pkg)
                        sub(/[<>=!~].*$/, "", pkg)
                        last=pkg
                    }
                    /ERROR: Command errored out with exit status/ {
                        if (last!="") { print last; exit }
                    }
                ' || true
            } | head -n 1
        )

        # echo ">>> pip output:"
        # echo "$out"

        failed="$(normalize_pkg_name "$failed")"

        if [[ -z "$failed" ]]; then
            echo "✗ Cannot determine failing package. Aborting."
            exit 1
        fi

        echo ">>> Failed package: $failed"

        sig="$(sha256sum /tmp/requirements.work "$constraints" 2>/dev/null | sha256sum | awk "{print \$1}")"
        if [[ "$sig" == "$last_sig" ]]; then
            echo "✗ No progress since last attempt. Giving up on this env."
            break
        fi
        last_sig="$sig"


        failed_esc="$(escape_ere "$failed")"
        # If it is TOP-LEVEL, record the original requirement line then remove it.
        if grep -qiE "^[[:space:]]*${failed_esc}([[:space:]]|$|[<>=!~])" /tmp/requirements.work; then
            orig_line="$(grep -iE "^[[:space:]]*${failed_esc}([[:space:]]|$|[<>=!~]).*" /tmp/requirements.work | head -n 1 || true)"

            # One-time reorder retry (don’t remove yet)
            if ! grep -qiFx "$failed" "$moved_first_file"; then
                printf '%s\n' "$failed" >> "$moved_first_file"
                [[ -z "$orig_line" ]] && orig_line="$failed"
                echo "⚠ Retrying TOP-LEVEL by moving to first: $orig_line"
                tmp_reordered="/tmp/requirements.reordered.$$"
                # Remove all lines for this package, then prepend the original line
                grep -viE "^[[:space:]]*${failed_esc}([[:space:]]|$|[<>=!~])" /tmp/requirements.work > "$tmp_reordered" || true
                {
                    printf '%s\n' "$orig_line"
                    cat "$tmp_reordered"
                } > /tmp/requirements.work
                rm -f "$tmp_reordered"

                continue
            fi

            # Second time failing: record and remove as before
            if [[ -n "$orig_line" ]]; then
                printf '%s\n' "$orig_line" >> "$removed_file"
            else
                printf '%s\n' "$failed" >> "$removed_file"
            fi

            echo "⚠ Removing incompatible TOP-LEVEL package: $failed"
            sed -i "/^$failed/d" /tmp/requirements.work
            sed -i "\|$failed|d" /tmp/requirements.work
            sed -i -E "/^[[:space:]]*${failed_esc}([[:space:]]|$|[<>=!~]).*$/Id" /tmp/requirements.work

            # If requirements is now empty, break
            if [[ ! -s /tmp/requirements.work ]]; then
                echo "All packages removed. Creating empty environment."
                break
            fi
            continue
        fi

        # Otherwise: TRANSITIVE -> try to pin a compatible version via constraints
        echo "⚠ '$failed' looks transitive. Trying to pin a compatible version via constraints..."
        pinned_ver="$(
            python3 - <<'PY' "$failed" 2>/tmp/pypi_versions.err || true
from __future__ import print_function
import sys
import json

# urllib compatibility
try:
    # Py3
    from urllib.request import urlopen
except ImportError:
    # Py2
    from urllib2 import urlopen

pkg = sys.argv[1]
url = "https://pypi.org/pypi/{}/json".format(pkg)

# Fetch + decode JSON (bytes -> text)
resp = urlopen(url, timeout=20)
raw = resp.read()
try:
    text = raw.decode("utf-8")
except Exception:
    # Py2: raw might already be 'str' (bytes); this still works
    text = raw

data = json.loads(text)
versions = list((data.get("releases") or {}).keys())

# Sort versions oldest->newest (reverse=False) in a "version-aware" way if possible
keyfunc = None

try:
    from packaging.version import Version as _V
    keyfunc = _V
except Exception:
    try:
        # Works in Py2.7..Py3.11; removed in 3.12 
        from distutils.version import LooseVersion as _V
        keyfunc = _V
    except Exception:
        keyfunc = None

if keyfunc is not None:
    try:
        versions.sort(key=keyfunc, reverse=False)
    except Exception:
        versions.sort(reverse=False)
else:
    versions.sort(reverse=False)

for v in versions:
    print(v)
PY
        )"
        if [[ -z "${pinned_ver:-}" ]]; then
            echo "✗ Could not fetch versions for '$failed' from PyPI."
            echo ">>> python3 error:"
            cat /tmp/pypi_versions.err || true
            break
        fi

        chosen=""
        while read -r ver; do
            [[ -z "$ver" ]] && continue
            echo "  - trying ${failed}==${ver}"
            if "$VENV_PY" -m pip install "${failed}==${ver}" -c "$constraints" >/tmp/pin_try.log 2>&1; then
                echo "✓ pinned transitive ${failed}==${ver}"
                chosen="$ver"
                break
            fi
        done <<< "$pinned_ver"

        if [[ -n "$chosen" ]]; then
            echo "${failed}==${chosen}" >> "$constraints"
            continue
        else
            while read -r ver; do
                [[ -z "$ver" ]] && continue
                echo "  - trying ${failed}==${ver} --no-deps"
                if "$VENV_PY" -m pip install --no-deps "${failed}==${ver}" -c "$constraints" >/tmp/pin_try.log 2>&1; then
                    echo "✓ pinned --no-deps transitive ${failed}==${ver}"
                    chosen="$ver"
                    break
                fi
            done <<< "$pinned_ver"

            if [[ -n "$chosen" ]]; then
                echo "${failed}==${chosen}" >> "$constraints"
                continue
            fi
        fi

        echo "✗ Could not find a working version for transitive '$failed'."
        echo ">>> last pin attempt log:"
        tail -200 /tmp/pin_try.log || true
        break
    done

    # Add a "removed packages retry install" per env
    # ---- Final pass: after base install, retry removed top-level packages using pinned_ver ----
    if [[ "$install_ok" -eq 1 && -s "$removed_file" ]]; then
        echo ">>> Final pass: retrying removed top-level packages for $env_name"
        while IFS= read -r reqline; do
            [[ -z "$reqline" ]] && continue

            pkg="$(printf '%s\n' "$reqline" | sed -nE 's/^[[:space:]]*([A-Za-z0-9_.-]+).*/\1/p')"
            [[ -z "$pkg" ]] && continue

            # echo ">>> Retrying removed API: $reqline"
            # pinned_ver="$(pypi_versions_for_reqline "$pkg" "$reqline")"
            # [[ -z "${pinned_ver:-}" ]] && { echo "  (no versions found for $pkg)"; continue; }

            # ok=""
            # while read -r ver; do
            #     [[ -z "$ver" ]] && continue
            #     echo "  - trying ${pkg}==${ver}"
            #     first_pkg=1
            #     # Start with desired then walk downward until the oldest version
            #     # To make reproducible builds
            #     if (( first_pkg )) && "$VENV_PY" -m pip install "${pkg}>${ver}" -c "$constraints" $pip_flags >/tmp/retry_removed.log 2>&1; then
            #         echo "✓ installed removed ${pkg}==${ver} with trying >=${ver}"
            #         echo "${pkg}>=${ver}" >> "$constraints"
            #         ok="1"
            #         break
            #     fi
            #     if (( first_pkg )) && "$VENV_PY" -m pip install --no-deps "${pkg}>${ver}" -c "$constraints" $pip_flags >/tmp/retry_removed.log 2>&1; then
            #         echo "✓ installed removed ${pkg}==${ver} with trying >=${ver} with --no-deps"
            #         echo "${pkg}>=${ver}" >> "$constraints"
            #         ok="1"
            #         break
            #     fi
            #     first_pkg=0

            #     if "$VENV_PY" -m pip install --no-deps "${pkg}==${ver}" -c "$constraints" $pip_flags >/tmp/retry_removed.log 2>&1; then
            #         echo "✓ installed removed ${pkg}==${ver} with --no-deps"
            #         echo "${pkg}==${ver}" >> "$constraints"
            #         ok="1"
            #         break
            #     fi
            # done <<< "$pinned_ver"

            # Extract version from the original requirement line (keeps rc/dev/etc, e.g., 1.2.3rc)
            ver="$(
                printf '%s\n' "$reqline" \
                | sed -nE 's/.*<=[[:space:]]*([^[:space:]#,;]+).*/\1/p'
            )"
            
            echo ">>> Retrying removed API: ${pkg}<=${ver} (from: $reqline)"
            if "$VENV_PY" -m pip install --no-deps "${pkg}<=${ver}" -c "$constraints" $pip_flags >/tmp/retry_removed.log 2>&1; then
                echo "✓ installed removed ${pkg}<=${ver} with --no-deps"
                echo "${pkg}<=${ver}" >> "$constraints"
            else
                echo "Backporting Failed. Aborting."
                tail -80 /tmp/retry_removed.log || true
                exit 1
            fi
        done < "$removed_file"
    fi

    # Download NLTK corpora only if nltk is actually installed in this venv
    # Provide corpora list via env var NLTK_CORPORA="punkt,wordnet,stopwords"
    if "$VENV_PY" -c "import nltk" >/dev/null 2>&1; then
        echo ">>> nltk detected; downloading corpora..."
        python3 - <<'PY'
import os
import sys
import ssl

try:
    import nltk
except Exception as e:
    sys.exit(0)

# allow unverified HTTPS context (for proxies or missing certs)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    _create_unverified_https_context = None
if _create_unverified_https_context is not None:
    ssl._create_default_https_context = _create_unverified_https_context

# corpora list
with open("/tmp/nltk_corpora.txt") as f:
    corpora = [line.strip() for line in f if line.strip() and not line.startswith("#")]

if  corpora:
    for pkg in corpora:
        nltk.download(pkg, quiet=True)
PY
    else
        echo ">>> nltk not installed; skipping corpora download."
    fi

    # set +u
    # deactivate
    # set -u

    unset PYENV_VERSION
    pyenv shell system
done

cd /kaggle/working
echo "Environment processed."