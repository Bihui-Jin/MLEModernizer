import requests
import json
import datetime
from pathlib import Path
import time
from collections import deque
from tqdm import tqdm
import os
import re
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from collections import defaultdict

CACHE_PATH = "apiDowngrade/pypi_versions.json"
try:
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        api_cache = json.load(f)
except Exception:
    api_cache = {}

with open("apiDowngrade/kernel_w_pyVersion.json", "r", encoding="utf-8") as f:
    kernel_content = json.load(f)


def save_cache():
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(api_cache, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CACHE_PATH)

def is_python_compatible(required_python: str, python_version: str) -> bool:
    """
    Check if python_version satisfies required_python specifier.
    Examples:
        required_python='>=3.8' and python_version='3.9' -> True
        required_python='>=3.6.1,<3.10' and python_version='3.11' -> False
    """
    if not required_python:
        return True  # No constraint means compatible
    
    # try:
    spec = SpecifierSet(required_python.replace(".*", ""))
    return spec.contains(python_version)
    # except Exception:
    #     return True  # If parsing fails, assume compatible

def adjust_python_version(python_version: str) -> str:
    """
    Adjust Python version by subtracting 0.1 from minor version.
    Example: '3.9.2' -> '3.8' -> '3.8.20' (latest patch for 3.8)
    """
    version_match = {"2.6":"2.6.9",
                     "2.7":"2.7.18",
                     "3.4":"3.4.10",
                     "3.5":"3.5.10",
                     "3.6":"3.6.15",
                     "3.7":"3.7.17",
                     "3.8":"3.8.20",
                     "3.9":"3.9.25",
                     "3.10":"3.10.19",
                     "3.11":"3.11.14",
                     "3.12":"3.12.12"}
    parts = python_version.split('.')
    if len(parts) >= 2:
        major = int(parts[0])
        minor = int(parts[1])
        patch = parts[2] if len(parts) > 2 else '0'
        
        # Subtract 1 from minor version
        adjusted_minor = max(0, minor - 1)
        if major == 2 or (major == 3 and minor == 5):
            return version_match.get(f"{major}.{minor}", python_version)
        return version_match.get(f"{major}.{adjusted_minor}", f"{major}.{adjusted_minor}")
    return python_version

def is_prerelease(version: str) -> bool:
    return Version(version).is_prerelease

def get_versions_from_pypi(pkg: str, compt: str, fname: str) -> dict:
    """
    Scrape PyPI.org history page and return dict of {version: datetime_str}
    Ignores pre-release versions (those with badge--warning tag) and yanked versions.
    Returns dict like {'0.4.5': '2025-07-15T19:08:29+0000', ...}
    """
    if pkg in api_cache and api_cache[pkg]:
        return api_cache[pkg]
    
    for attempt in range(3):
        try:
            url = f"https://pypi.org/pypi/{pkg}/json"
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            releases = {}
            
            for version, info in data["releases"].items():
                if not info:
                    continue
                # # Check if it's a pre-release/yanked
                # try:
                #     pre_release_badge = Version(version).is_prerelease
                # except Exception:
                #     # Skip versions with invalid format
                #     continue
                yanked_badge = info[0]['yanked']
                if len(data["releases"])>1 and yanked_badge:
                    continue
                
                # Extract datetime from <time> tag 
                upload_time = info[0]["upload_time_iso_8601"] # 2025-11-29T22:06:53.671416Z
                requires_python = info[0].get("requires_python", "")

                releases[version] = {"release_time": upload_time, "required_python": requires_python}
            
            api_cache[pkg] = releases
            if releases:
                save_cache()
            time.sleep(2 ** attempt)  # Rate limit: be polite to PyPI
            return releases
                
        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == 2:
                print(f"Fetch fail {pkg}: {compt}/{fname} due to {e}")
                with open("apiDowngrade/apiMatch_error.txt", "a", encoding="utf-8") as f:
                    f.write(f"{compt}/{fname} | {pkg} | {str(e)}\n")
                return {}
            continue
            
    return {}

def lower_version(a: str | None, b: str) -> str:
    """
    Compare two version strings and return the lower one.
    Handles pre-release (rc, alpha, beta) and post-release versions correctly.
    Examples:
        lower_version("1.3post1", "1.4rc1") -> "1.3post1"
        lower_version("1.0", "1.0rc1") -> "1.0rc1" (rc < release)
        lower_version("1.4.dev1", "1.4rc1") -> "1.4.dev1" (dev < rc < release)
        lower_version(None, "1.2.3") -> "1.2.3"
    """
    if a is None:
        return b
    try:
        ver_a = Version(a)
        ver_b = Version(b)
        return b if ver_b < ver_a else a
    except InvalidVersion:
        # Fallback: try to parse manually or use string comparison
        try:
            # Strip 'v' prefix if present
            a_stripped = a.lstrip('v').strip()
            b_stripped = b.lstrip('v').strip()

            # Try to extract base version (major.minor.patch)
            a_base = re.match(r'(\d+\.\d+(?:\.\d+)?)', a_stripped)
            b_base = re.match(r'(\d+\.\d+(?:\.\d+)?)', b_stripped)
            
            if a_base and b_base:
                a_tuple = tuple(map(int, a_base.group(1).split('.')))
                b_tuple = tuple(map(int, b_base.group(1).split('.')))
                
                if a_tuple == b_tuple:
                    # Same base version: prefer stable > post > rc > dev > alpha/beta
                    def priority(v_str):
                        if 'a' in v_str.lower() or 'alpha' in v_str.lower():
                            return 0
                        elif 'b' in v_str.lower() or 'beta' in v_str.lower():
                            return 1
                        elif 'dev' in v_str.lower():
                            return 2
                        elif 'rc' in v_str.lower():
                            return 3
                        elif 'post' in v_str.lower():
                            return 4
                        else:
                            return 5  # Stable release
                    
                    return a if priority(a_stripped) > priority(b_stripped) else b
                
                # Different base versions: compare numerically
                return b if b_tuple < a_tuple else a
            
            # Last resort: lexicographic comparison
            return b if b < a else a
        except (ValueError, AttributeError):
            # Last resort: string comparison
            print(f"Fallback comparison failed: {fallback_e}, using string comparison")
            return b if b < a else a

def normalize_version(v: str) -> str:
    parts = v.split(".")
    return ".".join(parts[:2])

if __name__ == "__main__":
    family_reqs: dict[str, dict[str, str]] = defaultdict(dict)
    family_empty: dict[str, str] = defaultdict(str)

    for compt, files in tqdm(kernel_content.items(), total=len(kernel_content),
        desc="Competitions", position=0):
        
        for fname, script_meta in tqdm(files.items(), total=len(files),
            desc=f"submissions to {compt}", leave=False, position=1):
            
            if "ps" in script_meta and float(script_meta['ps']) != 0.0 and script_meta['runtime'] <= 600 and \
               len(script_meta['datasets']) <= 1 and "R" not in script_meta:
                
                submission_date = datetime.datetime.strptime(
                    script_meta["datetime"], "%Y-%m-%dT%H:%M:%S.%fZ"
                ) #'2008-04-25T16:22:32.000Z'
                apis = script_meta.get("api") or []
                python_version = script_meta["python"]
                adjusted_python = adjust_python_version(python_version)

                results_per_file = ""

                # family key: compt + base name without trailing _vN
                base_name = fname.split('.')[0]
                family_base = re.sub(r'(_v\d+|_[A-Za-z]\d+)+$', '', base_name)
                family_key = f"{compt}_{family_base}"

                if apis:
                    if "keras" in apis:
                        apis.append("tensorflow") # based on official guide https://keras.io/getting_started/
                    for api in apis:
                        versions = get_versions_from_pypi(api, compt, fname)
                        
                        if not versions:
                            continue
                        
                        closest_v = None
                        closest_dt = None
                        oldest_v = None
                        oldest_dt = None
                        oldest_compatible_v = None
                        oldest_compatible_dt = None
                        
                        for version, info in versions.items():
                            required_python = info["required_python"]
                            dt_str = info["release_time"]
                            # Parse ISO datetime
                            try:
                                pub_date = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ") # 2025-11-29T22:06:53.671416Z
                            except ValueError:
                                pub_date = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ") # 2006-12-02T02:07:43Z
                            
                            is_compatible = is_python_compatible(required_python, adjusted_python)
                            
                            # Track oldest version (regardless of Python compatibility)
                            if oldest_dt is None or pub_date < oldest_dt:
                                oldest_dt = pub_date
                                oldest_v = version
                            
                            # Track oldest Python-compatible version
                            if is_compatible:
                                if oldest_compatible_dt is None or pub_date < oldest_compatible_dt:
                                    oldest_compatible_dt = pub_date
                                    oldest_compatible_v = version
                            
                            # Find closest version <= submission date AND Python-compatible
                            if pub_date <= submission_date and is_compatible:
                                if closest_dt is None or pub_date > closest_dt:
                                    closest_dt = pub_date
                                    closest_v = version
                        
                        if closest_v:
                            results_per_file += f"{api}<={closest_v}\n"
                        elif oldest_compatible_v:
                            # Fallback to oldest Python-compatible version
                            results_per_file += f"{api}<={oldest_compatible_v}\n"
                            with open("apiDowngrade/apiMatch_oldest_Python_compatible.txt", "a", encoding="utf-8") as f:
                                f.write(f"{compt}_{fname} | {api}<={oldest_compatible_v} | "
                                    f"oldest compatible {oldest_compatible_dt.date()} >= submission {submission_date.date()} | "
                                    f"adjusted_python={adjusted_python} |"
                                    f"required_python={required_python}\n")
                        elif oldest_v:
                            # Last resort: oldest version (may be incompatible)
                            results_per_file += f"{api}<={oldest_v}\n"
                            with open("apiDowngrade/apiMatch_oldest_version_compatible.txt", "a", encoding="utf-8") as f:
                                f.write(f"{compt}_{fname} | {api}<={oldest_v} | "
                                    f"oldest {oldest_dt.date()} >= submission {submission_date.date()} | "
                                    f"adjusted_python={adjusted_python} |"
                                    f"required_python={required_python} INCOMPATIBLE\n")
                        
                        chosen = closest_v or oldest_compatible_v or oldest_v
                        current = family_reqs[family_key].get(api)
                        family_reqs[family_key][api] = lower_version(current, chosen)
                    
                    output_path = f"apiDowngrade/apiDowngradeList/{compt}_{fname.split('.')[0]}.txt"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(results_per_file)
                
                else:
                    family_empty[family_key] = normalize_version(adjusted_python)  # No APIs, note Python version


    # Write one requirements per family (merged, smallest versions)
    for family_key, pkgs in family_reqs.items():
        out_path = f"apiDowngrade/apiDowngradeFamilyList/{family_key}.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for pkg in sorted(pkgs):
                f.write(f"{pkg}<={pkgs[pkg]}\n")


    with open("apiDowngrade/submission_noAPI.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(family_empty), ensure_ascii=False, indent=2))