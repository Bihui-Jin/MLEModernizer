import json
import datetime
from tqdm import tqdm
from collections import Counter
import nbformat
from pathlib import Path
import re
from packaging.version import Version

with open("kernel.json", "r", encoding="utf-8") as f:
    kernel_content = json.load(f)

with open("apiDowngrade/python_versions.json", "r", encoding="utf-8") as f:
    python_versions = json.load(f)


def classify_notebook(nb):
    """
    Robustly discern if the code is written in python 2 syntax or python 3.
    Returns '2', '3', or 'unknown'.
    """
    py2_hits = 0
    py3_hits = 0

    # Regex patterns for Python 2 specific syntax
    py2_patterns = [
        r'(^|[^a-zA-Z0-9_])print\s+["\']',           # print "string" (no parentheses)
        r'(^|[^a-zA-Z0-9_])print\s+[\w]+',           # print var (no parentheses)
        r'\bprint\s*>>',                             # print redirect syntax
        r'\bxrange\b',                               # xrange vs range
        r'\braw_input\b',                            # raw_input vs input
        r'\bunicode\b',                              # unicode type
        r'\blong\b',                                 # long type
        r'\bbasestring\b',                           # basestring type
        r'\.iteritems\(',                            # dict.iteritems()
        r'\.has_key\(',                              # dict.has_key()
        r'except\s+\w+\s*,\s*\w+:',                  # except Exception, e:
        r'from\s+__future__\s+import\s+division',    # __future__ division
        r'from\s+__future__\s+import\s+print_function', # __future__ print_function
        r'\bu["\']',                                 # u"string" literals
    ]
    # Regex patterns for Python 3 specific syntax
    py3_patterns = [
        r'\bprint\s*\(',                             # print() with parentheses
        r'\bf["\']',                                 # f-strings (Py 3.6+)
        r'\bf"""',                                   # f-strings triple quoted
        r"\bf'''",                                   # f-strings triple quoted
        r'\basync\s+def\b',                          # async/await (Py 3.5+)
        r'\bawait\b',                                # await keyword
        r'\bnonlocal\b',                             # nonlocal keyword
        r'\btyping\.',                               # typing module usage
        r'from\s+typing\s+import',                   # typing imports
        r'\)\s*->\s*[A-Za-z_]',                      # function return annotations
        r'\byield\s+from\b',                         # yield from (Py 3.3+)
        r'\basync\s+for\b',                          # async for (Py 3.5+)
        r'\basync\s+with\b',                         # async with (Py 3.5+)
    ]
    for cell in nb.cells:
        if cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', ''))
            if not src.strip():
                continue

            for p in py2_patterns:
                if re.search(p, src):
                    py2_hits += 1
            for p in py3_patterns:
                if re.search(p, src):
                    py3_hits += 1

    if py2_hits == 0 and py3_hits == 0:
        return 'unknown'
    if py2_hits > py3_hits:
        return '2'
    if py3_hits >= py2_hits: # Default to 3 if tied, as it's more likely for recent files from the distribution
        return '3'
    return 'unknown'

# Convert python_versions to datetime objects for comparison
python_versions_dt = {
    version: datetime.datetime.strptime(date, "%Y-%m-%d")
    for version, date in python_versions.items()
}

if __name__ == "__main__":
    updated_count = 0
    no_match_count = 0
    python_version_counts = Counter()

    for compt, files in tqdm(
            kernel_content.items(),
            total=len(kernel_content),
            desc="Competitions",
            position=0):
            for fname, script_meta in tqdm(
                files.items(),
                total=len(files),
                desc=f"Processing {compt}",
                leave=False,
                position=1
            ):
                if "ps" in script_meta and float(script_meta['ps']) != 0.0 and script_meta['runtime'] <= 600 and len(script_meta['datasets'])<=1 and "R" not in script_meta:
                    
                    kernel_path = Path(f"./baseline/notebooks/{compt}_{fname}")
                    with open(kernel_path, "r", encoding="utf-8") as fp:
                        nb = nbformat.read(fp, as_version=4)
                        nb.cells = [
                            cell
                            for cell in nb.cells
                            if cell.get("cell_type") == "code"
                        ]
                    detected_major = classify_notebook(nb)

                    submission_date = datetime.datetime.strptime(script_meta["datetime"], "%Y-%m-%dT%H:%M:%S.%fZ")
                    # Restrict search space by detected major version
                    if detected_major == '2':
                        candidate = {v: d for v, d in python_versions_dt.items() if v.startswith('2.') and d < submission_date and "p" not in v}
                    elif detected_major == '3':
                        candidate = {v: d for v, d in python_versions_dt.items() if v.startswith('3.') and d < submission_date and "p" not in v}
                    else:
                        candidate = {v: d for v, d in python_versions_dt.items() if d < submission_date and "p" not in v}


                    # Find the HIGHEST (most recent) Python version released before submission_date
                    # Within the detected major version (e.g., if detected_major='3', choose max among 3.x.y)
                    # Example: If 3.8.6 (2020-09-24) and 3.9.5 (2021-05-03) both < submission_date,
                    #          choose 3.9.5 (highest/most recent)
                    best_version = None
                    best_date = None
                    
                    if candidate:
                        best_version = max(candidate.keys(), key=Version)
                        best_date = candidate[best_version]
                    
                    if best_version:
                        script_meta["python"] = best_version
                        script_meta["detected_major"] = detected_major
                        python_version_counts[best_version] += 1
                        updated_count += 1
                    else:
                        no_match_count += 1
                        script_meta["detected_major"] = detected_major
                        print(f"No Python version found for {compt}/{fname} (submission: {submission_date.date()})")
                    
                    
    # Save updated kernel.json
    with open("apiDowngrade/kernel_w_pyVersion.json", "w", encoding="utf-8") as f:
        json.dump(kernel_content, f, indent=4, ensure_ascii=False)

    print(f"\nUpdated {updated_count} scripts with Python versions")
    print(f"No match found for {no_match_count} scripts")

    # Sort by version number (descending)
    sorted_versions = sorted(python_version_counts.items(), 
                            key=lambda x: tuple(map(int, x[0].split('.'))), 
                            reverse=True)
    
    # Generate Markdown output
    markdown_output = []
    markdown_output.append("# Python Version Statistics\n")
    markdown_output.append(f"**Total scripts updated:** {updated_count}<br>")
    markdown_output.append(f"**Scripts with no match:** {no_match_count}<br>")
    markdown_output.append(f"**Total unique versions:** {len(python_version_counts)}<br>\n")
    
    markdown_output.append("## Python Version Distribution\n")
    markdown_output.append("| Python Version | Count | Percentage |")
    markdown_output.append("|----------------|-------|------------|")
    
    for version, count in sorted_versions:
        percentage = (count / updated_count * 100) if updated_count > 0 else 0
        markdown_output.append(f"| {version} | {count} | {percentage:.2f}% |")
    
    markdown_output.append("")
    
    # Print to console
    print("\n" + "\n".join(markdown_output))
    
    print(f"{'-'*60}")
    print(f"Total unique versions: {len(python_version_counts)}")
    print(f"{'='*60}")


