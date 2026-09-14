import json
import re
import html
from tqdm import tqdm
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

output_path = "baseline/nltk_corpora.txt"


def extract_nltk_downloads(nb):
    """Return a list of all package names passed to nltk.download in the notebook."""

    downloads = set()
    # matches nltk.download('pkg') or nltk.download("pkg")
    pattern = re.compile(r"nltk\.download\(\s*['\"]([^'\"]+)['\"]")

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = "".join(cell.get('source', []))
        for m in pattern.finditer(src):
            downloads.add(m.group(1))

    return downloads

# 1. Load the JSON data
with open("kernel.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Flatten & filter entries
filtered = []
for comp, files in tqdm(data.items(), total=len(data),
        desc="Competitions",
        position=0):
    for fname, info in tqdm(files.items(), total=len(files),
        desc=f"submissions to {comp}",
        leave=False,
        position=1):
        if "ps" in info and float(info['ps']) != 0.0 and info['runtime'] <= 600 and len(info['datasets'])<=1 and "R" not in info and 'nltk' in info['api'] :
            # keep a reference to competition and filename if you need them
            filtered.append((comp, fname))

print(f"Total filtered entries w/ using nltk: {len(filtered)}")

corpora = set()

for comp, fname in tqdm(filtered):
    kernel_path = Path(f"./baseline/notebooks/{comp}_{fname}")
    if not kernel_path.exists():
        kernel_path = Path(f"./baseline/notebooks_2/{comp}_{fname}")
    with open(kernel_path, "r", encoding="utf-8") as fp:
        nb = nbformat.read(fp, as_version=4)
        nb.cells = [
            cell
            for cell in nb.cells
            if cell.get("cell_type") == "code"
        ]

    pkg = extract_nltk_downloads(nb)


    corpora.update(pkg)

with open(output_path, "w", encoding="utf-8") as f:
        for pkg in corpora:
            f.write(f"{pkg}\n")
    
print(f"Successfully saved {len(corpora)} unique corpora to {output_path}")