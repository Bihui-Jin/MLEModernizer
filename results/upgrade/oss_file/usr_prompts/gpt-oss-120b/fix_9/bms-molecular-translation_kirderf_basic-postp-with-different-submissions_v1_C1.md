# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Provided with images of chemicals, predict the corresponding International Chemical Identifier (InChI) text string of the image.

## Metric
Mean [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) between the InChi strings you submit and the ground truth InChi values.

## Submission Format
For each `image_id` in the test set, you must predict the InChi string of the molecule in the corresponding image. The file should contain a header and have the following format:

```
image_id,InChI
00000d2a601c,InChI=1S/H2O/h1H2
00001f7fc849,InChI=1S/H2O/h1H2
000037687605,InChI=1S/H2O/h1H2
etc.
```

## Dataset
- **train/** - the training images, arranged in a 3-level folder structure by `image_id`
- **test/** - the test images, arranged in the same folder structure as `train/`
- **train_labels.csv** - ground truth InChi labels for the training images
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        input/
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        working/
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
```

-> data/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/bms-molecular-translation/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/bms-molecular-translation/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> data/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> input/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> (stopped after 10 files for performance)

# 5. Target score

4.662682011302231

# 6. Current score

86.78095

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 89.17911) has done: 'I adjust the file‑search helper to locate the CSVs inside the nested competition folders, then reload the variables that depend on those paths. This fixes the FileNotFoundError and lets the script run to produce a valid `submission.csv` using the most common InChI from the training set.'
- What this solution (achieved 89.17911) has done: 'I keep the existing file‑search and most‑common fallback logic, but add a step that fills in the exact InChI for any test image that already appears in the training labels. This uses the correct labels where possible, which should substantially lower the Levenshtein distance while preserving the original workflow for unseen images.'
- What this solution (achieved 85.48218) has done: 'I add a small heuristic that uses the first two characters of each image_id to choose the most common InChI seen for that prefix in the training set, falling back to the overall most common InChI only when no prefix‑specific frequency is available. This keeps the original merge‑and‑fallback flow intact while giving more tailored predictions, which should reduce the Levenshtein distance toward the target score.'
- What this solution (achieved 86.71472) has done: 'I add a finer‑grained fallback that first looks for a training label with the same 3‑character prefix, then falls back to the existing 2‑character prefix mapping, and finally to the overall most common InChI. This hierarchical lookup keeps the core logic unchanged while giving more specific predictions for unseen images, which should reduce the average Levenshtein distance and move the score closer to the target.'
- What this solution (achieved 86.77668) has done: 'I extend the prefix‑based fallback to use longer prefixes (up to 5 characters) and choose the most specific prefix that appears in the training data. The exact‑match step stays unchanged, then we look for the longest prefix (5 → 4 → 3 → 2) with a known most‑common InChI, finally falling back to the overall most common InChI. This small heuristic refinement keeps the original logic while giving more tailored predictions, which should lower the average Levenshtein distance and move the score nearer to the target.'
- What this solution (achieved 86.78095) has done: 'I increase the prefix length used for the fallback heuristic from 5 to 10 characters, which gives a more specific most‑common InChI for each longer image‑id prefix while preserving the original workflow. This small change is expected to produce predictions that better match the true strings, thereby reducing the Levenshtein distance and moving the score closer to the target.'
- What this solution (achieved 86.78095) has done: 'I make the fallback prefix length dynamic, using the longest `image_id` seen in the training data instead of a fixed value of 10. This gives the heuristic the most specific prefix information available (up to the full ID length), while keeping the original exact‑match step and overall workflow unchanged. The change is limited to the prefix‑generation cells and should improve predictions, moving the Levenshtein score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path
from collections import Counter


def find_file(fname: str) -> Path:
    """
    Search for a file named `fname` in several common locations.
    If not found directly, perform a recursive search under the current
    working directory and return the first match.
    """
    possible = [
        Path.cwd() / fname,
        Path.cwd() / "data" / fname,
        Path.cwd() / "input" / fname,
        Path.cwd() / "data" / "bms-molecular-translation" / fname,
        Path.cwd() / "input" / "bms-molecular-translation" / fname,
        Path.cwd() / "working" / "bms-molecular-translation" / fname,
    ]
    for p in possible:
        if p.is_file():
            return p
    matches = list(Path.cwd().rglob(fname))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {fname} in expected locations.")


train_labels_path = find_file("train_labels.csv")
sample_submission_path = find_file("sample_submission.csv")




## === cell 1
train_labels = pd.read_csv(train_labels_path)

most_common_inchi = Counter(train_labels["InChI"]).most_common(1)[0][0]

max_prefix_len = train_labels["image_id"].str.len().max()

prefix_common_dicts = {}
for k in range(2, max_prefix_len + 1):
    prefix_col = f"prefix{k}"
    train_labels[prefix_col] = train_labels["image_id"].str[:k]
    prefix_common = (
        train_labels.groupby(prefix_col)["InChI"]
        .agg(lambda x: Counter(x).most_common(1)[0][0])
        .to_dict()
    )
    prefix_common_dicts[k] = prefix_common




## === cell 2
submission = pd.read_csv(sample_submission_path)

submission = submission.merge(
    train_labels[["image_id", "InChI"]],
    on="image_id",
    how="left",
    suffixes=("", "_train"),
)


def fill_inchi(row):
    if pd.notna(row["InChI_train"]):
        return row["InChI_train"]
    img_id = row["image_id"]
    for k in range(max_prefix_len, 1, -1):
        pref = img_id[:k]
        common_dict = prefix_common_dicts.get(k, {})
        if pref in common_dict:
            return common_dict[pref]
    return most_common_inchi


submission["InChI"] = submission.apply(fill_inchi, axis=1)
submission.drop(columns=["InChI_train"], inplace=True)




## === cell 3
submission.to_csv("submission.csv", index=False)
