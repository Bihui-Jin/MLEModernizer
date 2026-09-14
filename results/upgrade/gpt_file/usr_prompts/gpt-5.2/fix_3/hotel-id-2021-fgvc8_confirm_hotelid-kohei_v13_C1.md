# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.7698703376273452

# 6. Current score

0.00232

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00232) has done: 'I keep your current approach of running `python -m peko.subs.hotelid.v13`, but make the notebook reliably produce a valid `submission.csv` in the expected working directory and fail fast with actionable errors if it doesn’t. The main reason you’re getting “Not yielded” is likely that the script writes the submission somewhere else (or under a different name), so Kaggle can’t find/upload it. I add a minimal “find-and-copy” step that searches common output locations for a CSV with the required `image,hotel_id` header, copies it to `/kaggle/working/submission.csv`, and validates row count/columns against `sample_submission.csv`. This doesn’t change model logic, but it unblocks scoring and ensures the output is submission-valid.'
- What this solution (achieved 0.00232) has done: 'I fix the failing dependency installation by making it robust to the actual wheel layout in `/kaggle/input` (your current glob points to non-existent files), while keeping the core logic of running `python -m peko.subs.hotelid.v13` unchanged. Then I keep your existing “find-and-copy” validation step to guarantee a correctly formatted `/kaggle/working/submission.csv`. This should both unblock execution and recover the intended score (your current 0.00232 is consistent with a broken/placeholder submission rather than a real model output). The changes are minimal and confined to installation/path handling and submission file validation.'

# 9. Code solution

## === cell 0
import os, sys, glob, shutil, subprocess
from pathlib import Path

subprocess.run("ls -lha /kaggle/input/", shell=True, check=False)
subprocess.run("ls -lha /kaggle/input | sed -n '1,200p'", shell=True, check=False)



## === cell 1
import subprocess
from pathlib import Path
import sys


def run(cmd, check=True):
    print(f"\n[cmd] {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def find_wheels(root: Path):
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.whl") if p.is_file()])


deps_root = Path("/kaggle/input/pekolib-deps")
lib_root = Path("/kaggle/input/pekolib")

deps_wheels = find_wheels(deps_root)
lib_wheels = find_wheels(lib_root)

if not deps_wheels and not lib_wheels:
    fallback = Path("/kaggle/input")
    all_wheels = sorted([p for p in fallback.rglob("*.whl") if p.is_file()])
    lib_wheels = [p for p in all_wheels if "pekolib" in p.name.lower()]
    deps_wheels = [p for p in all_wheels if "pekolib" not in p.name.lower()]

print(f"Found deps wheels: {len(deps_wheels)}")
print(f"Found lib wheels: {len(lib_wheels)}")
if deps_wheels[:5]:
    print("deps sample:", [p.name for p in deps_wheels[:5]])
if lib_wheels[:5]:
    print("lib sample:", [p.name for p in lib_wheels[:5]])

if not deps_wheels and not lib_wheels:
    raise FileNotFoundError(
        "No .whl files found under /kaggle/input (expected pekolib wheels). "
        "Please ensure the pekolib/pekolib-deps datasets are attached."
    )

if deps_wheels:
    run("pip -q install " + " ".join([str(p) for p in deps_wheels]), check=True)
if lib_wheels:
    run("pip -q install " + " ".join([str(p) for p in lib_wheels]), check=True)

inplace_src_candidates = sorted(deps_root.rglob("inplace_abn-*/inplace_abn"))
if inplace_src_candidates:
    inplace_src = inplace_src_candidates[0].parent  # .../inplace_abn-1.0.12
    run(
        f"cp -r {inplace_src}/inplace_abn /tmp/inplace_abn && cd /tmp/inplace_abn && pip -q install .",
        check=True,
    )
else:
    print(
        "Warning: inplace_abn source folder not found; continuing (it may already be installed via wheel)."
    )

run("python -m peko.subs.hotelid.v13", check=True)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2232925072.py in <cell line: 0>()
     41 
     42 if not deps_wheels and not lib_wheels:
---> 43     raise FileNotFoundError(
     44         "No .whl files found under /kaggle/input (expected pekolib wheels). "
     45         "Please ensure the pekolib/pekolib-deps datasets are attached."

FileNotFoundError: No .whl files found under /kaggle/input (expected pekolib wheels). Please ensure the pekolib/pekolib-deps datasets are attached.

## === cell 2
import pandas as pd
import shutil
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT_SAMPLE = Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
if not INPUT_SAMPLE.exists():
    INPUT_SAMPLE = Path("/kaggle/input/sample_submission.csv")


def looks_like_submission(csv_path: Path) -> bool:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if list(df.columns) != ["image", "hotel_id"]:
        return False
    if df.shape[0] < 1:
        return False
    return True


candidate_paths = []
preferred = WORKING / "submission.csv"
if preferred.exists():
    candidate_paths.append(preferred)

search_roots = [
    WORKING,
    Path("/kaggle"),
    Path("/tmp"),
]
for root in search_roots:
    for p in root.rglob("*.csv"):
        if str(p).startswith("/kaggle/input"):
            continue
        candidate_paths.append(p)

seen = set()
candidate_paths_unique = []
for p in candidate_paths:
    ps = str(p)
    if ps not in seen:
        seen.add(ps)
        candidate_paths_unique.append(p)

valid_candidates = [
    p for p in candidate_paths_unique if p.exists() and looks_like_submission(p)
]

if not valid_candidates:
    raise FileNotFoundError(
        "No valid submission CSV found after running `python -m peko.subs.hotelid.v13`.\n"
        "Searched /kaggle/working, /kaggle (excluding /kaggle/input), and /tmp for *.csv with columns [image, hotel_id].\n"
        "If the module writes to a custom path, update the search_roots or copy step accordingly."
    )

valid_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
best = valid_candidates[0]

dst = WORKING / "submission.csv"
if best.resolve() != dst.resolve():
    shutil.copy2(best, dst)

sample = pd.read_csv(INPUT_SAMPLE)
sub = pd.read_csv(dst)

if list(sub.columns) != ["image", "hotel_id"]:
    raise ValueError(f"submission.csv has wrong columns: {list(sub.columns)}")

if sub.shape[0] != sample.shape[0]:
    raise ValueError(
        f"submission.csv row count {sub.shape[0]} != sample_submission row count {sample.shape[0]}"
    )

if not sub["image"].equals(sample["image"]):
    if set(sub["image"]) == set(sample["image"]):
        sub = sub.set_index("image").loc[sample["image"]].reset_index()
        sub.to_csv(dst, index=False)
    else:
        raise ValueError(
            "submission.csv images do not match sample_submission images (set mismatch)."
        )

print(f"Using submission file: {dst} (copied from: {best})")
print(sub.head(3).to_string(index=False))
print("\nsubmission.csv written to /kaggle/working/submission.csv")
