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

0.7433042855940036

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.00209) has done: 'Your current notebook doesn’t yield a score because it likely fails before producing `submission.csv` (missing private wheel inputs, compilation of `inplace_abn`, or the `peko` module not being available), so Kaggle never gets a valid submission to evaluate. To move toward the target score, the smallest meaningful change is to (1) stop depending on unavailable external wheels and (2) add a robust fallback that always creates a valid `submission.csv` in the required format using only the provided competition files. This let you submit and obtain a baseline score; once you report that score, we can make the smallest next-step adjustment toward the target. The fallback is metric-safe (MAP@5 expects 5 IDs) and uses the most frequent hotel IDs from `train.csv` as reasonable priors.'

# 9. Code solution

## === cell 0
import os
import subprocess
from pathlib import Path
import pandas as pd


def run(cmd):
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=False, text=True, capture_output=True)


DATA_CANDIDATES = [
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("/kaggle/data/hotel-id-2021-fgvc8"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]
DATA_DIR = None
for p in DATA_CANDIDATES:
    if (p / "sample_submission.csv").exists() and (p / "train.csv").exists():
        DATA_DIR = p
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv and train.csv in expected Kaggle paths."
    )

print("Using DATA_DIR:", DATA_DIR)

peko_deps = Path("/kaggle/input/pekolib-deps")
peko_lib = Path("/kaggle/input/pekolib")
can_try_peko = peko_deps.exists() and peko_lib.exists()

submission_path = Path("submission.csv")

if can_try_peko:
    r1 = run("ls -lha /kaggle/input/")
    r2 = run(
        "pip -q install /kaggle/input/pekolib-deps/*.whl && pip -q install /kaggle/input/pekolib/*.whl"
    )
    r3 = run(
        "cp -r /kaggle/input/pekolib-deps/inplace_abn-1.0.12/inplace_abn /tmp/inplace_abn && cd /tmp/inplace_abn && pip -q install ."
    )
    r4 = run("python -m peko.subs.hotelid.v9")

    if submission_path.exists():
        print("Found submission.csv generated by peko pipeline.")
    else:
        print(
            "peko pipeline did not produce submission.csv; will generate fallback submission."
        )
else:
    print("pekolib inputs not found; will generate fallback submission.")

if not submission_path.exists():
    train_csv = pd.read_csv(DATA_DIR / "train.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    top5 = train_csv["hotel_id"].value_counts().head(5).index.astype(str).tolist()
    if len(top5) < 5:
        top5 = (top5 + ["0"] * 5)[:5]

    pred_str = " ".join(top5)
    submission = sample_sub.copy()
    submission["hotel_id"] = pred_str
    submission.to_csv(submission_path, index=False)
    print("Wrote fallback submission.csv with top-5 frequent hotel_ids:", pred_str)

sub = pd.read_csv(submission_path)
assert list(sub.columns) == [
    "image",
    "hotel_id",
], f"Bad columns: {sub.columns.tolist()}"
assert len(sub) > 0, "Empty submission"
assert sub["hotel_id"].astype(str).str.split().map(len).min() >= 1, "Empty predictions"

print("\nsubmission.csv head:")
print(sub.head())
print("\nsubmission.csv rows:", len(sub))
print("\nSaved at:", submission_path.resolve())


## === cell 1
import pandas as pd

print(pd.read_csv("submission.csv").head().to_string(index=False))
