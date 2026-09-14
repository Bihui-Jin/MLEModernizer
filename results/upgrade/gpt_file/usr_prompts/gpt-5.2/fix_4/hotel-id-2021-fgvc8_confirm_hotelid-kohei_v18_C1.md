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

0.7449797928769869

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently cannot yield a Kaggle score because it depends on private Kaggle Dataset wheels (`/kaggle/input/pekolib*`) that are not present in your environment, so it fails before producing a valid `submission.csv`. I keep the “generate a submission.csv from the provided data” core goal, but replace the missing-package calls with a minimal, fully self-contained baseline that always runs using only the provided `train.csv` + `sample_submission.csv`. To move score upward from “not yielded” toward your target, the baseline predict the global top-5 most frequent `hotel_id` values from the training metadata (a common safe baseline for MAP@5) and write the correctly formatted submission file. This keeps runtime well under 600s and ensures the submission rows align exactly with `sample_submission.csv`.'
- What this solution (achieved 0.00209) has done: 'The timeout is dominated by `infer_chain_from_fs`, which for every test image repeatedly scans all subdirectories under `test_images_root` and does `os.path.exists` checks—an O(N_images × N_dirs) filesystem walk. We can preserve identical prediction logic by building a one-time mapping from `image_name -> chain` by scanning the directory tree once, then doing O(1) lookups per image. We also speed up the `chain_top5` computation by replacing the Python `apply(lambda ...)` with a vectorized `groupby().head(5)` pattern that produces the same top-5-per-chain list. All outputs/paths and fallback behavior remain the same.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str):
    for base in DATA_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in DATA_CANDIDATES:
        path = os.path.join(base, "hotel-id-2021-fgvc8", filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {DATA_CANDIDATES}"
    )


train_path = find_file("train.csv")
sub_path = find_file("sample_submission.csv")

print("Using train:", train_path)
print("Using sample_submission:", sub_path)

train = pd.read_csv(train_path)
sample_sub = pd.read_csv(sub_path)

train.head(), sample_sub.head(), train.shape, sample_sub.shape



## === cell 1
global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
global_pred_str = " ".join(global_top5)
print("Global top-5 fallback:", global_pred_str)

_chain_hotel_counts = (
    train.groupby(["chain", "hotel_id"], sort=False).size().rename("cnt").reset_index()
)
_chain_hotel_counts = _chain_hotel_counts.sort_values(
    ["chain", "cnt"], ascending=[True, False], kind="mergesort"
)
_chain_top = _chain_hotel_counts.groupby("chain", sort=False).head(5)
chain_top5 = (
    _chain_top.groupby("chain", sort=False)["hotel_id"]
    .apply(lambda s: s.astype(str).tolist())
    .to_dict()
)

TEST_IMAGES_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
    "/kaggle/input/hotel-id-2021-fgvc8/test_images/test_images",
    "/kaggle/data/hotel-id-2021-fgvc8/test_images/test_images",
    "/kaggle/input/test_images/test_images",
    "/kaggle/data/test_images/test_images",
]
test_images_root = None
for p in TEST_IMAGES_CANDIDATES:
    if os.path.isdir(p):
        test_images_root = p
        break
if test_images_root is None:
    print(
        "WARNING: Could not find test_images directory; will use global fallback for all rows."
    )
else:
    print("Using test_images root:", test_images_root)

_image_to_chain = None
if test_images_root is not None:
    _image_to_chain = {}
    try:
        with os.scandir(test_images_root) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                dname = entry.name
                try:
                    ch_int = int(dname)
                except Exception:
                    continue
                try:
                    with os.scandir(entry.path) as it2:
                        for f in it2:
                            if f.is_file():
                                _image_to_chain[f.name] = ch_int
                except Exception:
                    continue
    except Exception:
        _image_to_chain = {}


def infer_chain_from_fs(image_name: str):
    """
    Infer chain from the filesystem. Expected layout: test_images/<chain>/<image>.jpg
    Returns int chain or None if not found.
    """
    if _image_to_chain is None:
        return None
    return _image_to_chain.get(image_name)


preds = []
n_chain_found = 0
for img in sample_sub["image"].astype(str).tolist():
    ch = infer_chain_from_fs(img)
    if ch is not None:
        top = chain_top5.get(ch)
        if top is not None and len(top) > 0:
            n_chain_found += 1
            combined = []
            for x in top + global_top5:
                xs = str(x)
                if xs not in combined:
                    combined.append(xs)
                if len(combined) == 5:
                    break
            if len(combined) < 5:
                combined = (combined + global_top5)[:5]
            preds.append(" ".join(combined))
            continue
    preds.append(global_pred_str)

print(
    f"Inferred chain for {n_chain_found}/{len(sample_sub)} test images (others used global fallback)."
)

submission = sample_sub.copy()
submission["hotel_id"] = preds

assert list(submission.columns) == ["image", "hotel_id"]
assert len(submission) == len(sample_sub)
assert submission["hotel_id"].astype(str).str.split().map(len).eq(5).all()

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())



## === cell 2
import subprocess, shlex

print(subprocess.check_output(shlex.split("ls -lha submission.csv")).decode("utf-8"))
print(subprocess.check_output(shlex.split("head -n 5 submission.csv")).decode("utf-8"))
