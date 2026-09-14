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

0.7687168476888083

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on external wheels/modules (`pekolib`) that aren’t present in the provided filesystem, so it won’t reliably run end-to-end or create `submission.csv`. I replace that dependency with a minimal, self-contained baseline that always produces a valid `submission.csv` in the required `image,hotel_id` format using only the provided `train.csv` and `sample_submission.csv`. To move the score upward (vs. a blank/failed run) while keeping logic simple and within time, I predict the global top-5 most frequent `hotel_id` from the training data for every test image. This won’t hit the target MAP@5, but it generate a valid submission and produce a non-zero baseline score, which is the smallest legitimate step toward the target given the current “no submission” state.'
- What this solution (achieved 0.00209) has done: 'Your current baseline predicts the same global top-5 hotels for every image, which is valid but far from the target MAP@5. To move the score upward with minimal logic change and without using images/models, I generate *chain-aware* top-5 candidates by leveraging the `chain` field in `train.csv` and the fact that test images are stored in chain subfolders (the parent directory name). This keeps the approach as “frequency-based top-5” but makes it conditional on the inferred chain, which should materially increase MAP@5 vs the global prior while remaining fast and deterministic. I also keep a safe fallback to the global top-5 when the chain cannot be inferred or has no stats.'
- What this solution (achieved 0.00209) has done: 'Your current approach is valid but likely underperforms because many test images won’t have a numeric parent folder (or won’t match your `fn`-only mapping), causing frequent fallback to the global top-5. I make a minimal, score-relevant fix: build the test image “chain” mapping using the *relative path* (to robustly detect chain folders) and also match both `image` and `test_images/<chain>/<image>` naming patterns. I also add a tiny amount of smoothing by using global priors only when chain has too few unique hotels, otherwise keep chain top-5 unchanged—this preserves the same frequency-based core logic while improving recall. Output format and paths remain the same and it still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.00209) has done: 'Your current score is low because most test images likely don’t get a usable inferred `chain` from the directory structure, so you fall back to a weak global top-5 too often. I keep the same frequency-based “chain-aware top-5” core logic, but make the chain inference more robust by (1) also extracting chain from the immediate parent folder name (common layout), and (2) building the mapping keyed by the exact `image` strings in `sample_submission.csv` (so we match whether Kaggle provides bare filenames or paths). I also make the “chain too small” fallback less aggressive (only fall back when the chain has no candidates), which should increase MAP@5 while still being deterministic and fast. The output remains a valid `submission.csv` with `image,hotel_id` and exactly 5 IDs per row.'
- What this solution (achieved 0.00209) has done: 'Your current logic is sound but likely underperforms because many test images won’t get a chain assigned due to directory layout differences (e.g., `test_images/test_images/<chain>/<image>.jpg`), and because `os.walk` may traverse nested folders where the first numeric folder isn’t found reliably. I keep the exact same “frequency-based top-5, chain-aware with global fallback” core approach, but make chain inference deterministic and more robust by explicitly globbing for image files and extracting the nearest numeric parent chain id. I also add a safe “mixing” fallback that appends missing items from the global top-5 (instead of fully replacing with global), which preserves chain specificity while ensuring always-5 unique-ish candidates. This should increase MAP@5 materially from ~0.002 by reducing unnecessary global fallback without changing the fundamental model-free method.'
- What this solution (achieved 0.00209) has done: 'Your current frequency-based chain-aware baseline is valid but likely missing many chain matches because it only maps by basename and depends on globbing patterns that may not cover the nested `test_images/test_images/...` layout reliably. I keep the exact same core logic (per-chain top-5 with a global fallback) but make chain inference deterministic and higher-recall by scanning the actual test image files with `os.walk`, extracting the nearest numeric parent chain id, and building an `image -> chain` map keyed by the exact `image` strings from `sample_submission.csv` (plus basename fallback). I also make the fallback slightly less lossy by always “merge-filling” from global top-5 (instead of fully switching to global) when a chain exists but has no top-5 entry. These are minimal changes intended to materially increase MAP@5 from ~0.002 by reducing unnecessary global-only predictions, without changing the modeling approach.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should increase it with the smallest legitimate change that preserves your frequency-based, chain-aware top-5 logic. The biggest likely issue is that you’re inferring `chain` only from numeric parent folders, but the provided `test_images` layout often has no chain folders at all, making your chain-aware logic rarely apply and collapsing to a weak global prior. I keep the same approach but add a second, metadata-driven candidate source: a per-`hotel_id` co-occurrence prior conditioned on the chain (computed from `train.csv`), then merge-fill with chain-top5 and global-top5 to always produce 5 IDs. This is still “count-based top-5 retrieval” (no image model), but typically improves MAP@5 a lot versus global-only when chain inference is missing.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for base in BASE_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
        path2 = os.path.join(base, "hotel-id-2021-fgvcvc8", filename)
        if os.path.exists(path2):
            return path2
        path3 = os.path.join(base, "hotel-id-2021-fgvc8", filename)
        if os.path.exists(path3):
            return path3
    raise FileNotFoundError(
        f"Could not find {filename} under candidates: {BASE_CANDIDATES}"
    )


def find_dir(dirname: str) -> str:
    for base in BASE_CANDIDATES:
        p1 = os.path.join(base, dirname)
        if os.path.isdir(p1):
            return p1
        p2 = os.path.join(base, "hotel-id-2021-fgvc8", dirname)
        if os.path.isdir(p2):
            return p2
    raise FileNotFoundError(
        f"Could not find directory {dirname} under candidates: {BASE_CANDIDATES}"
    )


def top5_strict_from_counts(series: pd.Series) -> list:
    top = series.value_counts().head(5).index.astype(str).tolist()
    if len(top) == 0:
        return []
    while len(top) < 5:
        top.append(top[-1])
    return top[:5]


def merge_top5(primary: list, fallback: list) -> list:
    out = []
    for x in primary:
        xs = str(x)
        if xs not in out:
            out.append(xs)
        if len(out) == 5:
            return out
    for x in fallback:
        xs = str(x)
        if xs not in out:
            out.append(xs)
        if len(out) == 5:
            return out
    while len(out) < 5 and len(out) > 0:
        out.append(out[-1])
    return out[:5]


train_csv = find_file("train.csv")
sample_sub_csv = find_file("sample_submission.csv")

print("Using train.csv:", train_csv)
print("Using sample_submission.csv:", sample_sub_csv)

train = pd.read_csv(train_csv)
sub = pd.read_csv(sample_sub_csv)

train["hotel_id"] = train["hotel_id"].astype(str)
train["chain"] = train["chain"].astype(int)

global_top5 = top5_strict_from_counts(train["hotel_id"])
if len(global_top5) == 0:
    raise RuntimeError("No hotel_id found in training data.")

chain_top5 = {}
vc = train.groupby("chain")["hotel_id"].value_counts()
for chain_id in vc.index.get_level_values(0).unique().tolist():
    top = vc.loc[chain_id].head(5).index.astype(str).tolist()
    if len(top) == 0:
        continue
    while len(top) < 5:
        top.append(top[-1])
    chain_top5[int(chain_id)] = top[:5]

chain_hotelid_to_top5_other = {}
chain_hotel_counts = (
    train.groupby(["chain", "hotel_id"]).size().rename("cnt").reset_index()
)
for ch, g in chain_hotel_counts.groupby("chain", sort=False):
    ordered = g.sort_values("cnt", ascending=False)["hotel_id"].astype(str).tolist()
    ordered_set = ordered  # already ordered
    for hid in ordered_set:
        others = [x for x in ordered if x != hid][:5]
        if len(others) > 0:
            while len(others) < 5:
                others.append(others[-1])
            chain_hotelid_to_top5_other[(int(ch), str(hid))] = others[:5]

test_images_dir = find_dir("test_images")
print("Using test_images dir:", test_images_dir)

sub_images = sub["image"].astype(str).tolist()
sub_image_set = set(sub_images)
sub_basename_set = set(os.path.basename(x) for x in sub_images)


def infer_chain_from_path(p: str):
    d = os.path.dirname(p)
    while True:
        base = os.path.basename(d)
        if base.isdigit():
            return int(base)
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


img2chain_by_exact = {}
img2chain_by_basename = {}

hit_exact = 0
hit_base = 0
walk_count = 0

for root, _, files in os.walk(test_images_dir):
    for fn in files:
        if not fn.lower().endswith(".jpg"):
            continue
        walk_count += 1

        if fn not in sub_basename_set:
            continue

        full_path = os.path.join(root, fn)
        ch = infer_chain_from_path(full_path)
        if ch is None:
            continue

        if fn not in img2chain_by_basename:
            img2chain_by_basename[fn] = ch
            hit_base += 1

        rel = os.path.relpath(full_path, test_images_dir).replace("\\", "/")
        candidates = [
            rel,
            f"test_images/{rel}",
        ]
        for key in candidates:
            if key in sub_image_set and key not in img2chain_by_exact:
                img2chain_by_exact[key] = ch
                hit_exact += 1

print("os.walk scanned jpg files:", walk_count)
print(
    "Inferred chain for",
    hit_base,
    "unique test basenames out of",
    len(sub_basename_set),
)
print(
    "Inferred chain for",
    hit_exact,
    "exact submission image strings out of",
    len(sub_image_set),
)

sub = sub[["image"]].copy()

preds = []
for img in sub["image"].astype(str).tolist():
    ch = img2chain_by_exact.get(img, None)
    if ch is None:
        base = os.path.basename(img)
        ch = img2chain_by_basename.get(base, None)

    if ch is not None and ch in chain_top5:
        anchors = chain_top5[ch]
        mixed = []
        mixed = merge_top5(anchors, [])
        for a in anchors:
            others = chain_hotelid_to_top5_other.get((int(ch), str(a)), None)
            if others is not None:
                mixed = merge_top5(mixed, others)
        top5 = merge_top5(mixed, global_top5)
    else:
        top5 = global_top5

    preds.append(" ".join(top5))

sub["hotel_id"] = preds

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", list(sub.columns))




## === cell 1
import pandas as pd

chk = pd.read_csv("submission.csv")
assert list(chk.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be: image, hotel_id"
assert chk["image"].notnull().all(), "All images must be present"
assert chk["hotel_id"].notnull().all(), "All predictions must be present"
assert (
    chk["hotel_id"].astype(str).str.split().str.len() == 5
).all(), "Each prediction must contain 5 space-delimited hotel_ids"
print("submission.csv looks valid.")
print(chk.head())
