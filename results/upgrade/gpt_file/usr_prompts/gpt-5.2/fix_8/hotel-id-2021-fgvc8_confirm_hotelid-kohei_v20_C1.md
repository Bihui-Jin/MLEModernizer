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

0.7449776879683393

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a Kaggle score because it relies on private `pekolib` wheels and a `peko.subs.hotelid.v20` entrypoint that won’t be present/allowed in a standard environment; so no valid, reproducible `submission.csv` is guaranteed. To make it run end-to-end and produce a valid submission CSV with the required columns, I’m replacing those external dependencies with a minimal baseline that uses only the provided `train.csv`/`sample_submission.csv` and generates MAP@5-compatible predictions (top-5 hotel IDs). This won’t hit the 0.7449 target (that score typically requires image embeddings + retrieval), but it produce a valid submission so you can get a measurable score and iterate from there. The core “predict 5 hotel_ids per image” evaluation semantics are preserved, and the output format match Kaggle exactly.'
- What this solution (achieved 0.00209) has done: 'Your current 0.00209 score is consistent with predicting the exact same global top-5 hotels for every test image, which ignores any information correlated with the test distribution. To move toward your 0.7449 target without changing the “predict 5 hotel_ids per image” core logic, the smallest legitimate improvement is to make the top-5 list *conditional on chain*, because chain is a strong prior in this dataset. Since chain is not provided for test, we infer a plausible chain prior per test image by matching the sample/test image’s chain folder (0–99) from the `test_images/` directory structure, then predict the most frequent 5 hotels within that chain (falling back to global top-5 when needed). This keeps the approach simple (frequency baseline), preserves evaluation semantics, and typically yields a substantial lift versus a single global list.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, and the biggest issue is that the chain inference is effectively never used because `test_images/` is not organized into `0..99/` subfolders in this dataset layout; your `infer_chain_for_image` therefore returns `None` almost always, collapsing predictions back to the same global top-5 list. I fix chain inference to use the actual path structure by scanning the `test_images_root` recursively once and building an `image -> chain` map from the parent directory name when it’s numeric (and otherwise leaving it unknown). This keeps the same core logic (frequency top-5 per chain with global fallback) but makes it actually conditional when possible, which should move MAP@5 upward toward your target. I also make the path selection slightly more robust by preferring the dataset’s nested `.../hotel-id-2021-fgvc8/test_images` variant if present.'
- What this solution (achieved 0.00209) has done: 'Your current approach collapses to a near-constant prediction for most/all test images because the test images aren’t actually organized into numeric chain subfolders, so the chain-conditional priors rarely activate and MAP@5 stays near random. Keeping the same core logic (frequency-based top-5, with optional chain conditioning), I change the “conditioning” signal to something that exists for both train and test: the test image’s parent folder name under `test_images/` (and similarly for train images). Concretely, we build a `image -> folder_key` map for both train and test by scanning the image directories, compute top-5 hotels per `folder_key` from train, and use that for test with global fallback. This is a minimal, legitimate improvement that should move the score substantially upward toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score indicates the folder-key conditioning is not helping much, likely because the train/test directory parent folder names don’t align meaningfully (or the mapping coverage is weak), so predictions collapse toward the global top-5. To move the score upward with minimal logic change, I keep your “conditional top-5 priors with global fallback” approach but switch the conditioning key to the provided `chain` in `train.csv`, which is a much stronger and stable prior, and then infer a proxy `chain` for each test image by reading the immediate parent directory if it’s numeric. When chain can’t be inferred, we fall back to the global top-5 exactly as before. This preserves the same frequency-baseline semantics (no model/embeddings) while making the conditional prior far more likely to be useful, which should lift MAP@5 toward your target.'
- What this solution (achieved 0.00209) has done: 'Your current baseline is effectively falling back to the same global top-5 for almost all test images, because `infer_test_chain()` almost never finds a numeric parent folder in `test_images_root`. To move the score upward toward the target while preserving the same “frequency top-5 with conditional prior + global fallback” core logic, I infer a usable per-test-image “chain prior” by training a tiny filename-prefix→chain mapping on the training set (many images share a consistent prefix) and applying it to test images. When the prefix mapping can’t infer a chain, we keep the exact same global fallback behavior as before. This is a minimal change (only the chain inference signal changes) and should increase MAP@5 compared to the near-constant predictions.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target, so we should improve (higher is better) with minimal changes while keeping the same “frequency top-5 with conditional prior + global fallback” core logic. The biggest likely issue is that `infer_test_chain()` almost always returns `None` because test images typically aren’t stored in numeric chain folders; your prefix→chain map is also too strict (min_count/min_frac) and may have low coverage. I keep your exact pipeline but (1) build a robust `test image -> chain` map by scanning the test image directory and extracting the first numeric directory in the relative path (not just the immediate parent), and (2) relax prefix mapping thresholds slightly and add a small multi-length fallback (8 then 6) to increase inferred-chain coverage. These are small, legitimate inference-signal fixes that should move MAP@5 upward toward your target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE1 = "/kaggle/input/hotel-id-2021-fgvc8"
BASE2 = "/kaggle/input"


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the paths exist: {paths}")


train_csv = first_existing(
    os.path.join(BASE1, "train.csv"),
    os.path.join(BASE2, "train.csv"),
)

sample_csv = first_existing(
    os.path.join(BASE1, "sample_submission.csv"),
    os.path.join(BASE2, "sample_submission.csv"),
)

train_images_root = first_existing(
    os.path.join(BASE1, "hotel-id-2021-fgvc8", "train_images"),
    os.path.join(BASE1, "train_images"),
    os.path.join(BASE2, "hotel-id-2021-fgvc8", "train_images"),
    os.path.join(BASE2, "train_images"),
)

test_images_root = first_existing(
    os.path.join(BASE1, "hotel-id-2021-fgvc8", "test_images"),
    os.path.join(BASE1, "test_images"),
    os.path.join(BASE2, "hotel-id-2021-fgvc8", "test_images"),
    os.path.join(BASE2, "test_images"),
)

print("train_csv:", train_csv)
print("sample_csv:", sample_csv)
print("train_images_root:", train_images_root)
print("test_images_root:", test_images_root)

train = pd.read_csv(train_csv)
sample = pd.read_csv(sample_csv)

assert {"image", "hotel_id"}.issubset(sample.columns), sample.columns
assert {"image", "hotel_id"}.issubset(train.columns), train.columns
assert "chain" in train.columns, train.columns

print("train shape:", train.shape)
print("sample shape:", sample.shape)




## === cell 1
def pad5(lst, fallback):
    lst = list(lst) if lst is not None else []
    out = []
    seen = set()
    for x in lst:
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) == 5:
            return out
    for x in fallback:
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) == 5:
            return out
    if not out:
        out = ["0"]
    out = (out + [out[-1]] * 5)[:5]
    return out


def build_image_to_numeric_folder_map(root: str):
    img2num = {}
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        if not filenames:
            continue
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel in (".", "") else rel.split(os.sep)
        num = None
        for part in parts:
            if part.isdigit():
                num = int(part)
                break
        if num is None:
            continue
        for fn in filenames:
            if fn.lower().endswith(exts):
                img2num.setdefault(fn, num)
    return img2num


train["hotel_id"] = train["hotel_id"].astype(str)

global_top5 = train["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + [global_top5[-1]] * 5)[:5]
print("global_top5:", global_top5)

train_chain = train["chain"].astype("Int64")
chain_top5 = (
    train.assign(chain=train_chain)
    .dropna(subset=["chain"])
    .groupby("chain")["hotel_id"]
    .apply(lambda s: s.value_counts().head(5).index.tolist())
    .to_dict()
)
print("num chain priors:", len(chain_top5))

test_img2num = build_image_to_numeric_folder_map(test_images_root)
print(
    "mapped test images to numeric folder:",
    len(test_img2num),
    "out of sample rows:",
    len(sample),
)


def build_prefix_to_chain_map(
    train_df: pd.DataFrame,
    prefix_len: int = 8,
    min_count: int = 10,
    min_frac: float = 0.60,
):
    tmp = train_df[["image", "chain"]].copy()
    tmp["image"] = tmp["image"].astype(str)
    tmp["chain"] = tmp["chain"].astype("Int64")
    tmp = tmp.dropna(subset=["chain"])
    tmp["pref"] = tmp["image"].str.slice(0, prefix_len)

    vc = tmp.groupby("pref")["chain"].value_counts(dropna=False)
    vc = vc.rename("cnt").reset_index()
    totals = vc.groupby("pref")["cnt"].sum().rename("tot").reset_index()
    best = vc.sort_values(["pref", "cnt"], ascending=[True, False]).drop_duplicates(
        "pref", keep="first"
    )
    best = best.merge(totals, on="pref", how="left")
    best["frac"] = best["cnt"] / best["tot"]

    best = best[(best["tot"] >= min_count) & (best["frac"] >= min_frac)]
    return dict(zip(best["pref"].tolist(), best["chain"].astype(int).tolist()))


prefix2chain_8 = build_prefix_to_chain_map(
    train, prefix_len=8, min_count=10, min_frac=0.60
)
prefix2chain_6 = build_prefix_to_chain_map(
    train, prefix_len=6, min_count=15, min_frac=0.70
)
print("prefix2chain_8 size:", len(prefix2chain_8))
print("prefix2chain_6 size:", len(prefix2chain_6))


def infer_test_chain(img_name: str):
    num = test_img2num.get(img_name)
    if num is not None:
        return int(num)

    s = str(img_name)
    ch = prefix2chain_8.get(s[:8])
    if ch is not None:
        return int(ch)
    ch = prefix2chain_6.get(s[:6])
    return int(ch) if ch is not None else None




## === cell 2
preds = []
for img in sample["image"].astype(str).tolist():
    ch = infer_test_chain(img)
    top = chain_top5.get(ch, None) if ch is not None else None
    top5 = pad5(top, global_top5)
    preds.append(" ".join(top5))

sample["hotel_id"] = preds

submission = sample[["image", "hotel_id"]].copy()
assert len(submission) == len(sample)
assert submission["hotel_id"].astype(str).str.split().map(len).eq(5).all()

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())



## === cell 3
import pathlib

p = pathlib.Path("submission.csv")
print("exists:", p.exists(), "size:", p.stat().st_size if p.exists() else None)
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
