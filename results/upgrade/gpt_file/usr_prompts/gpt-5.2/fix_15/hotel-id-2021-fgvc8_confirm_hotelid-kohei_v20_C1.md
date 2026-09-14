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
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should increase MAP@5 with the smallest changes that keep your “chain-conditional top-5 with global fallback” logic intact. The main issue is that `test_img2num` likely has very low coverage because test images are effectively stored flat (no numeric chain folders), so almost all predictions fall back to the same global top-5. I keep the exact same prediction mechanism but improve the chain inference signal by (1) building a robust `image -> chain` mapping from the train image directory structure (which *does* contain numeric chain folders), and (2) learning a simple per-chain prefix dictionary from train images (instead of relying on `train.csv` filename-prefix statistics alone). This increases the fraction of test images getting a meaningful chain prior without changing the submission semantics or introducing any new modeling.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should improve MAP@5 (higher is better) with the smallest change that makes your existing “chain-conditional top-5 with global fallback” actually condition on something that exists for test. The main reason it collapses to the global top-5 is that `infer_test_chain()` rarely finds a usable chain signal for test images. I keep your same frequency-prior logic, but strengthen chain inference by (1) scanning the *train* image folders to learn a robust `prefix -> chain` map (more reliable than `train.csv`’s chain for this purpose) and (2) relaxing thresholds and using multiple prefix lengths (12/10/8/6) so far more test images get a chain prior. This should move the score upward while keeping runtime low and still producing a valid `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should improve MAP@5 (higher is better) with the smallest changes that keep your existing “chain-conditional top-5 + global fallback” logic intact. The biggest likely bottleneck is that `infer_test_chain()` still has very low coverage on the hidden test set, so most rows fall back to the same global top-5. I (1) make chain inference more robust by extracting the first numeric folder anywhere in the test image relative path (already done) but also correctly handling cases where test images are nested one level deeper, and (2) increase inferred-chain coverage by learning a stronger filename-prefix→chain mapping from the *train image directory structure* using multiple prefix lengths with slightly more permissive thresholds, while preserving the same prediction mechanism. These changes don’t alter the submission semantics (still top-5 hotels per image), but should materially increase the fraction of test images getting a chain prior and move the score upward toward your target.'
- What this solution (achieved 0.00209) has done: 'Your current 0.00209 indicates the predictions are still effectively “global top-5 for almost everyone”, meaning the chain inference is not activating on the hidden test set. Keeping your exact core logic (chain-conditional top-5 with global fallback), the minimal change is to strengthen `infer_test_chain()` by learning an additional, more reliable `image -> chain` map from the *train image folder structure* and using exact-filename matching (and, if needed, lowercase matching) to infer chain for test images—this tends to have much higher coverage than prefix heuristics when filenames overlap patterns. I also build the train-side folder map once and reuse it (no model changes), and keep all existing prefix fallbacks intact. This should increase the fraction of test images receiving a non-global chain prior, moving MAP@5 upward toward your target without changing evaluation semantics or adding dependencies.'
- What this solution (achieved 0.00209) has done: 'Your current 0.00209 suggests the chain-conditioned priors almost never activate on the hidden test set, so predictions collapse to the same global top-5. Keeping your exact “chain-conditional top-5 with global fallback” core logic, the smallest meaningful lift is to infer chain more reliably from the test images’ directory structure by building an `image -> chain` map that tolerates multiple nested layouts and uses a deterministic “best” chain when duplicates occur. I also make `chain_top5` robust to type mismatches by keying it with plain Python `int` chains, so inferred chains actually retrieve the intended prior list. These changes should increase the fraction of test rows that use a chain prior and move MAP@5 upward toward your target without changing the basic method or output format.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below target, so we should increase MAP@5 while keeping your same “chain-conditional top-5 with global fallback” baseline intact. The biggest likely issue is that the inferred “chain” often doesn’t match the `chain` values used to build `chain_top5` (train `chain` IDs are not guaranteed to be 0–99 folder indices), so `chain_top5.get(ch)` frequently misses and collapses to the same global top-5. I make a minimal, score-relevant fix by rebuilding `chain_top5` keyed by the *numeric folder chain* extracted from the training image directory (the same type of signal you infer for test), keeping everything else (pad5, inference order, CSV output) the same. I also add a small safety fallback: if a chain prior exists but has <5 unique hotels, it still pads correctly via `pad5`.'
- What this solution (achieved 0.00209) has done: 'Your current 0.00209 score is extremely far below the target (0.7449), and the most likely cause (given the logic) is that the “chain” you infer from image folders/prefixes does not correspond to the `chain` IDs used to compute per-chain hotel priors, so you almost always fall back to the same global top-5. To move the score upward while preserving your exact core approach (“conditional top-5 priors with global fallback”), I rebuild `chain_top5` keyed by the *train image folder numeric chain* (the same signal you’re trying to infer for test), and I also make `infer_test_chain()` return that same key type (with a safe `None` fallback). This is a minimal, directly score-relevant fix that should increase the fraction of test rows that use a non-global prior and improve MAP@5. The submission writing and formatting remain unchanged.'

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
    os.path.join(BASE1, "hotel-id-2021-fgvc8", "test_images", "test_images"),
    os.path.join(BASE1, "hotel-id-2021-fgvc8", "test_images"),
    os.path.join(BASE1, "test_images", "test_images"),
    os.path.join(BASE1, "test_images"),
    os.path.join(BASE2, "hotel-id-2021-fgvc8", "test_images", "test_images"),
    os.path.join(BASE2, "hotel-id-2021-fgvc8", "test_images"),
    os.path.join(BASE2, "test_images", "test_images"),
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
    """
    Scan root recursively and map each image filename -> first numeric folder in its relative path.

    Duplicate filenames: pick the most frequent numeric folder (and if tied, smallest) deterministically.
    """
    counts = {}  # img -> {num: cnt}
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
                d = counts.setdefault(fn, {})
                d[num] = d.get(num, 0) + 1

    img2num = {}
    for fn, d in counts.items():
        best_num = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        img2num[fn] = best_num
    return img2num


def build_prefix_to_chain_map_from_image_root(
    train_df: pd.DataFrame,
    train_images_root: str,
    prefix_len: int = 8,
    min_count: int = 3,
    min_frac: float = 0.52,
):
    """
    Learn prefix->numeric-folder-chain from TRAIN IMAGE FOLDER structure (strong signal).
    """
    img2chain = build_image_to_numeric_folder_map(train_images_root)
    tmp = train_df[["image"]].copy()
    tmp["image"] = tmp["image"].astype(str)
    tmp["chain_from_path"] = tmp["image"].map(img2chain)
    tmp = tmp.dropna(subset=["chain_from_path"])
    tmp["chain_from_path"] = tmp["chain_from_path"].astype(int)
    tmp["pref"] = tmp["image"].str.slice(0, prefix_len)

    vc = tmp.groupby("pref")["chain_from_path"].value_counts()
    vc = vc.rename("cnt").reset_index()
    totals = vc.groupby("pref")["cnt"].sum().rename("tot").reset_index()
    best = vc.sort_values(["pref", "cnt"], ascending=[True, False]).drop_duplicates(
        "pref", keep="first"
    )
    best = best.merge(totals, on="pref", how="left")
    best["frac"] = best["cnt"] / best["tot"]
    best = best[(best["tot"] >= min_count) & (best["frac"] >= min_frac)]
    return dict(
        zip(best["pref"].tolist(), best["chain_from_path"].astype(int).tolist())
    )


def build_prefix_to_chain_map(
    train_df: pd.DataFrame,
    prefix_len: int = 8,
    min_count: int = 3,
    min_frac: float = 0.52,
):
    """
    Keep CSV-derived prefix->chain as a weaker backstop (may not match numeric-folder chains).
    """
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


train["image"] = train["image"].astype(str)
train["hotel_id"] = train["hotel_id"].astype(str)

global_top5 = train["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + [global_top5[-1]] * 5)[:5]
print("global_top5:", global_top5)

train_img2num = build_image_to_numeric_folder_map(train_images_root)
train_img2num_lower = {k.lower(): v for k, v in train_img2num.items()}

test_img2num = build_image_to_numeric_folder_map(test_images_root)
print(
    "mapped test images to numeric folder:",
    len(test_img2num),
    "out of sample rows:",
    len(sample),
)
print(
    "mapped train images to numeric folder:",
    len(train_img2num),
    "out of train rows:",
    len(train),
)

train_chain_from_path = train["image"].map(train_img2num).astype("Int64")
chain_top5 = (
    train.assign(chain=train_chain_from_path)
    .dropna(subset=["chain"])
    .assign(chain=lambda df: df["chain"].astype(int))
    .groupby("chain")["hotel_id"]
    .apply(lambda s: s.value_counts().head(5).index.tolist())
    .to_dict()
)
print("num numeric-folder chain priors:", len(chain_top5))

prefix2chain_path_16 = build_prefix_to_chain_map_from_image_root(
    train,
    train_images_root=train_images_root,
    prefix_len=16,
    min_count=2,
    min_frac=0.55,
)
prefix2chain_path_14 = build_prefix_to_chain_map_from_image_root(
    train,
    train_images_root=train_images_root,
    prefix_len=14,
    min_count=2,
    min_frac=0.55,
)
prefix2chain_path_12 = build_prefix_to_chain_map_from_image_root(
    train,
    train_images_root=train_images_root,
    prefix_len=12,
    min_count=3,
    min_frac=0.52,
)
prefix2chain_path_10 = build_prefix_to_chain_map_from_image_root(
    train,
    train_images_root=train_images_root,
    prefix_len=10,
    min_count=3,
    min_frac=0.52,
)
prefix2chain_path_8 = build_prefix_to_chain_map_from_image_root(
    train, train_images_root=train_images_root, prefix_len=8, min_count=5, min_frac=0.52
)
prefix2chain_path_6 = build_prefix_to_chain_map_from_image_root(
    train, train_images_root=train_images_root, prefix_len=6, min_count=8, min_frac=0.58
)

prefix2chain_16 = build_prefix_to_chain_map(
    train, prefix_len=16, min_count=2, min_frac=0.55
)
prefix2chain_14 = build_prefix_to_chain_map(
    train, prefix_len=14, min_count=2, min_frac=0.55
)
prefix2chain_12 = build_prefix_to_chain_map(
    train, prefix_len=12, min_count=3, min_frac=0.52
)
prefix2chain_10 = build_prefix_to_chain_map(
    train, prefix_len=10, min_count=3, min_frac=0.52
)
prefix2chain_8 = build_prefix_to_chain_map(
    train, prefix_len=8, min_count=5, min_frac=0.52
)
prefix2chain_6 = build_prefix_to_chain_map(
    train, prefix_len=6, min_count=8, min_frac=0.58
)

print(
    "prefix2chain_path sizes:",
    {
        16: len(prefix2chain_path_16),
        14: len(prefix2chain_path_14),
        12: len(prefix2chain_path_12),
        10: len(prefix2chain_path_10),
        8: len(prefix2chain_path_8),
        6: len(prefix2chain_path_6),
    },
)
print(
    "prefix2chain_csv sizes:",
    {
        16: len(prefix2chain_16),
        14: len(prefix2chain_14),
        12: len(prefix2chain_12),
        10: len(prefix2chain_10),
        8: len(prefix2chain_8),
        6: len(prefix2chain_6),
    },
)


def infer_test_chain(img_name: str):
    num = test_img2num.get(img_name)
    if num is not None:
        return int(num)

    num = train_img2num.get(img_name)
    if num is not None:
        return int(num)
    num = train_img2num_lower.get(str(img_name).lower())
    if num is not None:
        return int(num)

    s = str(img_name)
    for L, m in (
        (16, prefix2chain_path_16),
        (14, prefix2chain_path_14),
        (12, prefix2chain_path_12),
        (10, prefix2chain_path_10),
        (8, prefix2chain_path_8),
        (6, prefix2chain_path_6),
    ):
        ch = m.get(s[:L])
        if ch is not None:
            return int(ch)

    for L, m in (
        (16, prefix2chain_16),
        (14, prefix2chain_14),
        (12, prefix2chain_12),
        (10, prefix2chain_10),
        (8, prefix2chain_8),
        (6, prefix2chain_6),
    ):
        ch = m.get(s[:L])
        if ch is not None:
            return int(ch)

    return None




## === cell 2
preds = []
used_chain = 0

for img in sample["image"].astype(str).tolist():
    ch = infer_test_chain(img)
    if ch is not None:
        used_chain += 1
    top = chain_top5.get(ch, None) if ch is not None else None
    top5 = pad5(top, global_top5)
    preds.append(" ".join(top5))

print("inferred chain for:", used_chain, "of", len(sample), "test images")

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
