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

0.7444325166287762

# 6. Current score

0.0013

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a Kaggle score because it relies on external wheel packages (`pekolib` / `pekolib-deps`) that are not present in the provided environment, so the pipeline fails before producing a valid `submission.csv`. I remove those installs and replace them with a minimal, fully self-contained baseline that always runs end-to-end using only the provided `train.csv` and `sample_submission.csv`. To keep changes minimal and focused on producing a valid submission (so you can get a score and iterate toward the target), the baseline predicts the 5 most frequent `hotel_id` values from the training data for every test image. This not maximize MAP@5, but it generate a correct-format `.csv` reliably within the time limit.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should improve MAP@5 with the smallest possible change to the existing “top-5 list” core logic. Instead of predicting the same global top-5 hotels for every test image, we use the available `chain` signal in `train.csv` to build per-chain top-5 hotel lists, and (because test has no chain) infer the chain from the test image’s folder name (the test images are arranged in subfolders in this dataset dump). If a test image has no folder/chain match, we fall back to the global top-5 exactly as before. This keeps the approach extremely lightweight and deterministic, but typically boosts MAP@5 substantially versus a pure global-frequency baseline.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below the target, so we should increase MAP@5 with the smallest change that keeps your existing “top-5 prior list” logic intact. The main issue is that `image_to_chain` is almost always `None` because the test set images are not organized into chain subfolders, so the per-chain priors rarely apply and you mostly fall back to the weak global top-5. We instead infer a better prior by extracting each test image’s **chain folder from the training images filesystem** (since train images are stored under `train_images/<chain>/...`), then use your already-built `chain_to_top5` for those mapped chains, falling back to global top-5 otherwise. This is lightweight, deterministic, preserves your approach, and should move the score meaningfully toward the target.'
- What this solution (achieved 0.00209) has done: 'Your current MAP@5 is far below target, and the main reason is that almost none of the test images can be mapped to a chain, so you fall back to the weak global top-5 nearly always. I keep your exact “top-5 prior list” logic but fix the chain-mapping: infer each test image’s chain by searching for the image filename inside the `train_images/<chain>/` folders using a lightweight precomputed set-per-chain, and only fall back to global when truly unknown. This preserves the same training signal (train.csv counts) and output semantics, but makes the per-chain priors actually apply to most test images in this dataset layout, which should move the score meaningfully toward your target. I also keep paths unchanged and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.00209) has done: 'Your current score is far below target, and the main issue is that the “map test image to chain by scanning all train chain folders” loop is both extremely expensive and (for the hidden test) unlikely to map many images, so you mostly fall back to the weak global top-5. I keep your exact “global top-5 + per-chain top-5 prior” core logic, but change the chain mapping to be correct and fast by deriving `image_to_chain` directly from `train.csv` itself (it already tells us each train image’s chain), then only using it when a test image name happens to coincide with a known train image name (which is the only legitimate way for this heuristic to fire). This removes the unreliable filesystem search and ensures deterministic behavior within the time limit, while preserving the same prediction semantics and submission format. If no chain can be inferred for a test image, it still cleanly falls back to your global top-5.'
- What this solution (achieved 0.0013) has done: 'Your current MAP@5 is extremely far below the target because the predictions are essentially a global/top-chain popularity prior with almost no test-specific signal; the biggest low-risk gain while preserving your “top-5 list” core logic is to make those priors much more specific using available metadata. I keep the same pipeline structure (count-based top-5 lists, no image model) but (1) build timestamp-conditioned priors (recent global + recent per-chain) and (2) combine them with your existing priors via a small deterministic rank-merge, so predictions better match the hidden test time distribution. This is still just counting/lookup logic and keeps submission semantics identical (5 IDs, space-delimited), but should move the score meaningfully upward toward your target without heavy compute. I also keep strict fallback behavior to guarantee every row has 5 valid IDs and always write `submission.csv`.'
- What this solution (achieved 0.0013) has done: 'Your current score is far below the target, so we should increase MAP@5 with the smallest possible change while keeping the same “count-based priors + deterministic rank-merge” core logic. The biggest issue is that your chain-conditioned priors almost never fire because test images won’t match any train image name, so you mostly fall back to the weak global/recent-global lists. I keep your exact prior-building and merge semantics, but add one more equally-lightweight signal: a per-chain prior inferred directly from the test image’s parent folder name (this dataset layout includes chain folders for test images in this dump), with clean fallback to your existing behavior. This should apply to most rows, improving relevance while preserving runtime and producing the same valid `submission.csv`.'
- What this solution (achieved 0.0013) has done: 'Your current score is far below the target, and the main reason is that (on the hidden test) your chain-conditioned priors almost never activate, so you effectively submit a weak global popularity list. I keep your exact “count-based priors + deterministic rank-merge” core logic, but make the test-time chain inference more robust by also reading the `chain/<image>.jpg` parent folder from the provided `test_images/` directory via a fast glob, rather than relying on deep `os.walk` (which is slow and can miss in alternate folder layouts). This should increase the fraction of test rows using `chain_to_top5` / `recent_chain_to_top5`, moving MAP@5 upward toward your target without changing the modeling approach. I also keep the existing fallbacks and guarantees (always 5 IDs, correct columns, always writes `submission.csv`).'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd

DATA_DIR = "/kaggle/input/hotel-id-2021-fgvc8"

train_csv = os.path.join(DATA_DIR, "train.csv")
sample_csv = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_csv), f"Missing: {train_csv}"
assert os.path.exists(sample_csv), f"Missing: {sample_csv}"

train = pd.read_csv(train_csv)
sample = pd.read_csv(sample_csv)

for col in ["hotel_id", "chain", "timestamp"]:
    assert col in train.columns, f"train.csv missing required column: {col}"
for col in ["image", "hotel_id"]:
    assert (
        col in sample.columns
    ), f"sample_submission.csv missing required column: {col}"

print("train shape:", train.shape)
print("sample shape:", sample.shape)
print("train columns:", list(train.columns))
print("sample columns:", list(sample.columns))



## === cell 1
train["hotel_id"] = train["hotel_id"].astype(str)
train["chain"] = train["chain"].fillna(0).astype(int)

ts = pd.to_datetime(train["timestamp"], errors="coerce", utc=True)
train = train.assign(_ts=ts)

global_top5 = train["hotel_id"].value_counts().head(5).index.tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + global_top5 * 5)[:5]

chain_to_top5 = {}
vc_chain = train.groupby("chain")["hotel_id"].value_counts()
for chain_id in vc_chain.index.get_level_values(0).unique():
    top = vc_chain.loc[chain_id].head(5).index.tolist()
    if len(top) < 5:
        top = (top + global_top5)[:5]
    chain_to_top5[int(chain_id)] = top

if train["_ts"].notna().any():
    cutoff = train.loc[train["_ts"].notna(), "_ts"].quantile(0.80)
    recent = train.loc[train["_ts"].notna() & (train["_ts"] >= cutoff)].copy()
else:
    cutoff = None
    recent = train.iloc[0:0].copy()

if len(recent) > 0:
    recent_global_top5 = recent["hotel_id"].value_counts().head(5).index.tolist()
    if len(recent_global_top5) < 5:
        recent_global_top5 = (recent_global_top5 + global_top5)[:5]
else:
    recent_global_top5 = global_top5[:]

recent_chain_to_top5 = {}
if len(recent) > 0:
    vc_recent_chain = recent.groupby("chain")["hotel_id"].value_counts()
    for chain_id in vc_recent_chain.index.get_level_values(0).unique():
        top = vc_recent_chain.loc[chain_id].head(5).index.tolist()
        if len(top) < 5:
            pad = chain_to_top5.get(int(chain_id), []) + global_top5
            top = (top + pad)[:5]
        recent_chain_to_top5[int(chain_id)] = top

print("Global top5:", global_top5)
print("Recent global top5:", recent_global_top5, "cutoff:", cutoff)
print("Num chains with priors:", len(chain_to_top5))
print("Num chains with recent priors:", len(recent_chain_to_top5))



## === cell 2
train_image_to_chain = (
    train.loc[:, ["image", "chain"]]
    .dropna()
    .assign(image=lambda d: d["image"].astype(str))
    .drop_duplicates("image")
    .set_index("image")["chain"]
    .astype(int)
    .to_dict()
)

image_set = set(sample["image"].astype(str).tolist())
image_to_chain = {
    img: train_image_to_chain[img] for img in image_set if img in train_image_to_chain
}

print(
    "Mapped test images to chain via train.csv image lookup (count):",
    len(image_to_chain),
)
if len(image_to_chain) > 0:
    ex_k = next(iter(image_to_chain))
    print("Example mapping:", ex_k, "->", image_to_chain[ex_k])


def _build_test_image_to_chain_from_folders(
    base_dir: str, test_images: pd.Series
) -> dict:
    """
    Change (score-relevant, minimal):
    - Prefer fast globbing for common Kaggle layouts where test images are stored as:
        test_images/<chain>/<image>.jpg  (chain is a digit folder)
      This increases the chance chain-conditioned priors fire on hidden test, improving MAP@5.
    - Keep the original os.walk fallback for unusual layouts.
    """
    candidates = [
        os.path.join(base_dir, "test_images"),
        os.path.join(base_dir, "test_images", "test_images"),
        os.path.join(base_dir, "test", "test"),
        os.path.join(base_dir, "test", "test", "test"),
    ]
    test_roots = [p for p in candidates if os.path.isdir(p)]
    if not test_roots:
        print(
            "No test image directory found under DATA_DIR; folder-based chain inference disabled."
        )
        return {}

    wanted = set(test_images.astype(str).tolist())
    out = {}

    for root in test_roots:
        pattern = os.path.join(root, "*", "*.jpg")
        for fp in glob.iglob(pattern):
            fn = os.path.basename(fp)
            if fn not in wanted or fn in out:
                continue
            ch = os.path.basename(os.path.dirname(fp))
            if ch.isdigit():
                out[fn] = int(ch)
            if len(out) >= len(wanted):
                break
        if len(out) >= len(wanted):
            break

    if len(out) < len(wanted):
        for root in test_roots:
            for dirpath, dirnames, filenames in os.walk(root):
                base = os.path.basename(dirpath)
                if not base.isdigit():
                    continue
                chain_id = int(base)
                for fn in filenames:
                    if fn in wanted and fn not in out:
                        out[fn] = chain_id
                if len(out) >= len(wanted):
                    break
            if len(out) >= len(wanted):
                break

    return out


folder_image_to_chain = _build_test_image_to_chain_from_folders(
    DATA_DIR, sample["image"]
)
print(
    "Mapped test images to chain via test folder name (count):",
    len(folder_image_to_chain),
)
if len(folder_image_to_chain) > 0:
    ex_k = next(iter(folder_image_to_chain))
    print("Example folder mapping:", ex_k, "->", folder_image_to_chain[ex_k])




## === cell 3
def _dedup_topk(*lists, k=5):
    out, seen = [], set()
    for lst in lists:
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
                if len(out) >= k:
                    return out
    if len(out) < k:
        out = (out + global_top5 + global_top5)[:k]
    return out[:k]


def make_pred(image_name: str) -> str:
    ch = folder_image_to_chain.get(image_name, None)
    if ch is None:
        ch = image_to_chain.get(image_name, None)

    if ch is not None:
        top_recent_chain = recent_chain_to_top5.get(ch, [])
        top_chain = chain_to_top5.get(ch, [])
        merged = _dedup_topk(
            top_recent_chain, top_chain, recent_global_top5, global_top5, k=5
        )
    else:
        merged = _dedup_topk(recent_global_top5, global_top5, k=5)

    return " ".join(merged)


submission = sample.copy()
submission["image"] = submission["image"].astype(str)
submission["hotel_id"] = submission["image"].map(make_pred)

assert (
    submission.shape[0] == sample.shape[0]
), "Row count mismatch vs sample_submission."
assert list(submission.columns) == [
    "image",
    "hotel_id",
], "Submission columns must be: image, hotel_id"
assert submission["hotel_id"].notna().all(), "Found NaNs in predictions."

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())



## === cell 4
chk = pd.read_csv("submission.csv")
print("submission.csv shape:", chk.shape)
print(chk.head())
print("Example prediction:", chk.loc[0, "hotel_id"])
