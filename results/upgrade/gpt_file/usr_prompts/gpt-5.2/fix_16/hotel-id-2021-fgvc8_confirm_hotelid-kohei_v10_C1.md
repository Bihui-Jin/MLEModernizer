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

0.00132

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
- What this solution (achieved 0.00159) has done: 'Your current score is far below the target, and the main reason is that this solution has almost no test-specific signal; the lowest-risk way to move MAP@5 upward without changing the “count-based priors + deterministic rank-merge” core logic is to make the priors more tailored to each test image using only information available at inference time. I keep your exact prior-building and merge approach, but add a lightweight “visual-similarity prior” by matching each test image to its nearest train image via a tiny, fast grayscale downsampled hash (no external packages), then injecting that matched train hotel/chain priors at the front of the top-5 list. This preserves evaluation semantics (still just producing 5 hotel IDs per image), stays within time by caching hashes and limiting candidate comparisons, and should materially improve MAP@5 versus pure popularity priors. The submission writing/format remains unchanged and guaranteed.'
- What this solution (achieved 0.00088) has done: 'Your current score is far below target, so we should improve MAP@5 with the smallest change that keeps your existing “priors + simple visual NN hash injection” logic intact. The main bottleneck is runtime/coverage: the visual nearest-neighbor step is too expensive (looping over up to 3000 reps per test image) and likely times out or under-computes on the hidden test, effectively degrading to weak global priors. I keep the exact aHash feature and prediction merge semantics, but make NN search fast and scalable by building a tiny in-memory index keyed by the first bytes of the packed hash, so each test image only compares against a small candidate subset (still exact Hamming within that subset, no approximation in the hash itself). I also make test image path resolution deterministic and faster by precomputing a filename→filepath map from the test directories (and similarly for train reps), increasing the fraction of images that successfully get a visual match.'
- What this solution (achieved 0.00099) has done: 'Your current score is far below the target, so we need a real MAP@5 lift without changing the core “priors + aHash nearest-rep injection + deterministic merge” logic. The biggest low-risk issue in your current code is that the aHash candidate indexing is too strict: when the prefix bucket is empty it falls back to comparing against *all* reps (slow) or restricts by chain in a way that can still be large, so on the hidden test it likely times out/under-computes and effectively reverts to weak priors. I keep the same aHash and exact Hamming distance, but add a deterministic multi-probe over nearby prefixes (flip 1 bit in the prefix bytes) so we almost always get a small candidate set without approximating the hash itself, and I also add a hard per-image candidate cap to guarantee runtime stays under 600s (still exact within the candidate set you deterministically define). Finally, I make test file indexing cover deeper layouts (`*/*/*.jpg`) so more test images actually get a visual match, which directly improves MAP@5 while preserving submission semantics.'
- What this solution (achieved 0.00113) has done: 'Your current score is far below the target, so the safest way to move MAP@5 upward without changing your core “priors + aHash NN injection + deterministic merge” logic is to (1) fix a major coverage issue: you only index representative train images at one directory depth (`train_images/<chain>/*.jpg`), but the provided dataset also contains `train_images/<chain>/train_images/*.jpg`, which currently makes many reps “missing” and weakens the NN signal. Then (2) make the NN signal slightly stronger but still consistent with your approach by keeping the same aHash+Hamming but using up to 2 reps per hotel for hashing (more robust to viewpoint noise) while keeping runtime bounded. These are minimal, localized changes that increase the fraction of test images getting a meaningful visual match, which should push the score toward your target. Submission format/semantics remain identical and a `submission.csv` is always written.'
- What this solution (achieved 0.00131) has done: 'Your current MAP@5 is far below the target, so the smallest meaningful improvement while preserving your exact “priors + aHash NN injection + deterministic merge” core logic is to strengthen the visual NN signal and ensure it reliably runs on the hidden test. I keep the same aHash+Hamming matcher, but (1) fix a coverage gap by indexing deeper `train_images/<chain>/**.jpg` layouts (not just 1-2 hardcoded patterns), (2) pick representative images more robustly by using up to 2 images per hotel but favoring **recent** images (better matching hidden test distribution) without changing the overall approach, and (3) make the multi-probe prefix search cheaper and more consistent by deduplicating candidate indices before Hamming evaluation. These changes keep semantics identical (still count-based priors + nearest aHash rep) but should increase the fraction/quality of NN matches, nudging MAP@5 upward toward your target while staying within the 600s limit and always writing a valid `submission.csv`.'
- What this solution (achieved 0.00105) has done: 'Your current score is far below the target, and the weakest link is that the visual NN step only indexes representatives from the top-3000 most frequent hotels, which is unlikely to match the hidden test distribution and yields near-random MAP@5. I keep your exact “priors + aHash nearest-rep injection + deterministic merge” core logic, but change the representative selection to be chain-aware (more diverse) while keeping runtime bounded, so more test images get a meaningful nearest-hotel candidate. I also ensure the nearest-hotel vote actually influences the final top-5 by injecting not only the best hotel but also that hotel’s chain priors ahead of global priors (same merge semantics, just better ordering). These minimal changes should materially increase MAP@5 while staying under 600 seconds and still writing a valid `submission.csv`.'
- What this solution (achieved 0.00132) has done: 'Your current score is far below the target, so we need a meaningful MAP@5 lift while keeping your existing “priors + aHash nearest-rep injection + deterministic merge” logic unchanged. The main performance leak is that the aHash hash size (16 → 256 bits) is too coarse for hotel-instance retrieval, so the “best_h” you inject is often effectively random; increasing to hash_size=32 (1024 bits) strengthens the same exact NN mechanism with minimal code change. To keep runtime within limits, we simultaneously tighten candidate evaluation per image (smaller cap) while keeping the same prefix-bucket multiprobe and exact Hamming selection inside the chosen candidate set. This should improve the quality of the injected first prediction (which MAP@5 rewards heavily) and move the score upward toward your target while still producing a valid `submission.csv`.'

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

train_image_to_hotel = (
    train.loc[:, ["image", "hotel_id"]]
    .dropna()
    .assign(
        image=lambda d: d["image"].astype(str),
        hotel_id=lambda d: d["hotel_id"].astype(str),
    )
    .drop_duplicates("image")
    .set_index("image")["hotel_id"]
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
    Score-relevant: increase coverage of chain priors by reliably mapping test images to chain
    using folder names when available; keep fallback semantics unchanged.
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
        for pat in [
            os.path.join(root, "*", "*.jpg"),
            os.path.join(root, "*", "*", "*.jpg"),
        ]:
            for fp in glob.iglob(pat):
                fn = os.path.basename(fp)
                if fn not in wanted or fn in out:
                    continue
                ch = os.path.basename(os.path.dirname(fp))
                if not ch.isdigit():
                    ch2 = os.path.basename(os.path.dirname(os.path.dirname(fp)))
                    if ch2.isdigit():
                        ch = ch2
                if str(ch).isdigit():
                    out[fn] = int(ch)
                if len(out) >= len(wanted):
                    break
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
from PIL import Image
import numpy as np


def _safe_open_image(fp: str):
    try:
        with Image.open(fp) as im:
            return im.convert("L")
    except Exception:
        return None


AHASH_SIZE = 32  # was 16


def _ahash_from_fp(fp: str, hash_size: int = AHASH_SIZE):
    im = _safe_open_image(fp)
    if im is None:
        return None
    im = im.resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    m = arr.mean()
    bits = (arr > m).astype(np.uint8).reshape(-1)
    return np.packbits(bits)


_BITCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_u8(packed_a: np.ndarray, packed_b: np.ndarray) -> int:
    x = np.bitwise_xor(packed_a, packed_b)
    return int(_BITCOUNT_LUT[x].sum())


train_images_root = os.path.join(DATA_DIR, "train_images")
test_images_roots = [
    os.path.join(DATA_DIR, "test_images"),
    os.path.join(DATA_DIR, "test_images", "test_images"),
    os.path.join(DATA_DIR, "test", "test"),
    os.path.join(DATA_DIR, "test", "test", "test"),
]
test_images_roots = [p for p in test_images_roots if os.path.isdir(p)]

assert os.path.isdir(
    train_images_root
), f"Missing train_images root: {train_images_root}"
assert len(test_images_roots) > 0, "No test images root found."


def _build_image_fp_map(roots):
    """
    Score-relevant: improve coverage of visual NN by indexing deeper layouts too.
    Minimal change: just add an extra glob depth.
    """
    mp = {}
    for root in roots:
        for pat in [
            os.path.join(root, "*.jpg"),
            os.path.join(root, "*", "*.jpg"),
            os.path.join(root, "*", "*", "*.jpg"),
        ]:
            for fp in glob.iglob(pat):
                mp.setdefault(os.path.basename(fp), fp)
    return mp


test_fp_map = _build_image_fp_map(test_images_roots)
print("Indexed test image files:", len(test_fp_map))

REPS_PER_HOTEL = 2
HOTELS_PER_CHAIN_FOR_HASH = 80
MAX_TOTAL_HOTELS_FOR_HASH = 9000

vc_chain_hotel = train.groupby("chain")["hotel_id"].value_counts()
chain_ids_sorted = sorted(chain_to_top5.keys())
selected_hotels = []
for ch in chain_ids_sorted:
    try:
        top_hotels = (
            vc_chain_hotel.loc[int(ch)]
            .head(HOTELS_PER_CHAIN_FOR_HASH)
            .index.astype(str)
            .tolist()
        )
    except KeyError:
        top_hotels = []
    selected_hotels.extend(top_hotels)

seen = set()
selected_unique = []
for h in (
    selected_hotels
    + train["hotel_id"].value_counts().head(300).index.astype(str).tolist()
):
    if h not in seen:
        selected_unique.append(h)
        seen.add(h)
    if len(selected_unique) >= MAX_TOTAL_HOTELS_FOR_HASH:
        break

train_hash_hotel_set = set(selected_unique)
print("Hotels selected for hashing:", len(train_hash_hotel_set))

rep_rows = train.loc[
    train["hotel_id"].astype(str).isin(train_hash_hotel_set),
    ["hotel_id", "image", "chain", "_ts"],
].assign(
    hotel_id=lambda d: d["hotel_id"].astype(str), image=lambda d: d["image"].astype(str)
)

if rep_rows["_ts"].notna().any():
    rep_rows = rep_rows.sort_values(
        ["hotel_id", "_ts", "image"], ascending=[True, False, True]
    )
else:
    rep_rows = rep_rows.sort_values(["hotel_id", "image"])

rep_rows = (
    rep_rows.groupby("hotel_id", as_index=False)
    .head(REPS_PER_HOTEL)
    .reset_index(drop=True)
)

rep_hotels = rep_rows["hotel_id"].tolist()
rep_images = rep_rows["image"].tolist()
rep_chains = rep_rows["chain"].fillna(0).astype(int).tolist()

print(
    "Selected rep images:", len(rep_rows), "for hotels:", rep_rows["hotel_id"].nunique()
)


def _build_train_rep_fp_map(train_root: str, rep_images: list[str]) -> dict:
    """
    Score-critical: increase rep image coverage by searching deeper under each chain directory.
    """
    wanted = set(rep_images)
    out = {}
    for ch_dir in glob.iglob(os.path.join(train_root, "*")):
        if not os.path.isdir(ch_dir):
            continue
        base = os.path.basename(ch_dir)
        if not base.isdigit():
            continue

        for fp in glob.iglob(os.path.join(ch_dir, "**", "*.jpg"), recursive=True):
            fn = os.path.basename(fp)
            if fn in wanted and fn not in out:
                out[fn] = fp
                if len(out) >= len(wanted):
                    return out
    return out


train_rep_fp_map = _build_train_rep_fp_map(train_images_root, rep_images)
print(
    "Indexed representative train image files:",
    len(train_rep_fp_map),
    "of",
    len(rep_images),
)

rep_train_hashes = []
rep_train_hotels = []
rep_train_chains = []

missing_rep = 0
for hotel_id, img, ch in zip(rep_hotels, rep_images, rep_chains):
    fp = train_rep_fp_map.get(img, None)
    if fp is None or not os.path.exists(fp):
        missing_rep += 1
        continue
    h = _ahash_from_fp(fp, hash_size=AHASH_SIZE)
    if h is None:
        missing_rep += 1
        continue
    rep_train_hashes.append(h)
    rep_train_hotels.append(str(hotel_id))
    rep_train_chains.append(int(ch))

rep_train_hashes = (
    np.stack(rep_train_hashes, axis=0)
    if len(rep_train_hashes)
    else np.zeros((0, (AHASH_SIZE * AHASH_SIZE) // 8), dtype=np.uint8)
)
rep_train_hotels = np.array(rep_train_hotels, dtype=object)
rep_train_chains = np.array(rep_train_chains, dtype=np.int32)

print("Rep train hashes:", rep_train_hashes.shape, "missing reps:", missing_rep)

hotel_to_chain = {}
for h, c in zip(rep_train_hotels.tolist(), rep_train_chains.tolist()):
    hotel_to_chain.setdefault(str(h), int(c))


def _build_hash_prefix_index(hashes: np.ndarray, prefix_bytes: int = 2) -> dict:
    idx = {}
    if hashes.shape[0] == 0:
        return idx
    for i in range(hashes.shape[0]):
        key = bytes(hashes[i][:prefix_bytes].tolist())
        idx.setdefault(key, []).append(i)
    return idx


PREFIX_BYTES = 2
prefix_index_all = _build_hash_prefix_index(rep_train_hashes, prefix_bytes=PREFIX_BYTES)

prefix_index_by_chain = {}
if rep_train_hashes.shape[0] > 0:
    for ch in np.unique(rep_train_chains):
        mask = rep_train_chains == ch
        sub_idx = np.where(mask)[0]
        if sub_idx.size == 0:
            continue
        d = {}
        for i in sub_idx.tolist():
            key = bytes(rep_train_hashes[i][:PREFIX_BYTES].tolist())
            d.setdefault(key, []).append(i)
        prefix_index_by_chain[int(ch)] = d

print(
    "Prefix index buckets (all):",
    len(prefix_index_all),
    "per-chain:",
    len(prefix_index_by_chain),
)


def _get_test_fp(img: str):
    return test_fp_map.get(img, None)


def _iter_probe_keys(key_bytes: bytes):
    """
    Deterministic multi-probe nearby prefix buckets to reduce empty-bucket fallbacks.
    """
    yield key_bytes
    b = bytearray(key_bytes)
    for i in range(len(b)):
        for bit in (1, 2, 4, 8, 16, 32, 64, 128):
            bb = bytearray(b)
            bb[i] ^= bit
            yield bytes(bb)


MAX_CANDIDATES_PER_IMAGE = 350  # was 600


def _nearest_rep_hotel(test_fp: str, restrict_chain: int | None = None):
    h = _ahash_from_fp(test_fp, hash_size=AHASH_SIZE)
    if h is None or rep_train_hashes.shape[0] == 0:
        return None, None

    key = bytes(h[:PREFIX_BYTES].tolist())

    if restrict_chain is not None and int(restrict_chain) in prefix_index_by_chain:
        idx = prefix_index_by_chain[int(restrict_chain)]
        cand = []
        for k in _iter_probe_keys(key):
            if k in idx:
                cand.extend(idx[k])
                if len(cand) >= MAX_CANDIDATES_PER_IMAGE:
                    break
        if not cand:
            cand = np.where(rep_train_chains == int(restrict_chain))[0].tolist()
    else:
        cand = []
        for k in _iter_probe_keys(key):
            if k in prefix_index_all:
                cand.extend(prefix_index_all[k])
                if len(cand) >= MAX_CANDIDATES_PER_IMAGE:
                    break
        if not cand:
            cand = list(range(rep_train_hashes.shape[0]))

    if cand:
        cand = list(dict.fromkeys(cand))

    if len(cand) > MAX_CANDIDATES_PER_IMAGE:
        cand = cand[:MAX_CANDIDATES_PER_IMAGE]

    best_d = 10**9
    best_hotel = None
    for i in cand:
        d = _hamming_u8(h, rep_train_hashes[i])
        if d < best_d:
            best_d = d
            best_hotel = str(rep_train_hotels[i])
            if best_d == 0:
                break
    return best_hotel, best_d


test_best_hotel = {}
test_best_chain = {}

wanted_test_images = sample["image"].astype(str).tolist()
for img in wanted_test_images:
    ch = folder_image_to_chain.get(img, None)
    fp = _get_test_fp(img)
    if fp is None:
        continue
    best_h, best_d = _nearest_rep_hotel(
        fp, restrict_chain=ch if ch is not None else None
    )
    if best_h is not None:
        test_best_hotel[img] = best_h
        test_best_chain[img] = int(
            hotel_to_chain.get(best_h, ch if ch is not None else 0)
        )

print("Test images with visual NN match:", len(test_best_hotel))




## === cell 4
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
    best_h = test_best_hotel.get(image_name, None)

    ch = folder_image_to_chain.get(image_name, None)
    if ch is None:
        ch = image_to_chain.get(image_name, None)
    if best_h is not None and ch is None:
        ch = test_best_chain.get(image_name, None)

    if ch is not None:
        top_recent_chain = recent_chain_to_top5.get(int(ch), [])
        top_chain = chain_to_top5.get(int(ch), [])
        merged = _dedup_topk(
            [best_h] if best_h is not None else [],
            top_recent_chain,
            top_chain,
            recent_global_top5,
            global_top5,
            k=5,
        )
    else:
        merged = _dedup_topk(
            [best_h] if best_h is not None else [],
            recent_global_top5,
            global_top5,
            k=5,
        )

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



## === cell 5
chk = pd.read_csv("submission.csv")
print("submission.csv shape:", chk.shape)
print(chk.head())
print("Example prediction:", chk.loc[0, "hotel_id"])
