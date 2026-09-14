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

0.7116317251831245

# 6. Current score

0.00231

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently cannot yield a Kaggle score because it depends on private wheel files (`/kaggle/input/pekolib*`) that are not present in the provided filesystem, so it never reliably produces a valid `submission.csv`. I replace that dependency with a minimal, self-contained baseline that (1) reads `train.csv` and `sample_submission.csv` from the existing competition paths, (2) generates a valid submission by predicting the global top-5 most frequent `hotel_id` for every test image (a common safe MAP@5 baseline), and (3) writes `submission.csv` with the correct columns and row order. This preserves evaluation semantics and guarantees an end-to-end run within the time limit, yielding a non-null score that you can iterate from toward the target.'
- What this solution (achieved 0.00325) has done: 'The timeout is dominated by per-image disk I/O and repeated full-vector Hamming computations for every test image; the current loop recomputes distances against all training hashes each time and repeatedly opens images with suboptimal PIL settings. I keep the exact same hashing + voting logic, but make it faster by (1) parallelizing train/test image hashing with a thread pool (I/O-bound), (2) caching Hamming results for duplicate test hashes (exactly equivalent), and (3) replacing the per-test full-distance scan with an exact “candidate-by-bucket then refine” approach: pre-bucket train hashes by high bits so we compute exact Hamming only for a small superset and still pick the true top-K by refining/expanding buckets as needed. All changes preserve correctness because we still use the same dhash64 and the final KNN is selected by exact Hamming distances; we just avoid unnecessary repeated work.'
- What this solution (achieved 0.00248) has done: 'Your current score is far below the target, so we should improve MAP@5 with the smallest change that keeps your hashing+KNN voting core intact. The biggest accuracy issue is that `dhash64` currently uses `np.packbits(..., bitorder="big")`, which changes the bit layout compared to the natural row-major “left-to-right” dhash convention and can severely degrade Hamming-nearest-neighbor matching. I fix this to a consistent little-endian bit packing (a negligible semantic fix: still dhash64 + exact Hamming + same voting), while leaving your bucketing, caching, KNN size, and training subset logic unchanged. This should substantially increase retrieval quality (and thus MAP@5) without changing the overall approach or runtime characteristics.'
- What this solution (achieved 0.00245) has done: 'Your current MAP@5 is far below target, so we should improve retrieval quality while keeping your exact dhash+Hamming+vote core intact. The biggest likely issue now is hash instability from EXIF orientation and alpha-channel handling: the same scene can hash very differently if not normalized, hurting nearest-neighbor matches. I add a minimal, deterministic image normalization step (EXIF transpose + RGB→L) used identically for both train and test before hashing, without changing the hashing method, KNN logic, bucketing, or voting. This should increase true neighbor hits and move the score upward toward the target while staying within the same semantics and runtime budget.'
- What this solution (achieved 0.00256) has done: 'Your current score is far below the target, so we should improve retrieval quality while preserving your dhash+exact-Hamming+vote core. The biggest remaining accuracy bottleneck is that the 8×8 dhash discards too much information for a 7,700-class problem; a minimal, semantics-preserving extension is to concatenate multiple dhashes from the same image at different scales (multi-scale dhash) and keep the same exact-Hamming KNN voting. Concretely, I keep your same pipeline, but compute two dhash64 hashes (8 and 16) and store them as a fixed 16-byte vector; distances remain exact Hamming (popcount over XOR) and bucketing/caching remain the same, just on the first 64 bits for speed. This is a small change to the feature representation (still dhash-based perceptual hashing) and should substantially lift MAP@5 toward the target without changing the overall approach or runtime beyond a modest constant factor.'
- What this solution (achieved 0.00231) has done: 'Your current MAP@5 is extremely far below the target, so we need a real retrieval-quality lift while keeping your exact “dhash → exact Hamming KNN → weighted voting” core intact. The biggest minimal-impact improvement is to make the hash descriptor more discriminative without changing the pipeline: extend the multi-scale dhash from 128-bit (8+16) to 256-bit (8+16+32) and keep the same exact Hamming computation and KNN voting, just over more bits. This preserves the same model/training semantics (still no learning; still exact Hamming KNN), but should increase true neighbor ranking substantially. I also fix a subtle correctness bug in your bucket indexing (`np.searchsorted` was used with an unsorted “all keys” array), which can silently produce wrong candidate sets and tank accuracy.'

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


def find_file(relpath: str) -> str:
    for base in BASE_CANDIDATES:
        p = os.path.join(base, relpath)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {relpath} under any of: {BASE_CANDIDATES}")


def find_dir(relpath: str) -> str:
    for base in BASE_CANDIDATES:
        p = os.path.join(base, relpath)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"Could not find dir {relpath} under any of: {BASE_CANDIDATES}"
    )


train_csv = find_file("train.csv")
sample_csv = find_file("sample_submission.csv")

train = pd.read_csv(train_csv)
sample = pd.read_csv(sample_csv)

assert {"image", "hotel_id"}.issubset(
    train.columns
), f"train columns: {train.columns.tolist()}"
assert {"image", "hotel_id"}.issubset(
    sample.columns
), f"sample columns: {sample.columns.tolist()}"

print("train:", train.shape, "sample:", sample.shape)
print("train head:\n", train.head())
print("sample head:\n", sample.head())



## === cell 1
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None
try:
    Image.draft
except Exception:
    pass

TRAIN_IMG_CANDIDATES = [
    "train_images",
    "hotel-id-2021-fgvc8/train_images",
    "hotel-id-2021-fgvc8/train_images/train_images",
]
TEST_IMG_CANDIDATES = [
    "test_images",
    "hotel-id-2021-fgvc8/test_images",
    "hotel-id-2021-fgvcvc8/test_images/test_images",  # harmless if missing
    "hotel-id-2021-fgvc8/test_images/test_images",
]

train_img_dir = None
for rp in TRAIN_IMG_CANDIDATES:
    try:
        train_img_dir = find_dir(rp)
        break
    except FileNotFoundError:
        pass
if train_img_dir is None:
    raise FileNotFoundError("Could not locate train_images directory.")

test_img_dir = None
for rp in TEST_IMG_CANDIDATES:
    try:
        test_img_dir = find_dir(rp)
        break
    except FileNotFoundError:
        pass
if test_img_dir is None:
    raise FileNotFoundError("Could not locate test_images directory.")

print("train_img_dir:", train_img_dir)
print("test_img_dir :", test_img_dir)

top5_global = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(top5_global) < 5:
    top5_global = top5_global + ["0"] * (5 - len(top5_global))
top5_global_str = " ".join(top5_global[:5])


def image_path_train(chain: int, image_name: str) -> str:
    return os.path.join(train_img_dir, str(chain), image_name)


def image_path_test(image_name: str) -> str:
    return os.path.join(test_img_dir, image_name)


def normalize_for_hash(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im)  # apply EXIF orientation consistently
    if im.mode != "RGB":
        im = im.convert("RGB")  # drop alpha/palette to stabilize grayscale conversion
    return im


def dhash64(img: Image.Image, hash_size: int = 8) -> np.uint64:
    img = ImageOps.grayscale(img).resize((hash_size + 1, hash_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).astype(np.uint8)
    bits = diff.reshape(-1)  # row-major
    packed = np.packbits(bits, bitorder="little")  # (hash_size*hash_size)/8 bytes
    return np.frombuffer(packed.tobytes(), dtype="<u8")[0].astype(np.uint64)


def dhash256(img: Image.Image) -> np.ndarray:
    h1 = dhash64(img, hash_size=8)
    h2 = dhash64(img, hash_size=16)
    h3 = dhash64(img, hash_size=32)
    return np.array(
        [h1, h2, h3, np.uint64(0)], dtype=np.uint64
    )  # fixed width (4,) => 32 bytes


_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_all_u256(query_u256: np.ndarray, train_u256: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(train_u256, query_u256)  # (N,4) uint64
    xb = x.view(np.uint8).reshape(-1, 32)  # 32 bytes per row
    return _POPCNT8[xb].sum(axis=1).astype(np.uint16)


MAX_TRAIN_HASH = 35000  # keep as-is (core logic / constant)
train_small = train.copy()
if len(train_small) > MAX_TRAIN_HASH:
    vc = train_small["hotel_id"].value_counts()
    keep_hotels = set(vc.head(2000).index)  # common hotels
    train_small = train_small[train_small["hotel_id"].isin(keep_hotels)]
    if len(train_small) > MAX_TRAIN_HASH:
        train_small = train_small.iloc[:MAX_TRAIN_HASH].copy()

print("Using train images for hashing:", len(train_small))


def _hash_train_row(row):
    img_name = row.image
    chain = row.chain
    hid = str(row.hotel_id)
    p = image_path_train(chain, img_name)
    try:
        with Image.open(p) as im:
            im = normalize_for_hash(im)
            try:
                im.draft("L", (33, 32))  # draft for the largest hash too
            except Exception:
                pass
            h = dhash256(im)
        return (h, hid)
    except Exception:
        return None


train_hashes_list = []
train_hotels_list = []
hash_to_hotel_votes = defaultdict(Counter)

failed = 0
rows = list(train_small.itertuples(index=False))
max_workers = min(32, (os.cpu_count() or 8) * 2)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(_hash_train_row, r) for r in rows]
    for fut in as_completed(futures):
        out = fut.result()
        if out is None:
            failed += 1
            continue
        h, hid = out
        train_hashes_list.append(h)
        train_hotels_list.append(hid)
        hash_to_hotel_votes[int(h[0])][hid] += 1

train_hashes = np.vstack(train_hashes_list).astype(np.uint64, copy=False)  # (N,4)
train_hotels = np.array(train_hotels_list, dtype=object)

print("Computed train hashes:", len(train_hashes), "failed:", failed)

_BUCKET_BITS = 14
_bucket_key = (train_hashes[:, 0] >> np.uint64(64 - _BUCKET_BITS)).astype(
    np.uint16, copy=False
)
order = np.argsort(_bucket_key, kind="mergesort")
bucket_sorted = _bucket_key[order]
hashes_sorted = train_hashes[order]
hotels_sorted = train_hotels[order]

nb = 1 << _BUCKET_BITS
counts = np.bincount(bucket_sorted.astype(np.int32), minlength=nb).astype(
    np.int32, copy=False
)
bucket_starts = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(
    np.int32, copy=False
)
bucket_ends = (bucket_starts + counts).astype(np.int32, copy=False)


def _get_bucket_range(bkey: int):
    s = int(bucket_starts[bkey])
    e = int(bucket_ends[bkey])
    return s, e


def _candidate_indices_for_query(th_u256: np.ndarray, target_n: int):
    bkey = int(
        (th_u256[0] >> np.uint64(64 - _BUCKET_BITS))
        & np.uint64((1 << _BUCKET_BITS) - 1)
    )
    total = 0
    ranges = []
    step = 0
    while total < target_n and (bkey - step >= 0 or bkey + step < (1 << _BUCKET_BITS)):
        if step == 0:
            s, e = _get_bucket_range(bkey)
            if e > s:
                ranges.append((s, e))
                total += e - s
            step = 1
            continue
        left = bkey - step
        right = bkey + step
        if left >= 0:
            s, e = _get_bucket_range(left)
            if e > s:
                ranges.append((s, e))
                total += e - s
        if total >= target_n:
            break
        if right < (1 << _BUCKET_BITS):
            s, e = _get_bucket_range(right)
            if e > s:
                ranges.append((s, e))
                total += e - s
        step += 1
        if step > (1 << _BUCKET_BITS):
            break

    if not ranges:
        return np.empty((0,), dtype=np.int64)

    idx_parts = [np.arange(s, e, dtype=np.int64) for (s, e) in ranges]
    return np.concatenate(idx_parts) if len(idx_parts) > 1 else idx_parts[0]


if len(train_hashes) < 1000:
    print("Too few train hashes computed; falling back to global top5 baseline.")
    sub = sample[["image"]].copy()
    sub["hotel_id"] = top5_global_str
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv:", sub.shape)
else:
    KNN = 120
    BEST = 5

    sub = sample[["image"]].copy()
    preds = []

    def _hash_test_image(img_name: str):
        p = image_path_test(img_name)
        try:
            with Image.open(p) as im:
                im = normalize_for_hash(im)
                try:
                    im.draft("L", (33, 32))
                except Exception:
                    pass
                th = dhash256(im)
            return (img_name, th, None)
        except Exception as e:
            return (img_name, None, e)

    test_names = sub["image"].tolist()
    test_hash_by_name = {}
    test_failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_hash_test_image, n) for n in test_names]
        for fut in as_completed(futures):
            img_name, th, err = fut.result()
            if err is not None or th is None:
                test_failed += 1
                test_hash_by_name[img_name] = None
            else:
                test_hash_by_name[img_name] = th

    pred_cache = {}

    for img_name in test_names:
        th = test_hash_by_name.get(img_name)
        if th is None:
            preds.append(top5_global_str)
            continue

        cache_key = (int(th[0]), int(th[1]), int(th[2]))
        cached = pred_cache.get(cache_key)
        if cached is not None:
            preds.append(cached)
            continue

        votes = Counter()
        exact = hash_to_hotel_votes.get(int(th[0]))
        if exact:
            votes.update(exact)

        if len(votes) < 20:
            target = max(KNN, 2048)
            cand_idx = _candidate_indices_for_query(th, target)

            if cand_idx.size < KNN:
                dists = hamming_all_u256(th, train_hashes)
                idx = np.argpartition(dists, KNN)[:KNN]
                for j in idx:
                    hid = train_hotels[j]
                    w = int(
                        257 - int(dists[j])
                    )  # same linear idea, adjusted for 256-bit max distance
                    if w < 1:
                        w = 1
                    votes[hid] += w
            else:
                cand_hashes = hashes_sorted[cand_idx]
                dists = hamming_all_u256(th, cand_hashes)
                idx_local = np.argpartition(dists, KNN)[:KNN]
                cand_hotels = hotels_sorted[cand_idx]
                for jl in idx_local:
                    hid = cand_hotels[jl]
                    w = int(257 - int(dists[jl]))
                    if w < 1:
                        w = 1
                    votes[hid] += w

        best = [hid for hid, _ in votes.most_common(BEST)]
        for hid in top5_global:
            if len(best) >= BEST:
                break
            if hid not in best:
                best.append(hid)
        if len(best) < BEST:
            best += ["0"] * (BEST - len(best))
        pred_str = " ".join(best[:BEST])
        pred_cache[cache_key] = pred_str
        preds.append(pred_str)

    sub["hotel_id"] = preds

    assert sub.shape[0] == sample.shape[0]
    assert list(sub.columns) == ["image", "hotel_id"]
    assert sub["hotel_id"].str.split().str.len().eq(5).all()

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv:", sub.shape, "test_failed:", test_failed)
    print(sub.head())



## === cell 2
import subprocess

subprocess.run(
    ["bash", "-lc", "ls -lha submission.csv && head -n 5 submission.csv"], check=False
)
