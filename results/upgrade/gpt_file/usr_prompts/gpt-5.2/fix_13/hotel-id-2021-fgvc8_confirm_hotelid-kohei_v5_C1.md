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

0.00339

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently cannot yield a Kaggle score because it depends on private wheel files (`/kaggle/input/pekolib*`) that are not present in the provided filesystem, so it never reliably produces a valid `submission.csv`. I replace that dependency with a minimal, self-contained baseline that (1) reads `train.csv` and `sample_submission.csv` from the existing competition paths, (2) generates a valid submission by predicting the global top-5 most frequent `hotel_id` for every test image (a common safe MAP@5 baseline), and (3) writes `submission.csv` with the correct columns and row order. This preserves evaluation semantics and guarantees an end-to-end run within the time limit, yielding a non-null score that you can iterate from toward the target.'
- What this solution (achieved 0.00325) has done: 'The timeout is dominated by per-image disk I/O and repeated full-vector Hamming computations for every test image; the current loop recomputes distances against all training hashes each time and repeatedly opens images with suboptimal PIL settings. I keep the exact same hashing + voting logic, but make it faster by (1) parallelizing train/test image hashing with a thread pool (I/O-bound), (2) caching Hamming results for duplicate test hashes (exactly equivalent), and (3) replacing the per-test full-distance scan with an exact “candidate-by-bucket then refine” approach: pre-bucket train hashes by high bits so we compute exact Hamming only for a small superset and still pick the true top-K by refining/expanding buckets as needed. All changes preserve correctness because we still use the same dhash64 and the final KNN is selected by exact Hamming distances; we just avoid unnecessary repeated work.'
- What this solution (achieved 0.00248) has done: 'Your current score is far below the target, so we should improve MAP@5 with the smallest change that keeps your hashing+KNN voting core intact. The biggest accuracy issue is that `dhash64` currently uses `np.packbits(..., bitorder="big")`, which changes the bit layout compared to the natural row-major “left-to-right” dhash convention and can severely degrade Hamming-nearest-neighbor matching. I fix this to a consistent little-endian bit packing (a negligible semantic fix: still dhash64 + exact Hamming + same voting), while leaving your bucketing, caching, KNN size, and training subset logic unchanged. This should substantially increase retrieval quality (and thus MAP@5) without changing the overall approach or runtime characteristics.'
- What this solution (achieved 0.00245) has done: 'Your current MAP@5 is far below target, so we should improve retrieval quality while keeping your exact dhash+Hamming+vote core intact. The biggest likely issue now is hash instability from EXIF orientation and alpha-channel handling: the same scene can hash very differently if not normalized, hurting nearest-neighbor matches. I add a minimal, deterministic image normalization step (EXIF transpose + RGB→L) used identically for both train and test before hashing, without changing the hashing method, KNN logic, bucketing, or voting. This should increase true neighbor hits and move the score upward toward the target while staying within the same semantics and runtime budget.'
- What this solution (achieved 0.00256) has done: 'Your current score is far below the target, so we should improve retrieval quality while preserving your dhash+exact-Hamming+vote core. The biggest remaining accuracy bottleneck is that the 8×8 dhash discards too much information for a 7,700-class problem; a minimal, semantics-preserving extension is to concatenate multiple dhashes from the same image at different scales (multi-scale dhash) and keep the same exact-Hamming KNN voting. Concretely, I keep your same pipeline, but compute two dhash64 hashes (8 and 16) and store them as a fixed 16-byte vector; distances remain exact Hamming (popcount over XOR) and bucketing/caching remain the same, just on the first 64 bits for speed. This is a small change to the feature representation (still dhash-based perceptual hashing) and should substantially lift MAP@5 toward the target without changing the overall approach or runtime beyond a modest constant factor.'
- What this solution (achieved 0.00231) has done: 'Your current MAP@5 is extremely far below the target, so we need a real retrieval-quality lift while keeping your exact “dhash → exact Hamming KNN → weighted voting” core intact. The biggest minimal-impact improvement is to make the hash descriptor more discriminative without changing the pipeline: extend the multi-scale dhash from 128-bit (8+16) to 256-bit (8+16+32) and keep the same exact Hamming computation and KNN voting, just over more bits. This preserves the same model/training semantics (still no learning; still exact Hamming KNN), but should increase true neighbor ranking substantially. I also fix a subtle correctness bug in your bucket indexing (`np.searchsorted` was used with an unsorted “all keys” array), which can silently produce wrong candidate sets and tank accuracy.'
- What this solution (achieved 0.00222) has done: 'Your current MAP@5 is far below the target, so the most effective minimal improvement is to keep your exact dhash→exact-Hamming KNN→vote pipeline but fix a correctness issue in `dhash64` for hash sizes 16/32 (it currently truncates to 64 bits, making your “256-bit” descriptor mostly zeros and destroying retrieval quality). I change `dhash64` to return the full packed bytes for any hash_size and then build a true 256-bit descriptor by concatenating 8/16/32-scale hashes (still dhash-based, still exact Hamming popcount). This preserves the same core logic and evaluation semantics, but makes the descriptor actually discriminative, which should move the score substantially upward toward your target. Everything else (bucketing, candidate expansion, caching, KNN, voting, submission format/paths) is kept the same.'
- What this solution (achieved 0.0026) has done: 'Main bottlenecks are (1) launching ~35k train-image hashing tasks individually (huge ThreadPoolExecutor scheduling overhead) and (2) building candidate index arrays per test image via repeated `np.arange`/`np.concatenate`, plus repeated per-neighbor Python loops. I keep the exact hashing and voting/KNN logic unchanged, but reduce constant factors by batching threaded work with `executor.map` + chunking, precomputing bucket index ranges once (so per-query candidate indices are built with cheap slicing), and vectorizing the “add KNN weighted votes” step via `np.unique` aggregation (exactly equivalent to the original per-neighbor loop). These changes remove the biggest Python overheads without changing any model/feature logic, and should bring runtime under 600s while preserving predictions up to negligible FP differences.'
- What this solution (achieved 0.00339) has done: 'Your current score (0.0026) is far below the target (0.7116), so we need a real retrieval-quality improvement while keeping your perceptual-hash → exact-Hamming KNN → voting core intact. The minimal high-impact fix is that your “256-bit” descriptor is currently not actually multi-scale (it truncates the 16/32 hashes to tiny prefixes), making it too weak; we instead build a true 256-bit multi-scale descriptor by concatenating full 8×8 (64b) + 16×16 (256b) and then selecting 256 total bits (64b + 192b) deterministically. We also ensure bucketing uses the same leading bits that the descriptor actually contains (still from the first uint64), preserving your candidate expansion and exact-distance refinement logic. Everything else (training subset, hashing normalization, Hamming popcount, KNN/voting, submission formatting/paths) stays the same.'

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
from concurrent.futures import ThreadPoolExecutor

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


def dhash_bytes(img: Image.Image, hash_size: int) -> bytes:
    img = ImageOps.grayscale(img).resize((hash_size + 1, hash_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).astype(np.uint8)
    bits = diff.reshape(-1)  # row-major
    packed = np.packbits(bits, bitorder="little")
    return packed.tobytes()


def dhash256(img: Image.Image) -> np.ndarray:
    b8 = dhash_bytes(img, 8)  # 8 bytes = 64 bits
    b16 = dhash_bytes(img, 16)  # 32 bytes = 256 bits
    desc = b8 + b16[:24]  # 8 + 24 = 32 bytes
    return np.frombuffer(desc, dtype="<u8", count=4).astype(np.uint64, copy=False)


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
                im.draft(
                    "L", (17, 16)
                )  # Change (runtime-only): align draft to largest used hash (16), no semantic change.
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
chunksize = 64
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for out in ex.map(_hash_train_row, rows, chunksize=chunksize):
        if out is None:
            failed += 1
            continue
        h, hid = out
        train_hashes_list.append(h)
        train_hotels_list.append(hid)
        hash_to_hotel_votes[int(h[0])][hid] += 1

if train_hashes_list:
    train_hashes = np.vstack(train_hashes_list).astype(np.uint64, copy=False)  # (N,4)
    train_hotels = np.array(train_hotels_list, dtype=object)
else:
    train_hashes = np.empty((0, 4), dtype=np.uint64)
    train_hotels = np.empty((0,), dtype=object)

print("Computed train hashes:", len(train_hashes), "failed:", failed)

_BUCKET_BITS = 14
if len(train_hashes) > 0:
    _bucket_key = (train_hashes[:, 0] >> np.uint64(64 - _BUCKET_BITS)).astype(
        np.uint16, copy=False
    )
    order = np.argsort(_bucket_key, kind="mergesort")
    bucket_sorted = _bucket_key[order]
    hashes_sorted = train_hashes[order]
    hotels_sorted = train_hotels[order]
else:
    bucket_sorted = np.empty((0,), dtype=np.uint16)
    hashes_sorted = train_hashes
    hotels_sorted = train_hotels

nb = 1 << _BUCKET_BITS
counts = np.bincount(bucket_sorted.astype(np.int32), minlength=nb).astype(
    np.int32, copy=False
)
bucket_starts = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(
    np.int32, copy=False
)
bucket_ends = (bucket_starts + counts).astype(np.int32, copy=False)

bucket_indices = [
    np.arange(int(bucket_starts[i]), int(bucket_ends[i]), dtype=np.int64)
    for i in range(nb)
]


def _candidate_indices_for_query(th_u256: np.ndarray, target_n: int):
    bkey = int(
        (th_u256[0] >> np.uint64(64 - _BUCKET_BITS))
        & np.uint64((1 << _BUCKET_BITS) - 1)
    )
    total = 0
    parts = []
    step = 0
    while total < target_n and (bkey - step >= 0 or bkey + step < nb):
        if step == 0:
            idx = bucket_indices[bkey]
            if idx.size:
                parts.append(idx)
                total += idx.size
            step = 1
            continue
        left = bkey - step
        right = bkey + step
        if left >= 0:
            idx = bucket_indices[left]
            if idx.size:
                parts.append(idx)
                total += idx.size
        if total >= target_n:
            break
        if right < nb:
            idx = bucket_indices[right]
            if idx.size:
                parts.append(idx)
                total += idx.size
        step += 1
        if step > nb:
            break

    if not parts:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


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
                    im.draft(
                        "L", (17, 16)
                    )  # Change (runtime-only): align draft to largest used hash (16), no semantic change.
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
        for img_name, th, err in ex.map(
            _hash_test_image, test_names, chunksize=chunksize
        ):
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

        cache_key = (int(th[0]), int(th[1]), int(th[2]), int(th[3]))
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

                sel_hotels = train_hotels[idx].astype(str, copy=False)
                sel_w = 257 - dists[idx].astype(np.int32, copy=False)
                sel_w[sel_w < 1] = 1
                uh, inv = np.unique(sel_hotels, return_inverse=True)
                sums = np.bincount(inv, weights=sel_w, minlength=uh.size).astype(
                    np.int64
                )
                for h, s in zip(uh.tolist(), sums.tolist()):
                    votes[h] += int(s)
            else:
                cand_hashes = hashes_sorted[cand_idx]
                dists = hamming_all_u256(th, cand_hashes)
                idx_local = np.argpartition(dists, KNN)[:KNN]
                cand_hotels = hotels_sorted[cand_idx]

                sel_hotels = cand_hotels[idx_local].astype(str, copy=False)
                sel_w = 257 - dists[idx_local].astype(np.int32, copy=False)
                sel_w[sel_w < 1] = 1
                uh, inv = np.unique(sel_hotels, return_inverse=True)
                sums = np.bincount(inv, weights=sel_w, minlength=uh.size).astype(
                    np.int64
                )
                for h, s in zip(uh.tolist(), sums.tolist()):
                    votes[h] += int(s)

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
