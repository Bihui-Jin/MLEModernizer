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

0.7445482866043589

# 6. Current score

0.00615

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a Kaggle score because it relies on private wheel inputs (`pekolib*`) that aren’t present in the provided `/kaggle/input/` tree, so it fails before producing a valid `submission.csv`. To move the score toward the target, the smallest safe change is to (1) make the pipeline self-contained with only standard installed packages and (2) generate a valid baseline submission CSV in the required format by predicting the 5 most frequent `hotel_id`s from `train.csv` for every test image in `sample_submission.csv`. This run end-to-end in the Kaggle environment, produce a valid submission, and yield a non-zero MAP@5 (likely below the target, but it enables iteration toward it). Paths remain on `/kaggle/input/...` and output is written to `submission.csv`.'
- What this solution (achieved 0.00186) has done: 'Your timeout is dominated by per-image PIL open/resize/convert work executed ~9.7k times for test and up to 30k times for train, plus Python-loop overhead in hashing and candidate lookup. I keep the exact hashing/matching logic but make it faster by (1) avoiding repeated function/global lookups, (2) speeding up image decoding/resizing via `Image.reduce()` + `draft()` when possible (equivalent output to current pipeline for 8x8 target), (3) parallelizing hash computation for train/test images with a thread pool (I/O bound), and (4) replacing dictionary bucket-building loops with vectorized numpy sorting/grouping while preserving identical bucket semantics. These changes reduce wall time substantially without changing the algorithm, thresholds, TOPK, or scoring.'
- What this solution (achieved 0.00335) has done: 'Your current score (0.00186) is far below the target (0.7445), so we need a real accuracy lift while keeping your hashing-based retrieval core intact. The biggest issue is that you only hash 30k train images (out of ~88k rows), which severely limits recall; increasing this cap (within time) should move MAP@5 up substantially without changing the algorithm. I also make the train path construction robust to the actual folder layout by deriving the subfolder from the real filesystem when `chain` is missing/incorrect, while keeping the same aHash, bucketing, and ranking logic. Finally, I keep determinism and submission formatting identical, still writing `submission.csv`.'
- What this solution (achieved 0.00317) has done: 'Your current score is far below the target, so the most likely reason is low retrieval quality rather than submission formatting. I keep your exact aHash + bucketing + Hamming-ranking core, but fix two high-impact issues: (1) the train image path resolution currently does expensive directory scans per image and can silently miss many files, and (2) your fallback candidate pool is very small and can dominate predictions when buckets are empty. The patch pre-indexes the train image filename→folder mapping once (fast, deterministic) so we hash many more correct train images, and it slightly enlarges the deterministic fallback pool so recalls improve when bucketing fails—both should move MAP@5 upward toward the target without changing model logic. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.00397) has done: 'Your score is far below the target, so we should improve retrieval quality while keeping your exact aHash→bucket→Hamming→hotel-vote logic intact. The largest gain with minimal semantic change is to (1) build tighter bucket neighborhoods (include all 1-bit bucket flips instead of only 2 LSB flips) to raise recall, and (2) slightly increase the per-query re-ranking depth so the hotel-vote has more evidence. To keep runtime within limits, we also cap neighbor bucket candidate counts deterministically (so huge buckets don’t explode compute) while preserving the same ranking method on that capped set. The submission formatting and file path behavior remain unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.00374) has done: 'Your current MAP@5 (0.00397) is far below the target (0.7445), so we should improve recall/quality with the smallest possible change that preserves your aHash→bucket→Hamming→hotel-vote core. The biggest easy gain is to stop relying on a single 16-bit prefix bucket: we use *multiple* bucket projections (several different 16-bit slices of the 64-bit hash), gather candidates from all of them (plus their 1-bit neighbors, as you already do), then run the exact same Hamming re-ranking + hotel vote on the merged candidate set. This keeps the same feature (aHash), same distance (Hamming), and same voting semantics, but typically increases the chance that true matches land in the candidate pool. To keep runtime bounded, we also keep a deterministic cap on total candidates per query and retain your global-top5 fallback.'
- What this solution (achieved 0.00359) has done: 'Your current score (0.00374) is far below the target (0.7445), so we should increase MAP@5 by improving retrieval quality while keeping your exact aHash→bucket→Hamming→hotel-vote core unchanged. The biggest minimal win is to remove the “precomputed neighbor cache only for keys seen in train” limitation: currently, if a test bucket key never appears in train for a given shift, you fall back to the global top5 even though 1-bit-neighbor buckets might exist; we query neighbors dynamically at inference so recall improves. To keep runtime bounded and deterministic, we keep the same per-shift and total candidate caps, and we keep all paths/I/O and the ranking/voting semantics identical. This should move the score upward toward the target without changing model architecture or training loops (there are none).'
- What this solution (achieved 0.00597) has done: 'Your current score is far below the target, so we should increase MAP@5 by improving retrieval quality while keeping your exact aHash→bucket→Hamming→hotel-vote logic intact. The smallest high-impact change is to fix a bug in bucket key computation: during bucket building you used all higher bits (`train_hashes >> sh`) instead of the intended 16-bit slice, which makes buckets extremely sparse and harms recall. I also keep the same multi-slice bucketing, neighbor expansion, and reranking, but remove an unnecessary `np.unique` on candidate indices (replaced by stable de-dup in a way that preserves determinism) to reduce overhead without changing semantics. Everything still runs end-to-end and writes a valid `submission.csv` with 5 space-delimited hotel IDs per image.'
- What this solution (achieved 0.00635) has done: 'Your current MAP@5 is far below the target, so we should increase recall/precision while keeping your aHash→bucket→Hamming→hotel-vote core unchanged. The smallest high-impact fix is to stop collapsing multiple occurrences of the same `hotel_id` inside the top neighbors: `np.unique(sel_hotels)` discards duplicates, which breaks the “vote” effect; we instead aggregate scores per hotel while preserving multiplicity. To further move the score up without changing the method, we also replace the O(n) Python set-based de-dup of candidate indices with a deterministic `np.unique` (and keep mergesort ordering) to allow raising `TOPK_NEIGHBORS` modestly within runtime. Everything still runs end-to-end, uses the same hashing, bucketing, neighbor expansion, and Hamming ranking, and writes a valid `submission.csv`.'
- What this solution (achieved 0.00617) has done: 'Your current MAP@5 (0.00635) is far below the target (0.7445), so we should improve recall while keeping your exact aHash→bucket→Hamming→hotel-vote core intact. The smallest likely win is to use the *same* retrieval logic but build the train hash index from more representative images: sort train by `timestamp` (newest-first) instead of by filename so we include more diverse hotels/cameras, which typically improves generalization on the hidden test set. I also make the train hashing deterministic by collecting threaded results back into the original order (no semantic change, but it stabilizes bucket construction and thus scoring). Everything else (hashing, bucket slices/neighbors, Hamming ranking, voting, submission formatting) remains the same and it still writes `submission.csv`.'
- What this solution (achieved 0.00615) has done: 'Your current score is far below the target, so we should make a small change that increases recall/precision without changing the aHash→bucket→Hamming→hotel-vote retrieval core. The minimal high-impact adjustment is to (1) allow a slightly larger re-ranking depth (more neighbors) and (2) make the vote aggregation slightly more robust by using a stable per-hotel “best-match boost” on top of your existing summed score, which often improves MAP@5 while preserving the same evidence (same candidates, same distances). To keep runtime bounded under 600s, we keep all existing candidate caps and thread parallelism unchanged, and only add O(topN) per query work. Submission format, paths, hashing, bucketing slices/neighbors, and Hamming ranking semantics remain intact and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

BASE_INPUT = Path("/kaggle/input")

print("Listing /kaggle/input (top-level):")
if BASE_INPUT.exists():
    for p in sorted(BASE_INPUT.iterdir()):
        try:
            sz = p.stat().st_size if p.is_file() else 0
        except Exception:
            sz = 0
        print(
            f" - {p.name}{'/' if p.is_dir() else ''}\t{sz/1024/1024:.2f} MB"
            if p.is_file()
            else f" - {p.name}/"
        )
else:
    raise FileNotFoundError("/kaggle/input not found")



## === cell 1
import pandas as pd

CANDIDATE_ROOTS = [
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("/kaggle/data/hotel-id-2021-fgvc8"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]


def find_first_existing(rel_path: str) -> Path:
    for r in CANDIDATE_ROOTS:
        p = r / rel_path
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {rel_path} under candidate roots: {CANDIDATE_ROOTS}"
    )


train_csv = find_first_existing("train.csv")
sample_sub_csv = find_first_existing("sample_submission.csv")

print("Using train.csv:", train_csv)
print("Using sample_submission.csv:", sample_sub_csv)

train_df = pd.read_csv(train_csv)
sub_df = pd.read_csv(sample_sub_csv)

required_train_cols = {"image", "hotel_id"}
required_sub_cols = {"image", "hotel_id"}
if not required_train_cols.issubset(train_df.columns):
    raise ValueError(
        f"train.csv missing required columns: {required_train_cols - set(train_df.columns)}"
    )
if not required_sub_cols.issubset(sub_df.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns: {required_sub_cols - set(sub_df.columns)}"
    )

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + global_top5 * 5)[:5]

print("Global fallback top-5 hotel_ids:", global_top5)



## === cell 2
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def find_images_dir(name: str) -> Path:
    candidates = []
    for r in CANDIDATE_ROOTS:
        candidates.append(r / name)
        candidates.append(r / "hotel-id-2021-fgvc8" / name)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find images dir for {name} under {CANDIDATE_ROOTS}"
    )


train_images_dir = find_images_dir("train_images")
test_images_dir = find_images_dir("test_images")

print("Using train_images_dir:", train_images_dir)
print("Using test_images_dir:", test_images_dir)

_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def ahash64(img_path: str) -> np.uint64:
    with Image.open(img_path) as im:
        try:
            im.draft("L", (8, 8))
        except Exception:
            pass
        im = im.convert("L")

        try:
            w, h = im.size
            if w > 64 or h > 64:
                factor = max(1, min(w // 64, h // 64))
                if factor > 1:
                    im = im.reduce(factor)
        except Exception:
            pass

        im = im.resize((8, 8), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)

    m = arr.mean()
    bits = (arr > m).astype(np.uint8).reshape(-1)
    packed = np.packbits(bits, bitorder="big")
    return packed.view(">u8")[0].astype(np.uint64)


def hamming_many_u64(hq: np.uint64, hs: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(hs, hq).view(np.uint64)
    xb = x.view(np.uint8).reshape(-1, 8)
    return _POPCNT8[xb].sum(axis=1).astype(np.int16)


train_df = train_df.copy()
train_df["image"] = train_df["image"].astype(str)
train_df["hotel_id"] = train_df["hotel_id"].astype(str)

MAX_TRAIN_HASHES = 87798

if "timestamp" in train_df.columns:
    train_df_sorted = train_df.sort_values(
        ["timestamp", "image"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
else:
    train_df_sorted = train_df.sort_values("image", kind="mergesort").reset_index(
        drop=True
    )

train_df_use = train_df_sorted.iloc[
    : min(MAX_TRAIN_HASHES, len(train_df_sorted))
].copy()

if "chain" in train_df_sorted.columns:
    train_df_use["chain"] = train_df_sorted.loc[train_df_use.index, "chain"].values
else:
    train_df_use["chain"] = 0

train_images_dir_str = str(train_images_dir)


def build_train_filename_to_folder_index(root_dir: str):
    idx = {}
    try:
        for d in sorted(os.listdir(root_dir)):
            if not d.isdigit():
                continue
            folder = os.path.join(root_dir, d)
            try:
                files = sorted(os.listdir(folder))
            except Exception:
                continue
            for fn in files:
                if fn.endswith(".jpg") and fn not in idx:
                    idx[fn] = d
    except Exception:
        return {}
    return idx


train_file_to_folder = build_train_filename_to_folder_index(train_images_dir_str)
print(f"Indexed train filenames: {len(train_file_to_folder)}")


def resolve_train_path(chain_val: str, img: str) -> str:
    p0 = os.path.join(train_images_dir_str, chain_val, img)
    if os.path.exists(p0):
        return p0
    folder = train_file_to_folder.get(img)
    if folder is not None:
        return os.path.join(train_images_dir_str, folder, img)
    return p0


chains = train_df_use["chain"].astype(str).values
imgs = train_df_use["image"].values
train_paths = [resolve_train_path(ch, img) for ch, img in zip(chains, imgs)]
train_df_use["path"] = train_paths
missing_train = sum(0 if os.path.exists(p) else 1 for p in train_paths)
print(f"Train subset size: {len(train_df_use)} (missing paths: {missing_train})")


def _hash_one_train(p: str, hid: str):
    if not os.path.exists(p):
        return None
    try:
        return (ahash64(p), hid)
    except Exception:
        return None


train_pairs = list(zip(train_df_use["path"].values, train_df_use["hotel_id"].values))

train_hashes_list = [None] * len(train_pairs)
train_hotels_list = [None] * len(train_pairs)
bad = 0

max_workers = min(32, (os.cpu_count() or 4) * 2)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    fut_to_i = {
        ex.submit(_hash_one_train, p, hid): i for i, (p, hid) in enumerate(train_pairs)
    }
    for fu in as_completed(fut_to_i):
        i = fut_to_i[fu]
        out = fu.result()
        if out is None:
            bad += 1
        else:
            h, hid = out
            train_hashes_list[i] = h
            train_hotels_list[i] = hid

keep = [i for i, h in enumerate(train_hashes_list) if h is not None]
train_hashes = np.array([train_hashes_list[i] for i in keep], dtype=np.uint64)
train_hotels = np.array([train_hotels_list[i] for i in keep], dtype=object)
print(f"Computed train hashes: {len(train_hashes)} (skipped: {bad})")

sub_df = sub_df.copy()
sub_df["image"] = sub_df["image"].astype(str)

test_images_dir_str = str(test_images_dir)
alt_test_dir_str = os.path.join(test_images_dir_str, "test_images")
test_paths = []
missing_test = 0
for x in sub_df["image"].values:
    p = os.path.join(test_images_dir_str, x)
    if os.path.exists(p):
        test_paths.append(p)
    else:
        alt = os.path.join(alt_test_dir_str, x)
        if os.path.exists(alt):
            test_paths.append(alt)
        else:
            test_paths.append(p)
            missing_test += 1
print(f"Test images: {len(test_paths)} (missing paths: {missing_test})")

BUCKET_BITS = 16
BUCKET_SHIFTS = (48, 32, 16, 0)  # 4 slices of 16 bits each
BUCKET_MASK = np.uint64((1 << BUCKET_BITS) - 1)

have_train = len(train_hashes) > 0
if have_train:
    neighbor_masks = [0] + [(1 << b) for b in range(BUCKET_BITS)]

    buckets_by_shift = {}

    for sh in BUCKET_SHIFTS:
        keys = (np.right_shift(train_hashes, np.uint64(sh)) & BUCKET_MASK).astype(
            np.uint32
        )
        order = np.argsort(keys, kind="mergesort")
        keys_sorted = keys[order]
        uniq_keys, start_idx, counts = np.unique(
            keys_sorted, return_index=True, return_counts=True
        )
        buckets = {}
        for k, s, c in zip(uniq_keys.tolist(), start_idx.tolist(), counts.tolist()):
            buckets[int(k)] = order[s : s + c].astype(np.int32, copy=False)
        buckets_by_shift[int(sh)] = buckets

    MAX_CAND_PER_KEY = 25000
else:
    buckets_by_shift = {}

_fallback_cand = None


def _unique_int32_deterministic(a: np.ndarray) -> np.ndarray:
    if a.size <= 1:
        return a.astype(np.int32, copy=False)
    return np.unique(a.astype(np.int32, copy=False))


def get_candidate_indices_multi(h: np.uint64):
    global _fallback_cand
    if not have_train:
        return None

    merged_list = []
    for sh in BUCKET_SHIFTS:
        buckets = buckets_by_shift[int(sh)]
        key = int(np.right_shift(h, np.uint64(sh)) & BUCKET_MASK)

        parts = []
        for m in neighbor_masks:
            arr = buckets.get(key ^ m)
            if arr is not None and arr.size:
                parts.append(arr)
        if parts:
            cand = np.concatenate(parts)
            if cand.size > 1:
                cand = _unique_int32_deterministic(cand)
            if cand.size > MAX_CAND_PER_KEY:
                cand = cand[:MAX_CAND_PER_KEY]
            merged_list.append(cand.astype(np.int32, copy=False))

    if not merged_list:
        if _fallback_cand is None:
            desired = 20000
            step = max(1, len(train_hashes) // desired) if len(train_hashes) else 1
            stop = min(len(train_hashes), step * desired)
            _fallback_cand = np.arange(0, stop, step, dtype=np.int32)
        return _fallback_cand

    if len(merged_list) == 1:
        return merged_list[0]

    merged = _unique_int32_deterministic(np.concatenate(merged_list))
    MAX_TOTAL_CANDS = 60000
    if merged.size > MAX_TOTAL_CANDS:
        merged = merged[:MAX_TOTAL_CANDS]
    return merged.astype(np.int32, copy=False)


TOPK_NEIGHBORS = 512

preds = []

global_top5_str = " ".join(global_top5)
train_hotels_local = train_hotels
train_hashes_local = train_hashes


def _hash_one_test(p):
    if not os.path.exists(p):
        return None
    try:
        return ahash64(p)
    except Exception:
        return None


test_hashes = [None] * len(test_paths)
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    fut_to_i = {ex.submit(_hash_one_test, p): i for i, p in enumerate(test_paths)}
    for fu in as_completed(fut_to_i):
        i = fut_to_i[fu]
        test_hashes[i] = fu.result()

for hq in test_hashes:
    try:
        if (hq is None) or (not have_train):
            preds.append(global_top5_str)
            continue

        cand_idx = get_candidate_indices_multi(hq)
        cand_hashes = train_hashes_local[cand_idx]

        dvec = hamming_many_u64(hq, cand_hashes)
        order = np.argsort(dvec, kind="mergesort")
        topn = min(TOPK_NEIGHBORS, order.size)
        top_local = order[:topn]

        sel_cand_idx = cand_idx[top_local]
        sel_hotels = train_hotels_local[sel_cand_idx]
        sel_scores = (64 - dvec[top_local]).astype(np.int32)

        hotel_sum = {}
        hotel_best = {}
        for hid, sc in zip(sel_hotels.tolist(), sel_scores.tolist()):
            prev = hotel_sum.get(hid)
            hotel_sum[hid] = (0 if prev is None else prev) + int(sc)
            pb = hotel_best.get(hid)
            if pb is None or sc > pb:
                hotel_best[hid] = int(sc)

        ordered = [
            str(k)
            for k, _ in sorted(
                hotel_sum.items(),
                key=lambda kv: (
                    -(kv[1] + hotel_best.get(kv[0], 0)),
                    -hotel_best.get(kv[0], 0),
                    str(kv[0]),
                ),
            )[:5]
        ]

        if len(ordered) < 5:
            for hid in global_top5:
                if hid not in ordered:
                    ordered.append(hid)
                if len(ordered) == 5:
                    break
        preds.append(" ".join(ordered[:5]))
    except Exception:
        preds.append(global_top5_str)

submission = pd.DataFrame({"image": sub_df["image"].values, "hotel_id": preds})

assert submission.shape[0] == sub_df.shape[0]
assert list(submission.columns) == ["image", "hotel_id"]
assert submission["hotel_id"].astype(str).str.split().map(len).eq(5).all()

out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print(submission.head())
