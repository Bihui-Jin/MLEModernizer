# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88

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

0.7065357413488229

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01648) has done: 'The timeout is dominated by building a full filename→path index for the entire `train_images` tree (millions of directory entries) and then reading many large images one-by-one. To keep the exact same feature extraction and kNN logic, the optimization focuses on (1) avoiding the global filesystem scan by resolving each needed image path directly from `chain` (the folder name) in `train.csv`, and (2) reading/featurizing images in parallel with a small thread pool (OpenCV releases the GIL in decode/resize/colorspace). We also keep determinism and preserve all computation semantics (same histograms, normalization, kNN) while reducing overhead via preallocation and tighter loops. Paths remain unchanged and the fallback behavior is identical.'
- What this solution (achieved 0.03239) has done: 'Your score is extremely low because the notebook is only indexing the first 20,000 training images, which makes the nearest-neighbor retrieval mostly irrelevant for the hidden test set; the smallest change that should move MAP@5 toward your target is to increase the indexed training set size while keeping the exact same HSV-histogram features and cosine kNN logic. To keep runtime under control without changing semantics, we also (a) ensure OpenCV uses a single thread per decode (since we already parallelize with a thread pool) and (b) deduplicate identical train image filenames (rare but safe) only for path resolution, not for labels. Finally, we keep the same submission formatting but make sure predictions always contain 5 unique hotel_ids (dedup within the top-5 list while preserving order), which can slightly improve MAP@5 without changing the model.'
- What this solution (achieved 0.03649) has done: 'Your current MAP@5 is far below the target because the index still excludes a meaningful fraction of the available training set; the smallest change likely to move score upward is to index all 87,798 training rows while keeping the exact same HSV-histogram feature and cosine kNN logic. To stay within the 600s budget without changing semantics, we avoid building a huge Python `train_args` list (stream through rows), preallocate the feature/label arrays, and reduce per-item overhead in the threadpool pipeline. We also fix a small inefficiency in the “fill to 5” logic by reusing a `seen` set instead of recreating it, keeping identical prediction semantics (5 unique IDs, order-preserving). Paths, feature extraction, kNN, and submission formatting remain the same.'
- What this solution (achieved 0.08408) has done: 'Your current MAP@5 is far below the target mainly because the retrieval features are too weak for this dataset; without changing the model family (still HSV-hist + cosine kNN), the smallest safe improvement is to enrich the histogram representation while keeping the same training/inference loop and metric semantics. I (1) extend the existing HSV histograms with a small 2D joint (H,S) histogram and (2) apply a standard Hellinger (sqrt) transform before L2-normalization, which often improves histogram cosine similarity while remaining the same “histogram→normalize→cosine kNN” core logic. Everything else (paths, full-train indexing, threadpool feature extraction, cosine topk, unique top-5, submission formatting) stays the same. This should move the score upward toward your target without introducing new dependencies or changing the overall approach.'
- What this solution (achieved 0.08435) has done: 'The timeout is dominated by computing features for ~88k train images and then doing a full cosine kNN against them for ~9.7k test images, which is far too much work for 600 seconds. To preserve the exact same feature extraction and similarity logic, the main speedup is to avoid recomputing train features on every run by caching them (memmap on disk) and to stream/train-feature extraction only once; subsequent runs load instantly. Additionally, we reduce Python overhead and memory pressure by using integer-encoded labels (while keeping identical hotel_id strings in the final output), and we make the top‑5 “most frequent hotels” computation linear-time without `np.unique` on an object array. Finally, we tune thread counts/chunk sizes for OpenCV I/O/resize throughput without changing any math.'
- What this solution (achieved 0.0136) has done: 'The timeout is dominated by reading and featurizing all ~88k train images at size=256 and then repeatedly re-reading candidate train images at size=384 during stage-2 reranking. To keep the same feature logic and ranking semantics, the main speedups are: (1) avoid wasted work by building the index on one representative image per hotel_id (still used only for retrieval; predictions remain hotel_id-based), (2) cache and persist the stage-2 (384) train features for the actually-used candidate subset so each candidate image is decoded at most once, and (3) replace Python dict-per-query assembly with array-based computation where equivalent. All paths, feature extraction math, cosine similarity, and 2-stage retrieval/rerank logic are preserved; we only remove redundant decoding and unnecessary train rows while keeping deterministic behavior.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import cv2

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
try:
    cv2.setNumThreads(
        1
    )  # deterministic throughput; avoid oversubscription with our threadpool
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from logging import getLogger
import numpy as np

logger = getLogger("peko")

import time
from contextlib import contextmanager


@contextmanager
def timer(name, logger=None):
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"{name}: {t1 - t0:.2f}s")


BASE = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

CACHE_DIR = "/kaggle/working/feat_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)


def _read_csv_simple(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rows.append(parts)
    return header, rows


def _train_img_path(chain, img_name):
    return os.path.join(TRAIN_IMG_DIR, str(chain), img_name)


def _test_img_path(img_name):
    return os.path.join(TEST_IMG_DIR, img_name)


def _lbp_hist(gray_u8, n_points=8, radius=1, n_bins=16):
    if gray_u8.ndim != 2:
        raise ValueError("gray_u8 must be 2D")
    h, w = gray_u8.shape
    if h < 2 * radius + 1 or w < 2 * radius + 1:
        return np.zeros((n_bins,), dtype=np.float32)

    c = gray_u8[radius : h - radius, radius : w - radius]
    n0 = gray_u8[0 : h - 2 * radius, 0 : w - 2 * radius]
    n1 = gray_u8[0 : h - 2 * radius, radius : w - radius]
    n2 = gray_u8[0 : h - 2 * radius, 2 * radius : w]
    n3 = gray_u8[radius : h - radius, 2 * radius : w]
    n4 = gray_u8[2 * radius : h, 2 * radius : w]
    n5 = gray_u8[2 * radius : h, radius : w - radius]
    n6 = gray_u8[2 * radius : h, 0 : w - 2 * radius]
    n7 = gray_u8[radius : h - radius, 0 : w - 2 * radius]

    code = (
        (n0 >= c).astype(np.uint8)
        | ((n1 >= c).astype(np.uint8) << 1)
        | ((n2 >= c).astype(np.uint8) << 2)
        | ((n3 >= c).astype(np.uint8) << 3)
        | ((n4 >= c).astype(np.uint8) << 4)
        | ((n5 >= c).astype(np.uint8) << 5)
        | ((n6 >= c).astype(np.uint8) << 6)
        | ((n7 >= c).astype(np.uint8) << 7)
    )

    q = (code.astype(np.int32) * n_bins) >> 8
    hist = np.bincount(q.reshape(-1), minlength=n_bins).astype(np.float32)
    return hist


_H_BINS_32 = [32]
_S_BINS_32 = [32]
_V_BINS_32 = [32]
_H_RANGE = [0, 180]
_SV_RANGE = [0, 256]
_HS_BINS = [16, 16]
_HS_RANGE = [0, 180, 0, 256]


def _tile_hist_feats(hsv_tile, gray_tile_u8):
    h, s, v = cv2.split(hsv_tile)

    hist_h = cv2.calcHist([h], [0], None, _H_BINS_32, _H_RANGE)
    hist_s = cv2.calcHist([s], [0], None, _S_BINS_32, _SV_RANGE)
    hist_v = cv2.calcHist([v], [0], None, _V_BINS_32, _SV_RANGE)
    hist_hs = cv2.calcHist([hsv_tile], [0, 1], None, _HS_BINS, _HS_RANGE)

    hist_lbp = _lbp_hist(gray_tile_u8, n_bins=16)

    return np.concatenate(
        (
            hist_h.reshape(-1),
            hist_s.reshape(-1),
            hist_v.reshape(-1),
            hist_hs.reshape(-1),
            hist_lbp.reshape(-1),
        ),
        axis=0,
    ).astype(np.float32)


def _center_crop_square(img_bgr, frac=0.90):
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return img_bgr
    side = int(min(h, w) * frac)
    side = max(1, side)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return img_bgr[y1 : y1 + side, x1 : x1 + side]


def _img_to_feat(img_bgr, size=256):
    img_bgr = _center_crop_square(img_bgr, frac=0.90)
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = []
    feats.append(_tile_hist_feats(hsv, gray))

    h0, w0 = hsv.shape[:2]
    ys = (0, h0 // 2, h0)
    xs = (0, w0 // 2, w0)
    z = np.zeros((32 + 32 + 32 + 16 * 16 + 16,), dtype=np.float32)
    for yi in range(2):
        y1, y2 = ys[yi], ys[yi + 1]
        for xi in range(2):
            x1, x2 = xs[xi], xs[xi + 1]
            tile_hsv = hsv[y1:y2, x1:x2]
            tile_g = gray[y1:y2, x1:x2]
            if tile_hsv.size == 0 or tile_g.size == 0:
                feats.append(z)
            else:
                feats.append(_tile_hist_feats(tile_hsv, tile_g))

    feat = np.concatenate(feats, axis=0).astype(np.float32)

    feat = np.sqrt(feat)  # Hellinger transform (as before)
    n = np.linalg.norm(feat) + 1e-12
    feat /= n
    return feat


def _batched_topk_cosine(query_feats, index_feats, topk=5, batch=256):
    query_feats = np.ascontiguousarray(query_feats, dtype=np.float32)
    index_feats = np.ascontiguousarray(index_feats, dtype=np.float32)
    qn = query_feats.shape[0]
    out = np.empty((qn, topk), dtype=np.int32)

    index_T = index_feats.T  # view, no copy

    for i in range(0, qn, batch):
        q = query_feats[i : i + batch]
        sims = q @ index_T  # (b, n_index)

        idx_part = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
        row = np.arange(idx_part.shape[0])[:, None]
        idx_sorted = idx_part[row, np.argsort(-sims[row, idx_part], axis=1)]
        out[i : i + q.shape[0]] = idx_sorted.astype(np.int32, copy=False)
    return out


def _unique_topk_preserve_order(ids, k=5):
    out = []
    seen = set()
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= k:
            break
    return out


with timer("Load CSVs", logger=logger):
    tr_header, tr_rows = _read_csv_simple(TRAIN_CSV)
    ss_header, ss_rows = _read_csv_simple(SAMPLE_SUB)

    col_idx = {c: i for i, c in enumerate(tr_header)}
    tr_images = [r[col_idx["image"]] for r in tr_rows]
    tr_hotels = [r[col_idx["hotel_id"]] for r in tr_rows]
    tr_chains = [r[col_idx["chain"]] for r in tr_rows]

    test_images = [r[0] for r in ss_rows]  # sample_submission columns: image, hotel_id



## === cell 1
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib


def _imread_fast(path, target_size):
    if target_size == 256:
        im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
        if im is None:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
        return im
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _read_and_feat_train(args):
    img_name, hid_code, chain, size = args
    p = _train_img_path(chain, img_name)
    im = _imread_fast(p, size)
    if im is None:
        return None
    return (_img_to_feat(im, size=size), hid_code)


def _read_and_feat_test(img_name, size=256):
    p = _test_img_path(img_name)
    if not os.path.exists(p):
        return None
    im = _imread_fast(p, size)
    if im is None:
        return None
    return (_img_to_feat(im, size=size), img_name)


base_dim = 32 + 32 + 32 + 16 * 16 + 16
feat_dim = 5 * base_dim

with timer("Build hotel_id <-> code maps", logger=logger):
    hid_to_code = {}
    code_to_hid = []
    tr_hotel_codes = np.empty((len(tr_hotels),), dtype=np.int32)
    for i, hid in enumerate(tr_hotels):
        c = hid_to_code.get(hid)
        if c is None:
            c = len(code_to_hid)
            hid_to_code[hid] = c
            code_to_hid.append(hid)
        tr_hotel_codes[i] = c
    code_to_hid = np.array(code_to_hid, dtype=object)

with timer("Compute/load train features", logger=logger):
    n_use = len(tr_images)

    feat_mm_path = os.path.join(CACHE_DIR, f"train_feat_full_{n_use}_{feat_dim}.mmap")
    label_npy_path = os.path.join(CACHE_DIR, f"train_labels_full_{n_use}.npy")
    ok_npy_path = os.path.join(CACHE_DIR, f"train_n_ok_full_{n_use}.npy")

    can_load = (
        os.path.exists(feat_mm_path)
        and os.path.exists(label_npy_path)
        and os.path.exists(ok_npy_path)
    )

    if can_load:
        n_ok = int(np.load(ok_npy_path))
        feat_index = np.memmap(
            feat_mm_path, mode="r", dtype=np.float32, shape=(n_ok, feat_dim)
        )
        label_index = np.load(label_npy_path, allow_pickle=False)
        if len(label_index) != n_ok:
            can_load = False

    if not can_load:
        cpu = os.cpu_count() or 1
        n_workers = min(12, max(2, cpu))

        feat_index_mm = np.memmap(
            feat_mm_path, mode="w+", dtype=np.float32, shape=(n_use, feat_dim)
        )
        label_index_tmp = np.empty((n_use,), dtype=np.int32)

        def _train_args_gen_full():
            for i in range(n_use):
                yield (tr_images[i], int(tr_hotel_codes[i]), tr_chains[i], 256)

        n_ok = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for res in ex.map(
                _read_and_feat_train, _train_args_gen_full(), chunksize=256
            ):
                if res is None:
                    continue
                f, hid_code = res
                feat_index_mm[n_ok] = f
                label_index_tmp[n_ok] = hid_code
                n_ok += 1

        if n_ok == 0:
            raise RuntimeError("No train images could be read to build the index.")

        np.save(ok_npy_path, np.array(n_ok, dtype=np.int64))
        np.save(
            label_npy_path,
            label_index_tmp[:n_ok].astype(np.int32, copy=False),
            allow_pickle=False,
        )
        feat_index_mm.flush()
        del feat_index_mm

        feat_index = np.memmap(
            feat_mm_path, mode="r", dtype=np.float32, shape=(n_ok, feat_dim)
        )
        label_index = np.load(label_npy_path, allow_pickle=False)

    cnt = Counter(label_index.tolist())
    top5_codes = [hid_code for hid_code, _ in cnt.most_common(5)]
    top5_str = " ".join(str(code_to_hid[c]) for c in top5_codes)

with timer("Build per-hotel mean features (for fast stage-1)", logger=logger):
    n_hotels = int(label_index.max()) + 1
    hotel_sum = np.zeros((n_hotels, feat_dim), dtype=np.float32)
    hotel_cnt = np.zeros((n_hotels,), dtype=np.int32)

    np.add.at(hotel_sum, label_index, feat_index)
    np.add.at(hotel_cnt, label_index, 1)

    hotel_mean = hotel_sum / np.maximum(hotel_cnt[:, None], 1).astype(np.float32)
    del hotel_sum
    norms = np.linalg.norm(hotel_mean, axis=1) + 1e-12
    hotel_mean = (hotel_mean / norms[:, None]).astype(np.float32, copy=False)

with timer("Build hotel->train-index lists", logger=logger):
    order = np.argsort(label_index, kind="mergesort")  # stable, deterministic
    sorted_labels = label_index[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_labels[1:] != sorted_labels[:-1], True]
    )
    uniq_labels = sorted_labels[boundaries[:-1]]
    starts = boundaries[:-1]
    ends = boundaries[1:]

    h_start = np.full((n_hotels,), -1, dtype=np.int32)
    h_end = np.full((n_hotels,), -1, dtype=np.int32)
    h_start[uniq_labels] = starts.astype(np.int32, copy=False)
    h_end[uniq_labels] = ends.astype(np.int32, copy=False)

with timer("Compute/load test features (stage-1, size=256)", logger=logger):
    test_256_mm = os.path.join(
        CACHE_DIR, f"test_feat256_{len(test_images)}_{feat_dim}.mmap"
    )
    test_256_names = os.path.join(
        CACHE_DIR, f"test_feat256_{len(test_images)}_names.npy"
    )

    feat_test_256 = None
    valid_test_images = None
    if os.path.exists(test_256_mm) and os.path.exists(test_256_names):
        valid_test_images = np.load(test_256_names, allow_pickle=False).tolist()
        feat_test_256 = np.memmap(
            test_256_mm,
            mode="r",
            dtype=np.float32,
            shape=(len(valid_test_images), feat_dim),
        )
    else:
        tfeats_256 = []
        valid_test_images = []

        cpu = os.cpu_count() or 1
        n_workers = min(12, max(2, cpu))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for res in ex.map(
                lambda x: _read_and_feat_test(x, size=256), test_images, chunksize=256
            ):
                if res is None:
                    continue
                f, img_name = res
                tfeats_256.append(f)
                valid_test_images.append(img_name)

        if len(tfeats_256) == 0:
            print(
                "[WARN] No test images found/read. Writing submission with most frequent hotels in train subset."
            )
            with open("submission.csv", "w", encoding="utf-8") as f:
                f.write("image,hotel_id\n")
                for img_name in test_images:
                    f.write(f"{img_name},{top5_str}\n")
        else:
            feat_test_256_arr = np.stack(tfeats_256, axis=0).astype(
                np.float32, copy=False
            )
            mm = np.memmap(
                test_256_mm, mode="w+", dtype=np.float32, shape=feat_test_256_arr.shape
            )
            mm[:] = feat_test_256_arr
            mm.flush()
            del mm
            np.save(
                test_256_names,
                np.array(valid_test_images, dtype=object),
                allow_pickle=False,
            )
            feat_test_256 = np.memmap(
                test_256_mm, mode="r", dtype=np.float32, shape=feat_test_256_arr.shape
            )

    if feat_test_256 is None or valid_test_images is None:
        pass
    else:
        with timer("kNN stage-1 search on hotel-means (top-200 hotels)", logger=logger):
            topk_stage1 = 200
            nn_hotel_200 = _batched_topk_cosine(
                feat_test_256, hotel_mean, topk=topk_stage1, batch=512
            )

        with timer("Expand stage-1 hotels -> train image candidates", logger=logger):
            MAX_CAND_IMAGES_PER_QUERY = 2000

            nn_idx_200 = np.empty((feat_test_256.shape[0], topk_stage1), dtype=np.int32)
            per_query_cands = []

            for qi in range(feat_test_256.shape[0]):
                hotels = nn_hotel_200[qi]
                cands = []
                for hc in hotels:
                    s = int(h_start[hc])
                    e = int(h_end[hc])
                    if s < 0:
                        continue
                    idxs = order[s:e]  # train indices for that hotel
                    if idxs.size:
                        cands.append(idxs)
                if cands:
                    cands = np.concatenate(cands, axis=0)
                else:
                    cands = np.empty((0,), dtype=np.int32)

                if cands.size > MAX_CAND_IMAGES_PER_QUERY:
                    cands = cands[:MAX_CAND_IMAGES_PER_QUERY]

                per_query_cands.append(cands)
                if cands.size >= topk_stage1:
                    nn_idx_200[qi] = cands[:topk_stage1]
                elif cands.size > 0:
                    pad = np.resize(cands, topk_stage1)
                    nn_idx_200[qi] = pad
                else:
                    nn_idx_200[qi].fill(0)

        with timer("Compute/load test features (stage-2, size=384)", logger=logger):
            test_384_mm = os.path.join(
                CACHE_DIR, f"test_feat384_{len(valid_test_images)}_{feat_dim}.mmap"
            )
            if os.path.exists(test_384_mm):
                feat_test_384 = np.memmap(
                    test_384_mm,
                    mode="r",
                    dtype=np.float32,
                    shape=(len(valid_test_images), feat_dim),
                )
            else:
                cpu = os.cpu_count() or 1
                n_workers = min(12, max(2, cpu))
                tfeats_384 = []
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    for res in ex.map(
                        lambda x: _read_and_feat_test(x, size=384),
                        valid_test_images,
                        chunksize=128,
                    ):
                        if res is None:
                            tfeats_384.append(None)
                        else:
                            f, _ = res
                            tfeats_384.append(f)

                for i in range(len(tfeats_384)):
                    if tfeats_384[i] is None:
                        tfeats_384[i] = np.array(feat_test_256[i], copy=False)
                feat_test_384_arr = np.stack(tfeats_384, axis=0).astype(
                    np.float32, copy=False
                )

                mm = np.memmap(
                    test_384_mm,
                    mode="w+",
                    dtype=np.float32,
                    shape=feat_test_384_arr.shape,
                )
                mm[:] = feat_test_384_arr
                mm.flush()
                del mm
                feat_test_384 = np.memmap(
                    test_384_mm,
                    mode="r",
                    dtype=np.float32,
                    shape=feat_test_384_arr.shape,
                )

        with timer("Stage-2 rerank candidates -> top-5", logger=logger):
            li = label_index  # int32 codes
            pred_lines = {}

            if per_query_cands:
                cand_union = np.unique(np.concatenate(per_query_cands, axis=0)).astype(
                    np.int32, copy=False
                )
            else:
                cand_union = np.unique(nn_idx_200.reshape(-1)).astype(
                    np.int32, copy=False
                )

            h = hashlib.md5(cand_union.tobytes()).hexdigest()[:16]
            cand_mm_path = os.path.join(
                CACHE_DIR, f"train_feat384_cands_{h}_{cand_union.size}_{feat_dim}.mmap"
            )
            cand_idx_path = os.path.join(
                CACHE_DIR, f"train_feat384_cands_{h}_{cand_union.size}_idx.npy"
            )

            can_load_cand = os.path.exists(cand_mm_path) and os.path.exists(
                cand_idx_path
            )
            if can_load_cand:
                prev_idx = np.load(cand_idx_path, allow_pickle=False)
                if prev_idx.shape == cand_union.shape and np.array_equal(
                    prev_idx, cand_union
                ):
                    cand_feats384 = np.memmap(
                        cand_mm_path,
                        mode="r",
                        dtype=np.float32,
                        shape=(cand_union.size, feat_dim),
                    )
                else:
                    can_load_cand = False

            if not can_load_cand:
                cpu = os.cpu_count() or 1
                n_workers = min(12, max(2, cpu))

                cand_feats384_mm = np.memmap(
                    cand_mm_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(cand_union.size, feat_dim),
                )

                def _cand_args_gen():
                    for idx_in_index in cand_union:
                        tr_i = int(idx_in_index)
                        yield (tr_images[tr_i], int(li[tr_i]), tr_chains[tr_i], 384)

                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    for k, res in enumerate(
                        ex.map(_read_and_feat_train, _cand_args_gen(), chunksize=128)
                    ):
                        if res is None:
                            cand_feats384_mm[k] = feat_index[int(cand_union[k])]
                        else:
                            f, _hid_code = res
                            cand_feats384_mm[k] = f

                np.save(
                    cand_idx_path,
                    cand_union.astype(np.int32, copy=False),
                    allow_pickle=False,
                )
                cand_feats384_mm.flush()
                del cand_feats384_mm

                cand_feats384 = np.memmap(
                    cand_mm_path,
                    mode="r",
                    dtype=np.float32,
                    shape=(cand_union.size, feat_dim),
                )

            max_idx = int(cand_union.max()) if cand_union.size else -1
            pos_map = np.full((max_idx + 1,), -1, dtype=np.int32)
            pos_map[cand_union] = np.arange(cand_union.size, dtype=np.int32)

            for q_i, img_name in enumerate(valid_test_images):
                cands = (
                    per_query_cands[q_i]
                    if q_i < len(per_query_cands)
                    else nn_idx_200[q_i]
                )
                if cands.size == 0:
                    pred_lines[img_name] = top5_str
                    continue

                pos = pos_map[cands]
                m = pos >= 0
                cands2 = cands[m]
                pos2 = pos[m]
                if cands2.size == 0:
                    pred_lines[img_name] = top5_str
                    continue

                cand_feats = cand_feats384[pos2]  # gather
                sims = cand_feats @ feat_test_384[q_i]
                best5_local = np.argsort(-sims)[:5]
                top_idxs = cands2[best5_local]
                hcodes = li[top_idxs].tolist()
                uniq5_codes = _unique_topk_preserve_order(hcodes, k=5)

                if len(uniq5_codes) < 5:
                    seen = set(uniq5_codes)
                    for extra in top5_codes:
                        if extra not in seen:
                            uniq5_codes.append(extra)
                            seen.add(extra)
                        if len(uniq5_codes) >= 5:
                            break

                pred_lines[img_name] = " ".join(
                    str(code_to_hid[c]) for c in uniq5_codes[:5]
                )

        with timer("Write submission.csv", logger=logger):
            with open("submission.csv", "w", encoding="utf-8") as f:
                f.write("image,hotel_id\n")
                for img_name in test_images:
                    f.write(f"{img_name},{pred_lines.get(img_name, top5_str)}\n")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1046541955.py in <cell line: 0>()
    203             mm.flush()
    204             del mm
--> 205             np.save(
    206                 test_256_names,
    207                 np.array(valid_test_images, dtype=object),

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in save(file, arr, allow_pickle, fix_imports)
    544     with file_ctx as fid:
    545         arr = np.asanyarray(arr)
--> 546         format.write_array(fid, arr, allow_pickle=allow_pickle,
    547                            pickle_kwargs=dict(fix_imports=fix_imports))
    548 

/usr/local/lib/python3.11/dist-packages/numpy/lib/format.py in write_array(fp, array, version, allow_pickle, pickle_kwargs)
    713         # directly.  Instead, we will pickle it out
    714         if not allow_pickle:
--> 715             raise ValueError("Object arrays cannot be saved when "
    716                              "allow_pickle=False")
    717         if pickle_kwargs is None:

ValueError: Object arrays cannot be saved when allow_pickle=False

## === cell 2
import os

if not os.path.exists("submission.csv"):
    raise FileNotFoundError("submission.csv was not created.")
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        line = f.readline()
        if not line:
            break
        print(line.rstrip("\n"))

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1175639347.py in <cell line: 0>()
      2 
      3 if not os.path.exists("submission.csv"):
----> 4     raise FileNotFoundError("submission.csv was not created.")
      5 with open("submission.csv", "r", encoding="utf-8") as f:
      6     for _ in range(5):

FileNotFoundError: submission.csv was not created.
