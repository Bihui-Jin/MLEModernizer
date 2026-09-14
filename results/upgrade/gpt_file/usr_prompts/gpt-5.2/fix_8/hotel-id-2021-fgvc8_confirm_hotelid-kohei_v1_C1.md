# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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


def _tile_hist_feats(hsv_tile):
    h, s, v = cv2.split(hsv_tile)

    hist_h = cv2.calcHist([h], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([s], [0], None, [32], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [32], [0, 256])
    hist_hs = cv2.calcHist([hsv_tile], [0, 1], None, [16, 16], [0, 180, 0, 256])

    return np.concatenate(
        (
            hist_h.reshape(-1),
            hist_s.reshape(-1),
            hist_v.reshape(-1),
            hist_hs.reshape(-1),
        ),
        axis=0,
    ).astype(np.float32)


def _img_to_feat(img_bgr, size=256):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    feats = []

    feats.append(_tile_hist_feats(hsv))

    h0, w0 = hsv.shape[:2]
    ys = [0, h0 // 2, h0]
    xs = [0, w0 // 2, w0]
    for yi in range(2):
        for xi in range(2):
            y1, y2 = ys[yi], ys[yi + 1]
            x1, x2 = xs[xi], xs[xi + 1]
            tile = hsv[y1:y2, x1:x2]
            if tile.size == 0:
                feats.append(np.zeros((32 + 32 + 32 + 16 * 16,), dtype=np.float32))
            else:
                feats.append(_tile_hist_feats(tile))

    feat = np.concatenate(feats, axis=0).astype(np.float32)

    feat = np.sqrt(feat)  # Hellinger transform (as before)
    n = np.linalg.norm(feat) + 1e-12
    feat /= n
    return feat


def _batched_topk_cosine(query_feats, index_feats, topk=5, batch=64):
    qn = query_feats.shape[0]
    out = np.empty((qn, topk), dtype=np.int32)
    index_T = index_feats.T
    for i in range(0, qn, batch):
        q = query_feats[i : i + batch]
        sims = q @ index_T
        idx_part = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
        row = np.arange(idx_part.shape[0])[:, None]
        idx_sorted = idx_part[row, np.argsort(-sims[row, idx_part], axis=1)]
        out[i : i + q.shape[0]] = idx_sorted
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


def _read_and_feat_train(args):
    img_name, hid, chain = args
    p = _train_img_path(chain, img_name)
    im = cv2.imread(p)
    if im is None:
        return None
    return (_img_to_feat(im), hid)


def _read_and_feat_test(img_name):
    p = _test_img_path(img_name)
    if not os.path.exists(p):
        return None
    im = cv2.imread(p)
    if im is None:
        return None
    return (_img_to_feat(im), img_name)


with timer("Compute train features", logger=logger):
    MAX_TRAIN = len(tr_images)
    n_use = min(len(tr_images), MAX_TRAIN)

    def _train_args_gen():
        for i in range(n_use):
            yield (tr_images[i], tr_hotels[i], tr_chains[i])

    n_workers = min(8, max(1, (os.cpu_count() or 1) // 2))

    base_dim = 32 + 32 + 32 + 16 * 16
    feat_dim = 5 * base_dim

    feat_index = np.empty((n_use, feat_dim), dtype=np.float32)
    label_index = np.empty((n_use,), dtype=object)

    n_ok = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_read_and_feat_train, _train_args_gen(), chunksize=64):
            if res is None:
                continue
            f, hid = res
            feat_index[n_ok] = f
            label_index[n_ok] = hid
            n_ok += 1

    if n_ok == 0:
        raise RuntimeError("No train images could be read to build the fallback index.")

    feat_index = feat_index[:n_ok]
    label_index = label_index[:n_ok]


with timer("Compute test features", logger=logger):
    tfeats = []
    valid_test_images = []

    n_workers = min(8, max(1, (os.cpu_count() or 1) // 2))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_read_and_feat_test, test_images, chunksize=64):
            if res is None:
                continue
            f, img_name = res
            tfeats.append(f)
            valid_test_images.append(img_name)

    uniq, cnt = np.unique(label_index, return_counts=True)
    top5 = uniq[np.argsort(-cnt)][:5].tolist()
    top5_str = " ".join(map(str, top5))

    if len(tfeats) == 0:
        print(
            "[WARN] No test images found/read. Writing submission with most frequent hotels in train subset."
        )
        with open("submission.csv", "w", encoding="utf-8") as f:
            f.write("image,hotel_id\n")
            for img_name in test_images:
                f.write(f"{img_name},{top5_str}\n")
    else:
        feat_test = np.stack(tfeats, axis=0)

        with timer("kNN search", logger=logger):
            topk = 5
            nn_idx = _batched_topk_cosine(feat_test, feat_index, topk=topk, batch=256)

        with timer("Write submission.csv", logger=logger):
            pred_lines = {}
            li = label_index
            for img_name, idxs in zip(valid_test_images, nn_idx):
                hids = li[idxs].tolist()
                uniq5 = _unique_topk_preserve_order(hids, k=5)

                if len(uniq5) < 5:
                    seen = set(uniq5)
                    for extra in top5:
                        if extra not in seen:
                            uniq5.append(extra)
                            seen.add(extra)
                        if len(uniq5) >= 5:
                            break

                pred_lines[img_name] = " ".join(map(str, uniq5[:5]))

            with open("submission.csv", "w", encoding="utf-8") as f:
                f.write("image,hotel_id\n")
                for img_name in test_images:
                    f.write(f"{img_name},{pred_lines.get(img_name, top5_str)}\n")



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
