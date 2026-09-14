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

0.7433042855940036

# 6. Current score

0.02053

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your current notebook doesn’t yield a score because it likely fails before producing `submission.csv` (missing private wheel inputs, compilation of `inplace_abn`, or the `peko` module not being available), so Kaggle never gets a valid submission to evaluate. To move toward the target score, the smallest meaningful change is to (1) stop depending on unavailable external wheels and (2) add a robust fallback that always creates a valid `submission.csv` in the required format using only the provided competition files. This let you submit and obtain a baseline score; once you report that score, we can make the smallest next-step adjustment toward the target. The fallback is metric-safe (MAP@5 expects 5 IDs) and uses the most frequent hotel IDs from `train.csv` as reasonable priors.'
- What this solution (achieved 0.00889) has done: 'The timeout is dominated by (1) repeatedly decoding/resizing images in pure-Python loops and (2) doing a full dense similarity scan for every test image. To keep identical logic (same grayscale thumbnail feature, same cosine similarity, same vote ranking), I speed it up by caching image features to disk, parallelizing feature extraction, batching matrix multiplications, and replacing the per-test Python `Counter` loop with a vectorized bincount-based vote/score aggregation that is mathematically equivalent. I also avoid unnecessary work (e.g., not sampling train rows via pandas when not needed, using fast path construction, and using `PIL.ImageFile.LOAD_TRUNCATED_IMAGES` to reduce decode failures). These changes preserve the algorithm and outputs up to negligible float differences while substantially reducing wall time.'
- What this solution (achieved 0.00714) has done: 'The timeout is dominated by per-image PIL decoding/resizing for tens of thousands of training images plus an O(test * train) similarity search done in pure NumPy with repeated allocations. I keep the same thumbnail-feature + L2-normalization + blockwise dot-product NN retrieval logic, but remove avoidable overhead by (1) eliminating per-image disk `.npy` cache I/O and MD5 hashing, (2) switching to a faster image decoding path (OpenCV if available, otherwise PIL) while keeping identical grayscale/resize semantics, (3) precomputing and reusing train-block transposes without storing extra tuples, and (4) avoiding `np.concatenate` inside the top‑k loop by maintaining fixed-size top‑k buffers. These changes are provably equivalent for the algorithm’s output given the same floating point math, and they cut constant factors substantially to fit under 600 seconds.'
- What this solution (achieved 0.00931) has done: 'The timeout is dominated by decoding and resizing tens of thousands of JPEGs from `train_images` plus repeated small `q @ block.T` calls per test image. To keep the same retrieval logic and scoring, I keep the exact thumbnail feature definition and NN voting/ranking, but speed it up by (1) caching features only for training images (avoids useless test-cache growth), (2) using OpenCV’s internal threading and fewer Python-layer threadpool overheads, (3) computing similarities in vectorized batches via matrix multiplication (same dot-products, just grouped), and (4) replacing per-query per-block top-k merging with a single global `argpartition` on the concatenated similarity vector (equivalent result). These changes preserve identical semantics (same feature vectors and similarity definition), but cut Python overhead and improve BLAS utilization so it fits under 600s.'
- What this solution (achieved 0.02053) has done: 'Your current fallback is a very weak image-retrieval baseline (tiny 32×32 grayscale thumbnails + crude vote/sim ranking), which explains the low MAP@5. To move the score substantially toward the 0.74 target without changing the overall approach (still NN retrieval over fixed image features + dot-product similarity + per-query top-5), the smallest meaningful upgrade is to use a stronger but still simple handcrafted global descriptor: a larger grayscale thumbnail plus a normalized HOG-like gradient histogram, concatenated and L2-normalized. This preserves the core logic (feature extraction → cosine similarity via dot product → aggregate NN hits per hotel → rank → fill with global priors) while improving visual matching signal a lot. I also fix a small bug in the test image path candidates list and keep runtime bounded by keeping the same train capping, batching, and vectorized similarity.'

# 9. Code solution

## === cell 0
import os
import subprocess
from pathlib import Path
import pandas as pd


def run(cmd):
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=False, text=True, capture_output=True)


DATA_CANDIDATES = [
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8"),
    Path("/kaggle/data/hotel-id-2021-fgvc8"),
    Path("/kaggle/data/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]
DATA_DIR = None
for p in DATA_CANDIDATES:
    if (p / "sample_submission.csv").exists() and (p / "train.csv").exists():
        DATA_DIR = p
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv and train.csv in expected Kaggle paths."
    )

print("Using DATA_DIR:", DATA_DIR)

peko_deps = Path("/kaggle/input/pekolib-deps")
peko_lib = Path("/kaggle/input/pekolib")
can_try_peko = peko_deps.exists() and peko_lib.exists()

submission_path = Path("submission.csv")

if can_try_peko:
    r1 = run("ls -lha /kaggle/input/")
    r2 = run(
        "pip -q install /kaggle/input/pekolib-deps/*.whl && pip -q install /kaggle/input/pekolib/*.whl"
    )
    r3 = run(
        "cp -r /kaggle/input/pekolib-deps/inplace_abn-1.0.12/inplace_abn /tmp/inplace_abn && cd /tmp/inplace_abn && pip -q install ."
    )
    r4 = run("python -m peko.subs.hotelid.v9")

    if submission_path.exists():
        print("Found submission.csv generated by peko pipeline.")
    else:
        print(
            "peko pipeline did not produce submission.csv; will generate improved fallback submission."
        )
else:
    print("pekolib inputs not found; will generate improved fallback submission.")

if not submission_path.exists():
    import time
    import numpy as np

    try:
        import cv2  # type: ignore

        _HAS_CV2 = True
        try:
            cv2.setNumThreads(max(1, min(8, os.cpu_count() or 4)))
        except Exception:
            pass
    except Exception:
        _HAS_CV2 = False
        from PIL import Image, ImageFile, ImageOps

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    np.random.seed(42)

    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()

    train_csv = pd.read_csv(DATA_DIR / "train.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    TRAIN_IMG_CANDIDATES = [
        DATA_DIR / "train_images",
        DATA_DIR / "train_images" / "train_images",
        Path("/kaggle/input/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/train_images/train_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/train_images"),
        Path(
            "/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/train_images/train_images"
        ),
        Path("/kaggle/data/hotel-id-2021-fgvc8/train_images"),
        Path("/kaggle/data/hotel-id-2021-fgvc8/train_images/train_images"),
        Path("/kaggle/input/train_images"),
        Path("/kaggle/data/train_images"),
    ]
    TEST_IMG_CANDIDATES = [
        DATA_DIR / "test_images",
        DATA_DIR / "test_images" / "test_images",
        Path("/kaggle/input/hotel-id-2021-fgvc8/test_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/test_images/test_images"),
        Path("/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/test_images"),
        Path(
            "/kaggle/input/hotel-id-2021-fgvc8/hotel-id-2021-fgvc8/test_images/test_images"
        ),
        Path("/kaggle/data/hotel-id-2021-fgvc8/test_images"),
        Path("/kaggle/data/hotel-id-2021-fgvc8/test_images/test_images"),
        Path("/kaggle/input/test_images"),
        Path("/kaggle/data/test_images"),
    ]

    TRAIN_IMG_DIR = None
    for p in TRAIN_IMG_CANDIDATES:
        if p.exists():
            TRAIN_IMG_DIR = p
            break
    TEST_IMG_DIR = None
    for p in TEST_IMG_CANDIDATES:
        if p.exists():
            TEST_IMG_DIR = p
            break

    def make_pred_str(ids):
        ids = [str(x) for x in ids if str(x).strip() != ""]
        if len(ids) < 5:
            ids = (ids + ["0"] * 5)[:5]
        else:
            ids = ids[:5]
        return " ".join(ids)

    if TRAIN_IMG_DIR is None or TEST_IMG_DIR is None:
        top5 = train_csv["hotel_id"].value_counts().head(5).index.astype(str).tolist()
        pred_str = make_pred_str(top5)
        submission = sample_sub.copy()
        submission["hotel_id"] = pred_str
        submission.to_csv(submission_path, index=False)
        print("Images not found; wrote simple fallback submission.csv with:", pred_str)
    else:
        print("Using TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
        print("Using TEST_IMG_DIR:", TEST_IMG_DIR)

        vc = train_csv["hotel_id"].value_counts()
        global_top = vc.head(50).index.astype(str).tolist()

        MAX_TRAIN_INDEX = int(os.environ.get("MAX_TRAIN_INDEX", "85000"))
        PER_HOTEL_CAP = int(os.environ.get("PER_HOTEL_CAP", "50"))

        THUMB_W, THUMB_H = 48, 48  # larger than 32x32 to retain more visual detail
        NBINS = 8  # gradient orientation bins
        GRID = 4  # 4x4 spatial grid for HOG-like pooling

        FEAT_DIM = THUMB_W * THUMB_H + (GRID * GRID * NBINS)

        _mem_feat_cache = {}
        _mem_cache_max = int(os.environ.get("MEM_FEAT_CACHE_MAX", "200000"))

        if not _HAS_CV2:
            try:
                _BILINEAR = Image.Resampling.BILINEAR
            except Exception:
                _BILINEAR = Image.BILINEAR

        def _hog_like(gray_01: np.ndarray) -> np.ndarray:
            gx = np.zeros_like(gray_01, dtype=np.float32)
            gy = np.zeros_like(gray_01, dtype=np.float32)
            gx[:, 1:-1] = gray_01[:, 2:] - gray_01[:, :-2]
            gy[1:-1, :] = gray_01[2:, :] - gray_01[:-2, :]

            mag = np.sqrt(gx * gx + gy * gy)
            ang = (np.arctan2(gy, gx) + np.pi) * (NBINS / (2.0 * np.pi))  # [0, NBINS)
            bin0 = np.floor(ang).astype(np.int32)
            bin0 = np.clip(bin0, 0, NBINS - 1)

            H, W = gray_01.shape
            cell_h = H // GRID
            cell_w = W // GRID
            hog = np.zeros((GRID, GRID, NBINS), dtype=np.float32)

            for cy in range(GRID):
                ys = cy * cell_h
                ye = (cy + 1) * cell_h if cy < GRID - 1 else H
                for cx in range(GRID):
                    xs = cx * cell_w
                    xe = (cx + 1) * cell_w if cx < GRID - 1 else W
                    b = bin0[ys:ye, xs:xe].reshape(-1)
                    m = mag[ys:ye, xs:xe].reshape(-1)
                    hog[cy, cx] = np.bincount(b, weights=m, minlength=NBINS).astype(
                        np.float32
                    )
            return hog.reshape(-1)

        def load_feat(img_path: Path, do_cache: bool):
            key = str(img_path)
            if do_cache:
                v = _mem_feat_cache.get(key)
                if v is not None:
                    return v
            try:
                if _HAS_CV2:
                    im = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        return None
                    im = cv2.resize(
                        im, (THUMB_W, THUMB_H), interpolation=cv2.INTER_LINEAR
                    )
                    gray = im.astype(np.float32, copy=False) * (1.0 / 255.0)  # (H,W)
                else:
                    with Image.open(img_path) as im:
                        im = ImageOps.grayscale(im)
                        im = im.resize((THUMB_W, THUMB_H), _BILINEAR)
                        gray = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)

                flat = gray.reshape(-1).astype(np.float32, copy=False)
                flat = flat - flat.mean()

                hog = _hog_like(gray)

                feat = np.concatenate([flat, hog], axis=0).astype(
                    np.float32, copy=False
                )
                n = np.linalg.norm(feat) + 1e-6
                feat = (feat / n).astype(np.float32, copy=False)

                if do_cache and (len(_mem_feat_cache) < _mem_cache_max):
                    _mem_feat_cache[key] = feat
                return feat
            except Exception:
                return None

        train_rows = train_csv[["image", "chain", "hotel_id"]].copy()

        if len(train_rows) > MAX_TRAIN_INDEX:
            train_rows = train_rows.sort_values(["hotel_id", "image"], kind="mergesort")
            capped = train_rows.groupby("hotel_id", sort=False).head(PER_HOTEL_CAP)

            if len(capped) > MAX_TRAIN_INDEX:
                capped = capped.sort_values(
                    ["hotel_id", "image"], kind="mergesort"
                ).head(MAX_TRAIN_INDEX)

            train_rows = capped.reset_index(drop=True)

        train_images = train_rows["image"].tolist()
        train_chains = train_rows["chain"].astype(int).tolist()
        train_hotel_ids_str = train_rows["hotel_id"].astype(str).tolist()
        train_paths = [
            TRAIN_IMG_DIR / str(c) / img for img, c in zip(train_images, train_chains)
        ]

        feats = np.zeros((len(train_paths), FEAT_DIM), dtype=np.float32)
        valid_mask = np.zeros(len(train_paths), dtype=bool)

        max_workers = min(16, (os.cpu_count() or 4))
        chunk_size = int(os.environ.get("IO_CHUNK_SIZE", "4096"))

        def _chunked_indices_paths(paths, chunk_size):
            n = len(paths)
            for s in range(0, n, chunk_size):
                yield s, paths[s : s + chunk_size]

        def _load_chunk_to_arrays(args):
            start_idx, paths_chunk = args
            for j, p in enumerate(paths_chunk):
                i = start_idx + j
                f = load_feat(p, do_cache=True)
                if f is not None:
                    feats[i] = f
                    valid_mask[i] = True

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(
                ex.map(
                    _load_chunk_to_arrays,
                    _chunked_indices_paths(train_paths, chunk_size),
                )
            )

        if valid_mask.sum() < 1000:
            top5 = (
                train_csv["hotel_id"].value_counts().head(5).index.astype(str).tolist()
            )
            pred_str = make_pred_str(top5)
            submission = sample_sub.copy()
            submission["hotel_id"] = pred_str
            submission.to_csv(submission_path, index=False)
            print(
                "Too few readable train images; wrote simple fallback submission.csv with:",
                pred_str,
            )
        else:
            feats = feats[valid_mask]
            hotel_ids = [
                h for h, m in zip(train_hotel_ids_str, valid_mask.tolist()) if m
            ]

            unique_h = pd.unique(pd.Series(hotel_ids, dtype="string"))
            h2code = {h: i for i, h in enumerate(unique_h.tolist())}
            code2h = unique_h.tolist()
            hotel_codes = np.fromiter(
                (h2code[h] for h in hotel_ids), dtype=np.int32, count=len(hotel_ids)
            )

            feats = np.ascontiguousarray(feats, dtype=np.float32)
            hotel_codes = np.ascontiguousarray(hotel_codes, dtype=np.int32)

            test_images = sample_sub["image"].tolist()

            k_nn = int(os.environ.get("K_NN", "200"))
            batch_size = 128

            preds = []
            global_top5_str = global_top[:5]

            featT = feats.T  # (D, N)

            def _load_batch_chunk(args):
                start_idx, paths_chunk = args
                out = []
                for j, p in enumerate(paths_chunk):
                    local_i = start_idx + j
                    f = load_feat(p, do_cache=False)
                    if f is not None:
                        out.append((local_i, f))
                return out

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for b0 in range(0, len(test_images), batch_size):
                    batch_names = test_images[b0 : b0 + batch_size]
                    batch_paths = [TEST_IMG_DIR / n for n in batch_names]

                    batch_feats = np.zeros(
                        (len(batch_paths), FEAT_DIM), dtype=np.float32
                    )
                    batch_valid = np.zeros(len(batch_paths), dtype=bool)

                    b_chunk = max(64, len(batch_paths) // max_workers or 1)
                    for out in ex.map(
                        _load_batch_chunk, _chunked_indices_paths(batch_paths, b_chunk)
                    ):
                        for local_i, f in out:
                            batch_feats[local_i] = f
                            batch_valid[local_i] = True

                    sims_batch = batch_feats @ featT

                    for i_in_batch in range(len(batch_names)):
                        if not batch_valid[i_in_batch]:
                            preds.append(make_pred_str(global_top5_str))
                            continue

                        sims = sims_batch[i_in_batch]

                        if k_nn >= sims.shape[0]:
                            nn_idx = np.argsort(-sims)
                        else:
                            nn_idx = np.argpartition(-sims, k_nn)[:k_nn]
                            nn_idx = nn_idx[np.argsort(-sims[nn_idx])]
                        nn_sims = sims[nn_idx].astype(np.float32, copy=False)

                        nn_codes = hotel_codes[nn_idx]

                        present_codes, inv = np.unique(nn_codes, return_inverse=True)
                        vote_counts = np.bincount(
                            inv, minlength=present_codes.shape[0]
                        ).astype(np.int32, copy=False)

                        best_sim = np.full(
                            present_codes.shape[0], -np.inf, dtype=np.float32
                        )
                        np.maximum.at(best_sim, inv, nn_sims)

                        order = np.lexsort((best_sim, vote_counts))[::-1]
                        ranked_codes = present_codes[order]

                        out_ids = []
                        seen = set()
                        for c in ranked_codes:
                            h = code2h[int(c)]
                            if h not in seen:
                                out_ids.append(h)
                                seen.add(h)
                            if len(out_ids) == 5:
                                break
                        if len(out_ids) < 5:
                            for h in global_top:
                                if h not in seen:
                                    out_ids.append(h)
                                    seen.add(h)
                                if len(out_ids) == 5:
                                    break

                        preds.append(make_pred_str(out_ids))

            submission = sample_sub.copy()
            submission["hotel_id"] = preds
            submission.to_csv(submission_path, index=False)

            dt = time.time() - t0
            print(
                f"Wrote improved fallback submission.csv using NN retrieval. Time: {dt:.1f}s"
            )

sub = pd.read_csv(submission_path)
assert list(sub.columns) == [
    "image",
    "hotel_id",
], f"Bad columns: {sub.columns.tolist()}"
assert len(sub) > 0, "Empty submission"
lens = sub["hotel_id"].astype(str).str.split().map(len)
assert (
    lens.min() == 5
), f"Predictions must have exactly 5 ids; got min len={lens.min()}, max len={lens.max()}"

print("\nsubmission.csv head:")
print(sub.head())
print("\nsubmission.csv rows:", len(sub))
print("\nSaved at:", submission_path.resolve())




## === cell 1
import pandas as pd

print(pd.read_csv("submission.csv").head().to_string(index=False))
