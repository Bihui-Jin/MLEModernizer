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

# 5. Code solution

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
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    np.random.seed(42)

    from concurrent.futures import ThreadPoolExecutor
    from itertools import islice
    import heapq

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
        (
            Path("/kaggle/data/hotel-id-2021-fgvcvc8/test_images/test_images")
            if False
            else Path("/kaggle/data/hotel-id-2021-fgvc8/test_images/test_images")
        ),
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
        global_top = [
            str(x)
            for x in heapq.nlargest(50, vc.index.tolist(), key=lambda k: int(vc.loc[k]))
        ]

        MAX_TRAIN_INDEX = int(os.environ.get("MAX_TRAIN_INDEX", "60000"))

        THUMB_W, THUMB_H = 32, 32
        FEAT_DIM = THUMB_W * THUMB_H

        cache_dir = Path("/kaggle/working/_feat_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        def _cache_key(p: Path) -> str:
            s = str(p)
            return s.replace("/", "_").replace("\\", "_")

        def load_feat(img_path: Path):
            try:
                cpath = cache_dir / (_cache_key(img_path) + ".npy")
                if cpath.exists():
                    arr = np.load(cpath)
                    if arr.shape == (FEAT_DIM,) and arr.dtype == np.float32:
                        return arr
                im = (
                    Image.open(img_path)
                    .convert("L")
                    .resize((THUMB_W, THUMB_H), Image.BILINEAR)
                )
                arr = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
                arr -= arr.mean()
                n = np.linalg.norm(arr) + 1e-6
                arr /= n
                try:
                    np.save(cpath, arr.astype(np.float32, copy=False))
                except Exception:
                    pass
                return arr
            except Exception:
                return None

        train_rows = train_csv[["image", "chain", "hotel_id"]].copy()

        if len(train_rows) > MAX_TRAIN_INDEX:
            per_hotel_cap = int(os.environ.get("PER_HOTEL_CAP", "12"))

            train_rows = train_rows.sort_values(["hotel_id", "image"], kind="mergesort")
            capped = train_rows.groupby("hotel_id", sort=False).head(per_hotel_cap)

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

        max_workers = min(32, (os.cpu_count() or 4))

        def _load_chunk(start_idx, paths_chunk):
            out = []
            for j, p in enumerate(paths_chunk):
                i = start_idx + j
                f = load_feat(p)
                if f is not None:
                    out.append((i, f))
            return out

        def _chunked_indices_paths(paths, chunk_size):
            n = len(paths)
            for s in range(0, n, chunk_size):
                yield s, paths[s : s + chunk_size]

        chunk_size = int(os.environ.get("IO_CHUNK_SIZE", "256"))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for out in ex.map(
                lambda sp: _load_chunk(sp[0], sp[1]),
                _chunked_indices_paths(train_paths, chunk_size),
            ):
                for i, f in out:
                    feats[i] = f
                    valid_mask[i] = True

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

            test_images = sample_sub["image"].tolist()

            k_nn = int(os.environ.get("K_NN", "50"))

            batch_size = 64
            preds = []

            global_top5_str = global_top[:5]

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for b0 in range(0, len(test_images), batch_size):
                    batch_names = test_images[b0 : b0 + batch_size]
                    batch_paths = [TEST_IMG_DIR / n for n in batch_names]

                    batch_feats = np.zeros(
                        (len(batch_paths), FEAT_DIM), dtype=np.float32
                    )
                    batch_valid = np.zeros(len(batch_paths), dtype=bool)

                    for out in ex.map(
                        lambda sp: _load_chunk(sp[0], sp[1]),
                        _chunked_indices_paths(
                            batch_paths, max(32, len(batch_paths) // max_workers or 1)
                        ),
                    ):
                        for i, f in out:
                            batch_feats[i] = f
                            batch_valid[i] = True

                    if batch_valid.any():
                        sims_batch = batch_feats[batch_valid] @ feats.T
                    else:
                        sims_batch = None

                    vb_idx = 0
                    for i_in_batch, img_name in enumerate(batch_names):
                        if not batch_valid[i_in_batch]:
                            preds.append(make_pred_str(global_top5_str))
                            continue

                        sims = sims_batch[vb_idx]
                        vb_idx += 1

                        if k_nn >= sims.shape[0]:
                            nn_idx = np.argsort(-sims)
                        else:
                            nn_idx = np.argpartition(-sims, k_nn)[:k_nn]
                            nn_idx = nn_idx[np.argsort(-sims[nn_idx])]

                        nn_codes = hotel_codes[nn_idx]
                        nn_sims = sims[nn_idx].astype(np.float32, copy=False)

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

                        out = []
                        seen = set()
                        for c in ranked_codes:
                            h = code2h[int(c)]
                            if h not in seen:
                                out.append(h)
                                seen.add(h)
                            if len(out) == 5:
                                break
                        if len(out) < 5:
                            for h in global_top:
                                if h not in seen:
                                    out.append(h)
                                    seen.add(h)
                                if len(out) == 5:
                                    break

                        preds.append(make_pred_str(out))

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
