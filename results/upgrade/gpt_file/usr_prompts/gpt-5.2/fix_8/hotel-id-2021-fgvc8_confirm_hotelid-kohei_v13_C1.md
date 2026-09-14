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
import os, sys, glob, shutil, subprocess
from pathlib import Path

subprocess.run("ls -lha /kaggle/input/", shell=True, check=False)
subprocess.run("ls -lha /kaggle/input | sed -n '1,200p'", shell=True, check=False)



## === cell 1
import pandas as pd
from pathlib import Path

WORKING = Path("/kaggle/working")
WORKING.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv")
if not TRAIN_CSV.exists():
    TRAIN_CSV = Path("/kaggle/input/train.csv")

SAMPLE_CSV = Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
if not SAMPLE_CSV.exists():
    SAMPLE_CSV = Path("/kaggle/input/sample_submission.csv")

TEST_IMAGES_DIR = Path("/kaggle/input/hotel-id-2021-fgvc8/test_images")
if not TEST_IMAGES_DIR.exists():
    TEST_IMAGES_DIR = Path("/kaggle/input/test_images")
if not TEST_IMAGES_DIR.exists():
    candidates = list(Path("/kaggle/input").rglob("test_images"))
    candidates = [p for p in candidates if p.is_dir()]
    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        TEST_IMAGES_DIR = candidates[0]

TRAIN_IMAGES_DIR = Path("/kaggle/input/hotel-id-2021-fgvc8/train_images")
if not TRAIN_IMAGES_DIR.exists():
    TRAIN_IMAGES_DIR = Path("/kaggle/input/train_images")
if not TRAIN_IMAGES_DIR.exists():
    candidates = list(Path("/kaggle/input").rglob("train_images"))
    candidates = [p for p in candidates if p.is_dir()]
    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        TRAIN_IMAGES_DIR = candidates[0]

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"train.csv not found at {TRAIN_CSV}")
if not SAMPLE_CSV.exists():
    raise FileNotFoundError(f"sample_submission.csv not found at {SAMPLE_CSV}")

train = pd.read_csv(TRAIN_CSV)
sample = pd.read_csv(SAMPLE_CSV)

required_cols = {"chain", "hotel_id", "image"}
missing = required_cols - set(train.columns)
if missing:
    raise ValueError(f"train.csv missing required columns: {sorted(missing)}")

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + global_top5[:1] * 5)[:5]
global_pred_str = " ".join(global_top5)

train["chain"] = train["chain"].fillna(0).astype(int)
chain_top5 = {}
vc = train.groupby("chain")["hotel_id"].value_counts()
for chain_id in vc.index.get_level_values(0).unique().tolist():
    topk = vc.loc[chain_id].head(5)
    ids = topk.index.astype(str).tolist()
    if len(ids) < 5:
        ids = (ids + global_top5)[:5]
    chain_top5[int(chain_id)] = ids


def parse_chain_from_image_field(image_field: str):
    s = str(image_field)
    if "/" not in s:
        return None
    parts = s.split("/")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0])
    except Exception:
        return None


image_to_chain = {}
parsed_from_sample = 0
for img in sample["image"].astype(str).tolist():
    ch = parse_chain_from_image_field(img)
    if ch is not None:
        image_to_chain[Path(img).name] = ch
        parsed_from_sample += 1

scanned = 0
if TEST_IMAGES_DIR.exists():
    need_scan = True
    if parsed_from_sample == len(sample):
        need_scan = False
    if need_scan:
        for img_path in TEST_IMAGES_DIR.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            ]:
                continue
            parent = img_path.parent.name
            try:
                chain_id = int(parent)
            except Exception:
                continue
            image_to_chain[img_path.name] = chain_id
            scanned += 1

print("Global top-5 hotel_ids used (fallback):", global_top5)
print(f"Parsed chain from sample image paths for {parsed_from_sample} rows.")
print(f"Scanned {scanned} image files under test_images for chain mapping.")
print(f"Total chain mappings available (by basename): {len(image_to_chain)}")
print(
    f"test_images dir used: {TEST_IMAGES_DIR if TEST_IMAGES_DIR.exists() else 'NOT FOUND (using global fallback)'}"
)
print(
    f"train_images dir used: {TRAIN_IMAGES_DIR if TRAIN_IMAGES_DIR.exists() else 'NOT FOUND (hash retrieval disabled)'}"
)



## === cell 2
import json
import time
from collections import defaultdict, Counter

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "PIL is required (usually available as Pillow on Kaggle)."
    ) from e

HASH_CACHE = WORKING / "phash_cache_v1.json"
INDEX_CACHE = WORKING / "phash_index_v1.json"

TEST_BASENAME_TO_PATH = {}
if TEST_IMAGES_DIR.exists():
    t_scan = time.time()
    for p in TEST_IMAGES_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        ]:
            TEST_BASENAME_TO_PATH[p.name] = p
    print(
        f"Built TEST_BASENAME_TO_PATH for {len(TEST_BASENAME_TO_PATH)} files in {time.time()-t_scan:.1f}s"
    )
else:
    print(
        "TEST_IMAGES_DIR not found; TEST_BASENAME_TO_PATH empty (retrieval may be limited)."
    )


def dhash64(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(g.getdata())
    h = 0
    bit = 0
    for y in range(hash_size):
        row = pixels[y * (hash_size + 1) : (y + 1) * (hash_size + 1)]
        for x in range(hash_size):
            v = 1 if row[x] > row[x + 1] else 0
            h |= v << bit
            bit += 1
    return h


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def safe_open_image(path: Path):
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy()
    except Exception:
        return None


def resolve_train_image_path(chain_id: int, image_name: str) -> Path:
    return TRAIN_IMAGES_DIR / str(chain_id) / image_name


def resolve_test_image_path(chain_id: int, image_name: str) -> Path:
    return TEST_IMAGES_DIR / str(chain_id) / image_name


MAX_HOTELS = 2000  # fewer hotels, but more/better exemplars each -> higher match rate
MAX_IMAGES_PER_HOTEL = (
    12  # more exemplars -> better recall for near-duplicate retrieval
)
TOPK_RETRIEVAL = 50  # take more closest items before voting -> more stable hotel vote
VOTE_TOPN = 5

t0 = time.time()

hotel_freq = train["hotel_id"].value_counts()
top_hotels = hotel_freq.head(MAX_HOTELS).index.tolist()
train_small = train[train["hotel_id"].isin(top_hotels)].copy()

train_small = train_small.sort_values(
    ["hotel_id", "timestamp", "image"], kind="mergesort"
)
selected_rows = []
for hid, g in train_small.groupby("hotel_id", sort=False):
    n = len(g)
    if n <= MAX_IMAGES_PER_HOTEL:
        selected_rows.append(g)
    else:
        idxs = [
            int(round(i * (n - 1) / (MAX_IMAGES_PER_HOTEL - 1)))
            for i in range(MAX_IMAGES_PER_HOTEL)
        ]
        selected_rows.append(g.iloc[idxs])
train_small = pd.concat(selected_rows, axis=0).reset_index(drop=True)

if HASH_CACHE.exists():
    with open(HASH_CACHE, "r") as f:
        phash_cache = json.load(f)
else:
    phash_cache = {}

if INDEX_CACHE.exists():
    with open(INDEX_CACHE, "r") as f:
        index_data = json.load(f)
    bucket_index = {
        int(k): [(int(h), str(hid)) for h, hid in v]
        for k, v in index_data["bucket_index"].items()
    }
    indexed_count = int(index_data.get("indexed_count", 0))
    meta = index_data.get("meta", {})
else:
    bucket_index = defaultdict(list)
    indexed_count = 0
    meta = {}

need_rebuild = (
    (not INDEX_CACHE.exists())
    or (indexed_count < len(train_small))
    or (meta.get("MAX_HOTELS") != MAX_HOTELS)
    or (meta.get("MAX_IMAGES_PER_HOTEL") != MAX_IMAGES_PER_HOTEL)
    or (meta.get("hash") != "dhash64")
)

if need_rebuild:
    bucket_index = defaultdict(list)
    computed = 0
    missing_imgs = 0
    for row in train_small.itertuples(index=False):
        img_name = str(row.image)
        chain_id = int(row.chain)
        hotel_id = str(row.hotel_id)
        key = f"tr::{chain_id}/{img_name}"
        if key in phash_cache:
            h = int(phash_cache[key])
        else:
            img_path = resolve_train_image_path(chain_id, img_name)
            im = safe_open_image(img_path)
            if im is None:
                missing_imgs += 1
                continue
            h = dhash64(im)
            phash_cache[key] = int(h)
        bucket = (h >> 48) & 0xFFFF
        bucket_index[bucket].append((int(h), hotel_id))
        computed += 1

    with open(HASH_CACHE, "w") as f:
        json.dump(phash_cache, f)
    serializable_index = {str(k): v for k, v in bucket_index.items()}
    with open(INDEX_CACHE, "w") as f:
        json.dump(
            {
                "bucket_index": serializable_index,
                "indexed_count": computed,
                "meta": {
                    "MAX_HOTELS": MAX_HOTELS,
                    "MAX_IMAGES_PER_HOTEL": MAX_IMAGES_PER_HOTEL,
                    "hash": "dhash64",
                },
            },
            f,
        )

    print(
        f"Built hash index: {computed} train images hashed, missing/unreadable={missing_imgs}, buckets={len(bucket_index)}"
    )
else:
    print(f"Loaded hash index from cache: buckets={len(bucket_index)}")

print(f"Hash/index prep time: {time.time() - t0:.1f}s")


def retrieval_top5_for_test(img_name: str, chain_id: int | None):
    test_path = None
    if chain_id is not None and TEST_IMAGES_DIR.exists():
        p = resolve_test_image_path(int(chain_id), img_name)
        if p.exists():
            test_path = p
    if test_path is None:
        test_path = TEST_BASENAME_TO_PATH.get(img_name, None)
    if test_path is None:
        return None

    key = f"te::{(chain_id if chain_id is not None else 'na')}/{img_name}"
    if key in phash_cache:
        th = int(phash_cache[key])
    else:
        im = safe_open_image(test_path)
        if im is None:
            return None
        th = dhash64(im)
        phash_cache[key] = int(th)

    bucket = (th >> 48) & 0xFFFF

    candidates = []
    for nb in (bucket, (bucket - 1) & 0xFFFF, (bucket + 1) & 0xFFFF):
        if nb in bucket_index:
            candidates.extend(bucket_index[nb])

    if not candidates:
        return None

    scored = []
    for h, hid in candidates:
        d = hamming64(th, int(h))
        scored.append((d, hid))
    scored.sort(key=lambda x: x[0])

    top = scored[:TOPK_RETRIEVAL]
    if not top:
        return None

    votes = Counter()
    for rank, (d, hid) in enumerate(top):
        w = max(1, 64 - int(d)) + max(0, TOPK_RETRIEVAL - rank)
        votes[hid] += w

    best = [hid for hid, _ in votes.most_common(VOTE_TOPN)]
    pad = []
    if chain_id is not None:
        pad = chain_top5.get(int(chain_id), global_top5)
    else:
        pad = global_top5
    out = []
    for x in best + list(map(str, pad)) + list(map(str, global_top5)):
        if x not in out:
            out.append(x)
        if len(out) == 5:
            break
    if len(out) < 5:
        out = (out + list(map(str, global_top5)))[:5]
    return out




## === cell 3
sub = sample.copy()

preds = []
unknown_chain = 0
used_retrieval = 0
retrieval_failed = 0

for img in sub["image"].astype(str).tolist():
    base = Path(img).name
    ch = image_to_chain.get(base, None)
    if ch is None:
        unknown_chain += 1

    top5 = retrieval_top5_for_test(base, ch)
    if top5 is not None:
        preds.append(" ".join(top5))
        used_retrieval += 1
    else:
        retrieval_failed += 1
        if ch is None:
            preds.append(global_pred_str)
        else:
            ids = chain_top5.get(int(ch), global_top5)
            preds.append(" ".join(list(map(str, ids))))

sub["hotel_id"] = preds

out_path = WORKING / "submission.csv"
sub.to_csv(out_path, index=False)

print(f"Unknown chain (fallback risk): {unknown_chain} / {len(sub)}")
print(f"Used retrieval for: {used_retrieval} / {len(sub)}")
print(f"Retrieval failed for: {retrieval_failed} / {len(sub)}")
print(f"Wrote submission to: {out_path}")
print(sub.head(3).to_string(index=False))

with open(HASH_CACHE, "w") as f:
    json.dump(phash_cache, f)



## === cell 4
import pandas as pd
import shutil
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT_SAMPLE = Path("/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv")
if not INPUT_SAMPLE.exists():
    INPUT_SAMPLE = Path("/kaggle/input/sample_submission.csv")


def looks_like_submission(csv_path: Path) -> bool:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if list(df.columns) != ["image", "hotel_id"]:
        return False
    if df.shape[0] < 1:
        return False
    return True


candidate_paths = []
preferred = WORKING / "submission.csv"
if preferred.exists():
    candidate_paths.append(preferred)

search_roots = [
    WORKING,
    Path("/kaggle"),
    Path("/tmp"),
]
for root in search_roots:
    for p in root.rglob("*.csv"):
        if str(p).startswith("/kaggle/input"):
            continue
        candidate_paths.append(p)

seen = set()
candidate_paths_unique = []
for p in candidate_paths:
    ps = str(p)
    if ps not in seen:
        seen.add(ps)
        candidate_paths_unique.append(p)

valid_candidates = [
    p for p in candidate_paths_unique if p.exists() and looks_like_submission(p)
]

if not valid_candidates:
    raise FileNotFoundError(
        "No valid submission CSV found.\n"
        "Searched /kaggle/working, /kaggle (excluding /kaggle/input), and /tmp for *.csv with columns [image, hotel_id]."
    )

valid_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
best = valid_candidates[0]

dst = WORKING / "submission.csv"
if best.resolve() != dst.resolve():
    shutil.copy2(best, dst)

sample = pd.read_csv(INPUT_SAMPLE)
sub = pd.read_csv(dst)

if list(sub.columns) != ["image", "hotel_id"]:
    raise ValueError(f"submission.csv has wrong columns: {list(sub.columns)}")

if sub.shape[0] != sample.shape[0]:
    raise ValueError(
        f"submission.csv row count {sub.shape[0]} != sample_submission row count {sample.shape[0]}"
    )

if not sub["image"].equals(sample["image"]):
    if set(sub["image"]) == set(sample["image"]):
        sub = sub.set_index("image").loc[sample["image"]].reset_index()
        sub.to_csv(dst, index=False)
    else:
        raise ValueError(
            "submission.csv images do not match sample_submission images (set mismatch)."
        )

print(f"Using submission file: {dst} (copied from: {best})")
print(sub.head(3).to_string(index=False))
print("\nsubmission.csv written to /kaggle/working/submission.csv")
