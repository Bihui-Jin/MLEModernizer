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
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from PIL import Image

DATA_DIR = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SUB_PATH = "submission.csv"

print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))

train = pd.read_csv(TRAIN_CSV)
sample = pd.read_csv(SAMPLE_SUB)

train["hotel_id"] = train["hotel_id"].astype(int)
train["chain"] = train["chain"].astype(int)
train["timestamp"] = (
    pd.to_numeric(train["timestamp"], errors="coerce").fillna(0).astype("int64")
)

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 * 5)[:5]

unknown_chain_top5 = (
    train.loc[train["chain"] == 0, "hotel_id"]
    .value_counts()
    .head(5)
    .index.astype(str)
    .tolist()
)
if len(unknown_chain_top5) < 5:
    unknown_chain_top5 = (unknown_chain_top5 + global_top5)[:5]
unknown_chain_top5 = (unknown_chain_top5 * 5)[:5]

K_PRIOR = 50
chain_prior = (
    train.groupby("chain")["hotel_id"]
    .value_counts()
    .groupby(level=0)
    .head(K_PRIOR)
    .reset_index(name="cnt")
)
chain_prior_map = {}
for ch, grp in chain_prior.groupby("chain"):
    chain_prior_map[int(ch)] = grp["hotel_id"].astype(int).tolist()

NEIGHBORS_PER_HOTEL = 10
neighbor_counts = {}
for ch, dfc in (
    train[["chain", "hotel_id", "timestamp"]]
    .sort_values(["chain", "timestamp"])
    .groupby("chain", sort=False)
):
    h = dfc["hotel_id"].to_numpy()
    for a, b in zip(h[:-1], h[1:]):
        if a == b:
            continue
        key_a = (int(ch), int(a))
        key_b = (int(ch), int(b))
        da = neighbor_counts.get(key_a)
        if da is None:
            da = {}
            neighbor_counts[key_a] = da
        db = neighbor_counts.get(key_b)
        if db is None:
            db = {}
            neighbor_counts[key_b] = db
        da[int(b)] = da.get(int(b), 0) + 1
        db[int(a)] = db.get(int(a), 0) + 1

neighbor_top_map = {}
for key, d in neighbor_counts.items():
    top = sorted(d.items(), key=lambda x: (-x[1], x[0]))[:NEIGHBORS_PER_HOTEL]
    neighbor_top_map[key] = [hid for hid, _ in top]

_chain_prefix_re = re.compile(r"^(\d{1,3})[^0-9]")
_chain_prefix_alt_re = re.compile(r"^(\d{1,3})")


def infer_chain_from_filename(fn: str):
    base = os.path.basename(fn)
    m = _chain_prefix_re.match(base)
    if m is None:
        m = _chain_prefix_alt_re.match(base)
    if m is None:
        return None
    try:
        ch = int(m.group(1))
    except Exception:
        return None
    if 0 <= ch <= 500:
        return ch
    return None


def build_test_image_chain_maps(test_img_dir: str):
    fn_to_chain = {}
    rel_to_chain = {}

    if not os.path.isdir(test_img_dir):
        return fn_to_chain, rel_to_chain

    candidate_roots = [test_img_dir]
    nested = os.path.join(test_img_dir, "test_images")
    if os.path.isdir(nested):
        candidate_roots.append(nested)

    for root_base in candidate_roots:
        for root, dirs, files in os.walk(root_base):
            parent = os.path.basename(root)
            try:
                ch = int(parent)
            except ValueError:
                ch = None

            for fn in files:
                if not fn.lower().endswith(".jpg"):
                    continue
                if ch is not None and fn not in fn_to_chain:
                    fn_to_chain[fn] = ch
                rel = os.path.relpath(os.path.join(root, fn), root_base).replace(
                    "\\", "/"
                )
                if ch is not None and rel not in rel_to_chain:
                    rel_to_chain[rel] = ch

    return fn_to_chain, rel_to_chain


image_to_chain_from_fn, image_to_chain_from_rel = build_test_image_chain_maps(
    TEST_IMG_DIR
)


def infer_chain_for_image(img_name: str):
    ch = image_to_chain_from_fn.get(img_name)
    if ch is not None:
        return int(ch)

    ch = image_to_chain_from_rel.get(img_name.replace("\\", "/"))
    if ch is not None:
        return int(ch)

    norm = img_name.replace("\\", "/")
    parts = norm.split("/")
    if len(parts) >= 2:
        try:
            ch2 = int(parts[-2])
            return ch2
        except Exception:
            pass

    return infer_chain_from_filename(img_name)


def make_top5_for_chain(ch: int):
    base = chain_prior_map.get(ch, [])
    if not base:
        return [int(x) for x in global_top5]

    seed = base[:10]
    cand = []
    for hid in seed:
        cand.append(int(hid))
        cand.extend([int(x) for x in neighbor_top_map.get((ch, int(hid)), [])])

    seen = set()
    out = []
    for hid in cand:
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
        if len(out) >= 5:
            return out

    for hid in base:
        hid = int(hid)
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
        if len(out) >= 5:
            return out

    for s in global_top5:
        hid = int(s)
        if hid in seen:
            continue
        seen.add(hid)
        out.append(hid)
        if len(out) >= 5:
            return out

    return (out * 5)[:5]


def finalize_top5(top5_int, fallback_list_str):
    seen = set()
    out = []
    for hid in top5_int:
        hid = int(hid)
        if hid in seen:
            continue
        seen.add(hid)
        out.append(str(hid))
        if len(out) == 5:
            return out
    for s in fallback_list_str:
        if s in out:
            continue
        out.append(s)
        if len(out) == 5:
            return out
    return (out * 5)[:5]


_cached_chain_top5_str = {}
for ch in chain_prior_map.keys():
    _cached_chain_top5_str[int(ch)] = finalize_top5(
        make_top5_for_chain(int(ch)), global_top5
    )

print("Loaded train/sample.")
print("Global top5:", global_top5)




## === cell 1
def dhash_64(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.uint8)  # (8, 9)
    diff = (a[:, :-1] > a[:, 1:]).reshape(-1).astype(np.uint8)  # 64 bits, row-major
    packed = np.packbits(diff, bitorder="little")  # 8 bytes
    return int.from_bytes(packed.tobytes(), byteorder="little", signed=False)


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_test_image_path_map(test_img_dir: str):
    path_map = {}
    if not os.path.isdir(test_img_dir):
        return path_map

    candidate_roots = [test_img_dir]
    nested = os.path.join(test_img_dir, "test_images")
    if os.path.isdir(nested):
        candidate_roots.append(nested)

    for root_base in candidate_roots:
        for root, _, files in os.walk(root_base):
            for fn in files:
                if fn.lower().endswith(".jpg") and fn not in path_map:
                    path_map[fn] = os.path.join(root, fn)
    return path_map


_test_path_map = build_test_image_path_map(TEST_IMG_DIR)


def resolve_image_path(base_dir: str, image_name: str) -> str:
    p = _test_path_map.get(image_name)
    if p is not None:
        return p
    p1 = os.path.join(base_dir, image_name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(base_dir, "test_images", image_name)
    if os.path.exists(p2):
        return p2
    p3 = os.path.join(base_dir, image_name.replace("\\", "/"))
    if os.path.exists(p3):
        return p3
    p4 = os.path.join(base_dir, "test_images", image_name.replace("\\", "/"))
    if os.path.exists(p4):
        return p4
    return p1  # fallback


def resolve_train_path(train_img_dir: str, chain: int, image_name: str) -> str:
    return os.path.join(train_img_dir, str(chain), image_name)


MAX_TRAIN_IMAGES_TOTAL = 25000  # overall cap
MAX_IMAGES_PER_HOTEL = 5  # was 3
TOP_MATCHES = 40  # retrieve top-N hashes then vote to top-5 hotels

t0 = time.time()

hotel_freq = train["hotel_id"].value_counts()
train_sorted = train.copy()
train_sorted["freq"] = train_sorted["hotel_id"].map(hotel_freq)
train_sorted = train_sorted.sort_values(["freq", "timestamp"], ascending=[False, False])

selected = []
per_hotel_count = defaultdict(int)
for row in train_sorted.itertuples(index=False):
    if len(selected) >= MAX_TRAIN_IMAGES_TOTAL:
        break
    hid = int(row.hotel_id)
    if per_hotel_count[hid] >= MAX_IMAGES_PER_HOTEL:
        continue
    ch = int(row.chain)
    img_name = str(row.image)
    path = resolve_train_path(TRAIN_IMG_DIR, ch, img_name)
    if not os.path.exists(path):
        continue
    selected.append((path, hid))
    per_hotel_count[hid] += 1


def _hash_one(item):
    path, hid = item
    try:
        with Image.open(path) as im:
            h = dhash_64(im)
        return (h, hid, 0)
    except Exception:
        return (0, 0, 1)


n_workers = min(32, (os.cpu_count() or 4) * 2)
train_hashes = []
read_fail = 0
with ThreadPoolExecutor(max_workers=n_workers) as ex:
    for h, hid, failed in ex.map(_hash_one, selected, chunksize=128):
        read_fail += failed
        if not failed:
            train_hashes.append((h, hid))

print(
    f"Built train hashes: {len(train_hashes)} (unique hotels: {len(per_hotel_count)}) read_fail={read_fail} time={time.time()-t0:.1f}s"
)

USE_RETRIEVAL = len(train_hashes) >= 500

if USE_RETRIEVAL:
    train_hash_arr = np.fromiter(
        (h for h, _ in train_hashes), dtype=np.uint64, count=len(train_hashes)
    )
    train_hid_arr = np.fromiter(
        (hid for _, hid in train_hashes), dtype=np.int32, count=len(train_hashes)
    )

    PREFIX_BITS = 16
    _prefix_shift = 64 - PREFIX_BITS
    train_prefix = (train_hash_arr >> np.uint64(_prefix_shift)).astype(np.uint16)

    order = np.argsort(train_prefix, kind="stable")
    train_prefix_sorted = train_prefix[order]
    uniq, start_idx, counts = np.unique(
        train_prefix_sorted, return_index=True, return_counts=True
    )
    prefix_to_range = {
        int(p): (int(s), int(s + c)) for p, s, c in zip(uniq, start_idx, counts)
    }
    train_order = order  # indices into original arrays

    _pc8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount16(xu16: np.ndarray) -> np.ndarray:
        lo = (xu16 & np.uint16(0xFF)).astype(np.uint8)
        hi = (xu16 >> np.uint16(8)).astype(np.uint8)
        return (_pc8[lo] + _pc8[hi]).astype(np.uint8)

    def popcount_u64(x: np.ndarray) -> np.ndarray:
        xb = x.view(np.uint8).reshape(-1, 8)
        return _pc8[xb].sum(axis=1).astype(np.uint8)

    existing_prefixes = uniq.astype(np.uint16)

    _prefix_neighbors_cache = {}

    def _get_prefix_neighbors_for(p: int):
        v = _prefix_neighbors_cache.get(p)
        if v is not None:
            return v
        d = _popcount16(np.bitwise_xor(existing_prefixes, np.uint16(p))).astype(
            np.uint8
        )
        ord2 = np.lexsort((existing_prefixes.astype(np.uint32), d.astype(np.uint32)))
        out = (existing_prefixes[ord2], d[ord2])
        _prefix_neighbors_cache[p] = out
        return out

    def get_candidate_indices_for_hash(
        th_u64: np.uint64, min_candidates: int
    ) -> np.ndarray:
        p = int(
            (th_u64 >> np.uint64(_prefix_shift)) & np.uint64((1 << PREFIX_BITS) - 1)
        )
        neigh, dists = _get_prefix_neighbors_for(p)

        total = 0
        last_dist = -1
        used = []
        for pref, dd in zip(neigh, dists):
            dd = int(dd)
            if dd != last_dist and total >= min_candidates:
                break
            last_dist = dd
            rng = prefix_to_range.get(int(pref))
            if rng is None:
                continue
            s, e = rng
            used.append((s, e))
            total += e - s

        if total == 0:
            return np.empty((0,), dtype=np.int64)

        out = np.empty((total,), dtype=np.int64)
        pos = 0
        for s, e in used:
            seg = train_order[s:e]
            n = seg.size
            out[pos : pos + n] = seg
            pos += n
        return out




## === cell 2
images = sample["image"].astype(str).tolist()

preds = []
t1 = time.time()
test_read_fail = 0

SIM_POWER = 2.0

W_PRIOR = 0.30  # was 0.15

for img_name in images:
    top5 = None

    if USE_RETRIEVAL:
        path = resolve_image_path(TEST_IMG_DIR, img_name)
        try:
            with Image.open(path) as im:
                th = np.uint64(dhash_64(im))

            cand_idx = get_candidate_indices_for_hash(
                th, min_candidates=max(TOP_MATCHES * 50, 2000)
            )
            if cand_idx.size == 0:
                raise RuntimeError("No candidates")

            cand_hash = train_hash_arr[cand_idx]
            cand_hid = train_hid_arr[cand_idx]

            d = popcount_u64(np.bitwise_xor(cand_hash, th)).astype(np.int16)

            if d.size > TOP_MATCHES:
                topk = np.argpartition(d, TOP_MATCHES - 1)[:TOP_MATCHES]
            else:
                topk = np.arange(d.size)
            topk = topk[np.argsort(d[topk], kind="stable")]

            top_hids = cand_hid[topk].astype(np.int32, copy=False)
            top_d = d[topk].astype(np.float32, copy=False)

            w = 1.0 / np.power(1.0 + top_d, SIM_POWER)

            uh, inv = np.unique(top_hids, return_inverse=True)
            score = np.bincount(inv, weights=w, minlength=uh.size).astype(np.float32)

            ch = infer_chain_for_image(img_name)
            if ch is not None:
                prior_list = chain_prior_map.get(int(ch), [])
                if prior_list:
                    prior_set = set(prior_list[:K_PRIOR])
                    prior_bonus = np.array(
                        [W_PRIOR if int(h) in prior_set else 0.0 for h in uh],
                        dtype=np.float32,
                    )
                    score = score + prior_bonus

            ordv = np.lexsort((uh.astype(np.int64), (-score).astype(np.float64)))
            top5_int = uh[ordv][:5].astype(int).tolist()

            top5 = finalize_top5(top5_int, global_top5)
        except Exception:
            test_read_fail += 1
            top5 = None

    if top5 is None:
        ch = infer_chain_for_image(img_name)
        if ch is not None:
            top5 = _cached_chain_top5_str.get(int(ch))
            if top5 is None:
                top5_int = make_top5_for_chain(int(ch))
                top5 = finalize_top5(top5_int, global_top5)
                _cached_chain_top5_str[int(ch)] = top5
        else:
            top5 = finalize_top5([int(x) for x in unknown_chain_top5], global_top5)

    preds.append(" ".join(top5))

sub = pd.DataFrame({"image": images, "hotel_id": preds})
assert list(sub.columns) == ["image", "hotel_id"]
assert len(sub) == len(sample)
sub.to_csv(SUB_PATH, index=False)

print("Wrote:", SUB_PATH, "rows:", len(sub))
print(sub.head())
print(
    f"Retrieval enabled: {USE_RETRIEVAL}, test_read_fail={test_read_fail}, time_pred={time.time()-t1:.1f}s"
)



## === cell 3
with open(SUB_PATH, "r", encoding="utf-8") as f:
    for i in range(6):
        print(f.readline().rstrip("\n"))
