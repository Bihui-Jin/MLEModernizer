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

0.7368948387639952

# 6. Current score

0.00122

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your current notebook doesn’t yield a score because it relies on private wheel files (`/kaggle/input/pekolib-deps/*.whl`) that are not present in the provided environment, so it can’t reliably run end-to-end to produce a valid `submission.csv`. To make it run and generate a valid submission, I remove the external installs and replace them with a minimal, deterministic baseline that uses only the provided `train.csv` to build a global “most frequent hotel_id” top-5 list, then writes predictions for every test image from `sample_submission.csv`. This preserves the “predict 5 hotel_ids per image” evaluation semantics and guarantees a correctly formatted CSV. While this baseline won’t reach your target MAP@5, it produce a valid scored submission so you can iterate from there.'
- What this solution (achieved 0.00209) has done: 'Your current score is low because the submission predicts the same global top-5 hotels for every image, which provides almost no retrieval power under MAP@5. To move the score up toward your target while keeping the approach simple and deterministic, I keep the same “frequency-based” core logic but condition predictions on the image’s chain folder (the parent directory name in `test_images/`), using `train.csv` to compute a top-5 per chain and falling back to the global top-5 when needed. This uses only provided files, preserves the “predict 5 hotel_ids per image” semantics, and should substantially increase MAP@5 versus a single global list. The output remains a valid `submission.csv` with exactly 5 space-delimited IDs per row.'
- What this solution (achieved 0.00184) has done: 'Your score is low because the test set does not include `chain` folders (unlike `train_images/`), so `infer_chain_id_from_path()` almost always returns `-1` and you fall back to the same global top-5 for every image. To move toward your target while keeping the same frequency-based core logic, I condition predictions on the **test image filename prefix** (first 2 hex chars), which is a stable, available signal, by learning a top-5 hotel list per prefix from `train.csv`. I keep a robust fallback cascade (prefix → global), and I also remove the expensive `os.walk` which risks timeouts and doesn’t help for this dataset layout. The result still produces a valid `submission.csv` with exactly 5 space-delimited hotel IDs per test image.'
- What this solution (achieved 0.00184) has done: 'Your current approach is a frequency-based top-5 lookup, but using a 2-char filename prefix is essentially noise, so almost every test row falls back to the same global top-5 and MAP@5 stays near-random. Keeping the same “learn top-5 lists from train.csv and apply them to test images” core logic, I switch the key from prefix2 to a more informative, still-available signal: the **image’s containing folder name** in `test_images/` (the 0–99 subdir), and learn top-5 hotel_id per that folder from training data. I also keep a robust fallback cascade (folder → prefix2 → global) so every row always gets 5 IDs, and I build the test image→folder mapping via a fast `os.listdir` over the 0–99 directories (no expensive full `os.walk`). This should substantially raise your score versus the prefix-only baseline while remaining minimal and deterministic.'
- What this solution (achieved 0.00184) has done: 'Your current score is low because the submission is effectively a weak frequency baseline: the “chain folder” lookup doesn’t match test distribution well and the 2-char prefix is mostly noise, so many images fall back to the same global top-5. To move toward the target while keeping the exact same core logic (train-derived top-5 lists + deterministic fallback cascade), I add one more train-derived key that is available at inference: the image’s 0–99 **folder id** (as used in `train_images/` and `test_images/`). Then I use a cascade `test_folder -> train_folder -> prefix2 -> global` so more test images get a more specific top-5 list without changing the fundamental approach. I also build the train/test image→folder mappings via fast `os.listdir` over the 0–99 directories (no full image-path inference beyond that), keeping runtime under the timeout and producing the same submission schema.'
- What this solution (achieved 0.0014) has done: 'Your current score is low because the folder-based key is likely not being used at inference: in this competition the `test_images/` directory is typically flat (images are not inside `0/..99/` subfolders), so `test_img_to_folder` stays mostly empty and you fall back to weak prefix/global top-5 for almost every row. Keeping the exact same “train-derived top-5 lists + deterministic fallback cascade” core logic, I (1) build a robust train `image->folder` map without scanning all filenames by leveraging the known `hotel_id` and `chain` folder structure, and (2) at test time infer a pseudo-folder for each test image by deterministically mapping it to the most likely train folder(s) for that image prefix, falling back to prefix/global. This should increase the fraction of rows using a more specific top-5 list and move MAP@5 upward while still producing a valid `submission.csv`. All I/O paths stay the same and runtime remains under the timeout.'
- What this solution (achieved 0.0014) has done: 'Your current score is far below the target, so we should increase MAP@5 while keeping the same “train-derived top-5 lookup + deterministic fallback cascade” core logic. The biggest issue is that `train["folder_dir"]` is being inferred incorrectly by scanning `train_images/` *as if it were organized by chain*, which makes most `folder_dir` values `-1` and collapses predictions back to weak global/prefix lists. I fix `folder_dir` inference to match the real dataset layout (subfolders `0..99` under `train_images/` and likely also under `test_images/`), by building an `image -> folder` map via fast per-folder `os.listdir` (no expensive `os.walk`). With correct folder keys, the existing cascade (test_folder -> prefix->candidate-train-folder -> prefix_top5 -> global_top5) actually use the stronger folder-conditioned priors much more often, moving the score upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.00122) has done: 'Your current logic is already the right kind of “train-derived top‑5 lookup + fallback cascade”, but it likely collapses at inference because `test_images/` is flat (no `0..99` subfolders), so `test_img_to_folder` stays empty and you mostly use weak prefix/global priors. I keep the same core approach, but fix the *test folder inference* by building a fast `image->folder` map directly from the on-disk structure whether it’s flat or nested, and I also add a simple stronger key: a `prefix4` (first 4 hex chars) top‑5 prior with fallback to `prefix2` then global. These are minimal changes that keep the semantics identical (always output 5 IDs) and should move MAP@5 upward toward your target. I also cap directory scanning to the filenames present in `sample_submission.csv` to stay within the 600s limit.'
- What this solution (achieved 0.00099) has done: 'Your score is far below the target, so we should increase MAP@5 while keeping the same frequency/lookup + fallback cascade core logic. The main issue is that your test-time “folder id” signal is effectively unused: when `test_images/` is flat you set folder to `-1` for all images, guaranteeing a fallback to weak prefix/global priors. I minimally change folder inference to always compute a deterministic pseudo-folder for test images using `hash(image_name) % 100`, which matches the 0–99 folder structure used on disk and lets your stronger `folder_top5` priors activate. I also build `folder_top5` in a more direct/robust way (still top-5 by frequency, padded with global top-5) to avoid edge cases and ensure every folder key produces exactly 5 IDs.'
- What this solution (achieved 0.00057) has done: 'Your current score is far below the target, so we should increase MAP@5, but with minimal changes and the same core “train-derived top‑5 lookup + deterministic fallback cascade” logic. The main weakness is that your pseudo-folder assignment for flat test directories is unrelated to the real on-disk foldering used to build `folder_top5`, so it activates essentially random folder priors. I change pseudo-folder inference to match the dataset’s actual sharding scheme by trying common deterministic schemes (CRC32 and MD5-based mod 100) and selecting the one that best matches the observed train `image->folder` mapping; this keeps everything frequency/lookup-based while making the folder-conditioned priors meaningfully aligned. I also avoid scanning all test files by only resolving folders for images listed in `sample_submission.csv`, keeping runtime within limits and preserving the same submission format.'
- What this solution (achieved 0.00122) has done: 'Your score is extremely low because the current logic still behaves almost like a weak global prior: the “pseudo-folder” calibration does not correspond to the real sharding used in `train_images/`, so folder-conditioned priors are effectively random for test and you frequently fall back to uninformative lists. I keep the exact same frequency-based top‑5 lookup + fallback cascade, but replace the pseudo-folder inference with a direct, deterministic mapping learned from `train_images`: for each filename prefix (4 then 2 chars), infer the most likely `train_images` folder id(s), then use the corresponding `folder_top5`. This is still the same core approach (train-derived priors keyed by available strings), but it makes the strongest `folder_top5` prior activate in a non-random, data-driven way. I also keep all current fallbacks (prefix4→prefix2→global) and ensure every row has exactly 5 space-delimited IDs and a valid `submission.csv`.'
- What this solution (achieved 0.00122) has done: 'Your current score is far below the target, so we should increase MAP@5 while keeping the same “train-derived top‑5 lookup + deterministic fallback cascade” core logic. The main issue is that the strongest signal you try to use (folder-conditioned priors from `train_images/0..99`) is not being activated for most test rows, because `test_images/` in this environment is largely flat and `test_img_to_folder` stays empty. I minimally fix this by (1) building a robust test `image->folder` map that works for both nested and flat layouts without scanning unnecessary files, and (2) adding one more train-derived, inference-available key (full basename without extension) to create a more informative top‑5 prior than short prefixes, with a safe fallback back to your existing prefix/folder/global cascade. This preserves evaluation semantics (still always outputs exactly 5 IDs) and keeps runtime within limits.'

# 9. Code solution

## === cell 0
import os
import hashlib
import zlib
import pandas as pd

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
train_path = os.path.join(BASE, "train.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")
test_images_dir = os.path.join(BASE, "test_images")

assert os.path.exists(train_path), f"Missing train.csv at {train_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.exists(test_images_dir), f"Missing test_images dir at {test_images_dir}"

train = pd.read_csv(train_path)
sample_sub = pd.read_csv(sample_sub_path)

assert "hotel_id" in train.columns, "train.csv must contain hotel_id"
assert "image" in train.columns, "train.csv must contain image"
assert (
    "image" in sample_sub.columns and "hotel_id" in sample_sub.columns
), "sample_submission.csv schema mismatch"




## === cell 1
train["hotel_id"] = train["hotel_id"].astype(str)
train["image"] = train["image"].astype(str)

train["stem"] = train["image"].str.replace(".jpg", "", regex=False)

train["prefix2"] = train["image"].str[:2]
train["prefix4"] = train["image"].str[:4]

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 + [global_top5[-1]] * 5)[:5]


def _pad_to_5(lst):
    lst = [str(x) for x in lst]
    if len(lst) >= 5:
        return lst[:5]
    need = 5 - len(lst)
    pad = [x for x in global_top5 if x not in lst][:need]
    lst = (lst + pad)[:5]
    if len(lst) < 5:
        lst = (lst + [lst[-1]] * 5)[:5]
    return lst


def _build_top5_map(key_col: str):
    mp = {}
    vc = train.groupby(key_col)["hotel_id"].value_counts().groupby(level=0).head(5)
    for (k, hid), _cnt in vc.items():
        mp.setdefault(str(k), []).append(str(hid))
    for k in list(mp.keys()):
        mp[k] = _pad_to_5(mp[k])
    return mp


stem_top5 = _build_top5_map("stem")
prefix2_top5 = _build_top5_map("prefix2")
prefix4_top5 = _build_top5_map("prefix4")

train_images_dir = os.path.join(BASE, "train_images")
assert os.path.exists(
    train_images_dir
), f"Missing train_images dir at {train_images_dir}"

folder_ids = [
    d
    for d in os.listdir(train_images_dir)
    if d.isdigit() and os.path.isdir(os.path.join(train_images_dir, d))
]

train_img_to_folder = {}
for d in folder_ids:
    p = os.path.join(train_images_dir, d)
    try:
        for fn in os.listdir(p):
            if fn.endswith(".jpg"):
                train_img_to_folder[fn] = d
    except PermissionError:
        continue

train["folder_dir"] = train["image"].map(train_img_to_folder).fillna("-1").astype(str)
folder_top5 = _build_top5_map("folder_dir")


def _build_prefix_folder_rank(prefix_col: str, topk: int):
    rank = {}
    vc = (
        train[train["folder_dir"].ne("-1")]
        .groupby(prefix_col)["folder_dir"]
        .value_counts()
        .groupby(level=0)
        .head(topk)
    )
    for (p, fd), _cnt in vc.items():
        rank.setdefault(str(p), []).append(str(fd))
    return rank


prefix4_folder_rank = _build_prefix_folder_rank("prefix4", topk=10)
prefix2_folder_rank = _build_prefix_folder_rank("prefix2", topk=10)




## === cell 2
def infer_prefix2(img_name: str) -> str:
    img_name = str(img_name)
    return img_name[:2] if len(img_name) >= 2 else ""


def infer_prefix4(img_name: str) -> str:
    img_name = str(img_name)
    return img_name[:4] if len(img_name) >= 4 else infer_prefix2(img_name)


def infer_stem(img_name: str) -> str:
    img_name = str(img_name)
    return img_name[:-4] if img_name.endswith(".jpg") else img_name


def infer_candidate_folders(img_name: str):
    p4 = infer_prefix4(img_name)
    p2 = infer_prefix2(img_name)
    cands = []
    for fd in prefix4_folder_rank.get(p4, []):
        if fd != "-1":
            cands.append(fd)
    for fd in prefix2_folder_rank.get(p2, []):
        if fd != "-1" and fd not in cands:
            cands.append(fd)
    return cands


need_imgs = set(sample_sub["image"].astype(str).tolist())

test_img_to_folder = {}

test_dir_entries = os.listdir(test_images_dir)
digit_subdirs = [
    d
    for d in test_dir_entries
    if d.isdigit() and os.path.isdir(os.path.join(test_images_dir, d))
]

if digit_subdirs:
    for d in digit_subdirs:
        p = os.path.join(test_images_dir, d)
        try:
            for fn in os.listdir(p):
                if fn in need_imgs:
                    test_img_to_folder[fn] = d
        except PermissionError:
            continue
else:
    test_img_to_folder = {}

preds = []
for img in sample_sub["image"].astype(str).tolist():
    fd = test_img_to_folder.get(img, None)

    if fd is not None and fd in folder_top5 and fd != "-1":
        top5 = folder_top5[fd]
    else:
        top5 = None

        for cfd in infer_candidate_folders(img):
            if cfd in folder_top5 and cfd != "-1":
                top5 = folder_top5[cfd]
                break

        if top5 is None:
            st = infer_stem(img)
            top5 = stem_top5.get(st, None)

        if top5 is None:
            p4 = infer_prefix4(img)
            p2 = infer_prefix2(img)
            top5 = prefix4_top5.get(p4, None)
            if top5 is None:
                top5 = prefix2_top5.get(p2, global_top5)

    preds.append(" ".join(_pad_to_5(top5)))

sub = sample_sub[["image"]].copy()
sub["hotel_id"] = preds

assert len(sub) == len(sample_sub), "Submission row count mismatch"
assert (
    sub["hotel_id"].str.split().map(len).eq(5).all()
), "Each prediction must contain exactly 5 IDs"
sub.to_csv("submission.csv", index=False)
print(sub.head())




## === cell 3
import subprocess

subprocess.run(["ls", "-lha", "submission.csv"], check=True)
subprocess.run(["head", "submission.csv"], check=True)
