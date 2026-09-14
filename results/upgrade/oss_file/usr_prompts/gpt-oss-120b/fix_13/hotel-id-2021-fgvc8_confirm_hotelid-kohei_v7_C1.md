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

0.7201545002946846

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the shell‑only execution with a small Python pipeline that (a) installs the required wheels, (b) loads the training metadata to compute the five most frequent hotel IDs, (c) builds a valid `submission.csv` for every test image using those IDs as a simple baseline, and (d) writes the file to `/kaggle/working/submission.csv`. This guarantees a correctly‑formatted submission and moves the score from “no result” toward the target without altering the core model code.'
- What this solution (achieved 0.00209) has done: 'I replace the single‑global top‑5 baseline with a per‑chain top‑5 heuristic: for each hotel chain I compute its five most frequent hotel IDs from the training data and assign that list to every test image that belongs to the same chain (derived from the test image’s folder). If a chain is missing or unknown, the global top‑5 list is used as a fallback. This small change respects the original pipeline while providing a much more relevant prediction, moving the MAP@5 score significantly toward the target.'
- What this solution (achieved 0.00209) has done: 'I make the test‑image‑to‑chain mapping robust by gathering JPEGs from **all** “test_images” folders found under /kaggle/input, instead of only the first match. This ensures each test image gets the correct chain ID (when available) and therefore receives the appropriate per‑chain top‑5 list, moving the MAP@5 score noticeably closer to the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.0014) has done: 'I make the fallback prediction a bit more specific: when a test image’s chain is missing we first try the “chain 0” (unknown‑chain) frequent hotels (if that list exists) before falling back to the overall global top‑5. I also ensure the returned list always contains exactly five distinct IDs by filling any gaps with global IDs while preserving order. This small adjustment keeps the original heuristic unchanged yet should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00171) has done: 'I adjust the top‑5 prediction logic so that the globally most frequent hotel ID is always placed first, then the chain‑specific frequent IDs (excluding duplicates), and finally fill any remaining slots with the next global frequent IDs. This small change preserves the original heuristic while giving higher weight to the best‑overall guess, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.0014) has done: 'I keep the package‑installation block unchanged and rename the cells to start at 1. In the prediction function I switch the ordering so that, when a chain‑specific list exists, it is used first (up to five IDs) and only then falls back to the globally most frequent hotels. This respects the original heuristic while giving the chain information higher priority, which should raise MAP@5 toward the target. The rest of the pipeline (reading data, mapping images to chains, writing submission.csv) stays the same.'
- What this solution (achieved 0.0014) has done: 'I tighten the test‑image‑to‑chain mapping so that the chain ID is correctly extracted even when the folder hierarchy is deeper, and I enlarge the global pool of frequent hotels (top 100) to give a richer fallback list while still preferring chain‑specific choices. These minimal adjustments keep the original heuristic intact but increase the chance that the true hotel appears among the five predicted IDs, moving the MAP@5 score closer to the target.'
- What this solution (achieved 0.0014) has done: 'I make the test‑image‑to‑chain mapping more robust by taking the immediate parent directory of each image (the chain folder) instead of relying on the relative parts count, which can mis‑detect the chain when the folder hierarchy is deeper. This ensures chain‑specific top‑5 lists are applied where possible, improving the relevance of predictions and moving the MAP@5 score upward toward the target. No other logic is altered.'
- What this solution (achieved 0.0014) has done: 'I add a direct‑match fallback (if a test image name appears in the training set we return its true hotel ID) and expand the per‑chain candidate pool to the top 20 most frequent hotels, while still using the global top 5/100 lists to fill remaining slots. This keeps the original heuristic but gives a higher‑precision guess for any overlapping images and a richer chain‑specific list, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.0014) has done: 'I keep the installation cell unchanged and revise the data‑processing cell so that (a) an exact‑match image now returns five IDs (the true hotel plus the most frequent others) and (b) chain‑specific candidates are drawn from the top 20 frequent hotels for that chain rather than only the top 5, which gives a higher chance of covering the correct hotel while still preserving the original heuristic flow. This small, targeted change should raise the MAP@5 score toward the target without altering the overall pipeline.'
- What this solution (achieved 0.0014) has done: 'To raise the MAP@5 score we add a cheap but effective duplicate‑image detection: we hash every training image and map the hash to its true hotel ID. When a test image has the same hash as any training image we treat it as an exact match (returning the true hotel ID first). This keeps the original heuristic untouched while giving many more correct predictions, moving the score much closer to the target. The rest of the pipeline (per‑chain and global fall‑backs) remains unchanged.'
- What this solution (achieved 0.00209) has done: 'I keep the overall heuristic pipeline but add a broader fallback list: compute the global top 20 most frequent hotels and use that when a test image’s chain is unknown, instead of the very limited “chain 0” list. This gives more relevant candidates for many images, modestly increasing the chance that the true hotel appears in the top‑5 and moving the MAP@5 score upward toward the target without changing the core logic.'

# 9. Code solution

## === cell 0
import subprocess, sys, os, glob

deps_path = "/kaggle/input/pekolib-deps"
if os.path.isdir(deps_path):
    for whl in glob.glob(os.path.join(deps_path, "*.whl")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", whl])

pk_path = "/kaggle/input/pekolib"
if os.path.isdir(pk_path):
    for whl in glob.glob(os.path.join(pk_path, "*.whl")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", whl])

abn_src = "/kaggle/input/pekolib-deps/inplace_abn-1.0.12/inplace_abn"
if os.path.isdir(abn_src):
    tmp_dir = "/tmp/inplace_abn"
    subprocess.run(["cp", "-r", abn_src, tmp_dir])
    subprocess.run(
        ["bash", "-c", f"cd {tmp_dir} && {sys.executable} -m pip install -q ."]
    )

print("Package installation completed.")



## === cell 1
import pandas as pd
from pathlib import Path
import hashlib

train_path = next(Path("/kaggle/input").rglob("train.csv"))
sample_sub_path = next(Path("/kaggle/input").rglob("sample_submission.csv"))
train_df = pd.read_csv(train_path)

img_exact_lookup = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))

global_top5 = train_df["hotel_id"].value_counts().nlargest(5).index.astype(str).tolist()
global_top20 = (
    train_df["hotel_id"].value_counts().nlargest(20).index.astype(str).tolist()
)
global_top100 = (
    train_df["hotel_id"].value_counts().nlargest(100).index.astype(str).tolist()
)

per_chain_top20 = (
    train_df.groupby(train_df["chain"].astype(str))["hotel_id"]
    .apply(lambda s: s.value_counts().nlargest(20).index.astype(str).tolist())
    .to_dict()
)

unknown_chain_fallback = global_top20

img_to_chain = {}
test_img_path = {}  # image name → full Path object
test_root_candidates = list(Path("/kaggle/input").rglob("test_images"))
if not test_root_candidates:
    raise FileNotFoundError("Cannot locate any test_images folder in /kaggle/input")

for root in test_root_candidates:
    for img_path in root.rglob("*.jpg"):
        img_name = img_path.name
        parent = img_path.parent
        chain_id = parent.name if parent != root else None
        img_to_chain[img_name] = chain_id
        test_img_path[img_name] = img_path

train_hash_to_hotel = {}
train_root_candidates = list(Path("/kaggle/input").rglob("train_images"))
if not train_root_candidates:
    raise FileNotFoundError("Cannot locate any train_images folder in /kaggle/input")

for root in train_root_candidates:
    for img_path in root.rglob("*.jpg"):
        img_name = img_path.name
        if img_name not in img_exact_lookup:
            continue
        try:
            with open(img_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            train_hash_to_hotel[h] = img_exact_lookup[img_name]
        except Exception:
            continue


def get_top5_for_image(img_name):
    """
    Return a space‑delimited string of exactly five hotel IDs.
    Preference order:
      1. Exact filename match (already in metadata).
      2. Duplicate‑image match via MD5 hash.
      3. Chain‑specific frequent IDs (up to 5 distinct IDs).
      4. Global top‑5 frequent IDs.
      5. Global top‑20 as fallback for unknown chains.
      6. Global top‑100 to fill any remaining slots.
    Duplicates are avoided.
    """
    if img_name in img_exact_lookup:
        true_id = img_exact_lookup[img_name]
        pred = [true_id]
        for h in global_top5:
            if h not in pred:
                pred.append(h)
            if len(pred) == 5:
                break
        return " ".join(pred)

    test_path = test_img_path.get(img_name)
    if test_path is not None:
        try:
            with open(test_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in train_hash_to_hotel:
                true_id = train_hash_to_hotel[h]
                pred = [true_id]
                for gh in global_top5:
                    if gh not in pred:
                        pred.append(gh)
                    if len(pred) == 5:
                        break
                return " ".join(pred)
        except Exception:
            pass

    pred = []
    chain_id = img_to_chain.get(img_name)
    chain_candidates = per_chain_top20.get(chain_id) if chain_id is not None else None
    if not chain_candidates:
        chain_candidates = unknown_chain_fallback

    if chain_candidates:
        for h in chain_candidates:
            if h not in pred:
                pred.append(h)
            if len(pred) == 5:
                break

    if len(pred) < 5:
        for h in global_top5:
            if h not in pred:
                pred.append(h)
            if len(pred) == 5:
                break

    if len(pred) < 5:
        for h in global_top100:
            if h not in pred:
                pred.append(h)
            if len(pred) == 5:
                break

    return " ".join(pred)


sample_sub = pd.read_csv(sample_sub_path)
submission = pd.DataFrame(
    {
        "image": sample_sub["image"],
        "hotel_id": sample_sub["image"].apply(get_top5_for_image),
    }
)

output_path = Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False)

print(f"Submission written to {output_path}")
print("Sample rows:")
print(submission.head())
