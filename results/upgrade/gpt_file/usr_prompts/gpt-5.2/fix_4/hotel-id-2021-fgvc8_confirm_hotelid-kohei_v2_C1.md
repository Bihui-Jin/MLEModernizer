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

0.6702281720973288

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on external wheels (`pekolib*`) that are not present in the provided dataset tree, so the pipeline fails before producing `submission.csv`. I remove those installs and replace them with a minimal, self-contained submission generator that always runs end-to-end on the provided `/kaggle/input/hotel-id-2021-fgvc8/` data and writes a valid `submission.csv` in the required `image,hotel_id` format. To get a non-trivial baseline MAP@5 without changing “modeling” complexity, it predict the 5 most frequent `hotel_id` values from `train.csv` for every test image (a common safe baseline that should score above random and, crucially, yields a valid submission). This moves you from “no score” to a real score, which is the necessary first step toward the target.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.6702), so we should improve the baseline while keeping the “predict 5 hotel_ids per image” core logic intact. The smallest meaningful step is to replace the global top-5 hotel prior with a chain-aware prior: for each test image, infer its `chain` from the test image folder name (0–89) and then predict the top-5 `hotel_id` within that chain from `train.csv`. This uses only provided metadata and file structure (no extra models), preserves evaluation semantics, and should move MAP@5 substantially toward the target compared to a single global prior. We also add safe fallbacks when a chain folder is missing/unknown, ensuring a valid submission is always produced.'
- What this solution (achieved 0.00209) has done: 'Your current score (0.00209) is far below the target (0.6702), so we need a meaningful lift while keeping your “chain-aware top-5 prior” core logic. The biggest issue is that the notebook likely fails to infer chains for most test images because the hidden test set uses a different folder layout than your existence-check loop, so you silently fall back to the global prior almost everywhere. I replace per-image `os.path.exists` probing with a fast, robust lookup that builds an `image -> chain` map by scanning the actual `test_images` directory structure once, and then uses that map for predictions. This preserves the same prediction strategy (top-5 within inferred chain, else global top-5) but should substantially increase chain coverage and move MAP@5 toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_DIR = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SUB_PATH = "submission.csv"

print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))



## === cell 1
train = pd.read_csv(TRAIN_CSV)
sample = pd.read_csv(SAMPLE_SUB)

global_top5 = train["hotel_id"].value_counts().head(5).index.astype(str).tolist()
if len(global_top5) < 5:
    global_top5 = (global_top5 * 5)[:5]
global_pred_str = " ".join(global_top5)

chain_to_top5 = (
    train.groupby("chain")["hotel_id"]
    .value_counts()
    .groupby(level=0)
    .head(5)
    .reset_index(name="cnt")
)

chain_top5_map = {}
for ch, grp in chain_to_top5.groupby("chain"):
    lst = grp["hotel_id"].astype(str).tolist()
    if len(lst) < 5:
        extras = [h for h in global_top5 if h not in lst]
        lst = (lst + extras)[:5]
        if len(lst) < 5:
            lst = (lst * 5)[:5]
    chain_top5_map[int(ch)] = lst


def build_test_image_chain_map(test_img_dir: str):
    image_to_chain = {}

    if not os.path.isdir(test_img_dir):
        return image_to_chain

    has_subdirs = any(
        os.path.isdir(os.path.join(test_img_dir, name))
        for name in os.listdir(test_img_dir)
    )
    if not has_subdirs:
        return image_to_chain

    for name in os.listdir(test_img_dir):
        sub = os.path.join(test_img_dir, name)
        if not os.path.isdir(sub):
            continue
        try:
            ch = int(name)
        except ValueError:
            continue
        try:
            for fn in os.listdir(sub):
                image_to_chain[fn] = ch
        except FileNotFoundError:
            continue

    return image_to_chain


image_to_chain = build_test_image_chain_map(TEST_IMG_DIR)

images = sample["image"].astype(str).tolist()
inferred_chains = [image_to_chain.get(img, None) for img in images]

preds = []
for ch in inferred_chains:
    if ch is not None and ch in chain_top5_map:
        preds.append(" ".join(chain_top5_map[ch]))
    else:
        preds.append(global_pred_str)

sub = pd.DataFrame({"image": images, "hotel_id": preds})

assert list(sub.columns) == ["image", "hotel_id"]
assert len(sub) == len(sample)

sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH, "rows:", len(sub))
print(sub.head())
print(
    "Chain inference coverage (non-None):",
    sum(c is not None for c in inferred_chains),
    "/",
    len(inferred_chains),
)
print("Unique inferred chains:", len({c for c in inferred_chains if c is not None}))



## === cell 2
with open(SUB_PATH, "r", encoding="utf-8") as f:
    for i in range(6):
        print(f.readline().rstrip("\n"))
