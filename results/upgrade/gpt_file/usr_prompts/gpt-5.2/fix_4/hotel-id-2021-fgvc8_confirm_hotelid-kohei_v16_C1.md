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

0.8164877494316718

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it depends on private wheel files and a `peko` module entrypoint that won’t exist in this environment, so it never reliably produces a valid `submission.csv`. To fix this with minimal, metric-aligned logic, I’m replacing that dependency with a simple, always-runnable baseline that creates a correct submission by predicting the global top-5 most frequent `hotel_id` values from `train.csv` for every test image (a standard fallback for MAP@5). This preserves evaluation semantics (space-delimited 5 IDs per image) and guarantees end-to-end execution within the time limit. Once you can submit and get a real score, we can make further minimal adjustments specifically to close the remaining gap to the target.'
- What this solution (achieved 0.00209) has done: 'Your current submission is far below the target, so we should improve it with the smallest metric-aligned change that preserves the “no-model” baseline core logic. Instead of predicting the same global top-5 hotels for every image, we use the provided `chain` metadata to make chain-conditioned top-5 predictions: for each test image, infer its chain from the folder name under `test_images/`, then predict the most frequent hotels within that chain from `train.csv`. For unknown chains (or chains with too few samples), we safely fall back to the global top-5 list to keep every row valid. This keeps the approach simple and fast, but should substantially raise MAP@5 compared to the unconditional global frequency baseline.'
- What this solution (achieved 0.00209) has done: 'Your current chain-conditioned frequency baseline likely maps most test images to the wrong “chain” because the test folder structure is not the chain id (it’s usually nested like `test_images/test_images/*.jpg`), causing lots of fallbacks and a very low MAP@5. I make the smallest fix: derive each test image’s chain by joining with `train.csv` on `image` (test images in this competition are drawn from known hotels/chains), and only fall back to the global top-5 when an image is truly unseen. This keeps the exact same core logic (top-5 frequency within chain), but fixes the key alignment bug so predictions are actually chain-conditioned. I also make the chain/top-5 padding deterministic and guarantee 5 unique ids per row.'

# 9. Code solution

## === cell 0
import os, glob
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "/kaggle/data",
]
base = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        base = b
        break
if base is None:
    raise FileNotFoundError("Could not find Kaggle input/data base directory.")


def pick_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


train_csv = pick_existing(
    os.path.join(base, "train.csv"),
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
)
sample_csv = pick_existing(
    os.path.join(base, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)

test_images_dir = pick_existing(
    os.path.join(base, "test_images"),
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
)
if test_images_dir is None:
    cand = glob.glob(os.path.join(base, "**", "test_images"), recursive=True)
    test_images_dir = cand[0] if cand else None

print("Using:")
print(" base:", base)
print(" train_csv:", train_csv)
print(" sample_csv:", sample_csv)
print(" test_images_dir:", test_images_dir)

if train_csv is None or sample_csv is None:
    raise FileNotFoundError(
        "Missing required CSV files (train.csv / sample_submission.csv)."
    )
if test_images_dir is None:
    raise FileNotFoundError("Could not locate test_images directory.")



## === cell 1
train = pd.read_csv(train_csv, usecols=["image", "chain", "hotel_id"])
train["image"] = train["image"].astype(str)
train["chain"] = train["chain"].astype(str)
train["hotel_id"] = train["hotel_id"].astype(str)

vc_global = train["hotel_id"].value_counts()
global_top5 = vc_global.sort_values(ascending=False).head(5).index.tolist()
global_top5 = list(dict.fromkeys(global_top5))
while len(global_top5) < 5:
    global_top5.append(global_top5[-1])
global_top5 = global_top5[:5]
global_pred_str = " ".join(global_top5)
print("Global top-5 hotel_id:", global_pred_str)

chain_top5 = {}
for ch, g in train.groupby("chain", sort=False):
    vc = g["hotel_id"].value_counts()
    top = vc.sort_values(ascending=False).head(5).index.tolist()
    top = list(dict.fromkeys(top + global_top5))
    top = top[:5]
    chain_top5[ch] = top
print("Precomputed chains:", len(chain_top5))

img_to_chain = dict(zip(train["image"].tolist(), train["chain"].tolist()))
print("Image->chain entries from train.csv:", len(img_to_chain))

all_test_files = glob.glob(os.path.join(test_images_dir, "**", "*.jpg"), recursive=True)
if len(all_test_files) == 0:
    raise FileNotFoundError(f"No .jpg found under {test_images_dir}")
print("Found test jpg files:", len(all_test_files))

sample = pd.read_csv(sample_csv)
if "image" not in sample.columns:
    raise ValueError("sample_submission.csv must contain 'image' column.")

preds = []
missing_chain = 0
for img in sample["image"].astype(str).tolist():
    ch = img_to_chain.get(img)
    if ch is None:
        missing_chain += 1
        top = global_top5
    else:
        top = chain_top5.get(str(ch), global_top5)
    top = list(dict.fromkeys(top + global_top5))[:5]
    while len(top) < 5:
        top.append(top[-1])
    preds.append(" ".join(top))

print(
    "Rows with missing chain mapping (fallback to global):",
    missing_chain,
    "/",
    len(sample),
)

sub = pd.DataFrame({"image": sample["image"].astype(str), "hotel_id": preds})
assert len(sub) == len(sample), (len(sub), len(sample))

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub.shape)



## === cell 2
check = pd.read_csv("submission.csv")
print(check.head())
print("Columns:", list(check.columns))
print("n_rows:", len(check))
assert list(check.columns) == ["image", "hotel_id"]
assert (
    check["hotel_id"].astype(str).str.split().map(len).eq(5).all()
), "Each prediction must have 5 space-delimited ids"
