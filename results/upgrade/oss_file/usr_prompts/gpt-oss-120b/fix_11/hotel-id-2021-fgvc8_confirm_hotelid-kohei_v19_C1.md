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

0.7449797928769869

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the external‑library call with a self‑contained baseline that reads the training metadata, picks the five most frequent `hotel_id`s, and writes those as the prediction for every test image. This guarantees a correctly formatted `submission.csv` (so the notebook produces a valid submission) and gives a non‑zero MAP@5 score, moving the result toward the target without altering any core modeling code.'
- What this solution (achieved 0.00209) has done: 'I replace the naive “global top‑5 for every image” prediction with a chain‑aware heuristic: for each test image I extract its chain folder (e.g., `0`, `1`, …), look up the five most frequent `hotel_id`s for that chain in the training data, and use those as the prediction. If the chain cannot be determined or is absent from the training set, I fall back to the global top‑5. This small change leverages existing metadata without altering any core modeling logic and is expected to raise the MAP@5 score much closer to the target.'
- What this solution (achieved 0.00209) has done: 'I add a direct lookup of the image filename in the training metadata so that whenever a test image is identical to a training image we can place its exact `hotel_id` first in the prediction list, padding with the global top‑5 IDs to keep five entries. For all other images the existing chain‑aware heuristic is kept. This small, targeted change should raise the MAP@5 score substantially toward the target while preserving the original workflow.'
- What this solution (achieved 0.00209) has done: 'I keep the existing data loading and directory discovery logic, but improve the prediction construction: 
1. Store the per‑chain top‑5 hotel IDs as lists (not pre‑joined strings). 
2. For each test image, first use an exact match if possible, then fill the remaining slots with the chain‑specific top‑5 (excluding duplicates), and finally pad with the global top‑5 IDs to guarantee exactly five IDs. 
3. Ensure every prediction string always contains five space‑separated IDs, which aligns with the MAP@5 metric and should raise the score toward the target while preserving the original workflow.'
- What this solution (achieved 0.00209) has done: 'I adjust the way the chain identifier is extracted from each test image path. Instead of assuming the immediate parent folder is the numeric chain id (which can fail if the directory layout includes extra levels), the code now scans all path components for a numeric folder name and uses that as the chain id. This lets the chain‑aware heuristic work for more images, improving the relevance of the top‑5 predictions and moving the MAP@5 score closer to the target while keeping the overall logic unchanged.'
- What this solution (achieved 0.00209) has done: 'I make the submission generation follow the official sample_submission.csv so that the output rows exactly match the expected test set, and tighten the prediction building to ensure five unique hotel IDs (using exact match, then chain‑specific top‑5, then global top‑5). This fixes the likely mismatch that kept the MAP@5 near 0 and should move the score noticeably toward the target while leaving the overall heuristic untouched.'
- What this solution (achieved 0.00209) has done: 'I broaden the candidate pools used for each image so the true hotel‑id has a higher chance of appearing in the top‑5 list.  
Instead of only the 5 most frequent hotels per chain (and globally), I keep the 20 most frequent IDs for each chain and globally, then fill the prediction with the first unique IDs from those lists (exact match first). This small change respects the existing workflow while giving the MAP@5 score a better chance to move toward the target.'
- What this solution (achieved 0.00209) has done: 'I reorder the prediction building so that after an exact‐match we first fill the list with the most frequent global hotel IDs (which have the highest chance of being correct) and only then use the chain‑specific popular IDs. I also expand the global pool to 100 IDs to give more candidates when the chain list is short, while still outputting exactly five unique IDs. This small tweak respects the original workflow but should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00209) has done: 'I keep the overall workflow but expand the candidate pools and reorder the heuristic so that after any exact‑match we first use the chain‑specific most‑frequent hotels (now up to 200), then the global most‑frequent hotels, and finally fill any remaining slots with random picks from the larger global pool. This adds more relevant alternatives per image while preserving the original logic, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00209) has done: 'I reorder the candidate pools so that after any exact‑match we first fill the prediction list with the most frequent global hotels (using a larger top‑20 list) before falling back to the chain‑specific most‑frequent hotels. This gives higher‑probability IDs a better chance to appear in the top‑5 while keeping the same overall workflow. The rest of the code stays unchanged, ensuring a valid `submission.csv` is still written.'

# 9. Code solution

## === cell 0
import os, subprocess, sys, glob, pandas as pd, random

print("Root input directories:")
subprocess.run(["ls", "-lha", "/kaggle/input"], check=False)

train_path = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
sample_sub_path = "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"

test_images_root_candidates = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/working/test_images",
]

test_dir = None
for cand in test_images_root_candidates:
    if os.path.isdir(cand):
        test_dir = cand
        break
if test_dir is None:
    sys.exit("Error: No test_images directory found among candidates.")

train_df = pd.read_csv(train_path)

GLOBAL_POOL_SIZE = 500  # pool used for random fallback
GLOBAL_TOP_N = 20  # number of most‑frequent globals we actively use
CHAIN_POOL_SIZE = 200  # per‑chain pool size

global_top_ids = (
    train_df["hotel_id"].value_counts().head(GLOBAL_POOL_SIZE).index.tolist()
)
global_top_n = global_top_ids[:GLOBAL_TOP_N]  # most frequent GLOBAL_TOP_N IDs
global_top5 = global_top_ids[:5]  # keep the original top‑5 for reference

chain_top200 = {}
for chain_val, grp in train_df.groupby("chain"):
    chain_top200[int(chain_val)] = (
        grp["hotel_id"].value_counts().head(CHAIN_POOL_SIZE).index.tolist()
    )

image_to_hotel = train_df.set_index("image")["hotel_id"].to_dict()


def extract_chain_id(img_path: str):
    """Return the first numeric folder name found in img_path, or None."""
    for part in img_path.split(os.sep):
        if part.isdigit():
            return int(part)
    return None


sample_sub_df = pd.read_csv(sample_sub_path)
test_filenames = sample_sub_df["image"].tolist()
print(f"Using {len(test_filenames)} test images from sample_submission.csv")

random.seed(42)




## === cell 1
predictions = []

for img_name in test_filenames:
    matches = glob.glob(os.path.join(test_dir, "**", img_name), recursive=True)
    img_path = matches[0] if matches else None

    pred_ids = []

    if img_name in image_to_hotel:
        pred_ids.append(str(image_to_hotel[img_name]))

    for gid in global_top_n:
        gid_str = str(gid)
        if gid_str not in pred_ids:
            pred_ids.append(gid_str)
        if len(pred_ids) >= 5:
            break

    if len(pred_ids) < 5 and img_path:
        chain_id = extract_chain_id(img_path)
        chain_ids = chain_top200.get(chain_id, []) if chain_id is not None else []
        for cid in chain_ids:
            cid_str = str(cid)
            if cid_str not in pred_ids:
                pred_ids.append(cid_str)
            if len(pred_ids) >= 5:
                break

    while len(pred_ids) < 5:
        random_gid = str(random.choice(global_top_ids))
        if random_gid not in pred_ids:
            pred_ids.append(random_gid)

    predictions.append(" ".join(pred_ids))

submission_df = pd.DataFrame({"image": test_filenames, "hotel_id": predictions})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission_df.head())
