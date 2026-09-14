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

0.00118

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I remove the stray explanatory text that caused a syntax error, replace the shell‑command cell with pure Python, and enhance the baseline by using the image‑folder (chain) information: for each test image we look up its chain directory, pick the most frequent hotel_ids for that chain from the training data (up to five), and fall back to the overall top‑5 when the chain is unknown. This keeps the original simple logic but adds a strong heuristic that should raise MAP@5 substantially while still producing a correctly formatted `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'I add a direct lookup from the training metadata: if a test image filename already appears in the training set we can use its known hotel_id as the first prediction, then fill the remaining slots with the most frequent hotels (excluding duplicates). This small heuristic leverages exact matches without changing the overall model logic and should raise MAP@5 toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.00209) has done: 'I keep the overall pipeline but improve the heuristic for generating the five hotel predictions.  
When a test image matches a training image we still use that exact hotel as the first prediction, but now we also try to fill the remaining slots with the most frequent hotels for the image’s chain (if known), falling back to the global most‑frequent hotels only for any still‑missing spots. This adds useful chain‑specific information without changing the core logic, and should raise MAP@5 toward the target.'
- What this solution (achieved 0.00209) has done: 'I replace the complex chain‑and‑exact‑match logic with a simpler, more reliable heuristic that always predicts the five most frequent hotels from the training set for every test image. This removes brittle folder‑lookup issues and ensures every submission row contains a valid five‑hotel list, which should raise the MAP@5 score significantly toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.00209) has done: 'I add a lightweight heuristic that leverages the known training metadata and the folder‑structure of the test images.  
For each test image we (1) use the exact hotel_id if the filename appears in the training set, (2) otherwise look up its chain directory (derived from the test_images folder) and fill the prediction with the five most frequent hotels for that chain, and finally (3) fall back to the global top‑5 for any remaining slots. This keeps the original simple pipeline but should raise MAP@5 far beyond the current 0.002 score while still writing a correct `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'I convert the chain identifiers to strings when building the per‑chain top‑5 list so they match the string folder names discovered in the test image directories. This minor type‑alignment lets the chain‑specific heuristic fire instead of always falling back to the global top‑5, which should raise the MAP@5 score toward the target while keeping the original logic unchanged.'
- What this solution (achieved 0.00209) has done: 'I broaden the chain‑specific heuristic by using the full frequency‑ordered list of hotels for each chain (instead of only the top‑5). The prediction builder now pull unique hotels from this longer list until five predictions are filled, then fall back to the global top‑5. This small change preserves the original workflow while increasing the chance that the true hotel appears in the top‑5, moving the MAP@5 score closer to the target.'
- What this solution (achieved 0.00209) has done: 'Implemented a clearer and slightly more aggressive chain‑specific heuristic.  
The prediction routine now prioritizes the top‑5 hotels for the detected chain before falling back to the global top‑5, ensuring a full five‑hotel list for every image while keeping the original simple workflow unchanged. This modest tweak should move the MAP@5 score closer to the target without altering the core model logic.'
- What this solution (achieved 0.00209) has done: 'I keep the overall pipeline but improve the heuristic that builds the 5‑hotel prediction list. Instead of always taking the most frequent hotels for a chain in the same order, I offset the start position in that list based on a deterministic hash of the image name. This spreads the predictions across more hotels within the same chain, giving each image a better chance of including its true hotel in the top‑5 while preserving determinism. The rest of the code stays unchanged, and the script still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.0003) has done: 'I add a deterministic fallback that assigns each test image to one of the known chains when the folder‑based lookup fails, then use that chain’s frequent hotels. This creates varied, chain‑informed predictions instead of the same global top‑5 for every image, moving the MAP@5 score toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.00209) has done: 'I simplify the prediction logic to always return the five most frequent hotels from the training data for every test image. Using the global top‑5 is a strong baseline that should raise the MAP@5 from the current near‑zero score toward the target, while keeping the rest of the pipeline unchanged and ensuring a correctly formatted `submission.csv`.'
- What this solution (achieved 0.00209) has done: 'I add a lightweight chain‑aware and exact‑match heuristic while keeping the overall simple pipeline. The script now builds a mapping from test images to their chain directories, computes per‑chain top‑5 hotels from the training data, and constructs each prediction list by (1) using the known hotel if the image appears in the train set, (2) filling with the chain’s most frequent hotels, and finally (3) falling back to the global top‑5. This modest, deterministic change should raise MAP@5 toward the target while preserving the original workflow and output format.'
- What this solution (achieved 0.00118) has done: 'I keep the overall pipeline unchanged but make the prediction step a bit more diverse: after trying an exact‑match and any chain‑specific hotels, I fill the remaining slots from a larger pool of the 20 most frequent hotels in the whole training set, starting at a deterministic offset based on the image name hash. This adds useful variety without altering the core logic and should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00118) has done: 'I extend the heuristic to use a larger pool of the most frequent hotels for each chain (top 20 instead of only top 5). After an exact‑match lookup we now fill the remaining slots with chain‑specific hotels, starting from a deterministic offset so the predictions are diverse yet reproducible. If the chain‑specific pool is exhausted we fall back to the global top 20 list as before. This small change keeps the overall workflow unchanged while giving a stronger, chain‑aware signal that should raise the MAP@5 score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import hashlib

possible_roots = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/working/hotel-id-2021-fgvc8",
    "/kaggle/working",
]
base_path = next((p for p in possible_roots if os.path.isdir(p)), "")

train_path = os.path.join(base_path, "train.csv")
sample_path = os.path.join(base_path, "sample_submission.csv")
test_images_root = os.path.join(base_path, "test_images")

train_df = pd.read_csv(train_path)
sample_df = pd.read_csv(sample_path)

global_top20 = train_df["hotel_id"].value_counts().head(20).index.astype(str).tolist()
global_top5 = global_top20[:5]

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))

chain_top20 = {}
for chain, grp in train_df.groupby("chain"):
    top20 = grp["hotel_id"].value_counts().head(20).index.astype(str).tolist()
    chain_top20[str(chain)] = top20

img_to_chain = {}
for root, _, files in os.walk(test_images_root):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img_to_chain[f] = os.path.basename(root)


def deterministic_offset(name: str, pool_size: int) -> int:
    """Hash the image name to obtain a reproducible offset within the pool."""
    h = hashlib.md5(name.encode()).hexdigest()
    return int(h, 16) % pool_size


def build_pred_list(img_name: str) -> str:
    """Create a space‑delimited list of five hotel IDs for a given image."""
    preds = []

    if img_name in image_to_hotel:
        preds.append(image_to_hotel[img_name])

    chain = img_to_chain.get(img_name)
    if chain is not None:
        chain_pool = chain_top20.get(chain, [])
        if chain_pool:
            offset = deterministic_offset(img_name, len(chain_pool))
            idx = offset
            while len(preds) < 5 and len(chain_pool) > 0:
                candidate = chain_pool[idx % len(chain_pool)]
                if candidate not in preds:
                    preds.append(candidate)
                idx += 1

    if len(preds) < 5:
        offset = deterministic_offset(img_name, len(global_top20))
        idx = offset
        while len(preds) < 5:
            candidate = global_top20[idx % len(global_top20)]
            if candidate not in preds:
                preds.append(candidate)
            idx += 1

    while len(preds) < 5:
        preds.append(global_top5[0])

    return " ".join(preds)


predictions = [build_pred_list(img) for img in sample_df["image"]]

submission = pd.DataFrame({"image": sample_df["image"], "hotel_id": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)



## === cell 1
print(f"Saved {submission_path} with {len(submission)} rows.")
print("First few rows of the submission:")
print(submission.head())
