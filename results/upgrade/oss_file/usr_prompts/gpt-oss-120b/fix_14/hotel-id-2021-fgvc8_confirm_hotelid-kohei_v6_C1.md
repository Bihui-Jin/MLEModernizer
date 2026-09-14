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

0.7948219247284642

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I replace the external module call with a lightweight, deterministic baseline that creates a valid `submission.csv`.  
The script loads the training metadata, computes the five most frequent hotel IDs per chain (and globally), then assigns those IDs to each test image using the chain folder inferred from its path. If a chain is unknown, the global top‑5 list is used. This ensures a proper CSV is written and yields a non‑zero MAP@5 score, moving the result toward the target.'
- What this solution (achieved 0.0) has done: 'I add a small lookup so that any test image that also appears in the training metadata gets its true hotel ID as the first prediction (followed by the most frequent IDs to fill five slots). This tiny change can raise the MAP@5 without altering the overall baseline logic, and I also make sure the CSV is written to the Kaggle working directory.'
- What this solution (achieved 0.0) has done: 'I make the submission list come from the full sample_submission file (so every required image gets a row) and simplify the prediction building to avoid duplicate IDs while still using the global top‑5 and any exact‑match lookup. This guarantees a complete CSV and a modest MAP@5 boost, moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I add a lightweight mapping that derives each test image’s chain from its folder name on disk, then use this chain to look up the chain‑specific top‑5 hotels (fall‑back to the global top‑5 as before). This small change keeps the original baseline logic but gives more relevant predictions, moving the MAP@5 score closer to the target. The rest of the script remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I adjust the chain‑fallback logic so that when a test image’s chain cannot be determined it uses the “unknown” chain (ID 0) top‑5 list instead of only the overall global top‑5. This small change keeps the original baseline intact but provides more relevant hotel‑IDs for many images, moving the MAP@5 score toward the target without altering the core model or feature extraction.'
- What this solution (achieved 0.0) has done: 'The update makes the data paths robust by checking common Kaggle input locations and falling back to the original relative path. This resolves the FileNotFoundError for both the training CSV and sample submission, allowing the script to load data, generate predictions, and write a valid `submission.csv` without altering the core prediction logic.'
- What this solution (achieved 0.0) has done: 'I replace the per‑chain top‑5 lookup with the per‑chain top‑10 list (still capped at five predictions after deduplication). This gives a richer, more relevant pool of hotel IDs for each chain while preserving the original baseline logic and output format, which should raise the MAP@5 score toward the target without altering the overall architecture.'
- What this solution (achieved 0.0) has done: 'I replace the use of the broader `chain_top10` lists with the more relevant `chain_top5` lists when building predictions. Using the top‑5 most frequent hotels for each chain (instead of the top‑10) keeps the candidate set tighter and therefore should raise the MAP@5 score, moving it closer to the target while leaving the overall pipeline unchanged. The fallback logic is also aligned to `chain_top5` so that unknown chains still receive the best‑available per‑chain suggestions.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged but replace the per‑chain candidate list with the richer `chain_top10` set (still limited to five predictions after deduplication) and compute a `global_top10` to use as a fallback. This small augmentation gives more relevant hotel IDs for each chain, which should raise the MAP@5 from the current 0 and move it toward the target 0.795 while preserving the original logic and output format.'
- What this solution (achieved 0.0) has done: 'I tighten the prediction logic by using the per‑chain top‑5 list (which is more specific than the previous top‑10) and fall back to the “unknown chain” top‑5 before using the global top‑5. I also write the submission to the standard Kaggle working directory (`/kaggle/working/submission.csv`). These minimal adjustments keep the original pipeline intact while giving more relevant hotel‑ID candidates, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.0) has done: 'I broaden the per‑chain candidate list by using the top 10 most frequent hotels for each chain (while still deduplicating and truncating to five predictions). This keeps the original pipeline intact but gives a richer, more relevant pool of IDs, which should raise the MAP@5 and move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

possible_paths = [
    "/kaggle/input/hotel-id-2021-fgvc8",
    "./kaggle/input/hotel-id-2021-fgvc8",
    "./input/hotel-id-2021-fgvc8",
]
BASE_PATH = next((p for p in possible_paths if os.path.isdir(p)), None)
if BASE_PATH is None:
    raise FileNotFoundError(
        "Dataset directory not found. Checked paths: " + ", ".join(possible_paths)
    )

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUBMISSION = os.path.join(BASE_PATH, "sample_submission.csv")
TEST_IMAGES_ROOT = os.path.join(BASE_PATH, "test_images")

train_df = pd.read_csv(TRAIN_CSV)

global_top5 = train_df["hotel_id"].value_counts().head(5).astype(str).tolist()
global_top10 = train_df["hotel_id"].value_counts().head(10).astype(str).tolist()

chain_top5 = (
    train_df.groupby("chain")["hotel_id"]
    .apply(lambda x: x.value_counts().head(5).astype(str).tolist())
    .to_dict()
)

chain_top10 = (
    train_df.groupby("chain")["hotel_id"]
    .apply(lambda x: x.value_counts().head(10).astype(str).tolist())
    .to_dict()
)

img_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))
img_to_chain = dict(zip(train_df["image"], train_df["chain"]))



## === cell 1
test_img_to_chain = {}
if os.path.isdir(TEST_IMAGES_ROOT):
    for entry in os.scandir(TEST_IMAGES_ROOT):
        if entry.is_dir():
            chain_name = entry.name
            try:
                chain_id = int(chain_name)
            except ValueError:
                chain_id = None
            for img_entry in os.scandir(entry.path):
                if img_entry.is_file():
                    test_img_to_chain[img_entry.name] = chain_id



## === cell 2
test_image_ids = pd.read_csv(SAMPLE_SUBMISSION)["image"].tolist()

pred_rows = []
for img_name in test_image_ids:
    start_ids = [img_to_hotel[img_name]] if img_name in img_to_hotel else []

    chain_id = img_to_chain.get(img_name)
    if chain_id is None:
        chain_id = test_img_to_chain.get(img_name)

    if chain_id is not None:
        chain_ids = chain_top10.get(chain_id, [])
        if not chain_ids:
            chain_ids = global_top10
    else:
        chain_ids = chain_top10.get(0, [])
        if not chain_ids:
            chain_ids = global_top10

    candidates = []
    for src in (start_ids, chain_ids, global_top5):
        for h in src:
            if h not in candidates:
                candidates.append(h)
            if len(candidates) == 5:
                break
        if len(candidates) == 5:
            break

    if len(candidates) < 5:
        for h in global_top10:
            if h not in candidates:
                candidates.append(h)
            if len(candidates) == 5:
                break

    pred_rows.append({"image": img_name, "hotel_id": " ".join(candidates)})

pred_df = pd.DataFrame(pred_rows)



## === cell 3
output_path = os.path.join("/kaggle/working", "submission.csv")
pred_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, rows: {len(pred_df)}")
