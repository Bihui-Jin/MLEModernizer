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

# 9. Code solution

## === cell 0
import os, subprocess, sys, glob, pandas as pd

print("Root input directories:")
subprocess.run(["ls", "-lha", "/kaggle/input"], check=False)




## === cell 1
train_path = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
test_images_root_candidates = [
    "/kaggle/input/hotel-id-2021-fgvc8/test_images",
    "/kaggle/input/test_images",
    "/kaggle/working/test_images",
]

train_df = pd.read_csv(train_path)

global_top5 = train_df["hotel_id"].value_counts().head(5).index.tolist()
global_top5_str = " ".join(map(str, global_top5))
print(f"Global top‑5 hotel_id values: {global_top5_str}")

chain_top5 = {}
for chain_val, grp in train_df.groupby("chain"):
    top5_list = grp["hotel_id"].value_counts().head(5).index.tolist()
    chain_top5[int(chain_val)] = top5_list

image_to_hotel = train_df.set_index("image")["hotel_id"].to_dict()

test_dir = None
for cand in test_images_root_candidates:
    if os.path.isdir(cand):
        test_dir = cand
        break
if test_dir is None:
    sys.exit("Error: No test_images directory found among candidates.")

test_image_paths = sorted(
    [p for p in glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)]
)
test_filenames = [os.path.basename(p) for p in test_image_paths]
print(f"Found {len(test_filenames)} test images in {test_dir}")

predictions = []
for img_path, img_name in zip(test_image_paths, test_filenames):
    if img_name in image_to_hotel:
        exact_id = str(image_to_hotel[img_name])
        pred_ids = [exact_id]
        for gid in global_top5:
            gid_str = str(gid)
            if gid_str != exact_id and len(pred_ids) < 5:
                pred_ids.append(gid_str)
        while len(pred_ids) < 5:
            pred_ids.append(global_top5_str.split()[0])
        pred = " ".join(pred_ids)
    else:
        parent_folder = os.path.basename(os.path.dirname(img_path))
        try:
            chain_id = int(parent_folder)
            chain_ids = chain_top5.get(chain_id, [])
        except Exception:
            chain_ids = []

        pred_ids = [str(cid) for cid in chain_ids]

        for gid in global_top5:
            gid_str = str(gid)
            if gid_str not in pred_ids and len(pred_ids) < 5:
                pred_ids.append(gid_str)

        while len(pred_ids) < 5:
            pred_ids.append(global_top5_str.split()[0])

        pred = " ".join(pred_ids)

    predictions.append(pred)

submission_df = pd.DataFrame({"image": test_filenames, "hotel_id": predictions})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission_df.head())
