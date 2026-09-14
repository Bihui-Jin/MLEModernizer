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

0.00238

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the external‑dependency code with a self‑contained baseline that reads the provided `train.csv`, finds the five most frequent hotel IDs, and writes a valid `submission.csv` where every test image receives this same list. This ensures the notebook finishes without errors and produces a submission file in the correct format, moving the solution from “no score” to a baseline MAP@5 that can be further tuned if needed.'
- What this solution (achieved 0.00209) has done: 'I fixed the type error when joining hotel IDs by converting all IDs to strings, and I rebuilt the submission using the official `sample_submission.csv` to guarantee the correct number and order of rows. Predictions are now matched to each test image via its chain folder, falling back to the global top‑5 when the chain is unknown, ensuring a valid CSV is written.'
- What this solution (achieved 0.00209) has done: 'I improve the per‑chain prediction by selecting the most frequent hotels **within each chain** rather than just using the global frequency order of unique IDs. This gives more relevant top‑5 lists for each chain while keeping the overall structure unchanged, and should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.0014) has done: 'I keep the existing simple frequency‑based approach but improve the fallback for test images whose chain is not seen in the training data. Instead of using the global most‑frequent hotels, I use the top‑5 hotels from the most common training chain (which is usually more representative than the overall global list). This small tweak is expected to raise the MAP@5 score toward the target while preserving the core logic.'
- What this solution (achieved 0.00209) has done: 'I replace the fallback prediction for images whose chain is not found with the globally most‑frequent five hotels (global_top5_str) instead of using the most common chain’s top‑5. This simple change keeps the core logic unchanged while providing a more universally relevant default list, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00209) has done: 'I adjust how the per‑chain top‑5 lists are built: each chain’s prediction now start with its most frequent hotel ID, then be filled with the globally most frequent hotels (skipping duplicates) until five IDs are present. This keeps the original frequency‑based logic but gives the globally popular hotels higher weight, which should raise the MAP@5 score toward the target while preserving the overall structure.'
- What this solution (achieved 0.00209) has done: 'I keep the original simple frequency‑based approach but make two small, score‑raising tweaks: (1) build a per‑chain list of the five most frequent hotels (instead of a single most‑common plus globals) and fill any short list with the global top‑5; (2) add an exact‑match fallback that, if a test image filename also appears in the training set, uses its true hotel ID as the first prediction (filled out with the global top‑5). These changes preserve the core logic while providing more relevant predictions, so the MAP@5 should move noticeably closer to the target.'
- What this solution (achieved 0.00244) has done: 'I add a lightweight “filename‑prefix” heuristic to increase the chance of guessing the correct hotel ID.  
After loading the training metadata I build a map from the first 6 characters of each training image name to the most frequent hotel ID observed for that prefix. During prediction, if the exact image is not in the training set we first try this prefix match (giving a likely correct ID), then fall back to the per‑chain top‑5 list or the global top‑5 list as before. This keeps the original simple frequency‑based logic intact while providing a modest boost toward the target MAP@5 score.'
- What this solution (achieved 0.00244) has done: 'I keep the existing frequency‑based logic but add a clearer prediction hierarchy: exact image match → chain‑based top‑5 list → prefix‑based guess → global top‑5 fallback, and I fill any missing slots with the most frequent hotels globally. This small extension stays within the original approach while providing more relevant predictions, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00238) has done: 'I slightly tweak the heuristics while keeping the overall frequency‑based design unchanged:  
1. Extend the filename‑prefix length from 6 to 8 characters so the prefix‑based guess is more specific.  
2. Re‑order the prediction cascade to try the prefix‑based guess **before** the chain‑based list (exact image → prefix → chain → global).  
These minimal changes preserve the core logic but give a better chance of picking the correct hotel, which should raise the MAP@5 score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path
import glob

INPUT_ROOT = Path("/kaggle/input")
TRAIN_CSV = next(INPUT_ROOT.rglob("train.csv"))
TEST_IMAGES_ROOT = next(INPUT_ROOT.rglob("test_images"))
SAMPLE_SUBMISSION_CSV = next(INPUT_ROOT.rglob("sample_submission.csv"))

print(f"Using train file: {TRAIN_CSV}")
print(f"Using test images root: {TEST_IMAGES_ROOT}")
print(f"Using sample submission file: {SAMPLE_SUBMISSION_CSV}")



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

global_counts = train_df["hotel_id"].value_counts()
global_top5_list = global_counts.nlargest(5).index.astype(str).tolist()
global_top5_str = " ".join(global_top5_list)
print("Global top‑5 frequent hotel IDs:", global_top5_str)

chain_top5_dict = {}
for chain_id, group in train_df.groupby("chain"):
    top5 = group["hotel_id"].value_counts().nlargest(5).index.astype(str).tolist()
    pred_list = top5.copy()
    for gid in global_top5_list:
        if len(pred_list) >= 5:
            break
        if gid not in pred_list:
            pred_list.append(gid)
    pred_list = pred_list[:5]
    chain_top5_dict[str(chain_id)] = pred_list  # store as list for easier use

sample_chains = list(chain_top5_dict.items())[:3]
print("Sample chain predictions (chain_id -> top‑5 hotels):")
for cid, preds in sample_chains:
    print(cid, preds)



## === cell 2
image_to_hotel = dict(
    zip(train_df["image"].astype(str), train_df["hotel_id"].astype(str))
)

PREFIX_LEN = 8
prefix_counts = {}
for img, hid in zip(train_df["image"].astype(str), train_df["hotel_id"].astype(str)):
    pref = img[:PREFIX_LEN]
    d = prefix_counts.setdefault(pref, {})
    d[hid] = d.get(hid, 0) + 1
prefix_best = {pref: max(cnts, key=cnts.get) for pref, cnts in prefix_counts.items()}

test_image_paths = sorted(glob.glob(str(TEST_IMAGES_ROOT / "**/*.jpg"), recursive=True))
image_to_chain = {}
for p in test_image_paths:
    img_name = Path(p).name
    chain_folder = Path(p).parent.name  # folder name corresponds to chain id
    image_to_chain[img_name] = chain_folder

sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)

predictions = []
for img_name in sample_sub["image"]:
    base_list = []

    if img_name in image_to_hotel:
        base_list.append(image_to_hotel[img_name])

    if not base_list:
        pref = img_name[:PREFIX_LEN]
        pref_hotel = prefix_best.get(pref)
        if pref_hotel:
            base_list.append(pref_hotel)

    if not base_list:
        chain_folder = image_to_chain.get(img_name)
        if chain_folder is not None and chain_folder in chain_top5_dict:
            base_list.extend(chain_top5_dict[chain_folder])

    for gid in global_top5_list:
        if len(base_list) >= 5:
            break
        if gid not in base_list:
            base_list.append(gid)

    while len(base_list) < 5:
        for gid in global_top5_list:
            if len(base_list) >= 5:
                break
            if gid not in base_list:
                base_list.append(gid)

    pred = " ".join(base_list[:5])
    predictions.append(pred)

submission = pd.DataFrame({"image": sample_sub["image"], "hotel_id": predictions})

submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} (first 5 rows):")
print(submission.head())
