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

0.00172

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'We replace the opaque external call with a straightforward baseline that always predicts the five most frequent hotel IDs from the training data for every test image. This guarantees a valid `submission.csv` in the required format, fixing the “no CSV produced” issue while keeping the core modeling approach unchanged (we simply use a deterministic heuristic instead of the missing package). The new script extracts the training frequencies, reads the sample submission to get test image names, builds the prediction string, writes the CSV, and prints its head for verification.'
- What this solution (achieved 0.00209) has done: 'I replace the naive “global top‑5” prediction with a chain‑aware heuristic: for each test image I locate its folder (the chain id) in the test_images directory and use the five most frequent hotels that appear in the training data for that same chain. If a chain cannot be determined or has no training data, I fall back to the overall top‑5 list. This small change keeps the original simple baseline while providing more relevant predictions, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00209) has done: 'I align the chain identifiers so the chain‑aware heuristic actually matches test images: convert the `chain` values to strings when building `chain_top5`. This lets the lookup find the per‑chain top‑5 hotels instead of always falling back to the global list, improving MAP@5 while keeping the overall simple baseline unchanged.'
- What this solution (achieved 0.00209) has done: 'The patch adds a direct lookup from training images to their known hotel IDs so that any test image that appears in the training set is predicted with its true label first, then filled out with the chain‑specific top‑5 (or global top‑5) to reach five unique IDs. This small but targeted heuristic should raise MAP@5 toward the target while keeping the original simple baseline unchanged.'
- What this solution (achieved 0.00209) has done: 'I tighten the heuristic so that when a test image’s chain is known we rely almost entirely on that chain’s most frequent hotels (only falling back to the global list if the chain‑specific list is shorter than five). This reduces unnecessary mixing of unrelated global predictions and should raise the MAP@5 score toward the target while keeping the original simple baseline unchanged.'
- What this solution (achieved 0.00092) has done: 'I add a lightweight heuristic that guesses a test image’s chain from the first two characters of its filename (using the same pattern observed in the training set). This lets us select the chain‑specific top‑5 hotels for many more images instead of always falling back to the global list, which should raise MAP@5 toward the target while keeping the overall baseline unchanged.'
- What this solution (achieved 0.00145) has done: 'I improve the chain‑guessing heuristic by building a robust prefix‑to‑chain map from the training data (using the first three filename characters and the most common chain for each prefix). This replaces the simplistic “first‑two‑chars” rule, allowing many more test images to be assigned the correct chain and thus receive the appropriate chain‑specific top‑5 hotels. The rest of the pipeline stays unchanged, so the script still writes a valid `submission.csv` while moving the MAP@5 score noticeably closer to the target.'
- What this solution (achieved 0.00136) has done: 'I strengthen the chain‑guessing fallback: use a longer filename prefix (5 chars) in addition to the existing 3‑char prefix and also try a simple digit‑prefix rule. This should assign the correct chain for more test images, letting the per‑chain top‑5 hotels be used more often and modestly lift the MAP@5 score toward the target.'
- What this solution (achieved 0.00172) has done: 'I add a tiny heuristic that always places the overall most‑frequent hotel (the first item of the global top‑5) at the front of every prediction list, then keep the existing chain‑aware logic and fall‑back to the global top‑5 to fill the remaining slots. This small change keeps the original baseline intact while increasing the chance that the true hotel ID appears in the top‑5, moving the MAP@5 score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import pathlib
from collections import Counter

train_path = pathlib.Path("/kaggle/input/hotel-id-2021-fgvc8/train.csv")
sample_sub_path = pathlib.Path(
    "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"
)
submission_path = pathlib.Path("/kaggle/working/submission.csv")
test_images_root = pathlib.Path("/kaggle/input/hotel-id-2021-fgvc8/test_images")

train_df = pd.read_csv(train_path)
sample_sub = pd.read_csv(sample_sub_path)

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()
most_common_global = global_top5[0]  # always insert this first

chain_top5 = (
    train_df.groupby(train_df["chain"].astype(str))["hotel_id"]
    .apply(lambda x: x.value_counts().head(5).index.astype(str).tolist())
    .to_dict()
)

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))

prefix_len_short = 3
prefix_len_long = 5
prefix_counter_short = {}
prefix_counter_long = {}

for img, chain in zip(train_df["image"], train_df["chain"].astype(str)):
    pref_short = img[:prefix_len_short]
    pref_long = img[:prefix_len_long]

    if pref_short not in prefix_counter_short:
        prefix_counter_short[pref_short] = Counter()
    prefix_counter_short[pref_short][chain] += 1

    if pref_long not in prefix_counter_long:
        prefix_counter_long[pref_long] = Counter()
    prefix_counter_long[pref_long][chain] += 1

prefix_to_chain_short = {
    pref: cnt.most_common(1)[0][0] for pref, cnt in prefix_counter_short.items()
}
prefix_to_chain_long = {
    pref: cnt.most_common(1)[0][0] for pref, cnt in prefix_counter_long.items()
}

image_to_chain = {}
for img_path in test_images_root.rglob("*.jpg"):
    image_to_chain[img_path.name] = img_path.parent.name

predictions = []
for img_name in sample_sub["image"]:
    first_hotel = image_to_hotel.get(img_name)

    chain_id = image_to_chain.get(img_name)
    if not chain_id or chain_id not in chain_top5:
        chain_id = prefix_to_chain_long.get(img_name[:prefix_len_long])
        if not chain_id or chain_id not in chain_top5:
            chain_id = prefix_to_chain_short.get(img_name[:prefix_len_short])
        if (
            (not chain_id or chain_id not in chain_top5)
            and img_name
            and img_name[0].isdigit()
        ):
            chain_id = img_name[0]

    chain_list = chain_top5.get(chain_id, []) if chain_id else []

    pred_list = []

    pred_list.append(most_common_global)

    if first_hotel and first_hotel not in pred_list:
        pred_list.append(first_hotel)

    for hid in chain_list:
        if hid not in pred_list:
            pred_list.append(hid)
        if len(pred_list) == 5:
            break

    if len(pred_list) < 5:
        for hid in global_top5:
            if hid not in pred_list:
                pred_list.append(hid)
            if len(pred_list) == 5:
                break

    predictions.append(" ".join(pred_list))

submission_df = pd.DataFrame({"image": sample_sub["image"], "hotel_id": predictions})
submission_df.to_csv(submission_path, index=False)

print("Submission preview:")
print(submission_df.head())



## === cell 1
import os

print("\nFile size:", os.path.getsize(submission_path), "bytes")
print("First 5 lines of the saved CSV:")
with open(submission_path, "r") as f:
    for _ in range(6):
        print(f.readline().strip())
