# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import pandas as pd
import random

TRAIN_CSV = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
SAMPLE_SUBMISSION = "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"
SUBMISSION_OUT = "submission.csv"

random.seed(42)

train_df = pd.read_csv(TRAIN_CSV)

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"].astype(str)))

top20_global = (
    train_df["hotel_id"].value_counts().nlargest(20).index.astype(str).tolist()
)
top5_global = top20_global[:5]  # primary fallback list (still 5 items)


def prefix_topk(df, length, k=5):
    """
    Build a dict: prefix → list of up to k most common hotel_id strings
    for a given prefix length.
    """
    pref_series = df["image"].str[:length]
    grouped = df.groupby(pref_series)["hotel_id"]
    topk = {}
    for pref, group in grouped:
        top_ids = group.value_counts().nlargest(k).index.astype(str).tolist()
        topk[pref] = top_ids
    return topk


def suffix_topk(df, length, k=5):
    """
    Build a dict: suffix → list of up to k most common hotel_id strings
    for a given suffix length.
    """
    suff_series = df["image"].str[-length:]
    grouped = df.groupby(suff_series)["hotel_id"]
    topk = {}
    for suff, group in grouped:
        top_ids = group.value_counts().nlargest(k).index.astype(str).tolist()
        topk[suff] = top_ids
    return topk


prefix_maps = {l: prefix_topk(train_df, l) for l in range(1, 11)}

suffix_maps = {l: suffix_topk(train_df, l) for l in range(1, 9)}


def build_prediction(image_name):
    """
    Return a space‑delimited string of 5 hotel IDs.
    Heuristic order (most specific → most generic):
        1) Exact image match
        2) Candidates from all matching prefixes (10 → 1)
        3) Candidates from all matching suffixes (8 → 1)
        4) Pad with a diverse set taken from the top‑20 global hotels.
    """
    if image_name in image_to_hotel:
        true_id = image_to_hotel[image_name]
        filler = [hid for hid in top20_global if hid != true_id][:4]
        return " ".join([true_id] + filler)

    candidates = []
    seen = set()

    for length in range(10, 0, -1):
        pref = image_name[:length]
        pref_dict = prefix_maps.get(length)
        if pref_dict and pref in pref_dict:
            for hid in pref_dict[pref]:
                if hid not in seen:
                    candidates.append(hid)
                    seen.add(hid)
                if len(candidates) == 5:
                    break
        if len(candidates) == 5:
            break

    if len(candidates) < 5:
        for length in range(8, 0, -1):
            suff = image_name[-length:]
            suff_dict = suffix_maps.get(length)
            if suff_dict and suff in suff_dict:
                for hid in suff_dict[suff]:
                    if hid not in seen:
                        candidates.append(hid)
                        seen.add(hid)
                    if len(candidates) == 5:
                        break
            if len(candidates) == 5:
                break

    if len(candidates) < 5:
        shuffled_global = top20_global.copy()
        random.shuffle(shuffled_global)
        filler = [hid for hid in shuffled_global if hid not in seen][
            : 5 - len(candidates)
        ]
        candidates.extend(filler)

    return " ".join(candidates)


sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
sample_sub["hotel_id"] = sample_sub["image"].apply(build_prediction)
sample_sub.to_csv(SUBMISSION_OUT, index=False)

print(f"Written submission with {len(sample_sub)} rows to {SUBMISSION_OUT}")
print(sample_sub.head())



## === cell 1
import os

print("File exists:", os.path.isfile(SUBMISSION_OUT))
print("First few lines of the submission:")
with open(SUBMISSION_OUT, "r") as f:
    for _ in range(5):
        print(f.readline().strip())
