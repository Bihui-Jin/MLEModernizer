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

0.7687168476888083

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the original installation‑and‑execution commands with a lightweight, deterministic baseline that reads the provided CSV metadata, derives the five most frequent hotel IDs per hotel chain, and uses those as the ordered predictions for each test image (falling back to the global top‑5 if a chain is unseen). This ensures the notebook runs end‑to‑end, writes a correctly formatted `submission.csv`, and improves the MAP@5 score compared to having no valid submission at all while preserving the core data‑handling logic.'
- What this solution (achieved 0.00209) has done: 'We keep the overall logic but improve the prediction list: for each chain we now merge its most‑common hotels with the global most‑common hotels, deduplicate, and keep the first five IDs. This gives a richer, more varied set of candidates while staying within the original frequency‑based approach, so the MAP@5 should move upward toward the target.'
- What this solution (achieved 0.00209) has done: 'I increase the number of candidate hotels considered per chain and globally from 5 to a larger pool (e.g., 20) before picking the first five unique IDs. This retains the original frequency‑based logic while giving a richer candidate set, which should raise the MAP@5 score toward the target without altering the core approach.'
- What this solution (achieved 0.00209) has done: 'We add a direct lookup so that if a test image filename already appears in the training metadata we predict its exact hotel_id first, then fill the remaining slots with the chain‑specific and global frequent hotels (deduplicated). This simple heuristic can capture duplicated images and should raise MAP@5 toward the target without altering the overall frequency‑based logic. We also increase the candidate pool size (`TOP_N`) to 100 to give a richer set of hotels when needed.'
- What this solution (achieved 0.00209) has done: 'I augment the frequency‑based heuristic by weighting each hotel’s popularity within its chain against its overall popularity, then use these combined scores to build the per‑chain candidate list. This keeps the original logic (frequency counts, exact‑match handling) while providing a more discriminative ordering that should raise MAP@5 toward the target. The rest of the pipeline (reading files, forming predictions, writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.00209) has done: 'I adjust the heuristic to rely more heavily on chain‑specific popularity, which is far more indicative of the correct hotel than the overall global popularity. By setting `CHAIN_WEIGHT` to 1.0 we use the pure chain frequency ordering, and we increase the candidate pool (`TOP_N`) so the final five can be chosen from a richer, still relevant list. These small changes keep the original frequency‑based logic intact while expectedly raising MAP@5 toward the target.'
- What this solution (achieved 0.00209) has done: 'I adjust the weighting between chain‑specific popularity and overall popularity by setting `CHAIN_WEIGHT` to 0.5 instead of 1.0. This blends the chain‑level counts with the global counts, giving a more balanced ranking that typically improves MAP@5 while preserving the original frequency‑based logic. The rest of the pipeline remains unchanged, ensuring a valid `submission.csv` is still produced.'
- What this solution (achieved 0.00209) has done: 'I raise the chain‑specific weight to 1.0 so predictions rely fully on the popularity of hotels within the same chain, and I increase the candidate pool (TOP_N) to give enough options. Moreover, when a chain folder is recognized I use only its chain‑specific list (appending the global list only if needed), which keeps the most relevant hotels first. These modest adjustments stay within the original frequency‑based approach while expectedly moving the MAP@5 score much closer to the target.'
- What this solution (achieved 0.00209) has done: 'The fix removes the stratified split which fails when some hotel IDs appear only once, allowing the code to run and generate a valid `submission.csv`. No core logic changes are made, so the prediction heuristic and scoring remain the same, keeping the solution’s intent while ensuring successful execution.'
- What this solution (achieved 0.00209) has done: 'I keep the original frequency‑based approach but make three small adjustments that are expected to raise MAP@5 toward the target: (1) force the chain weight to 1.0 so predictions rely fully on chain‑specific popularity, (2) expand the candidate pool (TOP_N) to give more options, and (3) change the prediction order to fill remaining slots with the global most‑common hotels before falling back to chain‑specific ones. These tweaks preserve the core logic while providing a richer, better‑ordered candidate list, which should improve the validation score and move the result closer to the target.'
- What this solution (achieved 0.00209) has done: 'I reorder the prediction construction so that, after an exact‑match lookup, the most frequent hotels for the image’s chain are used before falling back to the global most‑common hotels. This keeps the original frequency‑based logic (counts, weighting) unchanged but makes the validation and final predictions follow a more sensible hierarchy, which should raise MAP@5 toward the target score.'
- What this solution (achieved 0.00209) has done: 'I make the prediction lists use the full training data instead of only the 80 % split when building the frequency counters. This gives the heuristic access to every observed hotel‑id, dramatically increasing the chance that the true hotel appears in the top‑5 and moving the MAP@5 score much closer to the target while keeping the original frequency‑based logic unchanged. I also raise `TOP_N` to include all hotels so no correct id is omitted from the candidate pool.'
- What this solution (achieved 0.00209) has done: 'I broaden the search for the optimal weighting between chain‑specific and global hotel popularity and also try two ordering strategies (chain‑first vs. global‑first) when building the top‑5 list. By evaluating both on the validation split and picking the combination that gives the highest MAP@5, the heuristic becomes much more aligned with the target metric while keeping the overall frequency‑based approach unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

BASE_INPUT = "/kaggle/input/hotel-id-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_IMAGES_ROOT = os.path.join(BASE_INPUT, "test_images")
SUBMISSION_PATH = "submission.csv"

TOP_N = 20000

CANDIDATE_WEIGHTS = [0.0, 0.5, 1.0]

ORDER_OPTIONS = ["chain_first", "global_first"]

if not os.path.exists(TRAIN_CSV):
    ALT_BASE = "/kaggle/input/hotel-id-2021-fgvc8"
    TRAIN_CSV = os.path.join(ALT_BASE, "train.csv")

full_df = pd.read_csv(TRAIN_CSV)

image_to_hotel_full = dict(zip(full_df["image"], full_df["hotel_id"].astype(str)))

train_df, val_df = train_test_split(
    full_df, test_size=0.20, random_state=42, shuffle=True
)

global_counter_full = Counter(full_df["hotel_id"])
chain_counters_full = {
    chain: Counter(group["hotel_id"]) for chain, group in full_df.groupby("chain")
}


def map5_score(true_id: str, pred_str: str) -> float:
    """Return MAP@5 for a single example."""
    preds = pred_str.split()
    for rank, p in enumerate(preds, start=1):
        if p == true_id:
            return 1.0 / rank
    return 0.0


def evaluate_weight(chain_weight: float, order: str) -> float:
    """Compute MAP@5 on the validation split for a given weight & ordering."""
    chain_top_n = {}
    for chain, cnt in chain_counters_full.items():
        combined = {}
        for hotel_id, chain_cnt in cnt.items():
            global_cnt = global_counter_full[hotel_id]
            combined[hotel_id] = (
                chain_weight * chain_cnt + (1 - chain_weight) * global_cnt
            )
        top_n = [
            str(h)
            for h, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[
                :TOP_N
            ]
        ]
        chain_top_n[chain] = top_n

    global_top_n = [str(h) for h, _ in global_counter_full.most_common(TOP_N)]

    scores = []
    for _, row in val_df.iterrows():
        filename = row["image"]
        chain_id = row["chain"]
        true_hotel = str(row["hotel_id"])

        preds = []
        exact_match = image_to_hotel_full.get(filename)
        if exact_match:
            preds.append(exact_match)

        sources = (
            (chain_top_n.get(chain_id, []), global_top_n)
            if order == "chain_first"
            else (global_top_n, chain_top_n.get(chain_id, []))
        )

        for src in sources:
            for pid in src:
                if pid not in preds:
                    preds.append(pid)
                if len(preds) == 5:
                    break
            if len(preds) == 5:
                break

        pred_str = " ".join(preds)
        scores.append(map5_score(true_hotel, pred_str))

    return np.mean(scores)


best_weight = None
best_order = None
best_score = -1.0
for w in CANDIDATE_WEIGHTS:
    for ord_opt in ORDER_OPTIONS:
        score = evaluate_weight(w, ord_opt)
        if score > best_score:
            best_score = score
            best_weight = w
            best_order = ord_opt

CHAIN_WEIGHT = best_weight
PRED_ORDER = best_order
print(
    f"Chosen CHAIN_WEIGHT = {CHAIN_WEIGHT:.2f}, PRED_ORDER = '{PRED_ORDER}' (validation MAP@5 = {best_score:.5f})"
)

global_counter = global_counter_full
chain_counters = chain_counters_full

chain_top_n = {}
for chain, cnt in chain_counters.items():
    combined = {}
    for hotel_id, chain_cnt in cnt.items():
        global_cnt = global_counter[hotel_id]
        combined[hotel_id] = CHAIN_WEIGHT * chain_cnt + (1 - CHAIN_WEIGHT) * global_cnt
    top_n = [
        str(h)
        for h, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    ]
    chain_top_n[chain] = top_n

global_top_n = [str(h) for h, _ in global_counter.most_common(TOP_N)]



## === cell 1
test_image_paths = []
for root, _, files in os.walk(TEST_IMAGES_ROOT):
    for f in files:
        if f.lower().endswith(".jpg"):
            rel_path = os.path.relpath(os.path.join(root, f), BASE_INPUT)
            test_image_paths.append(rel_path)

rows = []
for rel_path in test_image_paths:
    parts = rel_path.split(os.sep)
    filename = os.path.basename(rel_path)

    preds = []
    exact_match = image_to_hotel_full.get(filename)
    if exact_match:
        preds.append(exact_match)

    chain_id = None
    if len(parts) >= 2:
        try:
            chain_id = int(parts[1])
        except ValueError:
            pass

    sources = (
        (chain_top_n.get(chain_id, []), global_top_n)
        if PRED_ORDER == "chain_first"
        else (global_top_n, chain_top_n.get(chain_id, []))
    )
    for src in sources:
        for pid in src:
            if pid not in preds:
                preds.append(pid)
            if len(preds) == 5:
                break
        if len(preds) == 5:
            break

    rows.append({"image": filename, "hotel_id": " ".join(preds)})

submission_df = pd.DataFrame(rows)[["image", "hotel_id"]]
submission_df.to_csv(SUBMISSION_PATH, index=False)



## === cell 2
print(f"Submission written to {SUBMISSION_PATH}")
print(submission_df.head())
