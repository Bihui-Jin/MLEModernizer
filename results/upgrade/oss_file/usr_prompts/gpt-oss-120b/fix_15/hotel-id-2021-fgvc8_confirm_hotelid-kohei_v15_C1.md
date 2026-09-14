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

0.7368948387639952

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the package‑install and opaque module call with a small self‑contained baseline that loads the training metadata, picks the five most frequent hotel IDs, and writes a valid `submission.csv` containing those IDs for every test image. The script also computes an internal MAP@5 on a hold‑out split of the training data so you can see how far the simple baseline is from the target score, without changing any core modeling logic. This guarantees a correctly formatted CSV and moves the score from “not yielded” toward the target.'
- What this solution (achieved 0.0) has done: 'We avoid the stratify error by removing stratification, and improve predictions by using the hotel‑frequency within each chain (derived from the folder name of each test image). This keeps the original simple‑baseline logic but makes the per‑image list much more relevant, moving the MAP@5 toward the target while still producing a correctly formatted CSV.'
- What this solution (achieved 0.0) has done: 'I keep the original simple frequency‑based logic but ensure every prediction list has exactly five IDs by padding missing slots with the global most‑frequent hotels that are not already present. This modest change can increase the chance that the true hotel appears in the top‑5, thus moving the internal MAP@5 closer to the target while preserving the overall approach and producing a valid submission file.'
- What this solution (achieved 0.0) has done: 'I keep the overall frequency‑based approach but make the fallback padding smarter: for a given chain we now use the full ordered list of its hotels (not just the first five) to fill any missing slots before resorting to the global top‑5. This gives each chain more relevant candidates while still guaranteeing exactly five IDs per image, which should raise the internal MAP@5 and move the Kaggle score nearer the target.'
- What this solution (achieved 0.0) has done: 'The update keeps the original frequency‑based strategy but adds a tiny scoring step when building each prediction: hotels that appear both in the chain‑specific list and in the global‑top‑5 receive a higher priority, then the remaining slots are filled with the most frequent global IDs that are not already chosen. This preserves the core logic, guarantees exactly five IDs per image, and should raise the internal MAP@5, moving the score closer to the target.'
- What this solution (achieved 0.0) has done: 'I keep the overall workflow unchanged but simplify the prediction logic: for each image I now take the most frequent hotels for its chain (up to five) and only then pad with the global top‑5 IDs that are not already used. This respects the original frequency‑based idea while removing the extra scoring step that could mis‑order the chain‑specific list, giving a modest boost to the internal MAP@5 and keeping the submission format identical. The rest of the notebook (data loading, validation split, MAP@5 computation, and CSV export) remains the same.'
- What this solution (achieved 0.0) has done: 'I keep the overall frequency‑based workflow but adjust the `build_prediction` function to mix a few chain‑specific hotels with the global most‑frequent ones (3 from the chain then fill up to 5 with global IDs). This still respects the original logic, guarantees exactly five IDs, and should raise the internal MAP@5, moving the score closer to the target while leaving the rest of the pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I tighten the prediction logic by using up to five of the most frequent hotels for each chain before falling back to the global top‑5 IDs. This small change keeps the overall frequency‑based approach but gives each image a better‑aligned candidate list, which should raise the internal MAP@5 from 0 toward the target while still writing a correctly formatted submission file.'
- What this solution (achieved 0.0) has done: 'I adjust the `build_prediction` function so that it first adds any chain‑specific hotels that also appear in the global top‑5 (preserving the chain order), then fills the remaining slots with the rest of the chain’s frequent hotels, and finally pads with global top‑5 IDs not already used. This small re‑ordering keeps the original frequency‑based logic but makes the most globally common hotels appear earlier when they belong to the same chain, which should raise the internal MAP@5 and move the score closer to the target.'
- What this solution (achieved 0.0) has done: 'I expand the fallback pool from the top 5 globally frequent hotels to the top 50, keeping the same chain‑specific ordering but allowing more candidates when a chain has few known hotels. This modest change preserves the original frequency‑based logic, guarantees exactly five IDs per image, and should raise the internal MAP@5 closer to the target score while still producing a valid submission CSV.'
- What this solution (achieved 0.0) has done: 'I expand the fallback pool from the top 50 to the top 200 most‑frequent hotels so that each prediction has a larger chance of containing the true hotel ID, and I use this broader list in the `build_prediction` function. This small change preserves the original frequency‑based logic while modestly improving the internal MAP@5, moving the score closer to the target.'
- What this solution (achieved 0.0) has done: 'I expand the fallback pool from the top 200 most‑frequent hotels to the top 5 000 so that when a chain does not provide enough candidates the model can draw from a much larger set of likely hotels. This small change keeps the original frequency‑based logic intact, only enlarges the global list used for padding, and is expected to raise the internal MAP@5 (moving the score toward the target) while still producing a correctly formatted submission CSV.'
- What this solution (achieved 0.0) has done: 'I adjust the prediction logic to prioritize the five most frequent hotels overall before adding chain‑specific hotels, while still padding with the broader global list when needed. This small re‑ordering keeps the original frequency‑based approach but should raise MAP@5 by giving higher weight to globally common hotels that often appear in the top‑5. The rest of the code stays unchanged, and a valid submission.csv is still written.'
- What this solution (achieved 0.0) has done: 'I keep the overall frequency‑based approach but reorder the prediction construction so that chain‑specific hotels are chosen first (up to five), then the global top‑5 IDs that haven’t been used, and finally the broader global list for any remaining slots. This small re‑ordering respects the original logic while giving each image more relevant candidates, which should raise the internal MAP@5 and move the Kaggle score toward the target.'

# 9. Code solution

## === cell 0
import os, glob, pandas as pd, numpy as np
from collections import defaultdict

print("Root input dirs:", os.listdir("/kaggle/input"))
print("Working dirs:", os.listdir("/kaggle/working"))

train_path = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
train_df = pd.read_csv(train_path)
print("Train rows:", len(train_df))

global_top5000_list = (
    train_df["hotel_id"].value_counts().head(5000).astype(str).tolist()
)
global_top5_list = global_top5000_list[:5]  # keep a separate list for the top‑5
global_top5_str = " ".join(global_top5_list)
print("Global top‑5 frequent hotel IDs (used for display):", global_top5_str)

chain_full_list = {}
for chain_id, grp in train_df.groupby("chain"):
    ordered = grp["hotel_id"].value_counts().astype(str).tolist()
    if ordered:
        chain_full_list[str(chain_id)] = ordered



## === cell 1
test_images_dir = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
test_image_paths = glob.glob(
    os.path.join(test_images_dir, "**", "*.jpg"), recursive=True
)
test_images = [os.path.basename(p) for p in test_image_paths]
test_chains = [os.path.basename(os.path.dirname(p)) for p in test_image_paths]


def build_prediction(chain):
    """
    Return a space‑separated string of exactly 5 hotel IDs.
    Ordering strategy (minimal change from original):
      1. Chain‑specific hotels (most frequent for that chain).
      2. Global top‑5 most frequent hotels not already selected.
      3. Global top‑5 000 hotels not already selected.
    This keeps the frequency‑based core while giving higher priority
    to chain‑relevant hotels, which should modestly improve MAP@5.
    """
    chain_list = chain_full_list.get(chain, [])
    ordered = []

    for hid in chain_list:
        if hid not in ordered:
            ordered.append(hid)
        if len(ordered) >= 5:
            break

    if len(ordered) < 5:
        for gid in global_top5_list:
            if gid not in ordered:
                ordered.append(gid)
            if len(ordered) >= 5:
                break

    if len(ordered) < 5:
        for gid in global_top5000_list:
            if gid not in ordered:
                ordered.append(gid)
            if len(ordered) >= 5:
                break

    return " ".join(ordered[:5])


predictions = [build_prediction(chain) for chain in test_chains]

submission_df = pd.DataFrame({"image": test_images, "hotel_id": predictions})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission written to:", submission_path)
print(submission_df.head())



## === cell 2
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    train_df.index, test_size=0.1, random_state=42, shuffle=True
)
val_df = train_df.loc[val_idx]


def map5_score(row, pred_str):
    preds = pred_str.split()
    true = str(row["hotel_id"])
    if true in preds:
        rank = preds.index(true) + 1  # 1‑based rank
        return 1.0 / rank
    return 0.0


val_predictions = [build_prediction(str(row["chain"])) for _, row in val_df.iterrows()]
val_scores = [
    map5_score(row, pred) for (_, row), pred in zip(val_df.iterrows(), val_predictions)
]
internal_map5 = np.mean(val_scores)
print(f"Internal MAP@5 on 10% hold‑out: {internal_map5:.5f}")
