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

0.7444325166287762

# 6. Current score

0.00169

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the fragile pip‑install and external module call with a small, self‑contained baseline that reads the training metadata, computes the five most frequent hotel_id values, and writes those same five IDs for every test image. This guarantees a valid submission.csv in the required format and yields a non‑zero MAP@5 score, moving the result toward the target without altering any core modeling logic.'
- What this solution (achieved 0.00209) has done: 'I keep the overall workflow but replace the single global “top‑5” heuristic with a per‑chain top‑5 prediction: for each test image I extract its chain folder name, look up the five most frequent hotel IDs for that chain in the training data, and use those IDs as the prediction (falling back to the global top‑5 when the chain is unseen). This simple contextual cue should raise MAP@5 dramatically while preserving the original simple baseline approach.'
- What this solution (achieved 0.00209) has done: 'I add a fallback that directly uses the known hotel_id when a test image filename also appears in the training set, and I ensure every prediction list always contains exactly five IDs by padding with the most frequent global IDs. This keeps the original per‑chain top‑5 logic but makes predictions more accurate for any overlapping images and guarantees a proper five‑ID format, which should move the MAP@5 score closer to the target.'
- What this solution (achieved 0.00147) has done: 'I add a lightweight prefix‑based heuristic: for each test image I first check if the exact filename exists in the training set, then I look for other training images that share the first 4 characters of the filename and use the most common hotel IDs among those as a prediction. If no prefix match is found I fall back to the per‑chain top‑5 (or the global top‑5). This small addition keeps the original workflow intact while providing a more informed guess for many images, moving the MAP@5 score closer to the target.'
- What this solution (achieved 0.00166) has done: 'I fix the prefix‑based lookup (the original code built an empty map due to a column‑name mistake) and add a longer‑prefix fallback (6 characters) before resorting to the per‑chain top‑5. This keeps the overall simple heuristic while giving the model more specific information, which should raise the MAP@5 toward the target.'
- What this solution (achieved 0.00209) has done: 'The changes keep the original lightweight heuristics but make the prediction richer:  
1. When a prefix match is found we now intersect those IDs with the chain‑specific top‑5 list, favouring IDs that are both prefix‑relevant and common in the same hotel chain.  
2. If the intersection yields fewer than five IDs we pad with the chain‑top‑5, then with the global top‑5, guaranteeing exactly five predictions.  
3. The helper `combine_ids` centralises this logic, improving the chance that the true hotel appears in the list while preserving the overall workflow and output format.'
- What this solution (achieved 0.00169) has done: 'The update adds a longer‑prefix heuristic (8 characters), improves the ID merging logic to keep useful prefix candidates before falling back to chain and global tops, and tweaks the prediction flow to try the extra prefix before shorter ones. These changes keep the original simple heuristic while giving the model more specific information, which should raise MAP@5 toward the target score.'
- What this solution (achieved 0.00169) has done: 'The update expands the per‑chain candidate pool from the five most common hotel IDs to the twenty most common. Providing a richer list gives the `combine_ids` logic more options to intersect with prefix‑based candidates, which raises the chance that the true hotel appears within the final five predictions and therefore moves the MAP@5 score closer to the target while keeping the overall workflow unchanged.'

# 9. Code solution

## === cell 0
import os, glob, pandas as pd

input_root = "/kaggle/input/hotel-id-2021-fgvc8"
train_path = os.path.join(input_root, "train.csv")
train_df = pd.read_csv(train_path)

global_top5 = train_df["hotel_id"].value_counts().nlargest(5).index.tolist()
global_top5_str = " ".join(map(str, global_top5))
print("Global top‑5 hotel IDs:", global_top5_str)


def top_n_per_chain(series, n=20):
    """Return the n most frequent hotel_ids for a given chain."""
    return series.value_counts().nlargest(n).index.tolist()


chain_top5_series = train_df.groupby("chain")["hotel_id"].apply(
    lambda s: top_n_per_chain(s, 20)
)
chain_top5_str = {
    str(chain): " ".join(map(str, ids)) for chain, ids in chain_top5_series.items()
}
print(
    f"Computed top‑{20} for {len(chain_top5_str)} chains (sample):",
    list(chain_top5_str.items())[:3],
)

image_to_hotel = dict(zip(train_df["image"], train_df["hotel_id"]))


def build_prefix_map(prefix_len):
    """
    Build a map: prefix → list of hotel_ids ordered by frequency.
    """
    pref_counts = (
        train_df.groupby(train_df["image"].str.slice(0, prefix_len))["hotel_id"]
        .value_counts()
        .reset_index(name="cnt")
    )
    prefix_map = {}
    for prefix, sub in pref_counts.groupby("image"):
        ordered = sub.sort_values("cnt", ascending=False)["hotel_id"].tolist()
        prefix_map[prefix] = ordered
    return prefix_map


prefix_len_short = 4
prefix_len_long = 6
prefix_len_extra = 8
prefix_to_ids_short = build_prefix_map(prefix_len_short)
prefix_to_ids_long = build_prefix_map(prefix_len_long)
prefix_to_ids_extra = build_prefix_map(prefix_len_extra)


def pad_to_five(ids):
    """Return exactly 5 hotel IDs, padding with global top‑5 if needed."""
    ids = ids[:5]  # truncate excess
    for gid in global_top5:
        if gid not in ids:
            ids.append(gid)
        if len(ids) == 5:
            break
    return ids


def combine_ids(base_ids, chain_ids):
    """
    Merge candidate IDs from a prefix match with the chain‑specific top‑IDs.
    Preserve intersecting IDs first, then any remaining base IDs, then chain IDs,
    finally pad with global top‑5 to reach exactly five entries.
    """
    result = [i for i in base_ids if i in chain_ids]  # intersect, keep order
    for i in base_ids:
        if i not in result:
            result.append(i)
        if len(result) >= 5:
            break
    for i in chain_ids:
        if i not in result:
            result.append(i)
        if len(result) >= 5:
            break
    return pad_to_five(result)




## === cell 1
test_images_dir = os.path.join(input_root, "test_images")
test_image_paths = sorted(
    glob.glob(os.path.join(test_images_dir, "**", "*.jpg"), recursive=True)
)
test_filenames = [os.path.basename(p) for p in test_image_paths]

predictions = []
for path, fname in zip(test_image_paths, test_filenames):
    if fname in image_to_hotel:
        base_ids = [image_to_hotel[fname]]
        pred_ids = pad_to_five(base_ids)
    else:
        prefix_extra = fname[:prefix_len_extra]
        ids_extra = prefix_to_ids_extra.get(prefix_extra, [])
        if ids_extra:
            chain_folder = os.path.basename(os.path.dirname(path))
            chain_ids_str = chain_top5_str.get(chain_folder, "")
            chain_ids = list(map(int, chain_ids_str.split())) if chain_ids_str else []
            pred_ids = combine_ids(ids_extra, chain_ids)
        else:
            prefix_long = fname[:prefix_len_long]
            ids_long = prefix_to_ids_long.get(prefix_long, [])
            if ids_long:
                chain_folder = os.path.basename(os.path.dirname(path))
                chain_ids_str = chain_top5_str.get(chain_folder, "")
                chain_ids = (
                    list(map(int, chain_ids_str.split())) if chain_ids_str else []
                )
                pred_ids = combine_ids(ids_long, chain_ids)
            else:
                prefix_short = fname[:prefix_len_short]
                ids_short = prefix_to_ids_short.get(prefix_short, [])
                if ids_short:
                    chain_folder = os.path.basename(os.path.dirname(path))
                    chain_ids_str = chain_top5_str.get(chain_folder, "")
                    chain_ids = (
                        list(map(int, chain_ids_str.split())) if chain_ids_str else []
                    )
                    pred_ids = combine_ids(ids_short, chain_ids)
                else:
                    chain_folder = os.path.basename(os.path.dirname(path))
                    chain_ids_str = chain_top5_str.get(chain_folder, "")
                    chain_ids = (
                        list(map(int, chain_ids_str.split())) if chain_ids_str else []
                    )
                    pred_ids = pad_to_five(chain_ids)
    predictions.append(" ".join(map(str, pred_ids)))

submission = pd.DataFrame({"image": test_filenames, "hotel_id": predictions})
print(f"Prepared submission for {len(submission)} images (sample):")
print(submission.head())




## === cell 2
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission written to:", submission_path)
