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

0.7698703376273452

# 6. Current score

0.00139

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the reliance on the unavailable `pekolib` package with a small pure‑Python baseline that reads the training metadata, computes the five most frequent hotel IDs overall, and writes a valid `submission.csv` containing those IDs for every test image. This guarantees a correctly‑formatted submission and gives a non‑zero MAP@5 score, moving the result toward the target.'
- What this solution (achieved 0.00201) has done: 'I keep the original file‑listing cell unchanged, then replace the naive “same five hotels for every image” logic with a small prefix‑based heuristic: for each image I look at the first two characters of its filename, compute the most frequent hotel IDs for that prefix in the training set, and use those as the personalised top‑5 list (falling back to the global most‑frequent hotels when a prefix is unseen). This adds only lightweight pandas operations, preserves the overall workflow, and is expected to increase MAP@5 toward the target without altering the core model‑free approach.'
- What this solution (achieved 0.00139) has done: 'I extend the prefix‑based heuristic by trying longer prefixes (4 and 3 characters) before falling back to the 2‑character prefix or the global most‑frequent hotels. This keeps the same lightweight, pure‑Python approach while giving a more specific prediction for many images, which should raise MAP@5 toward the target without altering the overall workflow.'
- What this solution (achieved 0.00139) has done: 'I add a one‑character prefix heuristic as an extra fallback before the global most‑frequent list. This gives the model a slightly more specific guess for images whose longer prefixes are unseen, while keeping the same lightweight, pure‑Python workflow. The change only extends the existing prefix‑based logic and does not alter any core modeling or data handling.'
- What this solution (achieved 0.00139) has done: 'I add a quick exact‑image lookup: if a test filename also appears in the training metadata we can output its true hotel ID as the first prediction (filled out with the global most‑frequent IDs). This small, deterministic tweak keeps the original prefix‑based logic unchanged while giving a guaranteed boost for any overlapping images, moving the MAP@5 score closer to the target.'
- What this solution (achieved 0.00139) has done: 'I fixed the runtime error caused by trying to convert a non‑numeric folder name (“test_images”) to an int when looking up chain‑specific hotel lists. The code now safely attempts the conversion and skips the chain‑based fallback if conversion fails, preserving the existing heuristic logic. This allows the script to run end‑to‑end, generate a correctly‑formatted `submission.csv`, and be ready for scoring.'
- What this solution (achieved 0.00139) has done: 'I add a lightweight “chain‑plus‑prefix” heuristic: for each hotel chain I compute the most frequent hotel IDs for the first 4 and 3 characters of the image filename. During prediction the script first tries an exact‑image match, then a chain‑+‑prefix match (4 → 3 chars), then the plain chain top‑5, and finally the generic prefix fallback. This keeps the overall simple, non‑model approach while giving much more specific guesses, which should raise MAP@5 toward the target without altering the core workflow.'
- What this solution (achieved 0.00139) has done: 'I adjust the prediction logic to prioritize chain‑specific frequent hotels (which are more informative than generic filename prefixes) before falling back to the prefix heuristics, and I simplify the fallback padding. This keeps the overall heuristic approach while giving a higher chance that the true hotel appears in the top‑5, moving the MAP@5 score closer to the target.'
- What this solution (achieved 0.00139) has done: 'I add a lightweight “hotel‑id prefix” heuristic: for each image I first check if the first five characters of its filename match the start of any hotel ID seen in training, and if so I use those frequent IDs (padded with the global top‑5). This adds useful signal without changing the overall workflow, and should raise the MAP@5 score toward the target. The rest of the code remains the same, ensuring a valid `submission.csv` is still produced.'

# 9. Code solution

## === cell 0
import subprocess, os, sys, json, pathlib, pandas as pd


def run_cmd(cmd):
    return subprocess.check_output(cmd, shell=True, text=True)


print(run_cmd("ls -R /kaggle/input/ | head -n 20"))




## === cell 1
train_path = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
sample_sub_path = "/kaggle/input/hotel-id-2021-fgvc8/sample_submission.csv"

train_df = pd.read_csv(train_path)
sample_sub_df = pd.read_csv(sample_sub_path)

global_top5 = train_df["hotel_id"].value_counts().head(5).index.astype(str).tolist()

exact_image_dict = train_df.set_index("image")["hotel_id"].astype(str).to_dict()


def build_prefix_dict(length):
    col_name = f"prefix_{length}"
    train_df[col_name] = train_df["image"].str.split(".").str[0].str[:length]
    return (
        train_df.groupby(col_name)["hotel_id"]
        .apply(lambda x: x.value_counts().head(5).index.astype(str).tolist())
        .to_dict()
    )


prefix4_top5 = build_prefix_dict(4)
prefix3_top5 = build_prefix_dict(3)
prefix2_top5 = build_prefix_dict(2)
prefix1_top5 = build_prefix_dict(1)

test_images_root = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
test_image_to_chain = {}
for img_path in pathlib.Path(test_images_root).rglob("*.jpg"):
    test_image_to_chain[img_path.name] = img_path.parent.name

chain_top5 = (
    train_df.groupby("chain")["hotel_id"]
    .apply(lambda x: x.value_counts().head(5).index.astype(str).tolist())
    .to_dict()
)


def build_chain_prefix_dict(length):
    """Returns a dict keyed by \"{chain}_{prefix}\" → top‑5 hotel_ids."""
    col_name = f"chain_prefix_{length}"
    train_df[col_name] = (
        train_df["chain"].astype(str)
        + "_"
        + train_df["image"].str.split(".").str[0].str[:length]
    )
    return (
        train_df.groupby(col_name)["hotel_id"]
        .apply(lambda x: x.value_counts().head(5).index.astype(str).tolist())
        .to_dict()
    )


chain_prefix4_top5 = build_chain_prefix_dict(4)
chain_prefix3_top5 = build_chain_prefix_dict(3)

hotel_counts = train_df["hotel_id"].value_counts()
prefix_hotel5_top5 = {}
for hid in hotel_counts.index.astype(str):
    pref = hid[:5]
    if pref not in prefix_hotel5_top5:
        prefix_hotel5_top5[pref] = []
    if len(prefix_hotel5_top5[pref]) < 5:
        prefix_hotel5_top5[pref].append(hid)


def pad_to_five(top_list):
    """Pad a list with global_top5 entries (excluding duplicates) to length 5."""
    if len(top_list) >= 5:
        return top_list[:5]
    needed = 5 - len(top_list)
    extra = [h for h in global_top5 if h not in top_list][:needed]
    return top_list + extra


def get_top5_for_image(img_name):
    if img_name in exact_image_dict:
        exact_hotel = exact_image_dict[img_name]
        rest = [h for h in global_top5 if h != exact_hotel][:4]
        return " ".join([exact_hotel] + rest)

    prefix5 = img_name.split(".")[0][:5]
    if prefix5 in prefix_hotel5_top5:
        return " ".join(pad_to_five(prefix_hotel5_top5[prefix5]))

    chain_str = test_image_to_chain.get(img_name)
    if chain_str is not None:
        try:
            chain_id = int(chain_str)
            top5 = chain_top5.get(chain_id)
            if top5:
                return " ".join(pad_to_five(top5))
        except ValueError:
            pass  # non‑numeric folder names are ignored

    if chain_str is not None:
        try:
            chain_id = int(chain_str)
            prefix4 = img_name.split(".")[0][:4]
            key4 = f"{chain_id}_{prefix4}"
            top5 = chain_prefix4_top5.get(key4)
            if top5:
                return " ".join(pad_to_five(top5))

            prefix3 = img_name.split(".")[0][:3]
            key3 = f"{chain_id}_{prefix3}"
            top5 = chain_prefix3_top5.get(key3)
            if top5:
                return " ".join(pad_to_five(top5))
        except ValueError:
            pass

    for prefix_len, mapping in [
        (4, prefix4_top5),
        (3, prefix3_top5),
        (2, prefix2_top5),
        (1, prefix1_top5),
    ]:
        prefix = img_name.split(".")[0][:prefix_len]
        top5 = mapping.get(prefix)
        if top5:
            return " ".join(pad_to_five(top5))

    return " ".join(global_top5)


submission = sample_sub_df.copy()
submission["hotel_id"] = submission["image"].apply(get_top5_for_image)

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)

print(f"Written submission with {len(submission)} rows to {output_path}")




## === cell 2
print(pd.read_csv(output_path).head())
