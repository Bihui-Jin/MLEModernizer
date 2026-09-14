# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the reliance on the unavailable `pekolib` package with a small pure‑Python baseline that reads the training metadata, computes the five most frequent hotel IDs overall, and writes a valid `submission.csv` containing those IDs for every test image. This guarantees a correctly‑formatted submission and gives a non‑zero MAP@5 score, moving the result toward the target.'
- What this solution (achieved 0.00201) has done: 'I keep the original file‑listing cell unchanged, then replace the naive “same five hotels for every image” logic with a small prefix‑based heuristic: for each image I look at the first two characters of its filename, compute the most frequent hotel IDs for that prefix in the training set, and use those as the personalised top‑5 list (falling back to the global most‑frequent hotels when a prefix is unseen). This adds only lightweight pandas operations, preserves the overall workflow, and is expected to increase MAP@5 toward the target without altering the core model‑free approach.'
- What this solution (achieved 0.00139) has done: 'I extend the prefix‑based heuristic by trying longer prefixes (4 and 3 characters) before falling back to the 2‑character prefix or the global most‑frequent hotels. This keeps the same lightweight, pure‑Python approach while giving a more specific prediction for many images, which should raise MAP@5 toward the target without altering the overall workflow.'
- What this solution (achieved 0.00139) has done: 'I add a one‑character prefix heuristic as an extra fallback before the global most‑frequent list. This gives the model a slightly more specific guess for images whose longer prefixes are unseen, while keeping the same lightweight, pure‑Python workflow. The change only extends the existing prefix‑based logic and does not alter any core modeling or data handling.'
- What this solution (achieved 0.00139) has done: 'I add a quick exact‑image lookup: if a test filename also appears in the training metadata we can output its true hotel ID as the first prediction (filled out with the global most‑frequent IDs). This small, deterministic tweak keeps the original prefix‑based logic unchanged while giving a guaranteed boost for any overlapping images, moving the MAP@5 score closer to the target.'

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
prefix1_top5 = build_prefix_dict(1)  # fine‑grained fallback

test_images_root = "/kaggle/input/hotel-id-2021-fgvc8/test_images"
test_image_to_chain = {}
for img_path in pathlib.Path(test_images_root).rglob("*.jpg"):
    test_image_to_chain[img_path.name] = img_path.parent.name

chain_top5 = (
    train_df.groupby("chain")["hotel_id"]
    .apply(lambda x: x.value_counts().head(5).index.astype(str).tolist())
    .to_dict()
)


def get_top5_for_image(img_name):
    if img_name in exact_image_dict:
        exact_hotel = exact_image_dict[img_name]
        needed = 5 - 1
        rest = [h for h in global_top5 if h != exact_hotel][:needed]
        return " ".join([exact_hotel] + rest)

    chain = test_image_to_chain.get(img_name)
    if chain is not None:
        top5 = chain_top5.get(int(chain))  # chain column is numeric
        if top5:
            if len(top5) < 5:
                needed = 5 - len(top5)
                top5 = top5 + [h for h in global_top5 if h not in top5][:needed]
            return " ".join(top5)

    for prefix_len, mapping in [
        (4, prefix4_top5),
        (3, prefix3_top5),
        (2, prefix2_top5),
        (1, prefix1_top5),
    ]:
        prefix = img_name.split(".")[0][:prefix_len]
        top5 = mapping.get(prefix)
        if top5:
            if len(top5) < 5:
                needed = 5 - len(top5)
                top5 = top5 + [h for h in global_top5 if h not in top5][:needed]
            return " ".join(top5)

    return " ".join(global_top5)


submission = sample_sub_df.copy()
submission["hotel_id"] = submission["image"].apply(get_top5_for_image)

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)

print(f"Written submission with {len(submission)} rows to {output_path}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1693509863.py in <cell line: 0>()
     83 
     84 submission = sample_sub_df.copy()
---> 85 submission["hotel_id"] = submission["image"].apply(get_top5_for_image)
     86 
     87 output_path = "/kaggle/working/submission.csv"

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/1693509863.py in get_top5_for_image(img_name)
     56     chain = test_image_to_chain.get(img_name)
     57     if chain is not None:
---> 58         top5 = chain_top5.get(int(chain))  # chain column is numeric
     59         if top5:
     60             if len(top5) < 5:

ValueError: invalid literal for int() with base 10: 'test_images'

## === cell 2
print(pd.read_csv(output_path).head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3545654737.py in <cell line: 0>()
----> 1 print(pd.read_csv(output_path).head())

NameError: name 'output_path' is not defined
