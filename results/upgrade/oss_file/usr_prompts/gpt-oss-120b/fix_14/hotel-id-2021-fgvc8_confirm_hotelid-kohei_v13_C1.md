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
import subprocess, os, sys, json, pathlib, pandas as pd
import numpy as np
import random
from PIL import Image


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


prefix1_top5 = build_prefix_dict(1)
prefix2_top5 = build_prefix_dict(2)
prefix3_top5 = build_prefix_dict(3)
prefix4_top5 = build_prefix_dict(4)
prefix5_top5 = build_prefix_dict(5)
prefix6_top5 = build_prefix_dict(6)

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


chain_prefix3_top5 = build_chain_prefix_dict(3)
chain_prefix4_top5 = build_chain_prefix_dict(4)
chain_prefix5_top5 = build_chain_prefix_dict(5)

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


def mean_rgb(image_path):
    """Return a 3‑element ndarray with mean R,G,B values."""
    with Image.open(image_path).convert("RGB") as img:
        img = img.resize((32, 32))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.mean(axis=(0, 1))


train_images_root = "/kaggle/input/hotel-id-2021-fgvc8/train_images"
all_train_paths = list(pathlib.Path(train_images_root).rglob("*.jpg"))
sample_size = 5000
if len(all_train_paths) > sample_size:
    random.seed(42)
    train_sample_paths = random.sample(all_train_paths, sample_size)
else:
    train_sample_paths = all_train_paths

train_means = []
train_hotel_ids = []
for p in train_sample_paths:
    img_name = p.name
    if img_name in exact_image_dict:
        try:
            vec = mean_rgb(p)
            train_means.append(vec)
            train_hotel_ids.append(exact_image_dict[img_name])
        except Exception:
            continue
train_means = np.stack(train_means) if train_means else np.empty((0, 3))


def get_top5_for_image(img_name):
    if img_name in exact_image_dict:
        exact_hotel = exact_image_dict[img_name]
        rest = [h for h in global_top5 if h != exact_hotel][:4]
        return " ".join([exact_hotel] + rest)

    if train_means.size > 0:
        try:
            test_path = pathlib.Path(test_images_root) / img_name
            test_vec = mean_rgb(test_path)
            dists = np.linalg.norm(train_means - test_vec, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_hotel = train_hotel_ids[nearest_idx]
            return " ".join(pad_to_five([nearest_hotel]))
        except Exception:
            pass  # fallback if image cannot be read

    chain_str = test_image_to_chain.get(img_name)
    if chain_str is not None:
        try:
            chain_id = int(chain_str)
            top5 = chain_top5.get(chain_id)
            if top5:
                return " ".join(pad_to_five(top5))

            for length, mapping in [
                (5, chain_prefix5_top5),
                (4, chain_prefix4_top5),
                (3, chain_prefix3_top5),
            ]:
                pref = img_name.split(".")[0][:length]
                key = f"{chain_id}_{pref}"
                top5 = mapping.get(key)
                if top5:
                    return " ".join(pad_to_five(top5))
        except ValueError:
            pass  # non‑numeric folder names are ignored

    prefix5 = img_name.split(".")[0][:5]
    if prefix5 in prefix_hotel5_top5:
        return " ".join(pad_to_five(prefix_hotel5_top5[prefix5]))

    for length, mapping in [
        (6, prefix6_top5),
        (5, prefix5_top5),
        (4, prefix4_top5),
        (3, prefix3_top5),
        (2, prefix2_top5),
        (1, prefix1_top5),
    ]:
        pref = img_name.split(".")[0][:length]
        top5 = mapping.get(pref)
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
