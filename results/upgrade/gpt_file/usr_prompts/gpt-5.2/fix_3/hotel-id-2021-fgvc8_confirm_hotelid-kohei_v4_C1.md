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
import os
import numpy as np
import pandas as pd
from PIL import Image

print("Listing /kaggle/input:")
print("\n".join(sorted(os.listdir("/kaggle/input"))[:50]))



## === cell 1

BASE = "/kaggle/input/hotel-id-2021-fgvc8"
train_csv_path = os.path.join(BASE, "train.csv")
sample_sub_path = os.path.join(BASE, "sample_submission.csv")
train_img_dir = os.path.join(BASE, "train_images")
test_img_dir = os.path.join(BASE, "test_images")

train_df = pd.read_csv(train_csv_path)
sub_df = pd.read_csv(sample_sub_path)


def img_to_feat(path, size=(32, 32)):
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize(size, Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr.reshape(-1)
    except Exception:
        return None


rng = np.random.default_rng(42)
max_hotels = 5000  # cap hotels to include (by frequency)
max_imgs_per_hotel = 3  # cap images per hotel
thumb_size = (32, 32)

hotel_counts = train_df["hotel_id"].value_counts()
selected_hotels = hotel_counts.head(max_hotels).index
train_df_small = train_df[train_df["hotel_id"].isin(selected_hotels)].copy()

train_df_small["_rand"] = rng.random(len(train_df_small))
train_df_small = (
    train_df_small.sort_values(["hotel_id", "_rand"])
    .groupby("hotel_id", as_index=False)
    .head(max_imgs_per_hotel)
    .drop(columns=["_rand"])
)

train_feats = []
train_hotels = []
train_images = []

for img_name, hid in zip(
    train_df_small["image"].values, train_df_small["hotel_id"].values
):
    img_path = (
        os.path.join(
            train_img_dir,
            str(
                int(
                    train_df_small.loc[
                        train_df_small["image"] == img_name, "chain"
                    ].values[0]
                )
            ),
            img_name,
        )
        if False
        else None
    )  # chain-folder mapping can vary; use robust search below

chain_dirs = [
    d
    for d in os.listdir(train_img_dir)
    if os.path.isdir(os.path.join(train_img_dir, d))
]
chain_dirs_set = set(chain_dirs)


def resolve_train_path(image_name):
    return None


train_df_small = train_df_small.merge(
    train_df[["image", "chain", "hotel_id"]],
    on=["image", "hotel_id"],
    how="left",
    suffixes=("", "_y"),
)
train_df_small = train_df_small.drop_duplicates(subset=["image", "hotel_id"])

missing = 0
for img_name, chain, hid in zip(
    train_df_small["image"].astype(str).values,
    train_df_small["chain"].values,
    train_df_small["hotel_id"].values,
):
    chain_str = str(int(chain)) if pd.notna(chain) else None
    if chain_str is None:
        missing += 1
        continue
    p = os.path.join(train_img_dir, chain_str, img_name)
    feat = img_to_feat(p, size=thumb_size)
    if feat is None:
        missing += 1
        continue
    train_feats.append(feat)
    train_hotels.append(str(hid))
    train_images.append(img_name)

train_feats = np.asarray(train_feats, dtype=np.float32)
train_hotels = np.asarray(train_hotels)

print("Gallery built:")
print(" - train feats shape:", train_feats.shape)
print(" - unique hotels in gallery:", len(np.unique(train_hotels)))
print(" - missing/failed images:", missing)

train_norm2 = (train_feats**2).sum(axis=1)

fallback_top5 = hotel_counts.head(5).index.astype(str).tolist()
if len(fallback_top5) < 5:
    pad = fallback_top5[0] if len(fallback_top5) > 0 else "0"
    fallback_top5 = fallback_top5 + [pad] * (5 - len(fallback_top5))


def predict_top5_for_test_image(test_path, k_nn=50):
    q = img_to_feat(test_path, size=thumb_size)
    if q is None or train_feats.shape[0] == 0:
        return fallback_top5[:5]
    q = q.astype(np.float32)
    q_norm2 = float((q**2).sum())

    dots = train_feats @ q
    d2 = train_norm2 + q_norm2 - 2.0 * dots

    k = min(k_nn, d2.shape[0])
    nn_idx = np.argpartition(d2, k - 1)[:k]
    nn_hotels = train_hotels[nn_idx]

    best_d2 = {}
    counts = {}
    for i in nn_idx:
        h = train_hotels[i]
        counts[h] = counts.get(h, 0) + 1
        di = float(d2[i])
        if (h not in best_d2) or (di < best_d2[h]):
            best_d2[h] = di

    ranked = sorted(counts.keys(), key=lambda h: (-counts[h], best_d2[h]))
    ranked5 = ranked[:5]

    if len(ranked5) < 5:
        for h in fallback_top5:
            if h not in ranked5:
                ranked5.append(h)
            if len(ranked5) == 5:
                break
    return ranked5[:5]


preds = []
for img_name in sub_df["image"].astype(str).values:
    test_path = os.path.join(test_img_dir, img_name)
    top5 = predict_top5_for_test_image(test_path, k_nn=50)
    preds.append(" ".join(top5))

submission = pd.DataFrame(
    {"image": sub_df["image"].astype(str).values, "hotel_id": preds}
)

assert list(submission.columns) == ["image", "hotel_id"]
assert len(submission) == len(sub_df)
assert submission["hotel_id"].str.split().map(len).eq(5).all()

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape {submission.shape}")
print("Fallback top-5 (used when needed):", fallback_top5)



## === cell 2
import pandas as pd

print(pd.read_csv("submission.csv").head())
print("Row count:", len(pd.read_csv("submission.csv")))
print("Example prediction:", pd.read_csv("submission.csv")["hotel_id"].iloc[0])
