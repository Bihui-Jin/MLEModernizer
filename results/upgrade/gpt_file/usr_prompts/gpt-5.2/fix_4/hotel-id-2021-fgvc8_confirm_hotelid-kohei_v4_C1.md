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

0.7110676096657378

# 6. Current score

0.0024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'Your notebook currently can’t yield a score because it relies on external wheels (`pekolib`, `inplace_abn`) that are not available in this environment, so it never produces a guaranteed-valid `submission.csv`. I replace that dependency with a minimal, self-contained baseline that always runs end-to-end using only the provided CSVs, and writes a correctly formatted submission file. Since MAP@5 requires 5 space-delimited hotel IDs, this baseline predict the 5 most frequent `hotel_id`s from `train.csv` for every test image, which is a legitimate and stable way to obtain a non-zero score and move toward the target. Paths are kept on `/kaggle/input/...` and the output be `submission.csv` in the working directory.'
- What this solution (achieved 0.0024) has done: 'The timeout is dominated by per-test-image Python overhead and repeated image decoding/resizing, plus unnecessary merges/row lookups when building the gallery. I keep the exact same feature definition (32×32 grayscale flatten) and the same kNN + vote/rerank logic, but vectorize the distance computation in batches and avoid recomputing/copying arrays in the inner loop. I also remove the unused/expensive merge and per-image DataFrame `.loc` lookup by using the already-present `chain` column in `train_df_small`. Finally, I speed up image I/O by using faster PIL resize settings and preallocations/caching while preserving identical pixel values and prediction semantics.'

# 9. Code solution

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
            try:
                im.draft("L", size)
            except Exception:
                pass
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

missing = 0
for img_name, chain, hid in zip(
    train_df_small["image"].astype(str).values,
    train_df_small["chain"].values,
    train_df_small["hotel_id"].values,
):
    if pd.isna(chain):
        missing += 1
        continue
    chain_str = str(int(chain))
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

train_norm2 = (train_feats**2).sum(axis=1).astype(np.float32, copy=False)

fallback_top5 = hotel_counts.head(5).index.astype(str).tolist()
if len(fallback_top5) < 5:
    pad = fallback_top5[0] if len(fallback_top5) > 0 else "0"
    fallback_top5 = fallback_top5 + [pad] * (5 - len(fallback_top5))


def predict_top5_batch(test_paths, k_nn=50, batch_size=256):
    n_gallery = train_feats.shape[0]
    if n_gallery == 0:
        return [" ".join(fallback_top5[:5])] * len(test_paths)

    feat_dim = train_feats.shape[1]
    out = [None] * len(test_paths)

    X = train_feats
    Xn2 = train_norm2
    H = train_hotels

    for b0 in range(0, len(test_paths), batch_size):
        b1 = min(b0 + batch_size, len(test_paths))
        paths = test_paths[b0:b1]

        Q = np.empty((b1 - b0, feat_dim), dtype=np.float32)
        valid = np.ones((b1 - b0,), dtype=bool)
        for i, p in enumerate(paths):
            q = img_to_feat(p, size=thumb_size)
            if q is None:
                valid[i] = False
            else:
                Q[i] = q

        for i in np.where(~valid)[0]:
            out[b0 + i] = " ".join(fallback_top5[:5])

        if not valid.any():
            continue

        Qv = Q[valid]
        qn2 = (Qv * Qv).sum(axis=1, dtype=np.float32)  # (bv,)
        dots = X @ Qv.T  # (n_gallery, bv)
        d2 = (Xn2[:, None] + qn2[None, :] - 2.0 * dots).astype(np.float32, copy=False)

        k = min(k_nn, n_gallery)
        nn_idx = np.argpartition(d2, kth=k - 1, axis=0)[:k, :]  # (k, bv)

        valid_pos = np.flatnonzero(valid)
        for j in range(nn_idx.shape[1]):
            idxs = nn_idx[:, j]
            best_d2 = {}
            counts = {}
            d2_col = d2[:, j]
            for ii in idxs:
                h = H[ii]
                counts[h] = counts.get(h, 0) + 1
                di = float(d2_col[ii])
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
            out[b0 + valid_pos[j]] = " ".join(ranked5[:5])

    return out


test_paths = [
    os.path.join(test_img_dir, img_name)
    for img_name in sub_df["image"].astype(str).values
]
preds = predict_top5_batch(test_paths, k_nn=50, batch_size=256)

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
