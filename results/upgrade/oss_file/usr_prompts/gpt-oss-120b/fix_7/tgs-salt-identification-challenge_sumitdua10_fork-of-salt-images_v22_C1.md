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
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Target score

0.68244

# 6. Current score

0.0575

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the TensorFlow/Keras imports that cause a protobuf error, convert all loaded images to grayscale so they have identical shapes, and fix variable usage so the arrays are built correctly. These minimal changes ensure the pipeline runs end‑to‑end and writes a valid `submission.csv` while keeping the original zero‑mask prediction logic unchanged.'
- What this solution (achieved 0.0575) has done: 'I fixed the IndexError when a test image’s depth isn’t present by falling back to the mean depth, and replaced the all‑zero predictions with a simple intensity‑based threshold (pixel > 0.5) which gives a sensible mask and raises the score toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.5221) has done: 'I replace the naive per‑image intensity threshold with a simple data‑driven baseline: compute the average binary mask from the training set and use this same mask for every test image. This leverages the training labels (which were previously ignored) and should raise the mean‑average‑precision toward the target without altering the overall pipeline or model architecture.'
- What this solution (achieved 0.5221) has done: 'I replace the single global average mask with a small depth‑aware baseline: the training set is split into a few depth bins, an average mask is computed per bin (using a slightly lower binary threshold 0.4 to increase recall), and each test image receives the mask from the bin that matches its depth. This keeps the overall pipeline unchanged while adding only lightweight logic and is expected to move the mean‑average‑precision from 0.522 toward the target 0.682.'
- What this solution (achieved 0.0575) has done: 'Improved the baseline by increasing depth granularity (10 bins instead of 5) to capture more depth‑specific mask patterns, and added a lightweight per‑image intensity threshold (pixel > 0.5) that is merged with the depth‑bin mask via logical OR. This modest augmentation keeps the original architecture unchanged while providing richer predictions, expected to move the mean‑average‑precision closer to the target score.'

# 9. Code solution

## === cell 0
import os, random, zipfile
import numpy as np
import pandas as pd
from PIL import Image

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(1234)
random.seed(1234)

print("input folders:", os.listdir("../input"))

df_train = pd.read_csv("../input/train.csv")
print("\nTrain CSV shape:", df_train.shape)

df_depths = pd.read_csv("../input/depths.csv")
df_depths["z"] = df_depths["z"] / df_depths["z"].max()
print("\nDepths CSV shape:", df_depths.shape)

df_train = df_train.merge(df_depths, on="id", how="inner")
df_train = df_train[["id", "rle_mask", "z"]]
df_train.rename(columns={"z": "depth"}, inplace=True)




## === cell 1
def load_image(path):
    return np.array(Image.open(path).convert("L"))


train_images = []
train_masks = []
for idx in df_train["id"]:
    img = load_image(f"../input/train/images/{idx}.png")
    msk = load_image(f"../input/train/masks/{idx}.png")
    train_images.append(img)
    train_masks.append(msk)

train_images = np.stack(train_images, axis=0)
train_masks = np.stack(train_masks, axis=0)

train_images = train_images[..., np.newaxis] / 255.0  # (N, 101, 101, 1)
train_masks = (train_masks[..., np.newaxis] > 127).astype(
    np.float32
)  # (N, 101, 101, 1)

print("Sample train image shape:", train_images.shape[1:])
print("Sample train mask shape:", train_masks.shape[1:])



## === cell 2
num_train = train_images.shape[0]
depth_np = np.full((num_train, 50, 50, 1), 0.0, dtype=np.float32)
for i, d in enumerate(df_train["depth"].values):
    depth_np[i] = np.full((50, 50, 1), d, dtype=np.float32)

depth_np = depth_np / depth_np.max()  # normalise depth channel

print("Depth tensor shape:", depth_np.shape)



## === cell 3
df_sub = pd.read_csv("../input/sample_submission.csv")
df_test = df_sub[["id"]].copy()

test_images = []
for idx in df_test["id"]:
    img = load_image(f"../input/test/images/{idx}.png")
    test_images.append(img)

test_images = np.stack(test_images, axis=0)
test_images = test_images[..., np.newaxis] / 255.0  # (N_test, 101, 101, 1)

num_test = test_images.shape[0]
depth_test = np.full((num_test, 50, 50, 1), 0.0, dtype=np.float32)

mean_depth = df_depths["z"].mean()

for i, idx in enumerate(df_test["id"]):
    depth_vals = df_depths.loc[df_depths["id"] == idx, "z"].values
    if depth_vals.size == 0:
        d = mean_depth
    else:
        d = depth_vals[0]
    depth_test[i] = np.full((50, 50, 1), d, dtype=np.float32)

depth_test = depth_test / depth_test.max()

print("Test image shape:", test_images.shape)
print("Test depth shape:", depth_test.shape)



## === cell 4
num_bins = 10
depth_vals_train = df_train["depth"].values
bin_edges = np.quantile(depth_vals_train, np.linspace(0, 1, num_bins + 1))

threshold = 0.4
bin_masks = []
for b in range(num_bins):
    in_bin = (depth_vals_train >= bin_edges[b]) & (depth_vals_train <= bin_edges[b + 1])
    idxs = np.where(in_bin)[0]
    if idxs.size == 0:
        avg = (train_masks.mean(axis=0) > threshold).astype(np.float32)
    else:
        avg = (train_masks[idxs].mean(axis=0) > threshold).astype(np.float32)
    bin_masks.append(avg)

y_pred = np.empty((num_test, *bin_masks[0].shape), dtype=np.float32)

for i in range(num_test):
    d = depth_test[i, 0, 0, 0]  # scalar depth value
    bin_idx = np.digitize(d, bin_edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    y_pred[i] = bin_masks[bin_idx]

intensity_mask = (test_images[..., 0] > 0.5).astype(np.float32)  # shape (N,101,101)
y_pred = np.maximum(y_pred, intensity_mask[..., np.newaxis])  # logical OR



## === cell 5
Height = 101
Width = 101


def post_process(pred_batch):
    num_files = pred_batch.shape[0]
    main_list = []
    pic_size = Height * Width
    for i in range(num_files):
        pic = pred_batch[i].astype(int).reshape(-1)  # flatten
        s = ""
        length = 0
        start = 0
        for j in range(pic_size):
            if j == pic_size - 1:
                if pic[j] == 1 and length == 0:
                    s = f"{s} {j+1} 1".strip()
                elif pic[j] == 1 and length != 0:
                    s = f"{s} {start} {length+1}".strip()
            else:
                if pic[j] == 1:
                    if length == 0:
                        start = j + 1
                    length += 1
                else:
                    if length > 0:
                        s = f"{s} {start} {length}".strip()
                        length = 0
        main_list.append(s.strip())
    return main_list


print("Generating RLEs for submission...")
rle_masks = post_process(np.round(y_pred).astype(int))

submission = pd.DataFrame({"id": df_test["id"], "rle_mask": rle_masks})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(f"Submission file written to {out_path}, rows:", submission.shape[0])
