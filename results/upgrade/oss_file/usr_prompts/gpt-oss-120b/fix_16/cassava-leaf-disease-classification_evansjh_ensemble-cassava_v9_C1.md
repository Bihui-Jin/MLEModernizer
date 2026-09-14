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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8890903596252644

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'We remove the failing TensorFlow imports and model‑loading steps, compute the most frequent label from the training data, and assign that label to every test image. This fixes the protobuf error, the missing model files, and ensures a valid `submission.csv` is written, allowing the notebook to run end‑to‑end.'
- What this solution (achieved 0.25561) has done: 'I replace the constant‑label baseline with a lightweight colour‑based classifier: compute the average RGB colour for each disease label on a random subset of the training images, then assign each test image the label whose average colour is closest (Euclidean distance). This keeps the original simple pipeline, adds only the Pillow library (available in the environment), and is expected to raise accuracy markedly toward the target without heavy modelling.'
- What this solution (achieved 0.12967) has done: 'I keep the overall colour‑based approach but make it more informative and use a larger, representative training subset. Instead of only the mean RGB per class, I also compute per‑class variance and classify each test image by a variance‑scaled (Mahalanobis‑like) distance to the class colour statistics. This small statistical upgrade usually raises accuracy noticeably while preserving the original pipeline.'
- What this solution (achieved 0.14798) has done: 'I add computation of per‑class RGB standard deviations and use both mean + std vectors (instead of variance‑scaled Mahalanobis distance) when matching test images, which is a minimal statistical upgrade that usually raises colour‑based classification accuracy. This keeps the overall pipeline unchanged while improving the distance metric, moving the score toward the target.'
- What this solution (achieved 0.19357) has done: 'I enhance the colour‑based baseline by adding a cheap visual feature: a down‑scaled RGB image prototype for each disease class. During training we accumulate the average 32×32 RGB image per label, and at inference we compare each test image (also resized to 32×32) to these prototypes using Euclidean distance. This keeps the original simple pipeline, adds only a few NumPy operations, and is expected to raise the validation accuracy noticeably, moving the score toward the target while preserving the overall logic.'
- What this solution (achieved 0.17862) has done: 'I use the entire training set instead of a random 10 k subset to compute more accurate colour means, variances, and image prototypes, and I combine the colour‑based and prototype‑based distances with a modest weighting (α ≈ 0.5) so that both cues influence the prediction. These small tweaks keep the original pipeline intact while giving the model richer statistics, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.18311) has done: 'The changes add parallel image loading and feature extraction for the training set (the main bottleneck) using a thread pool, which speeds up the I/O‑bound Pillow operations without altering any algorithmic logic. A small helper `_process_train_item` isolates the image work, and the results are aggregated exactly as before, preserving deterministic sums. The rest of the cells keep their original behavior.'
- What this solution (achieved 0.18386) has done: 'I increase the prototype image resolution to capture more visual detail and replace the simple colour‑difference metric with a Mahalanobis‑style distance that normalises by per‑class colour standard deviation. Both changes keep the original colour + prototype logic while giving the classifier a stronger, more discriminative signal, which should raise validation accuracy and move the Kaggle score closer to the target.'
- What this solution (achieved 0.18012) has done: 'I add a cheap texture feature (grayscale variance) to the existing colour‑plus‑prototype classifier, compute per‑class statistics for it, and combine the three distances with small weights that are tuned on a tiny validation split. This keeps the overall simple pipeline while giving the model more discriminative information, which should raise the validation accuracy and move the Kaggle score closer to the target.'
- What this solution (achieved 0.17601) has done: 'I added the missing imports (`os`, `pandas`, `numpy`, `concurrent.futures`, and `PIL.Image`) and made a tiny expansion of the hyper‑parameter search (adding a couple of non‑zero α/β values) to give the colour + prototype classifier a chance to improve over the simple majority‑class baseline while keeping the original logic intact. The script now runs end‑to‑end and writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'I expand the hyper‑parameter search to include a pure‑prototype option (α = 1.0, β = 0) and a larger validation split so the chosen weights better reflect the data. This small change keeps the overall pipeline unchanged while giving the model a chance to rely more on the discriminative prototype images, which should raise the validation accuracy and move the Kaggle score toward the target.'
- What this solution (achieved 0.05531) has done: 'The changes pre‑compute all validation image features in parallel once, then evaluate every α/β pair on those cached features instead of re‑reading the images for each combination. This removes the costly repeated disk I/O and per‑image processing inside the nested loops, turning the validation step from a sequential × 5 load to a single parallel load plus lightweight NumPy calculations, while keeping all model‑building logic unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import concurrent.futures
from PIL import Image

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

train_df = pd.read_csv(train_csv_path)

most_common_label = train_df["label"].mode().iloc[0]

label_sums = {}
label_sq_sums = {}
label_counts = {}
proto_sums = {}
proto_counts = {}
edge_sums = {}
edge_sq_sums = {}
proto_size = (64, 64)  # down‑scaled size for prototype images (height, width)


def _process_train_item(item):
    """Load one training image and return its colour, prototype and texture stats."""
    lbl, img_path = item
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            mean_rgb = arr.mean(axis=(0, 1))
            sq_rgb = (arr**2).mean(axis=(0, 1))
            gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
            var_gray = gray.var()
            img_small = img.resize(proto_size, Image.BILINEAR)
            img_small_arr = np.asarray(img_small, dtype=np.float32) / 255.0
            return (lbl, mean_rgb, sq_rgb, img_small_arr, var_gray, True)
    except Exception:
        return (lbl, None, None, None, None, False)


train_items = [
    (int(row["label"]), os.path.join(train_image_dir, row["image_id"]))
    for _, row in train_df.iterrows()
]

with concurrent.futures.ThreadPoolExecutor() as executor:
    for lbl, mean_rgb, sq_rgb, img_small_arr, var_gray, ok in executor.map(
        _process_train_item, train_items, chunksize=64
    ):
        if not ok:
            continue

        if lbl not in label_sums:
            label_sums[lbl] = np.zeros(3, dtype=np.float32)
            label_sq_sums[lbl] = np.zeros(3, dtype=np.float32)
            label_counts[lbl] = 0
        label_sums[lbl] += mean_rgb
        label_sq_sums[lbl] += sq_rgb
        label_counts[lbl] += 1

        if lbl not in edge_sums:
            edge_sums[lbl] = 0.0
            edge_sq_sums[lbl] = 0.0
        edge_sums[lbl] += var_gray
        edge_sq_sums[lbl] += var_gray**2

        if lbl not in proto_sums:
            proto_sums[lbl] = np.zeros(
                (proto_size[1], proto_size[0], 3), dtype=np.float32
            )
            proto_counts[lbl] = 0
        proto_sums[lbl] += img_small_arr
        proto_counts[lbl] += 1

label_means = {}
label_stds = {}
for lbl in label_sums:
    if label_counts[lbl] > 0:
        mean = label_sums[lbl] / label_counts[lbl]
        sq_mean = label_sq_sums[lbl] / label_counts[lbl]
        var = np.maximum(sq_mean - mean**2, 1e-6)
        std = np.sqrt(var)
        label_means[lbl] = mean
        label_stds[lbl] = std

label_edge_means = {}
label_edge_stds = {}
for lbl in edge_sums:
    cnt = label_counts.get(lbl, 0)
    if cnt > 0:
        mean = edge_sums[lbl] / cnt
        sq_mean = edge_sq_sums[lbl] / cnt
        var = max(sq_mean - mean**2, 1e-9)
        std = np.sqrt(var)
        label_edge_means[lbl] = mean
        label_edge_stds[lbl] = std

label_prototypes = {}
for lbl in proto_sums:
    if proto_counts[lbl] > 0:
        label_prototypes[lbl] = proto_sums[lbl] / proto_counts[lbl]  # (H, W, 3)

use_fallback = (len(label_means) == 0) or (len(label_prototypes) == 0)


def _predict_one(mean_rgb, std_rgb, img_small_arr, edge_var, alpha, beta):
    """
    Predict label using a weighted combination of:
      - colour Mahalanobis distance (weight 1‑alpha‑beta)
      - prototype Euclidean distance (weight alpha)
      - texture (grayscale variance) Mahalanobis distance (weight beta)
    """
    if use_fallback:
        return int(most_common_label)

    colour_distances = {}
    for lbl in label_means:
        norm_diff = (mean_rgb - label_means[lbl]) / (label_stds[lbl] + 1e-3)
        colour_distances[lbl] = np.linalg.norm(norm_diff)

    proto_distances = {}
    if alpha > 0:
        test_vec = img_small_arr.flatten()
        for lbl, proto_img in label_prototypes.items():
            proto_vec = proto_img.flatten()
            proto_distances[lbl] = np.linalg.norm(test_vec - proto_vec)

    texture_distances = {}
    if beta > 0:
        for lbl in label_edge_means:
            norm = (edge_var - label_edge_means[lbl]) / (label_edge_stds[lbl] + 1e-6)
            texture_distances[lbl] = abs(norm)  # 1‑D Mahalanobis

    combined = {}
    for lbl in colour_distances:
        c = (1 - alpha - beta) * colour_distances[lbl]
        p = alpha * proto_distances.get(lbl, np.inf)
        t = beta * texture_distances.get(lbl, np.inf)
        combined[lbl] = c + p + t

    return int(min(combined, key=combined.get))




## === cell 1
val_frac = 0.20
val_df = train_df.sample(frac=val_frac, random_state=42).reset_index(drop=True)

candidate_alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
candidate_betas = [0.0]  # texture gave little benefit in earlier runs


def _process_val_item(row):
    """Load a validation image and extract the same features used by the predictor."""
    img_path = os.path.join(train_image_dir, row["image_id"])
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_arr = np.asarray(img, dtype=np.float32) / 255.0
            mean_rgb = img_arr.mean(axis=(0, 1))
            img_small = img.resize(proto_size, Image.BILINEAR)
            img_small_arr = np.asarray(img_small, dtype=np.float32) / 255.0
            gray = (
                0.2989 * img_arr[..., 0]
                + 0.5870 * img_arr[..., 1]
                + 0.1140 * img_arr[..., 2]
            )
            edge_var = gray.var()
            label = int(row["label"])
            return (mean_rgb, img_small_arr, edge_var, label, True)
    except Exception:
        return (None, None, None, int(row["label"]), False)


with concurrent.futures.ThreadPoolExecutor() as executor:
    val_features = list(
        executor.map(
            _process_val_item, [row for _, row in val_df.iterrows()], chunksize=64
        )
    )

best_alpha = 0.0
best_beta = 0.0
best_acc = -1.0

for a in candidate_alphas:
    for b in candidate_betas:
        if a + b > 1.0:
            continue
        correct = 0
        total = 0
        for mean_rgb, img_small_arr, edge_var, true_label, ok in val_features:
            if not ok:
                pred = int(most_common_label)
            else:
                pred = _predict_one(mean_rgb, None, img_small_arr, edge_var, a, b)
            if pred == true_label:
                correct += 1
            total += 1
        acc = correct / total if total > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            best_alpha = a
            best_beta = b

alpha = best_alpha
beta = best_beta
print(
    f"Selected alpha = {alpha:.2f}, beta = {beta:.2f} (validation accuracy ≈ {best_acc:.4f})"
)




## === cell 2
test_images = [
    f
    for f in os.listdir(test_image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]


def _process_test_item(img_name):
    img_path = os.path.join(test_image_dir, img_name)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_arr = np.asarray(img, dtype=np.float32) / 255.0
            mean_rgb = img_arr.mean(axis=(0, 1))
            img_small = img.resize(proto_size, Image.BILINEAR)
            img_small_arr = np.asarray(img_small, dtype=np.float32) / 255.0
            gray = (
                0.2989 * img_arr[..., 0]
                + 0.5870 * img_arr[..., 1]
                + 0.1140 * img_arr[..., 2]
            )
            edge_var = gray.var()
        pred_label = _predict_one(mean_rgb, None, img_small_arr, edge_var, alpha, beta)
    except Exception:
        pred_label = int(most_common_label)
    return {"image_id": img_name, "label": pred_label}


with concurrent.futures.ThreadPoolExecutor() as executor:
    image_predictions = list(executor.map(_process_test_item, test_images))

submission_df = pd.DataFrame(image_predictions)




## === cell 3
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission_df)} rows.")
