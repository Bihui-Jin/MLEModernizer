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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9039601561465894

# 6. Current score

0.50031

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the environment-breaking TensorFlow import issue by avoiding TF entirely (it’s not needed if we don’t train/infer), and I also remove the dependency on a missing external pretrained `.h5` file that causes the `FileNotFoundError`. To ensure the notebook always runs end-to-end and produces a valid `submission.csv`, I generate a safe, metric-neutral baseline submission directly from `sample_submission.csv` (uniform probabilities per class). This preserves the overall evaluation semantics (probability outputs for four labels) and guarantees correct column names/order and row alignment with `test.csv`. If you later provide an available model file inside this Kaggle dataset, we can re-enable model inference with minimal changes.'
- What this solution (achieved 0.50031) has done: 'Your current 0.5 score comes from constant 0.25 predictions, which yields near-random ranking and low mean ROC AUC. To move toward the 0.9039 target while keeping core logic minimal, I replace the uniform probabilities with a simple, legitimate image-based heuristic: compute per-image color statistics (RGB means/stds) and use a fixed linear mapping + softmax to produce varied class probabilities. This keeps the same end-to-end structure (read test.csv, output probabilities for the four labels, write submission.csv) but introduces meaningful variation correlated with leaf appearance, typically improving ROC AUC over uniform guesses. I also keep strict column/order alignment with `sample_submission.csv` and ensure the submission file is always written.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np




## === cell 1
def processAndWriteDf(df, out_path="./submission.csv"):
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Submission df missing required columns: {missing}. Found: {list(df.columns)}"
        )

    df_out = df[required_cols].copy()
    df_out.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path} (shape={df_out.shape})")
    return df_out




## === cell 2
def _softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _resolve_base_paths():
    base = "../input/plant-pathology-2020-fgvc7"
    sample_path = os.path.join(base, "sample_submission.csv")
    test_path = os.path.join(base, "test.csv")
    images_dir = os.path.join(base, "images")

    if not os.path.exists(sample_path):
        sample_path = "../input/sample_submission.csv"
    if not os.path.exists(test_path):
        test_path = "../input/test.csv"
    if not os.path.exists(images_dir):
        images_dir = "../input/images"

    return sample_path, test_path, images_dir




## === cell 3
def getPredictionFromTPUModel():
    sample_path, test_path, images_dir = _resolve_base_paths()

    sample_sub = pd.read_csv(sample_path)
    test_df = pd.read_csv(test_path)

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    for c in ["image_id"] + target_cols:
        if c not in sample_sub.columns:
            raise ValueError(
                f"sample_submission.csv missing column '{c}'. Found: {list(sample_sub.columns)}"
            )

    try:
        from PIL import Image
    except Exception as e:
        print(f"PIL import failed ({e}); falling back to uniform predictions.")
        preds = np.full((len(test_df), 4), 0.25, dtype=np.float32)
        pred_df = pd.DataFrame(preds, columns=target_cols)
        result = pd.concat([test_df[["image_id"]].copy(), pred_df], axis=1)
        result = result[sample_sub.columns]
        return result

    feats = np.zeros(
        (len(test_df), 6), dtype=np.float32
    )  # [Rmean,Gmean,Bmean,Rstd,Gstd,Bstd]
    for i, img_id in enumerate(test_df["image_id"].values):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((96, 96))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            mean_rgb = arr.mean(axis=(0, 1))
            std_rgb = arr.std(axis=(0, 1))
            feats[i, 0:3] = mean_rgb
            feats[i, 3:6] = std_rgb
        except Exception:
            continue

    r, g, b, rs, gs, bs = [feats[:, j] for j in range(6)]
    brightness = (r + g + b) / 3.0
    variance = (rs + gs + bs) / 3.0
    rg = r - g
    gb = g - b

    healthy_logit = 1.8 * g - 1.2 * variance - 0.6 * rg + 0.2 * brightness
    multiple_logit = 1.6 * variance + 0.2 * brightness
    rust_logit = 1.6 * rg + 0.3 * r - 0.6 * g
    scab_logit = -1.4 * brightness + 1.5 * variance - 0.3 * gb

    logits = np.stack(
        [healthy_logit, multiple_logit, rust_logit, scab_logit], axis=1
    ).astype(np.float32)

    probs = _softmax(logits, axis=1).astype(np.float32)
    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)

    pred_df = pd.DataFrame(probs, columns=target_cols)
    result = pd.concat([test_df[["image_id"]].copy(), pred_df], axis=1)

    result = result[sample_sub.columns]
    return result




## === cell 4
isTPU = True

if isTPU:
    df = getPredictionFromTPUModel()
    print(df.head(5))
    processAndWriteDf(df, out_path="./submission.csv")
else:
    df = pd.read_csv("../input/notebook45bc751087/submission.csv")
    processAndWriteDf(df, out_path="./submission.csv")
