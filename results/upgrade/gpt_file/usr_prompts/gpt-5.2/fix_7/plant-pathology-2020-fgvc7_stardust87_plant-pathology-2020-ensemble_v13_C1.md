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

3.8

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.9699585629562538

# 6. Current score

0.44234

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code fails because it expects an external `/kaggle/input/submissions/submissions/` dataset that isn’t present, so `submissions_all` is empty and indexing `[0,1]` crashes. I make the input path robust by falling back to the competition’s `sample_submission.csv` and generate a valid baseline submission if no external submissions are found. This preserves your ensemble logic when files exist, but guarantees an end-to-end run and always writes `submission.csv` with the correct columns and row order. Since no score was yielded, the priority is producing a valid submission file; the fallback uses the sample’s 0.25 priors (a legitimate baseline).'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score indicates the fallback is essentially producing an uninformative (near-uniform) baseline; to move toward the 0.9699 target with minimal change, we replace that fallback with a simple data-driven prior computed from `train.csv` label frequencies, which usually beats uniform guessing on mean ROC AUC. This keeps your ensemble logic intact when external submissions exist, and only changes the “no external submissions found” branch. We also align rows strictly to `test.csv` order (instead of relying on sample submission ordering) to avoid any silent misalignment. Finally, we clip predictions into \[0,1\] to ensure valid probabilities without changing semantics.'
- What this solution (achieved 0.51815) has done: 'Your 0.5 score is consistent with producing constant probabilities per class (either uniform 0.25s or train-frequency priors tiled for every test row), which yields ROC AUC ≈ 0.5 because AUC is rank-based and constant predictions cannot rank positives above negatives. To move the score toward 0.9699 while keeping your “ensemble external submissions if present” core logic intact, the minimal legitimate improvement is to add a lightweight image-based inference fallback when no external submissions exist. This uses a pretrained ImageNet model (no training loop changes, just inference) to produce non-constant predictions, then calibrates them to match train label priors so outputs stay in a reasonable probability range. The rest (paths, submission columns/order, clipping, writing `submission.csv`) stays the same.'
- What this solution (achieved 0.54874) has done: 'Your score is stuck near 0.5 because the fallback still produces predictions that are effectively uninformative for ranking (near-constant or weakly varying), and ROC AUC is rank-based. With minimal change and without adding any training loop, we make the image-based fallback produce stronger, class-relevant per-image variation by using a pretrained ImageNet backbone that’s available via `torchvision` and mapping its features to 4 outputs using a deterministic, seeded linear head (same idea as your projection, but from a better feature space than raw logits). We also make the projection centered per-batch before the sigmoid to avoid saturation and improve ranking spread, while keeping your prior-mean calibration and all file/path/submission semantics intact. If torch/torchvision still aren’t available in your environment, it gracefully fall back exactly as before and still write a valid `submission.csv`.'
- What this solution (achieved 0.5462) has done: 'Your score is far below the target, so the safest way to move it upward without changing your overall “external ensemble else fallback” logic is to strengthen the fallback so it produces meaningful per-image ranking (AUC is rank-based). I keep your pretrained-backbone + deterministic projection idea, but make it more discriminative by (1) using a stronger ImageNet backbone (ResNet50) and (2) extracting stable pooled features via `create_feature_extractor` instead of manually averaging feature maps. I also fix a subtle batching bug: when an image file is missing mid-batch, your code appends priors immediately but still flushes a smaller tensor later, which can misalign predictions vs. image_ids; we batch only existing images and then place predictions back into the correct indices. These are minimal, contained changes limited to the fallback branch, preserving your ensemble path and submission semantics.'
- What this solution (achieved 0.44234) has done: 'Your current score (0.5462) is far below the target (0.9699), and because ROC AUC is rank-based, the most likely issue is that the fallback predictions still have weak per-image ranking signal. I keep your core logic (external-submission ensemble else pretrained-backbone deterministic projection + prior calibration) but strengthen the fallback’s discriminative power by (1) using a better backbone (EfficientNet-B0) and (2) extracting the actual classifier “pre-logits” features (not avgpool from ResNet) to get more informative embeddings. I also add test-time augmentation (simple horizontal flip averaging) to stabilize rankings without any training changes, and keep your strict test.csv ordering and output schema unchanged. These are contained to the fallback branch and should move AUC upward toward the target while still finishing quickly.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

COMP_DATA_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",  # contains train.csv/test.csv/sample_submission.csv directly in this environment
    "/kaggle/data",
]
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _find_first_existing_file(filename: str):
    for d in COMP_DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


SAMPLE_SUB_PATH = _find_first_existing_file("sample_submission.csv")
TEST_CSV_PATH = _find_first_existing_file("test.csv")
TRAIN_CSV_PATH = _find_first_existing_file("train.csv")

print(
    "SUBMISSIONS_PATH:", SUBMISSIONS_PATH, "exists:", os.path.exists(SUBMISSIONS_PATH)
)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)

if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input/data locations. "
        f"Searched: {COMP_DATA_DIR_CANDIDATES}"
    )
if TEST_CSV_PATH is None:
    raise FileNotFoundError(
        "Could not locate test.csv in expected Kaggle input/data locations. "
        f"Searched: {COMP_DATA_DIR_CANDIDATES}"
    )



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()

print("Found submission files:", len(submissions_all))
for p in submissions_all[:20]:
    print(" -", p)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted sum ensemble of submission files.

    Minimal bug-fix: validates indices/weights and raises a clear error if requested
    indices do not exist.
    """
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length. Got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError(
            "submissions_all is empty; cannot ensemble external submissions."
        )

    max_idx = max(sub_idx) if len(sub_idx) else -1
    if max_idx >= len(submissions_all):
        raise IndexError(
            f"Requested submission index {max_idx}, but only {len(submissions_all)} files were found."
        )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        w = weights[i]
        print(f"I'm taking submission {path} with weight {w}")
        submission = pd.read_csv(path)
        missing = [c for c in TARGET_COLS if c not in submission.columns]
        if missing:
            raise ValueError(f"Submission file {path} is missing columns: {missing}")
        submission = submission.loc[:, TARGET_COLS].values
        submission_with_weight.append(submission * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(
    submission_avg, base_submission_path, out_path="submission.csv"
):
    """
    Writes a valid submission.csv using the row order/image_id from test.csv (authoritative).
    """
    test_df = pd.read_csv(TEST_CSV_PATH)
    if "image_id" not in test_df.columns:
        raise ValueError(f"test.csv at {TEST_CSV_PATH} has no image_id column.")

    base_df = pd.read_csv(base_submission_path)
    for c in TARGET_COLS:
        if c not in base_df.columns:
            base_df[c] = 0.0

    sub_df = test_df[["image_id"]].copy()
    for c in TARGET_COLS:
        sub_df[c] = 0.0

    submission_avg = np.asarray(submission_avg, dtype=np.float64)
    submission_avg = np.clip(submission_avg, 0.0, 1.0)

    if submission_avg.shape != (len(sub_df), len(TARGET_COLS)):
        raise ValueError(
            f"Pred array shape {submission_avg.shape} does not match expected "
            f"({len(sub_df)}, {len(TARGET_COLS)})."
        )

    sub_df.loc[:, TARGET_COLS] = submission_avg
    sub_df.to_csv(out_path, index=False)
    return out_path




## === cell 5
def _find_images_dir():
    for root in [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "/kaggle/input",
        "/kaggle/data",
    ]:
        p = os.path.join(root, "images")
        if os.path.isdir(p):
            return p
    return None


def _infer_fallback_from_images(test_image_ids, priors):
    """
    Minimal, legitimate improvement over constant priors, while keeping the same core idea:
    pretrained backbone + deterministic projection + prior-mean calibration.

    Change (score-improving toward target):
    - Use EfficientNet-B0 ImageNet backbone and extract "pre-logits" features (more informative embedding
      for image ranking than a generic avgpool from ResNet in this setup).
    - Add a tiny test-time augmentation (horizontal flip averaging) to stabilize per-image ranking signal.
    - Keep strict prediction alignment by writing predictions back to the correct indices.
    """
    try:
        import torch
        import torchvision
        from PIL import Image
    except Exception as e:
        print("PyTorch/torchvision/PIL not available for image fallback:", repr(e))
        return None

    images_dir = _find_images_dir()
    if images_dir is None:
        print(
            "Could not locate images/ directory; cannot run image inference fallback."
        )
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    backbone = torchvision.models.efficientnet_b0(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    feat_dim = 1280

    g = torch.Generator(device="cpu").manual_seed(2020)
    W = torch.randn(feat_dim, 4, generator=g, dtype=torch.float32) / np.sqrt(
        float(feat_dim)
    )
    b = torch.randn(4, generator=g, dtype=torch.float32) * 0.01
    W = W.to(device)
    b = b.to(device)

    n = len(test_image_ids)
    pred = np.tile(np.asarray(priors, dtype=np.float64).reshape(1, -1), (n, 1))

    bs = 16
    batch_t = []
    batch_idx = []

    def _extract_feat(xb: "torch.Tensor") -> "torch.Tensor":
        feat_map = backbone.features(xb)
        pooled = backbone.avgpool(feat_map)
        feat = torch.flatten(pooled, 1)
        return feat

    for i, image_id in enumerate(test_image_ids):
        img_path = os.path.join(images_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        x = preprocess(img)
        batch_t.append(x)
        batch_idx.append(i)

        flush = (len(batch_t) == bs) or (i == n - 1)
        if flush and len(batch_t) > 0:
            xb = torch.stack(batch_t, dim=0).to(device)

            feat1 = _extract_feat(xb)
            xb_flip = torch.flip(xb, dims=[3])
            feat2 = _extract_feat(xb_flip)
            feat = 0.5 * (feat1 + feat2)

            s = feat @ W + b  # (B,4)
            s = s - s.mean(dim=1, keepdim=True)
            p = torch.sigmoid(s).detach().cpu().numpy()

            for j, idx in enumerate(batch_idx):
                pred[idx, :] = p[j, :]

            batch_t = []
            batch_idx = []

    eps = 1e-6
    col_mean = pred.mean(axis=0)
    scale = (np.asarray(priors, dtype=np.float64) + eps) / (col_mean + eps)
    pred = pred * scale.reshape(1, -1)

    return np.clip(pred, 0.0, 1.0)




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.18, 0.82])
    base_path = submissions_all[0]
elif len(submissions_all) == 1:
    submission_avg = ensemble(submissions_all, [0], [1.0])
    base_path = submissions_all[0]
else:
    print(
        "No external submissions found; using image-based inference fallback (pretrained) + train priors calibration."
    )
    test_df = pd.read_csv(TEST_CSV_PATH)

    if TRAIN_CSV_PATH is not None and os.path.exists(TRAIN_CSV_PATH):
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        missing = [c for c in (["image_id"] + TARGET_COLS) if c not in train_df.columns]
        if missing:
            raise ValueError(
                f"train.csv at {TRAIN_CSV_PATH} missing columns: {missing}"
            )
        priors = train_df[TARGET_COLS].mean(axis=0).astype(float).values
    else:
        base_df = pd.read_csv(SAMPLE_SUB_PATH)
        priors = base_df[TARGET_COLS].mean(axis=0).astype(float).values

    submission_avg = _infer_fallback_from_images(test_df["image_id"].tolist(), priors)

    if submission_avg is None:
        print(
            "Falling back to constant train-priors baseline (will likely score ~0.5 AUC)."
        )
        submission_avg = np.tile(priors.reshape(1, -1), (len(test_df), 1))

    base_path = SAMPLE_SUB_PATH  # schema for writer

out_file = make_submission_file(submission_avg, base_path, out_path="submission.csv")
print("Wrote:", out_file)

sub = pd.read_csv(out_file)
assert list(sub.columns) == ["image_id"] + TARGET_COLS
assert len(sub) == len(pd.read_csv(TEST_CSV_PATH))
print(sub.head())
print("Class means in submission:", sub[TARGET_COLS].mean().to_dict())
