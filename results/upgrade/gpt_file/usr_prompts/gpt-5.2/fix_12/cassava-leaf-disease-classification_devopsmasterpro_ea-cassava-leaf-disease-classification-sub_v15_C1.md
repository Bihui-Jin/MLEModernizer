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

3.14

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

0.7417648836506497

# 6. Current score

0.14275

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Your notebook fails early due to an incompatibility between the provided TensorFlow import and the environment’s protobuf version, and it also tries to load a model file that does not exist in the given input paths. To make this run end-to-end and still produce a valid `submission.csv`, I remove the TensorFlow/Keras dependency entirely and replace the missing-model inference with a deterministic, label-frequency baseline learned from `train.csv` (score be stable and non-zero, and the pipeline always write a correctly formatted CSV). I also fix path usage and ensure the submission rows align exactly with `sample_submission.csv`’s `image_id` ordering. The core “predict labels for each test image_id and write submission” semantics remain the same, but without external model files.'
- What this solution (achieved 0.20254) has done: 'Your current solution is a pure majority-class baseline, which caps accuracy near the train label prevalence (~0.61). To move toward the target score, the smallest legitimate step up (without changing the “train a rule from train.csv → predict test → write submission.csv” core semantics) is to replace the constant predictor with a very lightweight image-based nearest-centroid classifier computed from raw pixels. This keeps dependencies minimal (NumPy/Pandas + standard library), reads images directly from `train_images/` and `test_images/`, uses a small fixed-size grayscale downsample as features, and predicts by closest class mean—typically a meaningful bump over majority while still simple and deterministic. The submission format and ordering are preserved by copying `sample_submission.csv` and filling `label` in the same row order.'
- What this solution (achieved 0.22422) has done: 'Your current “feature extractor” is effectively random (CRC-seeded byte sampling), so it can’t learn meaningful visual centroids and ends up near-chance accuracy. To move your score upward toward the 0.74 target without changing the overall approach (centroid-per-class + nearest-centroid inference + same training loop structure), I replace the pseudo-feature with a real, dependency-light image decode + downsample grayscale feature using `PIL` (available in Kaggle’s base environment). I also L2-normalize features and centroids to make distance comparisons more stable, which typically improves accuracy for nearest-centroid with simple pixel features. Everything else (paths, centroid building, prediction loop, submission formatting) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.31988) has done: 'Your current nearest-centroid pipeline is sound, but it’s underpowered because grayscale-only features throw away strong class cues (leaf color patterns) and the centroid is built from a fixed cap without any balancing. To move accuracy up toward the 0.74 target while keeping the same core logic (extract fixed-size features → compute per-class centroids → nearest-centroid inference), I (1) switch the feature extractor from grayscale to small RGB (still PIL + resize) and normalize per-channel, and (2) make the centroid-building cap per class proportional to class frequency (so minority classes aren’t overly noisy while majority class isn’t overly clipped). These are minimal changes that keep the same training/inference structure and should improve separability without changing the overall approach or runtime drastically. The submission writing and ordering stays exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.31988) has done: 'Your current nearest-centroid approach is likely being held back by the distance metric on L2-normalized features: with unit vectors, squared Euclidean is equivalent to cosine distance, but computing it via `(a-b)^2` can be less stable and wastes signal that’s naturally “angular” after normalization. I keep the exact same pipeline (RGB downsample → normalize → per-class centroid → nearest centroid) and only change the inference scoring to use a direct dot-product similarity (cosine) against already-normalized centroids/features. I also normalize the feature at inference time with the same float64 pathway used for centroid accumulation to reduce small numeric drift. These are minimal, semantics-preserving changes that usually improve accuracy for this kind of normalized centroid classifier while keeping runtime within limits and still writing a valid `submission.csv`.'
- What this solution (achieved 0.29372) has done: 'I keep your nearest-centroid pipeline exactly the same (RGB downsample → standardized+L2-normalized feature → per-class centroid → cosine similarity), but make two minimal changes that typically increase accuracy: (1) compute centroids using a stratified, deterministic uniform sample across each class (instead of taking the first N after sorting), which reduces bias from any ordering artifacts and gives a more representative centroid; and (2) slightly increase the feature resolution from 24×24 to 32×32 while keeping the same extraction logic, which often adds enough spatial detail to move accuracy upward without changing the approach. I also ensure the fallback label is the true global majority label from `train.csv` (not the class with the most centroid samples), which is a tiny but correctness-aligned fix. The script still runs end-to-end and writes a valid `submission.csv` aligned to `sample_submission.csv` ordering.'
- What this solution (achieved 0.29858) has done: 'Your current gap to the target (0.29372 → 0.74176) is large, so we need a real accuracy lift while keeping the same nearest-centroid core pipeline. The smallest high-impact change is to make the centroid estimate more robust without changing the model: compute centroids from multiple deterministic augmentations (original + horizontal flip) and classify test images by averaging cosine similarities across the same augmentations. This preserves the exact “extract feature → centroid per class → cosine similarity → argmax” semantics, but typically improves accuracy for cassava leaves because flip invariance is helpful. I also keep runtime bounded by not increasing resolution further and by reusing the same feature extractor.'
- What this solution (achieved 0.29783) has done: 'Your current nearest-centroid + cosine + (orig, hflip) TTA pipeline is intact but is likely underperforming because the features are over-standardized (global mean/std across all RGB pixels) and the centroid estimate is noisy. I make two minimal, semantics-preserving tweaks that usually lift accuracy for simple centroid classifiers: (1) switch to per-channel standardization (keeps color information better than flatten-wide whitening), and (2) add one more deterministic TTA view (center-crop + resize) and average similarities across the views (same centroid/argmax logic, just a slightly more robust feature). These are small changes directly targeted at improving separability while keeping runtime reasonable and still writing a valid `submission.csv` in the required order.'
- What this solution (achieved 0.26046) has done: 'I keep your nearest-centroid + cosine + deterministic TTA core pipeline unchanged, but make two small, score-relevant adjustments aimed at improving separability without changing the overall approach. First, I switch the resize kernel to LANCZOS (still deterministic) to preserve more fine texture than bilinear at 32×32, which often helps simple pixel-feature classifiers. Second, I add a fourth deterministic view (vertical flip) alongside your existing (orig, hflip, center-crop) and average similarities across views exactly as you already do, which usually gives a modest robustness bump on leaf datasets while keeping runtime reasonable. The submission alignment/format and all paths remain the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.14275) has done: 'Your current nearest-centroid + cosine + TTA pipeline is conceptually fine, but it’s likely underperforming because the test-time features are being *re-standardized per image*, which removes global color/brightness cues that are actually discriminative for cassava classes and also makes train/test normalization inconsistent. I keep the exact same core logic (RGB downsample features → per-class centroid → cosine similarity → average over the same TTA views → argmax) and only change normalization to use *global per-channel mean/std computed from the same training subset you already use for centroids*, applied consistently to both train and test features. This is a minimal, score-relevant calibration fix that usually yields a meaningful accuracy lift without changing the model family or adding dependencies. I also keep the L2-normalization and all paths/submission ordering exactly the same.'
- What this solution (achieved 0.14275) has done: 'Your current score (0.14275) is far below the target (0.74176), so we should improve accuracy while keeping the same nearest-centroid + cosine + TTA pipeline. The main likely issue is that your centroid construction is accidentally over-weighting images from classes with more TTAs (and any missing/failed decodes), and the centroid mean/std pass and centroid accumulation pass duplicate image decoding work in a way that can introduce inconsistencies. I keep the exact same features and views, but (1) compute global channel mean/std on exactly the same sampled training images (once per image, no TTA) and (2) build centroids by averaging per-image (TTA-averaged) features so each training image contributes equally (not 4× more weight), which is a minimal semantics-preserving calibration fix that typically yields a large lift for centroid classifiers. Submission writing, ordering, paths, and the core “extract → centroid per class → cosine → argmax” logic remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/kaggle/input/cassava-leaf-disease-classification")
TRAIN_CSV = DATA_DIR / "train.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"
TEST_IMG_DIR = DATA_DIR / "test_images"

train_df = pd.read_csv(TRAIN_CSV)
submission = pd.read_csv(SAMPLE_SUB_CSV)

print("Train shape:", train_df.shape)
print("Sample submission shape:", submission.shape)
print(submission.head())


## === cell 1
from PIL import Image


def image_to_feature_vector(
    img_path: Path,
    out_hw=(32, 32),
    hflip: bool = False,
    vflip: bool = False,
    center_crop: bool = False,
    global_ch_mean: np.ndarray | None = None,
    global_ch_std: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fixed-size deterministic RGB feature extractor.

    Change (score-improvement, same core logic):
    - Use *global* per-channel mean/std (computed from the same sampled train images)
      to normalize both train and test features consistently.
    """
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")

            if center_crop:
                w, h = im.size
                s = min(w, h)
                left = (w - s) // 2
                top = (h - s) // 2
                im = im.crop((left, top, left + s, top + s))

            if hflip:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if vflip:
                im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            im = im.resize((out_hw[1], out_hw[0]), resample=Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
    except Exception:
        n = out_hw[0] * out_hw[1] * 3
        return np.zeros(n, dtype=np.float32)

    if global_ch_mean is not None and global_ch_std is not None:
        mu = global_ch_mean.reshape(1, 1, 3).astype(np.float32)
        sd = global_ch_std.reshape(1, 1, 3).astype(np.float32)
        arr = (arr - mu) / np.maximum(sd, 1e-6)
    else:
        ch_mean = arr.mean(axis=(0, 1), keepdims=True)
        ch_std = arr.std(axis=(0, 1), keepdims=True)
        arr = (arr - ch_mean) / np.maximum(ch_std, 1e-6)

    feat = arr.reshape(-1)

    nrm = float(np.linalg.norm(feat))
    if nrm > 1e-6:
        feat = feat / nrm

    return feat.astype(np.float32)




## === cell 2
FEATURE_HW = (32, 32)  # H, W; feature dim = H*W*3
feat_dim = FEATURE_HW[0] * FEATURE_HW[1] * 3

BASE_CAP = 1400
label_counts = train_df["label"].value_counts().to_dict()
max_count = max(label_counts.values())
per_class_cap = {
    int(lbl): int(min(BASE_CAP, max(200, round(BASE_CAP * (cnt / max_count)))))
    for lbl, cnt in label_counts.items()
}

labels = sorted(train_df["label"].unique().tolist())

VIEWS = (
    (False, False, False),  # original
    (True, False, False),  # hflip
    (False, True, False),  # vflip
    (False, False, True),  # center-crop
)

sum_ch = np.zeros(3, dtype=np.float64)
sum_sq_ch = np.zeros(3, dtype=np.float64)
n_pix = 0
missing_train = 0

sampled_rows = {int(lbl): [] for lbl in labels}
for lbl in labels:
    df_lbl = (
        train_df[train_df["label"] == lbl]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    cap = per_class_cap.get(int(lbl), BASE_CAP)
    n = len(df_lbl)
    if n == 0:
        continue
    take = min(cap, n)
    if take <= 0:
        continue
    idxs = np.linspace(0, n - 1, num=take, dtype=np.int64)
    sampled_rows[int(lbl)] = [df_lbl.iloc[int(j)] for j in idxs]

for lbl in labels:
    for row in sampled_rows[int(lbl)]:
        img_path = TRAIN_IMG_DIR / row["image_id"]
        if not img_path.exists():
            missing_train += 1
            continue
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = im.resize(
                    (FEATURE_HW[1], FEATURE_HW[0]), resample=Image.Resampling.LANCZOS
                )
                arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
            pix = arr.reshape(-1, 3).astype(np.float64)
            sum_ch += pix.sum(axis=0)
            sum_sq_ch += (pix * pix).sum(axis=0)
            n_pix += pix.shape[0]
        except Exception:
            continue

if n_pix > 0:
    global_ch_mean = (sum_ch / n_pix).astype(np.float32)
    global_ch_var = (sum_sq_ch / n_pix) - (global_ch_mean.astype(np.float64) ** 2)
    global_ch_std = np.sqrt(np.maximum(global_ch_var, 1e-6)).astype(np.float32)
else:
    global_ch_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    global_ch_std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

print("Global channel mean (train subset):", global_ch_mean)
print("Global channel std  (train subset):", global_ch_std)
print("Missing train images (during stats):", missing_train)

centroids = {int(lbl): np.zeros(feat_dim, dtype=np.float64) for lbl in labels}
counts = {int(lbl): 0 for lbl in labels}

for lbl in labels:
    for row in sampled_rows[int(lbl)]:
        img_path = TRAIN_IMG_DIR / row["image_id"]
        if not img_path.exists():
            continue

        feat_acc = None
        ok_views = 0
        for hflip, vflip, center_crop in VIEWS:
            feat = image_to_feature_vector(
                img_path,
                out_hw=FEATURE_HW,
                hflip=hflip,
                vflip=vflip,
                center_crop=center_crop,
                global_ch_mean=global_ch_mean,
                global_ch_std=global_ch_std,
            ).astype(np.float64)
            if float(np.linalg.norm(feat)) <= 1e-12:
                continue
            if feat_acc is None:
                feat_acc = feat
            else:
                feat_acc += feat
            ok_views += 1

        if feat_acc is None or ok_views == 0:
            continue

        feat_mean = feat_acc / float(ok_views)
        nrm = float(np.linalg.norm(feat_mean))
        if nrm > 1e-12:
            feat_mean = feat_mean / nrm

        centroids[int(lbl)] += feat_mean
        counts[int(lbl)] += 1

for lbl in labels:
    if counts[int(lbl)] > 0:
        c = (centroids[int(lbl)] / counts[int(lbl)]).astype(np.float32)
    else:
        c = centroids[int(lbl)].astype(np.float32)

    nrm = float(np.linalg.norm(c))
    if nrm > 1e-6:
        c = c / nrm
    centroids[int(lbl)] = c

print("Per-class caps:", per_class_cap)
print("Built centroids with image-counts per class:", counts)


## === cell 3
centroid_matrix = np.stack([centroids[lbl] for lbl in labels], axis=0).astype(
    np.float32
)  # (C, D)

preds = np.empty(len(submission), dtype=np.int64)
missing_test = 0

fallback = int(train_df["label"].value_counts().idxmax()) if len(train_df) else 0

for i, image_id in enumerate(submission["image_id"].values):
    img_path = TEST_IMG_DIR / image_id
    if not img_path.exists():
        missing_test += 1
        preds[i] = fallback
        continue

    sims_acc = None
    ok_views = 0
    for hflip, vflip, center_crop in VIEWS:
        feat = image_to_feature_vector(
            img_path,
            out_hw=FEATURE_HW,
            hflip=hflip,
            vflip=vflip,
            center_crop=center_crop,
            global_ch_mean=global_ch_mean,
            global_ch_std=global_ch_std,
        ).astype(np.float64)

        nrm = float(np.linalg.norm(feat))
        if nrm <= 1e-12:
            continue
        feat = (feat / nrm).astype(np.float32)

        sims = centroid_matrix @ feat  # cosine similarity due to normalization
        if sims_acc is None:
            sims_acc = sims.astype(np.float32, copy=True)
        else:
            sims_acc += sims.astype(np.float32, copy=False)
        ok_views += 1

    if sims_acc is None or ok_views == 0:
        preds[i] = fallback
        continue

    sims_mean = sims_acc / float(ok_views)
    preds[i] = int(labels[int(np.argmax(sims_mean))])

print("Missing test images:", missing_test)


## === cell 4
sub = submission.copy()
sub["label"] = preds.astype(np.int64)

assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(submission)
assert sub["label"].between(0, 4).all()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())


## === cell 5
print("Predicted label distribution:")
print(sub["label"].value_counts().sort_index())
