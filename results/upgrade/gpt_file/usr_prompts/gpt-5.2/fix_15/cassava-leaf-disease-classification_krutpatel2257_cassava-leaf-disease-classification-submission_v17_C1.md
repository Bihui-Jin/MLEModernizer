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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.8327289211242067

# 6. Current score

0.56577

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56764) has done: 'I fix the missing EfficientNet dependency by removing the external wheel install and switching to a torchvision EfficientNet-B4 model with an updated classification head so the checkpoint can load (with a safe fallback to non-strict loading if key names differ). I update the Albumentations transforms to the v2 API (notably `RandomResizedCrop(size=(h,w))` and replacing removed `Cutout` with `CoarseDropout`) so augmentation runs without validation errors. I also correct the TTA averaging logic (currently divides by 10 while looping 5 times) and ensure inference runs under `torch.no_grad()` with proper device handling and RGB conversion. Finally, I ensure `submission.csv` is written with the required `image_id,label` columns and correct row order from `sample_submission.csv`.'
- What this solution (achieved 0.11659) has done: 'The immediate failure is the missing checkpoint path, so I make checkpoint discovery robust by searching common Kaggle input locations for a `.pth`/`.pt` file and only asserting after we’ve tried to find a valid one. To move accuracy toward the target, I also fix the inference preprocessing to match EfficientNet’s expected input: use deterministic resize/center-crop (no random augmentations) and use the official EfficientNet-B4 normalization; then apply lightweight deterministic TTA (flips) with softmax-probability averaging (more stable than averaging logits). Finally, I keep the submission ordering exactly as `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.1988) has done: 'We need to unblock execution by removing the hard dependency on an external checkpoint that doesn’t exist in this environment; otherwise inference never runs and the submission is invalid/very low score. I keep the same EfficientNet-B4 architecture and inference pipeline, but load ImageNet pretrained weights when the custom checkpoint can’t be found (this is a minimal, legitimate fallback that should substantially improve accuracy vs random weights). I also fix pathing to use the provided `/kaggle/input/...` dataset root (while keeping the relative paths as secondary fallbacks), and guard checkpoint loading so it only runs if a real file is found. Finally, I ensure we always write `submission.csv` with `image_id,label` in the exact order of `sample_submission.csv`.'
- What this solution (achieved 0.24552) has done: 'Your current score (0.1988) is far below the target (0.8327), and the biggest issue is that when no cassava-trained checkpoint is found you still replace the classifier to 5 classes, which discards the ImageNet head and makes predictions essentially random. I keep the same EfficientNet-B4 + TTA inference core, but change the fallback path to *not* replace the classifier when using ImageNet weights, and instead map the 1000-class ImageNet output to 5 classes via a fixed, deterministic heuristic (argmax%5) so the score should move upward from random toward the target with minimal, safe changes. When a real 5-class cassava checkpoint is found, the existing 5-class head + checkpoint loading behavior remains unchanged. I also keep submission ordering exactly as `sample_submission.csv` and ensure `submission.csv` is always produced.'
- What this solution (achieved 0.65433) has done: 'Your score is far below the target, and the main reason is that in ImageNet-fallback mode the current “argmax % 5” mapping is essentially arbitrary, so accuracy stays near random. I keep the same EfficientNet-B4 inference core, but change only the fallback path to a legitimate, deterministic pseudo-classifier: extract penultimate-layer features and do nearest-centroid classification using centroids computed from the provided `train.csv` + `train_images` (no training loop, no architecture change). This uses the actual cassava distribution to produce meaningful 5-class predictions and should move accuracy much closer to the target while staying within time by caching features and limiting workers. The checkpoint path behavior remains unchanged: if a real cassava checkpoint is found, we use it exactly as before and skip the centroid logic.'
- What this solution (achieved 0.63154) has done: 'Your current score (0.65433) is still well below the target (0.83273), so we should improve accuracy while keeping your EfficientNet-B4 + centroid-fallback core intact. The biggest low-risk gain is to build stronger class centroids from *more* and *more representative* train images: instead of taking the first 800 per class after sorting, we sample a larger, deterministic, stratified subset per class (still no training loop). We also compute centroids using simple augmentation averaging (the same flip TTA you already use at test time) so train/test feature statistics match better, and batch feature extraction to keep runtime under control. Everything else (paths, preprocessing, model, and submission format) stays the same.'
- What this solution (achieved 0.63154) has done: 'We keep your EfficientNet-B4 + nearest-centroid fallback core intact and only tighten the feature pipeline so centroids and test features are computed in exactly the same way. Concretely: (1) normalize test fallback features after TTA-mean (you currently don’t), (2) compute fallback similarity using all TTA features (equivalently averaging then normalizing), and (3) make centroid accumulation numerically stable by accumulating in float64 then casting back. These are minimal changes that should improve alignment between train-derived centroids and test-time features, moving accuracy upward toward the target without changing architecture or adding any training. Submission writing/order stays identical.'
- What this solution (achieved 0.63154) has done: 'We keep your EfficientNet-B4 + nearest-centroid fallback core intact and make the smallest changes that should increase accuracy toward the 0.8327 target by improving centroid quality and matching train/test feature statistics. Concretely, we (1) compute centroids as the mean of per-image L2-normalized features (not mean-of-mean drift), (2) use class-balanced centroids with an explicit `counts`-based safe divide and a deterministic cap per class, and (3) during inference, aggregate TTA by normalizing each TTA feature then averaging then renormalizing (mirrors centroid construction). These are minimal, metric-aligned adjustments and keep the same architecture, preprocessing, and no-training approach, while remaining within Kaggle runtime by batching and (optionally) caching centroids to disk. Submission format and ordering remain exactly as `sample_submission.csv`, and we still always write `submission.csv`.'
- What this solution (achieved 0.62257) has done: 'We keep your EfficientNet‑B4 + nearest‑centroid fallback core intact and focus on one low-risk accuracy gain: remove train/test mismatch caused by different effective “views” of the image. Specifically, we build centroids using the same multi-crop strategy that your inference already approximates (center crop only), but add a deterministic 5-crop TTA (four corners + center) for both centroid construction and test-time fallback, then aggregate features with the same normalize→mean→normalize rule you already use. This tends to improve nearest-centroid classification accuracy without introducing training, changing the architecture, or altering evaluation semantics. We also batch test-time fallback inference (same logic, just fewer Python loops) to stay safely within the 600s runtime.'
- What this solution (achieved 0.61173) has done: 'Your current score (0.62257) is well below the target (0.83273), so we should increase accuracy while preserving your EfficientNet‑B4 + nearest‑centroid fallback core. The lowest-risk gain is to make the fallback features more cassava-relevant without adding any training loop: keep the same feature extractor, but replace the ImageNet normalization with dataset-derived mean/std computed from a small deterministic subset of training images, and then use that same normalization for both centroid building and test inference to remove a major train/test preprocessing mismatch. This keeps the architecture, inference approach, and centroid logic unchanged, but improves feature alignment to the cassava domain. I also keep submission ordering exactly as `sample_submission.csv` and still write `submission.csv` end-to-end.'
- What this solution (achieved 0.60949) has done: 'Your current score (0.61173) is far below the target (0.83273), so we should push accuracy upward while keeping the same EfficientNet‑B4 + nearest‑centroid fallback core. The biggest low-risk issue is that your cassava-derived mean/std is computed from resized images but not the same 380×380 crops used for centroid/test features, creating a normalization mismatch that can significantly degrade nearest-centroid similarity. I compute dataset mean/std from the exact same 5-crop pipeline used later (same resize + crops), and reuse it consistently for centroid building and test inference. I also make centroid cache loading robust to normalization changes by rebuilding if the cache shape is incompatible, preventing accidental use of stale centroids.'
- What this solution (achieved 0.60949) has done: 'We keep your EfficientNet‑B4 + 5‑crop TTA + nearest‑centroid fallback core unchanged, and make two small, score-relevant fixes aimed at improving feature/centroid alignment (which is the main driver of accuracy in fallback mode). First, we apply the exact same post-aggregation L2 normalization to test features that you already use for centroids, so similarity is purely cosine-like and not affected by feature norms. Second, we rebuild centroids with the same “normalize per-image feature then mean” rule you already intended, instead of summing already-aggregated features (tiny drift), and we add a lightweight cache version tag to avoid accidentally reusing older centroids built with different rules. These are minimal changes, should improve accuracy toward the 0.8327 target, and still produce a valid `submission.csv`.'
- What this solution (achieved 0.61248) has done: 'Your score is far below the target (0.60949 vs 0.83273; higher is better), so the safest way to move upward without changing the core model/inference idea is to improve the *fallback nearest-centroid classifier* quality. I keep EfficientNet‑B4 feature extraction + 5‑crop TTA + cosine-similarity-to-centroids exactly the same, but make the centroids more representative by (1) computing **per-class covariance (diagonal)** from the same normalized features and using a **whitened cosine** similarity (a small metric-aligned tweak, not a new model), and (2) building centroids from a slightly larger, still bounded, deterministic stratified subset while staying within the time limit. These changes directly target the fallback’s main weakness (domain mismatch / feature scaling) and should increase accuracy toward the target while preserving evaluation semantics and producing the same submission format.'
- What this solution (achieved 0.56577) has done: 'Your current score (0.61248) is far below the target (0.83273), so we should increase accuracy while keeping your EfficientNet‑B4 + 5‑crop TTA + nearest‑centroid fallback core intact. The smallest high-impact fix is that the “whitened cosine” currently whitens the **centroids but not the centroids’ variance domain correctly** because `centroid_var` is computed around an *unnormalized* mean (`mean_raw`) while the centroids used for similarity are *L2-normalized* (`mean_c`); this mismatch can distort similarity and hurt accuracy. I compute the diagonal variance around the **same normalized centroid** that you actually use at inference, and cache-bust with a new version tag so you don’t accidentally reuse the old inconsistent centroids. Everything else (paths, model, 5-crop TTA, submission writing/order) stays the same.'

# 9. Code solution

## === cell 0
import os
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)



## === cell 1
model_path = "../input/en-b4-tta-calr-15/model_15.pth"

sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_images_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"

if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "../input/cassava-leaf-disease-classification/sample_submission.csv"
    )
if not os.path.isdir(test_images_path):
    test_images_path = "../input/cassava-leaf-disease-classification/test_images"
if not os.path.exists(train_csv_path):
    train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
if not os.path.isdir(train_images_path):
    train_images_path = "../input/cassava-leaf-disease-classification/train_images"

assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing dir: {test_images_path}"


def _find_checkpoint(preferred_path: str) -> str | None:
    """Return best checkpoint path if found, else None (so we can fallback safely)."""
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data",
    ]
    exts = (".pth", ".pt", ".bin")

    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    candidates.append(os.path.join(dirpath, fn))

    def score(p):
        pl = p.lower()
        s = 0
        if "b4" in pl or "efficientnet" in pl:
            s += 5
        if "cassava" in pl:
            s += 2
        if "model" in pl or "checkpoint" in pl or "ckpt" in pl:
            s += 2
        if "tta" in pl:
            s += 1
        s -= pl.count(os.sep) * 0.01
        return s

    if not candidates:
        return None

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


resolved_model_path = _find_checkpoint(model_path)
print("Resolved model_path:", resolved_model_path)



## === cell 2
using_imagenet_fallback = resolved_model_path is None

if using_imagenet_fallback:
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    print(
        "No checkpoint found; using ImageNet pretrained EfficientNet-B4 weights (1000-class head)."
    )
else:
    model = models.efficientnet_b4(weights=None)
    print(
        "Checkpoint found; using weights=None then loading checkpoint:",
        resolved_model_path,
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)

model.to(DEVICE)



## === cell 3
if (
    (not using_imagenet_fallback)
    and resolved_model_path is not None
    and os.path.exists(resolved_model_path)
):
    ckpt = torch.load(resolved_model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(state_dict, dict):
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint with strict=True")
    except Exception as e:
        print("Strict load failed; retrying with strict=False. Error:", repr(e))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )
else:
    if using_imagenet_fallback:
        print("Skipping checkpoint load (ImageNet fallback mode).")
    else:
        print("Skipping checkpoint load (no valid checkpoint path).")

model.eval()




## === cell 4
def _compute_dataset_mean_std(
    train_df: pd.DataFrame,
    train_images_dir: str,
    resize_max: int = 380,
    crop_h: int = 380,
    crop_w: int = 380,
    n_images: int = 256,
    seed: int = 2021,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    ids = train_df["image_id"].values
    take = min(n_images, len(ids))
    sel = rng.choice(ids, size=take, replace=False)

    resize = A.SmallestMaxSize(max_size=resize_max, p=1.0)

    def crop5(img: np.ndarray) -> list[np.ndarray]:
        h, w = img.shape[:2]
        if h < crop_h or w < crop_w:
            img = resize(image=img)["image"]
            h, w = img.shape[:2]

        y0, x0 = 0, 0
        y1, x1 = 0, max(0, w - crop_w)
        y2, x2 = max(0, h - crop_h), 0
        y3, x3 = max(0, h - crop_h), max(0, w - crop_w)
        yc, xc = max(0, (h - crop_h) // 2), max(0, (w - crop_w) // 2)

        crops = [
            img[y0 : y0 + crop_h, x0 : x0 + crop_w, :],
            img[y1 : y1 + crop_h, x1 : x1 + crop_w, :],
            img[y2 : y2 + crop_h, x2 : x2 + crop_w, :],
            img[y3 : y3 + crop_h, x3 : x3 + crop_w, :],
            img[yc : yc + crop_h, xc : xc + crop_w, :],
        ]
        return [np.ascontiguousarray(c) for c in crops]

    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    n_pix = 0

    for img_id in sel:
        p = os.path.join(train_images_dir, img_id)
        im = Image.open(p).convert("RGB")
        im = np.array(im)
        im = resize(image=im)["image"]
        crops = crop5(im)
        for c in crops:
            x = c.astype(np.float64) / 255.0  # HWC, [0,1]
            flat = x.reshape(-1, 3)
            sum_c += flat.sum(axis=0)
            sumsq_c += (flat**2).sum(axis=0)
            n_pix += x.shape[0] * x.shape[1]

    mean = sum_c / max(1, n_pix)
    var = sumsq_c / max(1, n_pix) - mean**2
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    mean_t = (float(mean[0]), float(mean[1]), float(mean[2]))
    std_t = (float(std[0]), float(std[1]), float(std[2]))
    return mean_t, std_t


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

infer_resize = A.SmallestMaxSize(max_size=380, p=1.0)
to_tensor = transforms.ToTensor()

if using_imagenet_fallback:
    assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
    assert os.path.isdir(train_images_path), f"Missing dir: {train_images_path}"
    train_df_stats = pd.read_csv(train_csv_path)
    try:
        ds_mean, ds_std = _compute_dataset_mean_std(
            train_df_stats,
            train_images_path,
            resize_max=380,
            crop_h=380,
            crop_w=380,
            n_images=256,
            seed=2021,
        )
        mean, std = ds_mean, ds_std
        print(
            "Using cassava-derived normalization (fallback, crop-matched): mean=",
            mean,
            "std=",
            std,
        )
    except Exception as e:
        print(
            "Failed to compute cassava mean/std; using ImageNet normalization. Error:",
            repr(e),
        )

infer_norm = A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0)


def _crop5(img: np.ndarray, crop_h: int = 380, crop_w: int = 380) -> list[np.ndarray]:
    h, w = img.shape[:2]
    if h < crop_h or w < crop_w:
        img_r = infer_resize(image=img)["image"]
        h, w = img_r.shape[:2]
        img = img_r

    y0, x0 = 0, 0
    y1, x1 = 0, max(0, w - crop_w)
    y2, x2 = max(0, h - crop_h), 0
    y3, x3 = max(0, h - crop_h), max(0, w - crop_w)
    yc, xc = max(0, (h - crop_h) // 2), max(0, (w - crop_w) // 2)

    crops = [
        img[y0 : y0 + crop_h, x0 : x0 + crop_w, :],
        img[y1 : y1 + crop_h, x1 : x1 + crop_w, :],
        img[y2 : y2 + crop_h, x2 : x2 + crop_w, :],
        img[y3 : y3 + crop_h, x3 : x3 + crop_w, :],
        img[yc : yc + crop_h, xc : xc + crop_w, :],
    ]
    return [np.ascontiguousarray(c) for c in crops]


def _tta_views(img: np.ndarray) -> list[np.ndarray]:
    img_r = infer_resize(image=img)["image"]
    return _crop5(img_r, 380, 380)


def _imgs_to_tensor_batch(imgs_np: list[np.ndarray]) -> torch.Tensor:
    xs = []
    for im in imgs_np:
        imn = infer_norm(image=im)["image"]
        xs.append(to_tensor(imn))
    return torch.stack(xs, dim=0)


def _extract_feat_batch(
    model_feat: nn.Module, imgs_np: list[np.ndarray]
) -> torch.Tensor:
    xb = _imgs_to_tensor_batch(imgs_np).to(DEVICE)
    with torch.no_grad():
        feats = model_feat(xb)  # [B, C]
    return feats.detach().cpu()


def _aggregate_tta_feats(feats_cpu: torch.Tensor, tta: int) -> torch.Tensor:
    """
    Expected feats_cpu shape: [N*tta, C]
    Return: [N, C] built as normalize(each TTA feature) -> mean -> normalize.
    """
    feats_cpu = feats_cpu.view(-1, tta, feats_cpu.shape[1])  # [N, T, C]
    feats_cpu = torch.nn.functional.normalize(feats_cpu, p=2, dim=2)
    feat_mean = feats_cpu.mean(dim=1)
    feat_mean = torch.nn.functional.normalize(feat_mean, p=2, dim=1)
    return feat_mean


centroids = None  # [5, C]
centroid_var = None  # [5, C] diagonal variance (for whitening)
if using_imagenet_fallback:
    model_feat = nn.Sequential(model.features, model.avgpool, nn.Flatten(1)).to(DEVICE)
    model_feat.eval()

    norm_tag = f"m{mean[0]:.4f}_{mean[1]:.4f}_{mean[2]:.4f}_s{std[0]:.4f}_{std[1]:.4f}_{std[2]:.4f}"
    centroid_cache_path = f"centroids_enb4_imagenet_tta5crop_380_max3000_seed2021_{norm_tag}_v4_whiten_consistent.npz"

    cache_ok = False
    if os.path.exists(centroid_cache_path):
        try:
            cached = np.load(centroid_cache_path)
            centroids_np = cached["centroids"]
            counts_np = cached["counts"]
            var_np = cached["var"]
            if (
                centroids_np.ndim == 2
                and centroids_np.shape[0] == 5
                and counts_np.shape[0] == 5
                and var_np.shape == centroids_np.shape
            ):
                centroids = torch.from_numpy(centroids_np).to(torch.float32)
                centroid_var = torch.from_numpy(var_np).to(torch.float32)
                counts = torch.from_numpy(counts_np).to(torch.long)
                cache_ok = True
                print(
                    "Loaded cached centroids+var:",
                    centroid_cache_path,
                    "counts:",
                    counts.tolist(),
                )
            else:
                print(
                    "Cached centroids/var shape mismatch; rebuilding:",
                    centroids_np.shape,
                    counts_np.shape,
                    var_np.shape,
                )
        except Exception as e:
            print("Failed to load cached centroids; rebuilding. Error:", repr(e))

    if not cache_ok:
        train_df = pd.read_csv(train_csv_path)

        rng = np.random.default_rng(2021)
        max_per_class = 3000

        class_ids = {}
        for c in range(5):
            ids = train_df.loc[train_df["label"] == c, "image_id"].values
            if len(ids) == 0:
                raise RuntimeError(f"No training images for class {c}.")
            take = min(max_per_class, len(ids))
            sel = rng.choice(ids, size=take, replace=False)
            class_ids[c] = sel.tolist()

        feats_sum = None
        feats_sumsq = None
        counts = torch.zeros(5, dtype=torch.long)

        batch_size_imgs = 28  # 28*5=140 crops per batch
        tta_n = 5

        for c in range(5):
            img_ids = class_ids[c]
            for j in range(0, len(img_ids), batch_size_imgs):
                batch_ids = img_ids[j : j + batch_size_imgs]
                tta_imgs = []
                for img_id in batch_ids:
                    p = os.path.join(train_images_path, img_id)
                    im = Image.open(p).convert("RGB")
                    im = np.array(im)
                    tta_imgs.extend(_tta_views(im))

                feats_raw = _extract_feat_batch(model_feat, tta_imgs)  # [B*5, C]
                feats_img = _aggregate_tta_feats(
                    feats_raw, tta=tta_n
                )  # [B, C] normalized

                if feats_sum is None:
                    feats_sum = torch.zeros(5, feats_img.shape[1], dtype=torch.float64)
                    feats_sumsq = torch.zeros(
                        5, feats_img.shape[1], dtype=torch.float64
                    )
                x = feats_img.to(torch.float64)
                feats_sum[c] += x.sum(dim=0)
                feats_sumsq[c] += (x * x).sum(dim=0)
                counts[c] += feats_img.shape[0]

        denom = counts.clamp_min(1).unsqueeze(1).to(torch.float64)

        mean_un = (feats_sum / denom).to(torch.float32)
        mean_c = torch.nn.functional.normalize(mean_un, p=2, dim=1)

        mean_c64 = mean_c.to(torch.float64)
        var_c = (feats_sumsq / denom) - (mean_c64 * mean_c64)
        var_c = torch.clamp(var_c.to(torch.float32), min=1e-6)

        centroids = mean_c
        centroid_var = var_c

        np.savez_compressed(
            centroid_cache_path,
            centroids=centroids.numpy(),
            var=centroid_var.numpy(),
            counts=counts.numpy(),
        )
        print(
            "Built and cached centroids+var:",
            centroid_cache_path,
            "counts:",
            counts.tolist(),
        )



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

pred_labels = []

if using_imagenet_fallback:
    model_feat = nn.Sequential(model.features, model.avgpool, nn.Flatten(1)).to(DEVICE)
    model_feat.eval()

tta_n = 5
test_batch_size_imgs = 32  # 32*5=160 crops per step

with torch.no_grad():
    if using_imagenet_fallback:
        eps = 1e-6
        inv_std = torch.rsqrt(centroid_var + eps)  # [5, C]

        image_ids = sample_sub["image_id"].tolist()
        for i in range(0, len(image_ids), test_batch_size_imgs):
            batch_ids = image_ids[i : i + test_batch_size_imgs]
            tta_imgs = []
            for img_id in batch_ids:
                image_path = os.path.join(test_images_path, img_id)
                im = Image.open(image_path).convert("RGB")
                im = np.array(im)
                tta_imgs.extend(_tta_views(im))

            feats_raw = _extract_feat_batch(model_feat, tta_imgs)  # [B*5, C] CPU
            feat_mean = _aggregate_tta_feats(feats_raw, tta=tta_n)  # [B, C] normalized

            f = feat_mean.to(torch.float32)  # already normalized, keep float32
            f_w = f.unsqueeze(1) * inv_std.unsqueeze(0)
            c_w = centroids.unsqueeze(0) * inv_std.unsqueeze(0)
            f_w = torch.nn.functional.normalize(f_w, p=2, dim=2)
            c_w = torch.nn.functional.normalize(c_w, p=2, dim=2)
            sims = (f_w * c_w).sum(dim=2)  # [B,5]

            preds = torch.argmax(sims, dim=1).cpu().numpy().astype(int).tolist()
            pred_labels.extend(preds)
    else:
        for _, row in sample_sub.iterrows():
            image_path = os.path.join(test_images_path, row.image_id)
            img = Image.open(image_path).convert("RGB")
            img = np.array(img)

            views = _tta_views(img)  # 5 deterministic crops

            prob_sum = None
            for im in views:
                imn = infer_norm(image=im)["image"]
                x = to_tensor(imn).to(DEVICE)
                logits = model(x.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                prob_sum = probs if prob_sum is None else (prob_sum + probs)
            prob_mean = prob_sum / float(len(views))
            pred = int(torch.argmax(prob_mean, dim=1).item())
            pred_labels.append(int(pred))

sub_df = sample_sub.copy()
sub_df["label"] = pred_labels
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Saved submission.csv with", len(sub_df), "rows")
print("Unique labels predicted:", sorted(sub_df["label"].unique().tolist()))
print(
    "Mode:",
    (
        "ImageNet fallback (nearest-centroid on train features, 5-crop TTA, whitened cosine)"
        if using_imagenet_fallback
        else "Cassava checkpoint (5-class head, 5-crop prob-avg)"
    ),
)
