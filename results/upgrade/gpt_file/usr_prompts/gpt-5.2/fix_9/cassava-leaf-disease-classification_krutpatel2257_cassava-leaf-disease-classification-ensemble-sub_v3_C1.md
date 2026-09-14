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
scipy==1.15.3
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

0.8930190389845875

# 6. Current score

0.4574

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4275) has done: 'The timeout is dominated by per-image Python overhead: iterating with `iterrows()`, running Albumentations 10× per image, converting HWC→tensor each time, and doing two separate forward passes (ResNeXt then EfficientNet) per TTA with small batch size 1. I keep the exact same models, weights, TTA pipeline, and averaging logic, but reduce overhead by (1) precomputing the test file list and iterating by index (no pandas row objects), (2) stacking all TTA views for an image into one tensor and running each model once per image in a single batched forward, and (3) using pinned-memory + non_blocking transfers and lightweight numpy conversions. These changes are mathematically equivalent (same TTA samples and same averaging), but drastically cut Python/CUDA launch overhead and improve GPU utilization, bringing runtime under 600s.'
- What this solution (achieved 0.35015) has done: 'Your current score (0.4275) is far below the target (0.8930), and the biggest accuracy issue is that the inference preprocessing does not match the pretrained model expectations: you’re normalizing twice (Albumentations Normalize + `ToTensor()` which rescales again), and you’re also feeding 512×512 to models that were trained/initialized with 224/380 conventions. I keep the exact same two-model ensemble and 10x TTA averaging logic, but fix the input pipeline to produce correct tensors: apply Albumentations augmentations without normalization, then use the proper torchvision weights’ preprocessing (resize/crop + normalize) for each model separately. This is a minimal semantic fix (still TTA + average logits) that should move accuracy sharply toward the target band. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.21637) has done: 'Your current score is far below the target, so we should push accuracy up with the smallest changes that fix likely correctness mismatches. The biggest issue is that your inference-time TTA uses heavy random geometric transforms and RandomResizedCrop that almost certainly don’t match how these checkpoints were trained, so it injects distribution shift and destroys accuracy; we keep “10x TTA + average logits + two-model ensemble” intact but make TTA deterministic and label-preserving (resize + flips only). Second, we ensure the input scaling is consistent by always converting augmented numpy images to uint8 RGB before PIL and preprocessing (avoids subtle dtype/range issues). These are minimal, metric-aligned changes that should move accuracy sharply toward your target while preserving the overall inference logic and producing the same submission schema.'
- What this solution (achieved 0.26495) has done: 'Your current score (0.21637) is far below the target (0.8930), so we should improve accuracy with minimal, metric-aligned fixes while keeping the same two-model ensemble + 10x TTA + logits-averaging core logic. The biggest likely correctness bug is that the current TTA includes transforms (Transpose/VerticalFlip) that change leaf orientation in a way these checkpoints likely weren’t trained to be invariant to, which can severely hurt accuracy; we keep 10 TTA views but make them deterministic and limited to safe label-preserving transforms (identity + horizontal flip). Second, we match each model’s expected input resolution by resizing to the model’s native size before preprocessing (224 for ResNeXt, 380 for EfficientNet-B4) instead of forcing 512 then cropping, which reduces harmful resampling artifacts. These changes keep the inference semantics (TTA, same models, same averaging) but should move the score substantially toward the target band.'
- What this solution (achieved 0.4574) has done: 'Your current score is far below the target, and the main accuracy blocker is that the fallback path (when external checkpoints aren’t found) replaces the classifier heads with random weights, which collapses accuracy to near-random. I keep your exact ensemble + 10x TTA + logits-averaging inference logic, but change the fallback to use the pretrained ImageNet heads (no head replacement) and then map the 1000-class predictions into 5 classes using a calibrated mapping learned from train.csv (via a simple confusion-based assignment). This is a minimal, metric-aligned fix that should move accuracy sharply upward when custom checkpoints aren’t available, while still producing a valid `submission.csv`. If the external checkpoints are found, the mapping is not used and behavior remains the same.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v3/eff_epoch_11.pth"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"


def _find_first_existing(path_candidates):
    for p in path_candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _autofind_checkpoint(patterns, search_root="/kaggle/input"):
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(search_root, "**", pat), recursive=True))
    hits = [h for h in hits if os.path.isfile(h)]
    hits.sort()
    return hits[0] if hits else None


_resnet_found = _find_first_existing([resnet_model_path])
_effnet_found = _find_first_existing([effnet_model_path])

if _resnet_found is None:
    _resnet_found = _autofind_checkpoint(
        patterns=[
            "model(24).pth",
            "*resnext*50*.pth",
            "*resnext*.pth",
            "*resnet*.pth",
        ]
    )
if _effnet_found is None:
    _effnet_found = _autofind_checkpoint(
        patterns=[
            "eff_epoch_11.pth",
            "*efficientnet*b4*.pth",
            "*eff*epoch*.pth",
        ]
    )

resnet_model_path = _resnet_found
effnet_model_path = _effnet_found

USE_EXTERNAL_CKPTS = (resnet_model_path is not None) and (effnet_model_path is not None)

print("External resnet checkpoint:", resnet_model_path)
print("External effnet checkpoint:", effnet_model_path)
print("USE_EXTERNAL_CKPTS =", USE_EXTERNAL_CKPTS)




## === cell 2
def load_checkpoint_safely(model, ckpt_path, device=DEVICE):
    if ckpt_path is None or (not os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(
            f"[WARN] Missing keys when loading {os.path.basename(ckpt_path)}: {len(missing)}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading {os.path.basename(ckpt_path)}: {len(unexpected)}"
        )
    return model




## === cell 3
if USE_EXTERNAL_CKPTS:
    resnet_model = models.resnext50_32x4d(weights=None)
    resnet_model.fc = nn.Linear(2048, 5)
    resnet_model.to(DEVICE)
    resnet_model = load_checkpoint_safely(
        resnet_model, resnet_model_path, device=DEVICE
    )
    resnet_model.eval()

    effnet_model = models.efficientnet_b4(weights=None)
    in_features = effnet_model.classifier[1].in_features
    effnet_model.classifier[1] = nn.Linear(in_features, 5)
    effnet_model.to(DEVICE)
    effnet_model = load_checkpoint_safely(
        effnet_model, effnet_model_path, device=DEVICE
    )
    effnet_model.eval()

    resnet_weights = None
    effnet_weights = None
    FALLBACK_USES_MAPPING = False
else:
    resnet_weights = models.ResNeXt50_32X4D_Weights.DEFAULT
    effnet_weights = models.EfficientNet_B4_Weights.DEFAULT

    resnet_model = models.resnext50_32x4d(weights=resnet_weights).to(DEVICE).eval()
    effnet_model = models.efficientnet_b4(weights=effnet_weights).to(DEVICE).eval()

    FALLBACK_USES_MAPPING = True

print("Models ready on:", DEVICE)
print("FALLBACK_USES_MAPPING =", FALLBACK_USES_MAPPING)



## === cell 4
try:
    sub_aug_id = A.Compose([A.NoOp()], p=1.0)
    sub_aug_hflip = A.Compose([A.HorizontalFlip(p=1.0)], p=1.0)
except Exception as e:
    print(
        "[WARN] Failed to build albumentations TTA pipeline, falling back to no-op only. Error:"
    )
    print(str(e))
    sub_aug_id = A.Compose([A.NoOp()], p=1.0)
    sub_aug_hflip = A.Compose([A.NoOp()], p=1.0)

RESNET_INPUT_SIZE = 224
EFFNET_INPUT_SIZE = 380

if resnet_weights is not None:
    resnet_preprocess = resnet_weights.transforms()
else:
    resnet_preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

if effnet_weights is not None:
    effnet_preprocess = effnet_weights.transforms()
else:
    effnet_preprocess = transforms.Compose(
        [
            transforms.Resize(426, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )



## === cell 5
if not os.path.isfile(sample_sub_path):
    raise FileNotFoundError(f"sample_submission.csv not found at: {sample_sub_path}")
if not os.path.isdir(test_images_path):
    raise FileNotFoundError(f"test_images directory not found at: {test_images_path}")

sample_sub = pd.read_csv(sample_sub_path)
assert "image_id" in sample_sub.columns and "label" in sample_sub.columns
print("Test rows:", len(sample_sub))




## === cell 6
def _build_imagenet_to_cassava_mapping(
    model_r,
    model_e,
    preprocess_r,
    preprocess_e,
    train_csv,
    train_images_dir,
    device=DEVICE,
    max_samples_per_class=60,
    seed=0,
):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(train_csv)
    if not {"image_id", "label"}.issubset(df.columns):
        raise ValueError("train.csv must contain image_id and label")

    parts = []
    for y in range(5):
        sub = df[df["label"] == y]
        n = min(max_samples_per_class, len(sub))
        if n <= 0:
            continue
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        parts.append(sub.loc[idx])
    df_small = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    num_classes = 1000
    accum_r = np.zeros((5, num_classes), dtype=np.float64)
    accum_e = np.zeros((5, num_classes), dtype=np.float64)
    counts = np.zeros(5, dtype=np.int64)

    model_r.eval()
    model_e.eval()

    with torch.no_grad():
        for row in df_small.itertuples(index=False):
            img_id = row.image_id
            y = int(row.label)
            img_path = os.path.join(train_images_dir, img_id)
            if not os.path.isfile(img_path):
                continue

            pil = Image.open(img_path).convert("RGB")
            x_r = preprocess_r(pil).unsqueeze(0).to(device, dtype=torch.float32)
            x_e = preprocess_e(pil).unsqueeze(0).to(device, dtype=torch.float32)

            logits_r = model_r(x_r).float().detach().cpu().numpy()[0]
            logits_e = model_e(x_e).float().detach().cpu().numpy()[0]
            pr = softmax(logits_r, axis=0)
            pe = softmax(logits_e, axis=0)

            accum_r[y] += pr
            accum_e[y] += pe
            counts[y] += 1

    if (counts == 0).any():
        raise RuntimeError(
            f"Mapping build failed: missing samples for labels {np.where(counts==0)[0].tolist()}"
        )

    accum_r /= counts[:, None]
    accum_e /= counts[:, None]

    combined = (accum_r + accum_e) / 2.0  # [5, 1000]
    remaining = set(range(num_classes))
    mapping = np.full(num_classes, -1, dtype=np.int16)

    for _ in range(num_classes):
        best_c = None
        best_y = None
        best_s = -1.0
        for c in remaining:
            y = int(np.argmax(combined[:, c]))
            s = float(combined[y, c])
            if s > best_s:
                best_s = s
                best_c = c
                best_y = y
        mapping[best_c] = best_y
        remaining.remove(best_c)

    priors = counts / counts.sum()
    return mapping, priors


imagenet_to_cassava = None
cassava_priors = None

if FALLBACK_USES_MAPPING:
    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
    train_images_dir = os.path.join(os.path.dirname(sample_sub_path), "train_images")
    if not os.path.isdir(train_images_dir):
        train_images_dir = "../input/cassava-leaf-disease-classification/train_images"
    if not os.path.isdir(train_images_dir):
        raise FileNotFoundError(
            f"train_images directory not found for mapping at: {train_images_dir}"
        )

    imagenet_to_cassava, cassava_priors = _build_imagenet_to_cassava_mapping(
        resnet_model,
        effnet_model,
        resnet_preprocess,
        effnet_preprocess,
        train_csv=train_csv_path,
        train_images_dir=train_images_dir,
        device=DEVICE,
        max_samples_per_class=50,  # keep under runtime limits
        seed=0,
    )
    print("Built ImageNet->Cassava mapping. Priors:", cassava_priors)



## === cell 7
tta_count = 10

image_ids = sample_sub["image_id"].to_numpy()
img_paths = [os.path.join(test_images_path, img_id) for img_id in image_ids]

out_dim = 5 if not FALLBACK_USES_MAPPING else 1000
predictions = np.empty((len(img_paths), out_dim), dtype=np.float32)

resnet_model.eval()
effnet_model.eval()

pin_memory = DEVICE.type == "cuda"

with torch.no_grad():
    for i, img_path in enumerate(img_paths):
        base_pil = Image.open(img_path).convert("RGB")
        base_np = np.asarray(base_pil)

        tta_r = []
        tta_e = []

        for t in range(tta_count):
            aug = sub_aug_id if (t % 2 == 0) else sub_aug_hflip
            aug_img = aug(image=base_np)["image"]

            if not isinstance(aug_img, np.ndarray):
                aug_img = np.array(aug_img)
            if aug_img.dtype != np.uint8:
                aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)

            aug_pil_r = Image.fromarray(aug_img, mode="RGB").resize(
                (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE), resample=Image.BILINEAR
            )
            aug_pil_e = Image.fromarray(aug_img, mode="RGB").resize(
                (EFFNET_INPUT_SIZE, EFFNET_INPUT_SIZE), resample=Image.BILINEAR
            )

            tta_r.append(resnet_preprocess(aug_pil_r))
            tta_e.append(effnet_preprocess(aug_pil_e))

        batch_r_cpu = torch.stack(tta_r, dim=0)
        batch_e_cpu = torch.stack(tta_e, dim=0)

        if pin_memory:
            batch_r_cpu = batch_r_cpu.pin_memory()
            batch_e_cpu = batch_e_cpu.pin_memory()

        batch_r = batch_r_cpu.to(DEVICE, dtype=torch.float32, non_blocking=True)
        batch_e = batch_e_cpu.to(DEVICE, dtype=torch.float32, non_blocking=True)

        out_r = resnet_model(batch_r)
        out_e = effnet_model(batch_e)

        avg_logits = (out_r.mean(dim=0) + out_e.mean(dim=0)) / 2.0
        predictions[i] = avg_logits.detach().cpu().numpy()



## === cell 8
if FALLBACK_USES_MAPPING:
    probs_1000 = softmax(predictions, axis=1)  # [N, 1000]
    probs_5 = np.zeros((probs_1000.shape[0], 5), dtype=np.float64)
    for c in range(1000):
        y = int(imagenet_to_cassava[c])
        probs_5[:, y] += probs_1000[:, c]
    probs_5 = probs_5 / probs_5.sum(axis=1, keepdims=True)
    pred_labels = probs_5.argmax(axis=1).astype(int)
else:
    pred_labels = softmax(predictions, axis=1).argmax(axis=1).astype(int)

sub_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
