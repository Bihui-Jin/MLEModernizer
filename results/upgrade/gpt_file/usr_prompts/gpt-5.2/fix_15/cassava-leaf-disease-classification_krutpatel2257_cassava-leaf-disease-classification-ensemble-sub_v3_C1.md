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

0.65508

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4275) has done: 'The timeout is dominated by per-image Python overhead: iterating with `iterrows()`, running Albumentations 10× per image, converting HWC→tensor each time, and doing two separate forward passes (ResNeXt then EfficientNet) per TTA with small batch size 1. I keep the exact same models, weights, TTA pipeline, and averaging logic, but reduce overhead by (1) precomputing the test file list and iterating by index (no pandas row objects), (2) stacking all TTA views for an image into one tensor and running each model once per image in a single batched forward, and (3) using pinned-memory + non_blocking transfers and lightweight numpy conversions. These changes are mathematically equivalent (same TTA samples and same averaging), but drastically cut Python/CUDA launch overhead and improve GPU utilization, bringing runtime under 600s.'
- What this solution (achieved 0.35015) has done: 'Your current score (0.4275) is far below the target (0.8930), and the biggest accuracy issue is that the inference preprocessing does not match the pretrained model expectations: you’re normalizing twice (Albumentations Normalize + `ToTensor()` which rescales again), and you’re also feeding 512×512 to models that were trained/initialized with 224/380 conventions. I keep the exact same two-model ensemble and 10x TTA averaging logic, but fix the input pipeline to produce correct tensors: apply Albumentations augmentations without normalization, then use the proper torchvision weights’ preprocessing (resize/crop + normalize) for each model separately. This is a minimal semantic fix (still TTA + average logits) that should move accuracy sharply toward the target band. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.21637) has done: 'Your current score is far below the target, so we should push accuracy up with the smallest changes that fix likely correctness mismatches. The biggest issue is that your inference-time TTA uses heavy random geometric transforms and RandomResizedCrop that almost certainly don’t match how these checkpoints were trained, so it injects distribution shift and destroys accuracy; we keep “10x TTA + average logits + two-model ensemble” intact but make TTA deterministic and label-preserving (resize + flips only). Second, we ensure the input scaling is consistent by always converting augmented numpy images to uint8 RGB before PIL and preprocessing (avoids subtle dtype/range issues). These are minimal, metric-aligned changes that should move accuracy sharply toward your target while preserving the overall inference logic and producing the same submission schema.'
- What this solution (achieved 0.26495) has done: 'Your current score (0.21637) is far below the target (0.8930), so we should improve accuracy with minimal, metric-aligned fixes while keeping the same two-model ensemble + 10x TTA + logits-averaging core logic. The biggest likely correctness bug is that the current TTA includes transforms (Transpose/VerticalFlip) that change leaf orientation in a way these checkpoints likely weren’t trained to be invariant to, which can severely hurt accuracy; we keep 10 TTA views but make them deterministic and limited to safe label-preserving transforms (identity + horizontal flip). Second, we match each model’s expected input resolution by resizing to the model’s native size before preprocessing (224 for ResNeXt, 380 for EfficientNet-B4) instead of forcing 512 then cropping, which reduces harmful resampling artifacts. These changes keep the inference semantics (TTA, same models, same averaging) but should move the score substantially toward the target band.'
- What this solution (achieved 0.4574) has done: 'Your current score is far below the target, and the main accuracy blocker is that the fallback path (when external checkpoints aren’t found) replaces the classifier heads with random weights, which collapses accuracy to near-random. I keep your exact ensemble + 10x TTA + logits-averaging inference logic, but change the fallback to use the pretrained ImageNet heads (no head replacement) and then map the 1000-class predictions into 5 classes using a calibrated mapping learned from train.csv (via a simple confusion-based assignment). This is a minimal, metric-aligned fix that should move accuracy sharply upward when custom checkpoints aren’t available, while still producing a valid `submission.csv`. If the external checkpoints are found, the mapping is not used and behavior remains the same.'
- What this solution (achieved 0.66704) has done: 'Your score gap is large (0.4574 vs target 0.8930), so we need a minimal change that plausibly fixes a correctness mismatch rather than tuning. The most likely issue is that in the fallback (no external cassava checkpoints), you’re using ImageNet weights but mapping 1000→5 via a greedy “one-class-per-cassava-label” assignment that forces all 1000 classes to map and tends to smear probability mass; instead, we learn a direct 5-class linear head on top of each frozen backbone using train.csv images (fast, deterministic), then keep your exact ensemble + 10x TTA + averaging logic at inference. This preserves the core architecture (same two torchvision models), same loss semantics (cross-entropy), and same inference averaging, but replaces the brittle mapping with a legitimate supervised calibration step that should move accuracy sharply upward toward the target. Runtime stays within limits by training only the final layers for a small number of steps with a straightforward DataLoader and no augmentation.'
- What this solution (achieved 0.67526) has done: 'The timeout is dominated by per-image Python overhead in the test-time loop: opening/decoding images one-by-one, running albumentations + PIL conversions 10 times per image, and repeatedly calling the preprocess pipelines. I preserve the exact same models, checkpoints, TTA semantics (10 passes alternating identity/hflip), preprocessing, and averaging logic, but batch the entire test set through a DataLoader so decoding/augmentation runs in parallel workers and GPU inference runs on larger contiguous batches. I also eliminate redundant conversions (PIL→np→PIL) and avoid per-image `.pin_memory()` calls by letting DataLoader handle pinned memory. These changes are provably equivalent in outputs (same transforms, same TTA images, same averaging) while greatly reducing interpreter overhead and improving GPU utilization.'
- What this solution (achieved 0.65994) has done: 'Your current score (0.67526) is well below the target (0.8930), so we should improve accuracy with a minimal, metric-aligned fix rather than tuning. The biggest likely issue is that your inference is averaging *logits* across TTA and across models; when checkpoints are trained separately, logit scales can differ and this often hurts accuracy—averaging *probabilities* is usually more stable and improves top-1 accuracy without changing the model or augmentations. I keep the same two-model ensemble, same 10-view (id/hflip) TTA, same preprocessing, and the same DataLoader pipeline, but change only the aggregation: compute softmax per-model per-TTA, average probabilities, then argmax. This preserves evaluation semantics (still predicting a single class per image) and should move the score upward toward your target.'
- What this solution (achieved 0.65508) has done: 'Your current score (0.65994) is far below the target (0.8930), so we should improve accuracy with a minimal, metric-aligned fix rather than changing the model/loops. The biggest likely correctness issue is that your inference preprocessing uses the ImageNet weight transforms (good) but your TTA feeds full-resolution images directly into those transforms, which may apply different resize/crop behavior than what the cassava checkpoints expect; we make the TTA explicitly resize to each model’s native input size before preprocessing (keeping the same 10-view id/hflip TTA and the same ensemble averaging of probabilities). This keeps the core inference logic identical (same two models, same TTA count/pattern, same softmax-then-average aggregation) while reducing harmful resampling/cropping artifacts that can depress accuracy. I also enable `torch.cuda.amp.autocast` only for inference (not changing semantics, just numerical dtype) to keep runtime safely under 600s.'
- What this solution (achieved 0.65508) has done: 'Your gap to the target is large (0.65508 vs 0.8930), so we need a minimal change that improves correctness rather than speed. The biggest likely issue is a preprocessing mismatch when external cassava checkpoints are used: you currently resize to a fixed square (224/380) before applying `weights.transforms()`, but those transforms already include their own resize/crop policy and were intended to run on the original PIL image; double-resizing/cropping can destroy accuracy. I keep the exact same models, 10x (id/hflip) TTA, DataLoader batching, and probability-averaging logic, but change the dataset to apply the weights’ preprocess directly to the augmented PIL image when using external checkpoints (and only use the explicit resize path in the fallback ImageNet case). This is small, metric-aligned, and should move accuracy upward toward your target while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
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

    FALLBACK_USES_MAPPING = False  # replaced by supervised linear-head calibration

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

_resnet_resize_pil = transforms.Resize(
    (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE),
    interpolation=transforms.InterpolationMode.BILINEAR,
)
_effnet_resize_pil = transforms.Resize(
    (EFFNET_INPUT_SIZE, EFFNET_INPUT_SIZE),
    interpolation=transforms.InterpolationMode.BILINEAR,
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
def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _train_linear_head_on_trainset(
    resnet_model,
    effnet_model,
    resnet_preprocess,
    effnet_preprocess,
    train_csv,
    train_images_dir,
    device=DEVICE,
    batch_size=32,
    num_workers=2,
    max_train_images=6000,
    seed=0,
):
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"train.csv not found at: {train_csv}")
    if not os.path.isdir(train_images_dir):
        raise FileNotFoundError(f"train_images dir not found at: {train_images_dir}")

    df = pd.read_csv(train_csv)
    df = df[["image_id", "label"]].copy()
    df["path"] = df["image_id"].apply(lambda x: os.path.join(train_images_dir, x))
    df = df[df["path"].apply(os.path.isfile)].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    if len(df) > max_train_images:
        idx = rng.choice(len(df), size=max_train_images, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    class CassavaDataset(torch.utils.data.Dataset):
        def __init__(self, df_):
            self.df = df_

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            r = self.df.iloc[i]
            pil = Image.open(r["path"]).convert("RGB")
            xr = resnet_preprocess(pil)
            xe = effnet_preprocess(pil)
            y = int(r["label"])
            return xr, xe, y

    ds = CassavaDataset(df)
    pin = device.type == "cuda"
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    resnet_model.train()
    effnet_model.train()

    _set_requires_grad(resnet_model, False)
    _set_requires_grad(effnet_model, False)

    if isinstance(resnet_model.fc, nn.Linear) and resnet_model.fc.out_features != 5:
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
    if (
        hasattr(effnet_model, "classifier")
        and isinstance(effnet_model.classifier, nn.Sequential)
        and isinstance(effnet_model.classifier[-1], nn.Linear)
        and effnet_model.classifier[-1].out_features != 5
    ):
        effnet_model.classifier[-1] = nn.Linear(
            effnet_model.classifier[-1].in_features, 5
        )

    resnet_model.to(device)
    effnet_model.to(device)

    _set_requires_grad(resnet_model.fc, True)
    _set_requires_grad(effnet_model.classifier[-1], True)

    params = list(resnet_model.fc.parameters()) + list(
        effnet_model.classifier[-1].parameters()
    )
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    epochs = 2 if len(df) >= 3000 else 3

    for ep in range(epochs):
        total = 0
        correct = 0
        running_loss = 0.0
        for xr, xe, y in dl:
            y = torch.as_tensor(y, device=device, dtype=torch.long)
            xr = xr.to(device, dtype=torch.float32, non_blocking=True)
            xe = xe.to(device, dtype=torch.float32, non_blocking=True)

            with torch.no_grad():
                x = resnet_model.conv1(xr)
                x = resnet_model.bn1(x)
                x = resnet_model.relu(x)
                x = resnet_model.maxpool(x)
                x = resnet_model.layer1(x)
                x = resnet_model.layer2(x)
                x = resnet_model.layer3(x)
                x = resnet_model.layer4(x)
                x = resnet_model.avgpool(x)
                feat_r = torch.flatten(x, 1).detach()
            logits_r = resnet_model.fc(feat_r)

            with torch.no_grad():
                feat_e = effnet_model.features(xe)
                feat_e = effnet_model.avgpool(feat_e)
                feat_e = torch.flatten(feat_e, 1).detach()
            logits_e = effnet_model.classifier[-1](feat_e)

            logits = (logits_r + logits_e) / 2.0
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.detach().cpu()) * y.size(0)
            total += y.size(0)
            correct += int((logits.detach().argmax(dim=1) == y).sum().cpu())

        print(
            f"[head-train] epoch {ep+1}/{epochs}  loss={running_loss/max(total,1):.4f}  acc={correct/max(total,1):.4f}  n={total}"
        )

    resnet_model.eval()
    effnet_model.eval()
    return resnet_model, effnet_model


if (
    (not USE_EXTERNAL_CKPTS)
    and (resnet_weights is not None)
    and (effnet_weights is not None)
):
    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
    train_images_dir = os.path.join(os.path.dirname(sample_sub_path), "train_images")
    if not os.path.isdir(train_images_dir):
        train_images_dir = "../input/cassava-leaf-disease-classification/train_images"
    if not os.path.isdir(train_images_dir):
        raise FileNotFoundError(
            f"train_images directory not found at: {train_images_dir}"
        )

    torch.set_grad_enabled(True)
    resnet_model, effnet_model = _train_linear_head_on_trainset(
        resnet_model,
        effnet_model,
        resnet_preprocess,
        effnet_preprocess,
        train_csv=train_csv_path,
        train_images_dir=train_images_dir,
        device=DEVICE,
        batch_size=32 if DEVICE.type == "cuda" else 8,
        num_workers=2,
        max_train_images=6000,
        seed=0,
    )
    torch.set_grad_enabled(False)




## === cell 7
tta_count = 10

image_ids = sample_sub["image_id"].to_numpy()
img_paths = [os.path.join(test_images_path, img_id) for img_id in image_ids]

out_dim = 5

predictions_proba = np.empty((len(img_paths), out_dim), dtype=np.float32)

resnet_model.eval()
effnet_model.eval()


class CassavaTTATestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        resnet_preprocess,
        effnet_preprocess,
        tta_count,
        sub_aug_id,
        sub_aug_hflip,
        resnet_resize_pil,
        effnet_resize_pil,
        use_explicit_resize: bool,
    ):
        self.paths = paths
        self.resnet_preprocess = resnet_preprocess
        self.effnet_preprocess = effnet_preprocess
        self.tta_count = tta_count
        self.sub_aug_id = sub_aug_id
        self.sub_aug_hflip = sub_aug_hflip
        self.resnet_resize_pil = resnet_resize_pil
        self.effnet_resize_pil = effnet_resize_pil
        self.use_explicit_resize = use_explicit_resize

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        base_np = np.asarray(Image.open(p).convert("RGB"))
        tta_r = []
        tta_e = []
        for t in range(self.tta_count):
            aug = self.sub_aug_id if (t % 2 == 0) else self.sub_aug_hflip
            aug_img = aug(image=base_np)["image"]
            if not isinstance(aug_img, np.ndarray):
                aug_img = np.array(aug_img)
            if aug_img.dtype != np.uint8:
                aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
            aug_pil = Image.fromarray(aug_img, mode="RGB")

            if self.use_explicit_resize:
                aug_pil_r = self.resnet_resize_pil(aug_pil)
                aug_pil_e = self.effnet_resize_pil(aug_pil)
                tta_r.append(self.resnet_preprocess(aug_pil_r))
                tta_e.append(self.effnet_preprocess(aug_pil_e))
            else:
                tta_r.append(self.resnet_preprocess(aug_pil))
                tta_e.append(self.effnet_preprocess(aug_pil))

        batch_r = torch.stack(tta_r, dim=0)  # (T, C, H, W)
        batch_e = torch.stack(tta_e, dim=0)
        return idx, batch_r, batch_e


USE_EXPLICIT_RESIZE_FOR_PREPROCESS = not USE_EXTERNAL_CKPTS

ds_test = CassavaTTATestDataset(
    img_paths,
    resnet_preprocess,
    effnet_preprocess,
    tta_count,
    sub_aug_id,
    sub_aug_hflip,
    _resnet_resize_pil,
    _effnet_resize_pil,
    use_explicit_resize=USE_EXPLICIT_RESIZE_FOR_PREPROCESS,
)

pin_memory = DEVICE.type == "cuda"
num_workers = min(4, os.cpu_count() or 2)
dl_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=(8 if DEVICE.type == "cuda" else 2),
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

use_amp = DEVICE.type == "cuda"
amp_dtype = torch.float16

with torch.no_grad():
    for idxs, batch_r, batch_e in dl_test:
        bsz, tta, c, h, w = batch_r.shape
        batch_r = batch_r.view(bsz * tta, c, h, w).to(
            DEVICE, dtype=torch.float32, non_blocking=True
        )

        bsz2, tta2, c2, h2, w2 = batch_e.shape
        batch_e = batch_e.view(bsz2 * tta2, c2, h2, w2).to(
            DEVICE, dtype=torch.float32, non_blocking=True
        )

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits_r = resnet_model(batch_r).view(bsz, tta, out_dim)
            logits_e = effnet_model(batch_e).view(bsz, tta, out_dim)

            proba_r = torch.softmax(logits_r, dim=-1).mean(dim=1)  # (B, 5)
            proba_e = torch.softmax(logits_e, dim=-1).mean(dim=1)  # (B, 5)
            avg_proba = (proba_r + proba_e) / 2.0

        idxs_np = idxs.numpy()
        predictions_proba[idxs_np] = avg_proba.detach().float().cpu().numpy()




## === cell 8
pred_labels = predictions_proba.argmax(axis=1).astype(int)

sub_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
