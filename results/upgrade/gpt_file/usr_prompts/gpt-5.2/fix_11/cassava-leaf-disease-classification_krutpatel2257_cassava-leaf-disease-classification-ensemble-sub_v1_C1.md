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

0.8822907222725899

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the runtime failure by making the checkpoint paths resolve against the actual Kaggle dataset directory you have (`/kaggle/data/...` or `/kaggle/input/...`) and by skipping any missing fold checkpoints instead of crashing. I also add a safe fallback so the script still produces a valid `submission.csv` even if none of the external `.pth` files are present (it then use the randomly initialized EfficientNet, which is score-poor but valid). These changes keep the model architecture and inference logic the same; they only harden file I/O and ensure end-to-end execution. Finally, I ensure the submission columns exactly match `image_id,label` and that label dtype is integer.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.88229), so we should cautiously improve accuracy without changing the core model or inference approach. The biggest issue is that your TTA uses *random* augmentations (RandomResizedCrop/ShiftScaleRotate/etc.), which makes predictions noisy and typically worse at test time; switching to deterministic, test-time-safe transforms (resize/center-crop + normalize) keeps the same EfficientNet+B4 + checkpoint ensemble logic but should move score substantially upward. I also fix a subtle tensor conversion bug (feeding ToTensor a NumPy array after Normalize) and add a batched DataLoader inference path (same semantics, faster and more stable) while keeping the same argmax-over-softmax final labeling. Paths and submission format remain unchanged, and the script still falls back safely if checkpoints are missing.'
- What this solution (achieved 0.11024) has done: 'Your current score (0.61099) is far below the target (0.88229), so we should improve accuracy with the smallest changes that don’t alter the core model/ensemble logic. The biggest likely cause is a mismatch between the training checkpoints’ expected input preprocessing and the current inference preprocessing (512 center-crop + ImageNet mean/std), which can heavily degrade accuracy. I keep the EfficientNet-B4 fold ensemble exactly as-is, but switch the test transform to the common EfficientNet inference recipe (resize shorter side then center-crop to 380, ImageNet normalize) and ensure evaluation uses `torch.inference_mode()` for stability. The script still robustly skip missing checkpoints and always write a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11024) is far below the target (0.88229), so we should make a small, low-risk fix that plausibly restores the intended pretrained-checkpoint inference behavior without changing the model/ensemble logic. The most likely cause is that the EfficientNet-B4 checkpoints were trained with a different input scaling than the current test pipeline (albumentations Normalize to ImageNet mean/std), so I switch the preprocessing to the standard torchvision EfficientNet normalization (mean/std of 0.5/0.5) while keeping the same 380 resize+center-crop and the same fold-mean logits + softmax+argmax. I also keep everything else identical (architecture, ensemble, inference) and still robustly skip missing checkpoints and always write a valid `submission.csv`. This should materially increase accuracy toward the target if the checkpoint expects torchvision EfficientNet-style normalization.'
- What this solution (achieved 0.11024) has done: 'Your current score (0.11584) is far below the target (0.88229), so we should make a minimal inference-only fix that most plausibly restores the checkpoints’ intended preprocessing rather than changing the model/ensemble logic. The largest red flag is the normalization: torchvision EfficientNet-B4 checkpoints are almost always trained with ImageNet mean/std, and using 0.5/0.5 can collapse accuracy. I switch the test transform normalization back to ImageNet mean/std while keeping the same resize+center-crop, same EfficientNet-B4 architecture, same fold-averaging, and same argmax decoding. I also set deterministic seeds to stabilize results (no semantic change) and keep the same submission writing.'
- What this solution (achieved 0.11024) has done: 'Your current score (0.11024) is so far below the target (0.88229) that the most likely issue is not “model quality” but that you’re effectively submitting random predictions because the fold checkpoints are not being found/loaded correctly. I make a minimal, I/O-only fix: resolve checkpoint paths by *searching the actual Kaggle input/data directories* for the filename (and common subfolders), instead of relying on hardcoded dataset slugs. I also ensure we build a fresh EfficientNet instance per fold load (same architecture) so partial/mismatched loads don’t compound across folds, and keep the exact same preprocessing and argmax decoding semantics. This should move the score sharply upward toward the target if the intended checkpoints exist in your environment.'
- What this solution (achieved 0.11024) has done: 'Your score is extremely low versus the target, which strongly suggests the fold checkpoints are not being loaded (or are being loaded incorrectly), causing near-random predictions. I make a minimal, inference-only fix that (1) correctly finds the fold checkpoint files by also searching common locations under the Cassava dataset directory and (2) loads checkpoints more robustly by handling the common “saved full model object” case in addition to plain `state_dict` dicts. I also force RGB contiguous arrays before Albumentations and enable safe `persistent_workers` for more stable dataloader behavior, without changing model architecture, loss, or prediction semantics. The rest (EfficientNet-B4, resize+center-crop, ImageNet normalize, fold-logit averaging, softmax+argmax, submission format) stays the same.'
- What this solution (achieved 0.11024) has done: 'Your score (0.11024) is far below the target (0.88229), so the most likely problem is that the fold checkpoints are not being loaded correctly (or not being found), resulting in essentially random predictions. I make a minimal, inference-only fix to checkpoint loading: handle common nested keys like `model_state_dict`/`net` and the “full model object” case, and try a strict load first (fall back to non-strict only if needed) so we don’t silently run with mismatched heads. I also ensure we only average predictions from successfully loaded checkpoints (and otherwise fall back), and I keep the same EfficientNet-B4 architecture, preprocessing, and argmax decoding. This should restore the intended ensemble behavior and move accuracy substantially toward the target if the checkpoints are present.'
- What this solution (achieved 0.11024) has done: 'Your score (0.11024) is far below the target (0.88229), so the most likely issue is still that you are not actually using the intended trained weights (or you’re loading them into a mismatched head), resulting in near-random predictions. I keep the exact same EfficientNet-B4 fold-ensemble inference and the same resize+center-crop + ImageNet normalization, but I (1) automatically discover any `.pth` fold checkpoints present in the environment and (2) make checkpoint loading robust to the very common “classifier named differently” case by remapping classifier keys only when shapes match. This is a minimal, inference-only fix aimed at restoring the original trained model behavior without changing architecture, loss, or prediction decoding. The script still always produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings
import random
import glob

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v2-8/model(15).pth"
effnet_model_folds_path = [
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENTIENT_NET_B4_0_11.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENTIENT_NET_B4_1_10.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENTIENT_NET_B4_2_13.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENTIENT_NET_B4_3_12.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENTIENT_NET_B4_4_11.pth",
]

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 3
def _clean_state_dict(state_dict):
    """Make checkpoint loading robust to common wrappers (DataParallel/Lightning)."""
    if not isinstance(state_dict, dict):
        return state_dict
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "encoder.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v
    return cleaned




## === cell 4
def build_effnet_b4(num_classes=5):
    m = models.efficientnet_b4(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m




## === cell 5
_EFFNET_B4_CROP = 380

sub_aug = A.Compose(
    [
        A.SmallestMaxSize(max_size=_EFFNET_B4_CROP),
        A.CenterCrop(height=_EFFNET_B4_CROP, width=_EFFNET_B4_CROP),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 6
def _resolve_path(p: str) -> str:
    if os.path.isabs(p) and os.path.exists(p):
        return p

    candidates = [p]

    if p.startswith("../input/"):
        rel = p[len("../input/") :]
        candidates.append(os.path.join("/kaggle/input", rel))
        candidates.append(os.path.join("/kaggle/data", rel))

    candidates.append(os.path.join("/kaggle/input", p.lstrip("./")))
    candidates.append(os.path.join("/kaggle/data", p.lstrip("./")))

    for c in candidates:
        if os.path.exists(c):
            return c
    return p  # caller may handle missing


def _resolve_checkpoint_path(p: str) -> str:
    rp = _resolve_path(p)
    if os.path.exists(rp):
        return rp

    base = os.path.basename(p)
    search_roots = [
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    common_patterns = [
        "*/" + base,
        "*/*/" + base,
        "*/*/*/" + base,
        "*/*/*/*/" + base,
        "**/" + base,
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pat in common_patterns:
            hits = glob.glob(os.path.join(root, pat), recursive=("**/" in pat))
            if hits:
                hits = [h for h in hits if os.path.isfile(h)]
                if hits:
                    hits.sort()
                    return hits[0]
    return rp


def _auto_discover_effnet_b4_fold_checkpoints():
    roots = ["/kaggle/input", "/kaggle/data"]
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        patterns = [
            "**/*B4*.pth",
            "**/*b4*.pth",
            "**/*efficient*net*b4*.pth",
            "**/*EFFICIENT*NET*B4*.pth",
        ]
        for pat in patterns:
            hits = glob.glob(os.path.join(root, pat), recursive=True)
            for h in hits:
                if os.path.isfile(h):
                    found.append(h)

    def score_path(p):
        bn = os.path.basename(p).lower()
        s = 0
        if "fold" in bn:
            s += 2
        if any(f"_{i}_" in bn for i in range(5)):
            s += 3
        if "efficient" in bn and "b4" in bn:
            s += 2
        return s

    found = sorted(list(set(found)), key=lambda x: (-score_path(x), x))
    return found


sample_sub_path = _resolve_path(sample_sub_path)
test_images_path = _resolve_path(test_images_path)

print("Resolved sample_sub_path:", sample_sub_path)
print("Resolved test_images_path:", test_images_path)
print("test_images exists:", os.path.isdir(test_images_path))

sample_sub = pd.read_csv(sample_sub_path)
assert {"image_id", "label"}.issubset(sample_sub.columns)

tta_count = 1  # keep identical semantics as provided code




## === cell 7
class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, images_dir, aug):
        self.image_ids = list(image_ids)
        self.images_dir = images_dir
        self.aug = aug

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        img = np.ascontiguousarray(np.array(img))
        img = self.aug(image=img)["image"]  # float32 HWC normalized
        x = torch.from_numpy(img).permute(2, 0, 1).float()  # CHW float32
        return x, image_id


def predict_dataset_logits(model, image_ids, batch_size=64, num_workers=2):
    ds = CassavaTestDataset(
        image_ids=image_ids, images_dir=test_images_path, aug=sub_aug
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    preds = np.zeros((len(ds), 5), dtype=np.float32)
    model.eval()
    start = 0
    with torch.inference_mode():
        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).detach().cpu().numpy().astype(np.float32)
            end = start + out.shape[0]
            preds[start:end] = out
            start = end
    return preds




## === cell 8
def _extract_state_dict_from_checkpoint(ckpt):
    """
    Load-only robustness:
    - supports 'full model' saved objects, plus dict checkpoints with nested keys.
    """
    if hasattr(ckpt, "state_dict") and callable(getattr(ckpt, "state_dict")):
        try:
            sd = ckpt.state_dict()
            sd = _clean_state_dict(sd)
            if isinstance(sd, dict) and len(sd) > 0:
                return sd
        except Exception:
            pass

    if not isinstance(ckpt, dict):
        return None

    direct = _clean_state_dict(ckpt)
    if isinstance(direct, dict) and any(
        isinstance(v, torch.Tensor) for v in direct.values()
    ):
        return direct

    for key in (
        "state_dict",
        "model_state_dict",
        "net_state_dict",
        "model",
        "net",
        "weights",
        "ema_state_dict",
    ):
        if key in ckpt:
            cand = ckpt[key]
            if hasattr(cand, "state_dict") and callable(getattr(cand, "state_dict")):
                try:
                    cand = cand.state_dict()
                except Exception:
                    cand = None
            cand = _clean_state_dict(cand)
            if isinstance(cand, dict) and any(
                isinstance(v, torch.Tensor) for v in cand.values()
            ):
                return cand
    return None


def _remap_effnet_classifier_keys_if_needed(model, state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    msd = model.state_dict()
    tgt_w = "classifier.1.weight"
    tgt_b = "classifier.1.bias"
    if tgt_w in state_dict and tgt_b in state_dict:
        return state_dict

    candidates = [
        ("_fc.weight", "_fc.bias"),
        ("fc.weight", "fc.bias"),
        ("head.weight", "head.bias"),
        (
            "classifier.weight",
            "classifier.bias",
        ),  # sometimes classifier is Linear directly
        ("classifier.0.weight", "classifier.0.bias"),
    ]
    out = dict(state_dict)
    for cw, cb in candidates:
        if cw in state_dict and cb in state_dict and tgt_w in msd and tgt_b in msd:
            if tuple(state_dict[cw].shape) == tuple(msd[tgt_w].shape) and tuple(
                state_dict[cb].shape
            ) == tuple(msd[tgt_b].shape):
                out[tgt_w] = state_dict[cw]
                out[tgt_b] = state_dict[cb]
                return out
    return state_dict


def _load_effnet_checkpoint_into_model(model, state_dict):
    """
    try strict=True first; if that fails, fall back to strict=False.
    """
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        return True, missing, unexpected, True
    except Exception:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return True, missing, unexpected, False


resolved_fold_paths = []
for p in effnet_model_folds_path:
    rp = _resolve_checkpoint_path(p)
    if os.path.exists(rp):
        resolved_fold_paths.append(rp)

if len(resolved_fold_paths) == 0:
    auto_found = _auto_discover_effnet_b4_fold_checkpoints()
    if len(auto_found):
        print(
            f"INFO: No provided fold checkpoints resolved; auto-discovered {len(auto_found)} candidate .pth files."
        )
        resolved_fold_paths = auto_found
    else:
        print("WARNING: No EfficientNet-B4 checkpoints auto-discovered either.")

image_ids = sample_sub["image_id"].values
usable_fold_logits = []

if len(resolved_fold_paths) == 0:
    print(
        "WARNING: No fold checkpoints found. Will run inference with randomly initialized EfficientNet (valid submission, low score)."
    )
    effnet_model = build_effnet_b4(num_classes=5).to(device)
    effnet_model.eval()
    fold_logits = predict_dataset_logits(
        effnet_model, image_ids, batch_size=64, num_workers=2
    )
else:
    for model_path in resolved_fold_paths:
        effnet_model = build_effnet_b4(num_classes=5)
        ckpt = torch.load(model_path, map_location="cpu")
        sd = _extract_state_dict_from_checkpoint(ckpt)

        if sd is None:
            continue

        sd = _remap_effnet_classifier_keys_if_needed(effnet_model, sd)
        sd = _clean_state_dict(sd)

        ok, missing, unexpected, was_strict = _load_effnet_checkpoint_into_model(
            effnet_model, sd
        )

        if (not was_strict) and (
            ("classifier.1.weight" in missing) or ("classifier.1.bias" in missing)
        ):
            continue
        if (not was_strict) and (len(missing) > 200):
            continue

        effnet_model.to(device)
        effnet_model.eval()

        preds = predict_dataset_logits(
            effnet_model, image_ids, batch_size=64, num_workers=2
        )
        usable_fold_logits.append(preds)

    if len(usable_fold_logits) == 0:
        print(
            "WARNING: All found checkpoints were unusable. Falling back to randomly initialized EfficientNet."
        )
        effnet_model = build_effnet_b4(num_classes=5).to(device)
        effnet_model.eval()
        fold_logits = predict_dataset_logits(
            effnet_model, image_ids, batch_size=64, num_workers=2
        )
    else:
        fold_logits = np.mean(np.stack(usable_fold_logits, axis=0), axis=0)  # (N, 5)

labels = softmax(fold_logits, axis=1).argmax(axis=1)

sub_df = pd.DataFrame({"image_id": image_ids, "label": labels.astype(np.int64)})
sub_df = sub_df[["image_id", "label"]]
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Unique labels:", np.unique(sub_df["label"].values))
print("Found fold checkpoints:", len(resolved_fold_paths))
print(
    "Usable fold checkpoints:",
    (len(usable_fold_logits) if isinstance(usable_fold_logits, list) else 0),
)
if len(resolved_fold_paths):
    print("Example checkpoint path:", resolved_fold_paths[0])
