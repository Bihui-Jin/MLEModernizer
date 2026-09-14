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

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the runtime failure by making the checkpoint paths resolve against the actual Kaggle dataset directory you have (`/kaggle/data/...` or `/kaggle/input/...`) and by skipping any missing fold checkpoints instead of crashing. I also add a safe fallback so the script still produces a valid `submission.csv` even if none of the external `.pth` files are present (it then use the randomly initialized EfficientNet, which is score-poor but valid). These changes keep the model architecture and inference logic the same; they only harden file I/O and ensure end-to-end execution. Finally, I ensure the submission columns exactly match `image_id,label` and that label dtype is integer.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.88229), so we should cautiously improve accuracy without changing the core model or inference approach. The biggest issue is that your TTA uses *random* augmentations (RandomResizedCrop/ShiftScaleRotate/etc.), which makes predictions noisy and typically worse at test time; switching to deterministic, test-time-safe transforms (resize/center-crop + normalize) keeps the same EfficientNet+B4 + checkpoint ensemble logic but should move score substantially upward. I also fix a subtle tensor conversion bug (feeding ToTensor a NumPy array after Normalize) and add a batched DataLoader inference path (same semantics, faster and more stable) while keeping the same argmax-over-softmax final labeling. Paths and submission format remain unchanged, and the script still falls back safely if checkpoints are missing.'
- What this solution (achieved 0.11024) has done: 'Your current score (0.61099) is far below the target (0.88229), so we should improve accuracy with the smallest changes that don’t alter the core model/ensemble logic. The biggest likely cause is a mismatch between the training checkpoints’ expected input preprocessing and the current inference preprocessing (512 center-crop + ImageNet mean/std), which can heavily degrade accuracy. I keep the EfficientNet-B4 fold ensemble exactly as-is, but switch the test transform to the common EfficientNet inference recipe (resize shorter side then center-crop to 380, ImageNet normalize) and ensure evaluation uses `torch.inference_mode()` for stability. The script still robustly skip missing checkpoints and always write a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11024) is far below the target (0.88229), so we should make a small, low-risk fix that plausibly restores the intended pretrained-checkpoint inference behavior without changing the model/ensemble logic. The most likely cause is that the EfficientNet-B4 checkpoints were trained with a different input scaling than the current test pipeline (albumentations Normalize to ImageNet mean/std), so I switch the preprocessing to the standard torchvision EfficientNet normalization (mean/std of 0.5/0.5) while keeping the same 380 resize+center-crop and the same fold-mean logits + softmax+argmax. I also keep everything else identical (architecture, ensemble, inference) and still robustly skip missing checkpoints and always write a valid `submission.csv`. This should materially increase accuracy toward the target if the checkpoint expects torchvision EfficientNet-style normalization.'

# 9. Code solution

## === cell 0
import os
import warnings

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



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v2-8/model(15).pth"
effnet_model_folds_path = [
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_0_11.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_1_10.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_2_13.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_3_12.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_4_11.pth",
]

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 3
effnet_model = models.efficientnet_b4(weights=None)
in_features = effnet_model.classifier[1].in_features
effnet_model.classifier[1] = nn.Linear(in_features, 5)
effnet_model = effnet_model.to(device)




## === cell 4
def _clean_state_dict(state_dict):
    """Make checkpoint loading robust to common wrappers (DataParallel/Lightning)."""
    if not isinstance(state_dict, dict):
        return state_dict
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v
    return cleaned




## === cell 5
_EFFNET_B4_CROP = 380

sub_aug = A.Compose(
    [
        A.SmallestMaxSize(max_size=_EFFNET_B4_CROP),
        A.CenterCrop(height=_EFFNET_B4_CROP, width=_EFFNET_B4_CROP),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 6
def predict_one_image(model, img_path, tta_count=1):
    image_pred = None
    for _ in range(tta_count):
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = sub_aug(image=img)["image"]  # float32 HWC normalized
        x = torch.from_numpy(img).permute(2, 0, 1).float().to(device)  # CHW float32

        with torch.inference_mode():
            out = model(x.unsqueeze(0))
        out_np = out.detach().cpu().numpy()

        if image_pred is None:
            image_pred = out_np
        else:
            image_pred += out_np

    image_pred /= float(tta_count)
    return image_pred[0]  # shape: (5,)




## === cell 7
def _resolve_path(p: str) -> str:
    if os.path.isabs(p) and os.path.exists(p):
        return p

    candidates = []
    candidates.append(p)

    if p.startswith("../input/"):
        candidates.append(os.path.join("/kaggle/input", p[len("../input/") :]))
        candidates.append(os.path.join("/kaggle/data", p[len("../input/") :]))

    candidates.append(os.path.join("/kaggle/input", p.lstrip("./")))
    candidates.append(os.path.join("/kaggle/data", p.lstrip("./")))

    for c in candidates:
        if os.path.exists(c):
            return c
    return p  # return original; caller may handle missing


sample_sub_path = _resolve_path(sample_sub_path)
test_images_path = _resolve_path(test_images_path)

print("Resolved sample_sub_path:", sample_sub_path)
print("Resolved test_images_path:", test_images_path)
print("test_images exists:", os.path.isdir(test_images_path))

sample_sub = pd.read_csv(sample_sub_path)
assert {"image_id", "label"}.issubset(sample_sub.columns)

tta_count = 1  # keep identical semantics as provided code




## === cell 8
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
        img = np.array(img)
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




## === cell 9
resolved_fold_paths = []
for p in effnet_model_folds_path:
    rp = _resolve_path(p)
    if os.path.exists(rp):
        resolved_fold_paths.append(rp)
    else:
        print("WARNING: missing checkpoint (skipping):", p, "->", rp)

fold_predictions = []

image_ids = sample_sub["image_id"].values

if len(resolved_fold_paths) == 0:
    print(
        "WARNING: No fold checkpoints found. Will run inference with randomly initialized EfficientNet (valid submission, low score)."
    )
    effnet_model.eval()
    preds = predict_dataset_logits(
        effnet_model, image_ids, batch_size=64, num_workers=2
    )
    fold_predictions = preds
else:
    for model_path in resolved_fold_paths:
        ckpt = torch.load(model_path, map_location="cpu")
        sd = _clean_state_dict(ckpt)

        missing, unexpected = effnet_model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                f"Loaded {os.path.basename(model_path)} with missing={len(missing)} unexpected={len(unexpected)}"
            )

        effnet_model.to(device)
        effnet_model.eval()

        preds = predict_dataset_logits(
            effnet_model, image_ids, batch_size=64, num_workers=2
        )
        fold_predictions.append(preds)

    fold_predictions = np.mean(np.stack(fold_predictions, axis=0), axis=0)  # (N, 5)

labels = softmax(fold_predictions, axis=1).argmax(axis=1)

sub_df = pd.DataFrame({"image_id": image_ids, "label": labels.astype(np.int64)})
sub_df = sub_df[["image_id", "label"]]
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Unique labels:", np.unique(sub_df["label"].values))
