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

0.8649138712601995

# 6. Current score

0.56278

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09641) has done: 'I fix the missing `efficientnet_pytorch` dependency by removing the offline wheel install cell and replacing the model with `torchvision.models.efficientnet_b4`, then load the provided checkpoint with a compatible key-mapping so inference can run. I also fix the broken cell numbering/order and ensure the model is actually created before prediction. To prevent runtime issues in modern PyTorch, I load the checkpoint on the correct device and run inference under `torch.inference_mode()` with proper tensor shapes. Finally, I keep the same core idea (EfficientNet-B4 + CenterCrop/Resize/Normalize + argmax labels) and write a valid `submission.csv`.'
- What this solution (achieved 0.11211) has done: 'I fix the failing path logic by removing the hard dependency on the missing external checkpoint and instead use a robust fallback that runs with the competition’s available files. To move the very low score (0.096) toward the target, I minimally switch the model to use torchvision’s ImageNet-pretrained EfficientNet-B4 weights (same architecture family) and keep the same preprocessing/inference loop semantics (center-crop/resize/normalize + argmax). I also make the input normalization correct for Albumentations (keep it in float HWC) to avoid dtype/range issues that can silently hurt predictions. Finally, I keep submission formatting aligned to `sample_submission.csv` and always write `submission.csv`.'
- What this solution (achieved 0.14985) has done: 'Your score is far below the target, so the most likely issue is not the model family but a mismatch between how inputs are normalized and what EfficientNet-B4 expects. I make the smallest change that preserves your core inference logic: replace `transforms.ToTensor()` (which re-scales again and effectively double-normalizes) with a direct HWC->CHW tensor conversion so Albumentations’ normalization is applied exactly once. I also switch to using the EfficientNet-B4 weights’ own recommended preprocessing stats (still Normalize+CenterCrop/Resize, same semantics) to better match the pretrained model, which should move accuracy upward toward the target. Everything else (architecture, argmax labels, loop structure, submission format/path) stays the same and it still write `submission.csv`.'
- What this solution (achieved 0.56278) has done: 'Your current score is far below the target, so we should make the smallest changes that legitimately improve accuracy without changing the model family or inference semantics. The biggest likely issue is using a randomly initialized 5-class head when the external checkpoint is missing, which makes predictions nearly random; I instead keep the ImageNet-pretrained backbone and simply keep the classifier at 1000 classes (no fine-tuning) and map its outputs to 5 cassava classes using a fixed, deterministic nearest-prototype mapping computed from the training set. This preserves the “EfficientNet-B4 + center-crop/resize/normalize + argmax” core logic, adds no training loops, and uses only provided files. I also ensure the preprocessing resolution matches EfficientNet-B4’s default (380) to better align with pretrained features, which should move the score upward toward your target band.'

# 9. Code solution

## === cell 0
import os
import json

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 1
model_path = "../input/en-b4-tta-calr-15-v2/model(13).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

assert os.path.exists(sample_sub_path), f"Missing sample submission: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing test images dir: {test_images_path}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"
assert os.path.isdir(
    train_images_path
), f"Missing train images dir: {train_images_path}"

has_ckpt = os.path.exists(model_path)
print("Checkpoint exists:", has_ckpt, "| path:", model_path)



## === cell 2
if has_ckpt:
    model = models.efficientnet_b4(weights=None)
    eff_weights = None
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
else:
    eff_weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=eff_weights)

model.to(device)

if has_ckpt:
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    def _remap_keys_for_torchvision_efficientnet(state_dict):
        """
        Map common EfficientNet-PyTorch naming (._fc / _conv_stem / _bn0 / _blocks / _bn1 / _conv_head)
        to torchvision EfficientNet naming (classifier.1 / features.*).
        This is minimal and only to make checkpoint loadable for inference.
        """
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ("module.", "model.", "net."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]

            nk = nk.replace("_fc.", "classifier.1.")
            nk = nk.replace("_conv_stem.", "features.0.0.")
            nk = nk.replace("_bn0.", "features.0.1.")
            nk = nk.replace("_blocks.", "features.1.")
            nk = nk.replace("_conv_head.", "features.8.0.")
            nk = nk.replace("_bn1.", "features.8.1.")

            new_sd[nk] = v
        return new_sd

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        print(
            "Direct load_state_dict failed; attempting key remap. Error:", str(e)[:300]
        )
        state = _remap_keys_for_torchvision_efficientnet(state)
        missing, unexpected = model.load_state_dict(state, strict=False)

    print("Checkpoint loaded.")
    print("Missing keys (count):", len(missing))
    print("Unexpected keys (count):", len(unexpected))

model.eval()



## === cell 3
if eff_weights is not None:
    mean = eff_weights.transforms().mean
    std = eff_weights.transforms().std
    input_size = 380
else:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_size = 256

sub_aug = albumentations.Compose(
    [
        albumentations.CenterCrop(512, 512, p=1.0),
        albumentations.Resize(input_size, input_size),
        albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
    ],
    p=1.0,
)




## === cell 4
def _load_image_tensor(img_path: str) -> torch.Tensor:
    image = np.array(Image.open(img_path).convert("RGB"))
    image = sub_aug(image=image)["image"]  # float32 HWC normalized
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    return image


cassava_prototypes = None  # shape [5, 1000] in fallback mode

if not has_ckpt:
    train_df = pd.read_csv(train_csv_path)
    per_class = 64
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    feats_sum = torch.zeros(5, 1000, device=device)
    feats_cnt = torch.zeros(5, device=device)

    with torch.inference_mode():
        for cls in range(5):
            rows = train_df[train_df["label"] == cls].head(per_class)
            for img_id in rows["image_id"].tolist():
                p = os.path.join(train_images_path, img_id)
                x = _load_image_tensor(p).to(device)
                logits = model(x.unsqueeze(0)).squeeze(0)  # [1000]
                feats_sum[cls] += logits
                feats_cnt[cls] += 1

    feats_cnt = torch.clamp(feats_cnt, min=1.0).unsqueeze(1)
    cassava_prototypes = feats_sum / feats_cnt  # [5,1000]
    cassava_prototypes = F.normalize(cassava_prototypes, dim=1)
    print(
        "Built cassava prototypes from train set. counts:",
        feats_cnt.squeeze(1).to("cpu").tolist(),
    )



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

tta_count = 1  # keep identical semantics to original (no extra TTA)

predictions = []
with torch.inference_mode():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        image_pred = None

        for _ in range(tta_count):
            x = _load_image_tensor(img_path).to(device)

            outputs = model(x.unsqueeze(0))  # [1, C]

            if cassava_prototypes is not None:
                o = F.normalize(outputs, dim=1)  # [1,1000]
                outputs = o @ cassava_prototypes.T  # [1,5]

            image_pred = outputs if image_pred is None else (image_pred + outputs)

        image_pred = image_pred / tta_count
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with rows:", len(sub_df))
print("Submission columns:", list(sub_df.columns))
