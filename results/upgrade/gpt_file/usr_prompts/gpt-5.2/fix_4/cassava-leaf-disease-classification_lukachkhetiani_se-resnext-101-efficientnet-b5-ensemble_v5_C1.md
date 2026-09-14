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

0.8618918102145663

# 6. Current score

0.11996

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23057) has done: 'I remove the hard failure on a missing checkpoint and instead fall back to using EfficientNet-B5 ImageNet weights when `b5.pth` is not available, so the notebook can run end-to-end and generate a valid `submission.csv`. I also fix the device/dtype mismatch by ensuring the loaded model weights are moved to the same device as the input tensor after loading. These changes preserve the existing inference pipeline (same architecture, same preprocessing, same argmax-over-softmax decision) while making it runnable and typically much higher-scoring than an untrained random model when no checkpoint is present. Finally, I add a small amount of defensive I/O checking so missing images don’t crash submission generation.'
- What this solution (achieved 0.11996) has done: 'Your current score is low because when the checkpoint is missing, the model’s final classifier layer is randomly initialized, so predictions are essentially random despite using an ImageNet backbone. To move the score toward the target while preserving your core inference pipeline (EfficientNet-B5 + CLAHE + argmax over softmax), I keep the architecture identical but, in the fallback case, I replace the random head with the pretrained 1000-class ImageNet head and map its logits to 5 cassava classes via a fixed grouping (no training, no extra data). I also normalize inputs using the official EfficientNet-B5 ImageNet mean/std so the pretrained weights behave as intended; this is a minimal preprocessing correction that typically yields a large accuracy gain versus unnormalized uint8 tensors. If your `b5.pth` exists, the code path remains effectively unchanged (still loads your checkpoint and uses the 5-class head).'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

CKPT_PATH = "/kaggle/working/b5.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 1
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def build_model(
    num_classes: int = 5,
    use_imagenet_weights: bool = True,
    keep_imagenet_head: bool = False,
):
    weights = (
        torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
        if use_imagenet_weights
        else None
    )
    model = torchvision.models.efficientnet_b5(weights=weights)

    if not keep_imagenet_head:
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


use_imagenet_fallback = not os.path.exists(CKPT_PATH)

keep_imagenet_head = use_imagenet_fallback
model = build_model(
    num_classes=5,
    use_imagenet_weights=True if use_imagenet_fallback else True,
    keep_imagenet_head=keep_imagenet_head,
)

if os.path.exists(CKPT_PATH):
    state = torch.load(CKPT_PATH, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state_dict = state["state_dict"]
    else:
        state_dict = state

    clean_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(
        "Loaded checkpoint. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )
else:
    print(
        f"WARNING: Checkpoint not found at {CKPT_PATH}. Using ImageNet-pretrained EfficientNet-B5 with pretrained head + fixed 1000->5 mapping fallback."
    )

model = model.to(device).eval()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def imagenet_logits_to_5(logits_1000: torch.Tensor) -> torch.Tensor:
    splits = [logits_1000[:, i * 200 : (i + 1) * 200] for i in range(5)]
    logits_5 = torch.stack([torch.logsumexp(s, dim=1) for s in splits], dim=1)  # [B,5]
    return logits_5




## === cell 2
def process(image_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_bgr, (512, 512))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (
        torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    )  # 1x3x512x512
    x = x.to(device)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


if not os.path.exists(SAMPLE_SUB_PATH):
    raise FileNotFoundError(f"sample_submission.csv not found at: {SAMPLE_SUB_PATH}")
if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"test_images directory not found at: {TEST_IMG_DIR}")

sub_df = pd.read_csv(SAMPLE_SUB_PATH)
test_names = sub_df["image_id"].tolist()
test_paths = [os.path.join(TEST_IMG_DIR, n) for n in test_names]

labels = []
for path in tqdm.tqdm(test_paths, total=len(test_paths)):
    img = cv2.imread(path)
    if img is None:
        labels.append(0)
        continue

    topred = process(img)
    with torch.no_grad():
        out = model(topred)
        if use_imagenet_fallback and out.shape[1] == 1000:
            out = imagenet_logits_to_5(out)
        pred = int(torch.argmax(F.softmax(out, dim=1), dim=1).detach().cpu().item())
    labels.append(pred)

sub_df["label"] = labels
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape:", sub_df.shape)
print(sub_df.head())
