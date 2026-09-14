# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7824930747922458

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e3/epoch-3"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

if len(dataset_labels) != 6:
    print(
        "WARNING: Expected 6 classes but found",
        len(dataset_labels),
        "classes:",
        dataset_labels,
    )

print("Classes:", dataset_labels)
print(
    "Test dir exists:",
    os.path.isdir(test_dir),
    "num test images:",
    len(os.listdir(test_dir)),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 1

import timm


def _find_any_torch_weight_candidate(path: str):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None

    exts = (".pth", ".pt", ".bin")
    candidates = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(exts):
                candidates.append(os.path.join(root, f))

    if not candidates:
        return None

    def score(p):
        base = os.path.basename(p).lower()
        pref = 0
        if "best" in base:
            pref -= 5
        if "epoch" in base:
            pref -= 2
        if "model" in base:
            pref -= 1
        return (pref, len(p), p)

    candidates.sort(key=score)
    return candidates[0]


class MultiLabelTorch(nn.Module):
    """
    Minimal PyTorch equivalent for inference:
    EfficientNet-B7 backbone (global pooled) -> 6 independent logits -> sigmoid.
    """

    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b7", pretrained=False, num_classes=0, global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        probs = torch.sigmoid(logits)
        return probs


def load_weights_robust_torch(model: nn.Module, model_path: str):
    cand = _find_any_torch_weight_candidate(model_path)
    if cand is None:
        raise OSError(
            f"Could not load weights from '{model_path}'. No .pth/.pt/.bin found recursively."
        )
    ckpt = torch.load(cand, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        raise OSError(f"Unsupported checkpoint type at {cand}: {type(ckpt)}")

    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print("Loaded weights from:", cand)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("Sample missing:", missing[:10])
    if len(unexpected) > 0:
        print("Sample unexpected:", unexpected[:10])


def preprocess_pil(img: Image.Image, size=(300, 300)):
    img = img.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)  # [0,255]
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr)

    x = x / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x




## === cell 2
if __name__ == "__main__":
    model = MultiLabelTorch(num_classes=len(dataset_labels)).to(device)
    model.eval()

    load_weights_robust_torch(model, model_dir)

    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )

    values = []
    classes = dataset_labels
    threshold = 0.7  # preserve original decision threshold

    with torch.no_grad():
        for fname in images_path_list:
            fpath = os.path.join(test_dir, fname)
            img = Image.open(fpath)
            x = preprocess_pil(img, size=(image_dims[0], image_dims[1]))
            x = x.unsqueeze(0).to(device)

            probs = model(x)[0].detach().cpu().numpy().reshape(-1)
            index_values = [i for i, v in enumerate(probs) if v > threshold]

            if len(index_values) == 0:
                classes_img = "healthy"
            else:
                classes_img = " ".join([str(classes[i]) for i in index_values])

            values.append([fname, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to", out_path)
    print(csv_pd.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3216714755.py in <cell line: 0>()
      5 
      6     # Load weights
----> 7     load_weights_robust_torch(model, model_dir)
      8 
      9     images_path_list = sorted(

/tmp/ipykernel_11/654210056.py in load_weights_robust_torch(model, model_path)
     64     cand = _find_any_torch_weight_candidate(model_path)
     65     if cand is None:
---> 66         raise OSError(
     67             f"Could not load weights from '{model_path}'. No .pth/.pt/.bin found recursively."
     68         )

OSError: Could not load weights from '../input/conve01/eff7-e3/epoch-3'. No .pth/.pt/.bin found recursively.
