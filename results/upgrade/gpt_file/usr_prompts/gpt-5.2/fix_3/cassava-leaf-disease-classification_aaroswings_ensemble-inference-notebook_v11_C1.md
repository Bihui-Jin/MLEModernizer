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

0.8733756421879721

# 6. Current score

0.44544

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09155) has done: 'I fix the albumentations import breakage by switching from the removed `albumentations.augmentations.transforms` wildcard import to explicit top-level albumentations transforms (this also restores `CenterCrop`/`Normalize`). I also make model loading robust: your referenced pickle files are not available in the provided data paths, so the code currently can’t define `models`; I fall back to a standard torchvision ResNeXt101 model (same family as your intended model) so inference can run end-to-end without external inputs. Finally, I ensure the test dataset iterates images in a deterministic order and that the generated `submission.csv` matches `sample_submission.csv` exactly in length and ordering, preventing the “same length as answers” error.'
- What this solution (achieved 0.44544) has done: 'Your score is low because the current code is doing inference with an ImageNet-pretrained ResNeXt that has a randomly initialized 5-class head (since the intended pickled weights are missing), so predictions are essentially random. To move toward the target score with minimal core-logic changes, I (1) keep the same ResNeXt family and ensemble/inference flow, but (2) load a readily-available, competition-relevant pretrained checkpoint (ResNeXt101_32x8d WSL weights) if present inside the Kaggle dataset, and (3) make preprocessing match what those weights expect (resize to 512 then center-crop + ImageNet mean/std). This keeps your model architecture and inference approach intact while making predictions meaningful and should increase accuracy substantially toward the target band. The script still writes a valid `submission.csv` aligned exactly to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")



## === cell 1
batch_size = 32
valid_input_size = (512, 512)

DATA_ROOT = Path("/kaggle/input/cassava-leaf-disease-classification")
test_img_path = DATA_ROOT / "test_images"
sample_sub_path = DATA_ROOT / "sample_submission.csv"

assert test_img_path.exists(), f"Missing test images path: {test_img_path}"
assert sample_sub_path.exists(), f"Missing sample submission: {sample_sub_path}"



## === cell 2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

test_tfms = A.Compose(
    [
        A.LongestMaxSize(max_size=max(valid_input_size), always_apply=True),
        A.PadIfNeeded(
            min_height=valid_input_size[0],
            min_width=valid_input_size[1],
            border_mode=0,
            value=(0, 0, 0),
            always_apply=True,
        ),
        A.CenterCrop(
            height=valid_input_size[0], width=valid_input_size[1], always_apply=True
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, path, tfms):
        super().__init__()
        path = Path(path)
        self.files = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        self.tfms = tfms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = np.array(Image.open(file).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return file.name, img


test_ds = TestDataset(test_img_path, test_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=min(8, os.cpu_count() or 1),
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)



## === cell 3
import torchvision


def _make_fallback_model(num_classes: int = 5) -> nn.Module:
    m = torchvision.models.resnext101_32x8d(
        weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
    )
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _try_load_state_dict_into_resnext101(
    checkpoint_path: Path, num_classes: int = 5
) -> nn.Module:
    """
    Minimal, robust loader for common WSL checkpoints (often provided as state_dict with various keys).
    Keeps architecture as ResNeXt101_32x8d; replaces fc to 5 classes and ignores mismatched fc weights.
    """
    obj = torch.load(checkpoint_path, map_location="cpu")
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        sd = obj
    else:
        if hasattr(obj, "eval") and hasattr(obj, "to"):
            return obj
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v

    m = torchvision.models.resnext101_32x8d(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)

    for bad_key in ["fc.weight", "fc.bias"]:
        if bad_key in cleaned:
            cleaned.pop(bad_key, None)

    missing, unexpected = m.load_state_dict(cleaned, strict=False)
    return m


model_fnames = [
    "../input/cassava-notebook-15-resnext-models/resnext101wsl_epoch_6.pickle",
    "../input/cassava-notebook-16-models/resnext101wsl_epoch_8.pickle",
]

models = []
for x in model_fnames:
    try:
        p = Path(x)
        if p.exists():
            mdl = torch.load(p, map_location="cpu")
            models.append(mdl)
    except Exception:
        pass

if len(models) == 0:
    search_roots = [Path("/kaggle/input"), Path("/kaggle/data/input")]
    candidate_paths = []
    for root in search_roots:
        if root.exists():
            try:
                for pat in (
                    "*wsl*resnext101*.pth",
                    "*resnext101*wsl*.pth",
                    "*wsl*.pth",
                    "*resnext101*.pth",
                ):
                    candidate_paths.extend(list(root.rglob(pat)))
            except Exception:
                pass

    def _score_path(p: Path) -> int:
        s = p.name.lower()
        score = 0
        score += 10 if "wsl" in s else 0
        score += 10 if "resnext101" in s else 0
        score += 2 if s.endswith(".pth") else 0
        score += 1 if s.endswith(".pt") else 0
        return score

    candidate_paths = sorted(set(candidate_paths), key=_score_path, reverse=True)

    loaded = False
    for cp in candidate_paths[:10]:
        try:
            mdl = _try_load_state_dict_into_resnext101(cp, num_classes=5)
            models = [mdl]
            loaded = True
            print("Loaded checkpoint:", str(cp))
            break
        except Exception:
            continue

    if not loaded:
        models = [_make_fallback_model(num_classes=5)]
        print(
            "No suitable checkpoint found; using ImageNet-pretrained backbone with random 5-class head."
        )

models = [m.to(device).eval() for m in models]




## === cell 4
class EnsemblePredictor:
    def __init__(self, models):
        super().__init__()
        self.models = models

    def predict_on_loader(self, loader):
        predictions = []
        filenames = []

        for file, img in loader:
            img = img.to(device, non_blocking=True)
            pred = None
            for model in self.models:
                with torch.no_grad():
                    out = model(img)
                    out = out.float().detach().cpu().numpy()
                pred = out if pred is None else (pred + out)

            predictions.extend([int(np.argmax(x)) for x in pred])
            filenames.extend(list(file))

        return predictions, filenames


predictor = EnsemblePredictor(models)
predictions, filenames = predictor.predict_on_loader(test_loader)



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(filenames, predictions))

missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if missing:
    for img_id in missing:
        pred_map[img_id] = 0

submission_df = sample_sub.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map).astype(int)

out_path = Path("submission.csv")
submission_df.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print("Submission shape:", submission_df.shape)
print(submission_df.head())
