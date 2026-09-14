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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.7881535207011182

# 6. Current score

0.33931

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14013) has done: 'I fix the pipeline so it always produces a valid `submission.csv` even when no external checkpoint files are available in `../input` (the current hard failure). To preserve the core inference logic, I keep the same model, transforms, and inference loop, but add a safe fallback: use a pretrained timm backbone when checkpoints are missing and avoid raising an exception. I also fix the test-time augmentation bug (your “inference” transforms are random), making inference deterministic and improving accuracy stability by switching to a pure resize+normalize transform at test time. Finally, I guard against empty/invalid prediction arrays so `np.argmax(..., axis=1)` never crashes and submission rows stay aligned with `sample_submission.csv`.'
- What this solution (achieved 0.33931) has done: 'Your current score (0.14013) is far below the target (0.78815), so we should increase performance with minimal, low-risk changes while keeping your inference-only core logic intact. The main issue is that when no checkpoints are found you fall back to ImageNet-pretrained weights but still use a custom 5-class randomly initialized head, which makes predictions near-random; instead we should use timm’s built-in pretrained 5-class cassava weights if available. I change the fallback to use `timm.create_model(..., pretrained=True, num_classes=5)` first (this uses timm’s pretrained classifier when it exists), and only if that fails it revert to your current ImageNet-backbone + random head behavior. I also wrap inference in `torch.cuda.amp.autocast` to stabilize/accelerate on GPU without changing semantics, and keep submission formatting/alignment unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import cv2
import timm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch as Apy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

package_path = "../input/pytorchimagemodels"
if os.path.isdir(package_path):
    sys.path.append(package_path)



## === cell 1
config = {
    "fold_num": 1,
    "seed": 719,
    "model_arch": "resnext50d_32x4d",
    "img_size": 512,
    "valid_bs": 256,
    "num_workers": 4,
    "accum_iter": 1,
    "verbose_step": 1,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}



## === cell 2
DATA_DIR = "../input/cassava-leaf-disease-classification"
submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
submission.head()




## === cell 3
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        raise FileNotFoundError(f"Failed to read image at: {path}")
    return im_bgr[:, :, ::-1]


seed_everything(config["seed"])




## === cell 4
class CassavaDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        if self.output_label:
            target = int(self.df.iloc[index]["label"])

        path = "{}/{}".format(
            self.data_root.rstrip("/"), self.df.iloc[index]["image_id"]
        )
        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        if self.output_label:
            return img, target
        else:
            return img




## === cell 5
def get_inference_transforms():
    return A.Compose(
        [
            A.Resize(config["img_size"], config["img_size"]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            Apy.ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 6
class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        if hasattr(self.model, "fc") and getattr(self.model, "fc") is not None:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)
        else:
            self.model.reset_classifier(n_class)

    def forward(self, x):
        return self.model(x)




## === cell 7
def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    use_amp = device.type == "cuda"
    for step, imgs in pbar:
        imgs = imgs.to(device).float()
        with torch.cuda.amp.autocast(enabled=use_amp):
            image_preds = model(imgs)
            probs = torch.softmax(image_preds, 1)
        image_preds_all.append(probs.detach().cpu().numpy())

    if len(image_preds_all) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def _find_checkpoint_path(model_arch: str, fold: int):
    candidate_dirs = [
        "../input/cassava-leaf-disease",
        "../input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-models",
        "../input",
    ]
    candidate_names = [
        f"{model_arch}_fold_{fold}",
        f"{model_arch}_fold_{fold}.pth",
        f"{model_arch}_fold_{fold}.pt",
        f"{model_arch}_fold_{fold}.bin",
        f"{model_arch}_fold{fold}.pth",
        f"{model_arch}_fold{fold}.pt",
        f"{model_arch}_fold{fold}.bin",
    ]
    for d in candidate_dirs:
        for n in candidate_names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


def _build_pretrained_cassava_if_available(model_arch: str, n_class: int, device):
    """
    Score-improving minimal fix:
    If no local checkpoints exist, using ImageNet-pretrained backbone with a random 5-class head
    yields near-random predictions (~0.2 acc). timm provides cassava-pretrained weights for some
    architectures via pretrained=True *with num_classes=5*, which includes a trained 5-class head.
    We try that first; if unavailable, we fall back to the previous behavior.
    """
    try:
        m = timm.create_model(model_arch, pretrained=True, num_classes=n_class)
        return m.to(device), True
    except Exception:
        m = CassvaImgClassifier(model_arch, n_class, pretrained=True).to(device)
        return m, False




## === cell 8
test_images_dir = f"{DATA_DIR}/test_images"
test = submission[["image_id"]].copy()

test_ds = CassavaDataset(
    test, test_images_dir, transforms=get_inference_transforms(), output_label=False
)

tst_loader = DataLoader(
    test_ds,
    batch_size=config["valid_bs"],
    num_workers=config["num_workers"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device(config["device"])

model = CassvaImgClassifier(config["model_arch"], 5, pretrained=False).to(device)

tst_preds = []
used_any_checkpoint = False

for fold in range(config["fold_num"]):
    ckpt_path = _find_checkpoint_path(config["model_arch"], fold)
    if ckpt_path is None:
        continue

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and any(
        k in state for k in ["state_dict", "model_state_dict", "model"]
    ):
        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state["model"]
    else:
        state_dict = state

    if isinstance(state_dict, dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
            new_state_dict[nk] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=True)
    used_any_checkpoint = True

    with torch.no_grad():
        tst_preds.append(inference_one_epoch(model, tst_loader, device))

if not used_any_checkpoint:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, used_cassava_head = _build_pretrained_cassava_if_available(
        config["model_arch"], 5, device
    )
    with torch.no_grad():
        tst_preds = [inference_one_epoch(model, tst_loader, device)]
else:
    used_cassava_head = False

tst_preds = np.mean(np.stack(tst_preds, axis=0), axis=0)

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 9
if not isinstance(tst_preds, np.ndarray):
    tst_preds = np.asarray(tst_preds)

if tst_preds.ndim == 1:
    tst_preds = np.zeros((len(test), 5), dtype=np.float32)

if tst_preds.shape[0] != len(test) or tst_preds.shape[1] != 5:
    raise ValueError(
        f"Prediction shape mismatch: got {tst_preds.shape}, expected ({len(test)}, 5)."
    )

test["label"] = np.argmax(tst_preds, axis=1).astype(int)
test.to_csv("submission.csv", index=False)

print(test.head())
print("Wrote submission.csv with shape:", test.shape)
print("Columns:", list(test.columns))
print("Used checkpoint:", used_any_checkpoint)
print(
    "Used timm cassava-pretrained 5-class head (fallback path only):", used_cassava_head
)
