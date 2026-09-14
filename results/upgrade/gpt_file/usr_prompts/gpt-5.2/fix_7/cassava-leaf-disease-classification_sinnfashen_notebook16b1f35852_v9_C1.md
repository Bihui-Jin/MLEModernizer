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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.866424901783016

# 6. Current score

0.57324

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53176) has done: 'I fix the two blockers preventing a valid submission: the missing pretrained model file path and the test_images directory listing that mistakenly includes a nested `test_images/` directory. To keep the core logic (ResNet50 inference) intact while making it runnable end-to-end, I load a standard torchvision ResNet50 ImageNet backbone when the custom `.mdl` file isn’t available, and I filter the test file list to images only. I also ensure proper device placement (CPU/GPU) and that predictions are produced in the exact `sample_submission.csv` order so the output CSV is valid for Kaggle.'
- What this solution (achieved 0.55792) has done: 'Your current score is far below the target (gap ≈ -0.3347), and the main reason is that when the custom `.mdl` isn’t found you fall back to an ImageNet ResNet50 with a randomly initialized 5-class head, which produces near-random predictions. To move the score toward the target while keeping the same inference-only ResNet50 core logic, I load a standard ImageNet ResNet50 backbone and replace the random head with a deterministic, training-free head built from simple class prototypes computed on the provided `train_images/` (using the same backbone features). I also fix preprocessing to match torchvision’s ResNet50 weights (Resize(256) + CenterCrop(224)), which is a minimal semantic alignment improvement for inference stability. The output submission format and ordering remain identical to `sample_submission.csv`.'
- What this solution (achieved 0.56988) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end and writes a valid `submission.csv`. The main issue is that in this torchvision version `ResNet50_Weights.IMAGENET1K_V2.meta` doesn’t contain `mean/std`, causing a crash that prevents `default_loader` and `ImageDataset` from being defined, cascading into later NameErrors. I use the official ImageNet normalization constants (which match ResNet training) as a stable fallback while keeping your preprocessing/model/inference logic the same. I also add a small safety fallback for transforms and DataLoader workers to avoid environment-specific crashes, without changing the modeling approach.'
- What this solution (achieved 0.57324) has done: 'Your current gap to the target is large (0.56988 vs 0.86642), and the easiest score gain without changing your core “ResNet50 backbone + prototype head + TTA averaging” logic is to make the prototype computation less noisy and more robust. I keep the same architecture and inference flow, but compute class prototypes using the same 5-crop TTA features you already use at test time, then use those prototypes for the cosine-similarity logits; this is a semantic alignment fix, not a new model. I also compute prototypes in mixed batches efficiently (vectorized accumulation) and add a tiny diagonal covariance shrink (prototype smoothing) via temperature scaling on logits to improve calibration slightly without changing the decision rule. The submission ordering/format stays identical and it still write `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
TEST_PATH_CANDIDATES = [
    "../input/cassava-leaf-disease-classification/test_images/",
    "/kaggle/input/cassava-leaf-disease-classification/test_images/",
]
TRAIN_PATH_CANDIDATES = [
    "../input/cassava-leaf-disease-classification/train_images/",
    "/kaggle/input/cassava-leaf-disease-classification/train_images/",
]
TRAIN_CSV_CANDIDATES = [
    "../input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
]

TEST_PATH = None
for p in TEST_PATH_CANDIDATES:
    if os.path.isdir(p):
        TEST_PATH = p
        break
if TEST_PATH is None:
    raise FileNotFoundError(
        f"Could not find test_images dir in any of: {TEST_PATH_CANDIDATES}"
    )

TRAIN_PATH = None
for p in TRAIN_PATH_CANDIDATES:
    if os.path.isdir(p):
        TRAIN_PATH = p
        break
if TRAIN_PATH is None:
    raise FileNotFoundError(
        f"Could not find train_images dir in any of: {TRAIN_PATH_CANDIDATES}"
    )

TRAIN_CSV_PATH = None
for p in TRAIN_CSV_CANDIDATES:
    if os.path.isfile(p):
        TRAIN_CSV_PATH = p
        break
if TRAIN_CSV_PATH is None:
    raise FileNotFoundError(
        f"Could not find train.csv in any of: {TRAIN_CSV_CANDIDATES}"
    )

test_files = sorted(
    [
        f
        for f in os.listdir(TEST_PATH)
        if os.path.isfile(os.path.join(TEST_PATH, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)
if len(test_files) == 0:
    raise RuntimeError(f"No image files found in {TEST_PATH}")

train_df = pd.read_csv(TRAIN_CSV_PATH)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise RuntimeError(
        f"train.csv missing required columns; got {train_df.columns.tolist()}"
    )



## === cell 2
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2

try:
    base_preprocess = weights.transforms()
except Exception:
    base_preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

norm_mean = getattr(weights, "meta", {}).get("mean", (0.485, 0.456, 0.406))
norm_std = getattr(weights, "meta", {}).get("std", (0.229, 0.224, 0.225))

resize_size = 256
center_crop_size = 224

tta_preprocess = transforms.Compose(
    [
        transforms.Resize(
            resize_size, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.FiveCrop(center_crop_size),  # returns 5 PIL images
        transforms.Lambda(
            lambda crops: torch.stack([transforms.ToTensor()(c) for c in crops])
        ),
        transforms.Lambda(
            lambda x: transforms.Normalize(mean=norm_mean, std=norm_std)(x)
        ),
    ]
)


def default_loader(path):
    img_pil = Image.open(path).convert("RGB")
    return base_preprocess(img_pil)


def tta_loader(path):
    img_pil = Image.open(path).convert("RGB")
    return tta_preprocess(img_pil)  # [5, 3, H, W]




## === cell 3
class ImageDataset(Dataset):
    def __init__(self, root_dir, files, labels=None, loader=default_loader):
        self.root_dir = root_dir
        self.files = list(files)
        self.labels = None if labels is None else list(labels)
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = os.path.join(self.root_dir, img_name)
        img = self.loader(img_path)
        if self.labels is None:
            return img_name, img
        return img_name, img, int(self.labels[index])

    def __len__(self):
        return len(self.files)




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "../input/cassavaleaffsolmodels/resnet2.mdl"

model = torchvision.models.resnet50(weights=None)
model.fc = torch.nn.Linear(2048, 5, bias=True)

custom_weights_loaded = False
if os.path.isfile(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    custom_weights_loaded = True
else:
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()




## === cell 5
def compute_class_prototypes(
    backbone, df, train_root, max_per_class=None, batch_size=32, use_tta=True
):
    parts = []
    for c in sorted(df["label"].unique()):
        dfi = df[df["label"] == c]
        if (max_per_class is not None) and (len(dfi) > max_per_class):
            dfi = dfi.sample(n=max_per_class, random_state=42)
        parts.append(dfi)
    sdf = pd.concat(parts, axis=0).reset_index(drop=True)

    loader_fn = tta_loader if use_tta else default_loader
    ds = ImageDataset(
        train_root,
        sdf["image_id"].tolist(),
        labels=sdf["label"].tolist(),
        loader=loader_fn,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = 5
    feat_dim = 2048
    feats_sum = torch.zeros((num_classes, feat_dim), dtype=torch.float32)
    feats_cnt = torch.zeros((num_classes,), dtype=torch.long)

    with torch.no_grad():
        for _, imgs, labs in dl:
            labs = labs.to(torch.long)

            if use_tta:
                b, ncrops, c, h, w = imgs.shape
                imgs = imgs.view(b * ncrops, c, h, w).to(device, non_blocking=True)
                f = backbone(imgs)  # [B*5, 2048]
                f = F.normalize(f, dim=1)
                f = f.view(b, ncrops, -1).mean(dim=1)  # [B, 2048]
                f = F.normalize(f, dim=1)
            else:
                imgs = imgs.to(device, non_blocking=True)
                f = backbone(imgs)  # [B, 2048]
                f = F.normalize(f, dim=1)

            f_cpu = f.detach().cpu()
            labs_cpu = labs.detach().cpu()

            for cls in range(num_classes):
                mask = labs_cpu == cls
                if mask.any():
                    feats_sum[cls] += f_cpu[mask].sum(dim=0)
                    feats_cnt[cls] += int(mask.sum().item())

    protos = []
    for c in range(num_classes):
        if feats_cnt[c].item() == 0:
            proto = torch.zeros(feat_dim, dtype=torch.float32)
        else:
            proto = feats_sum[c] / float(feats_cnt[c].item())
        proto = F.normalize(proto, dim=0)
        protos.append(proto)
    protos = torch.stack(protos, dim=0)  # [5, 2048]
    return protos


prototypes = None
if not custom_weights_loaded:
    prototypes = compute_class_prototypes(
        model, train_df, TRAIN_PATH, max_per_class=None, batch_size=32, use_tta=True
    ).to(device)



## === cell 6
testset = ImageDataset(TEST_PATH, test_files, labels=None, loader=tta_loader)
batch_size = 32  # 5 crops increases tensor size; keep within memory/time
test_loader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

LOGIT_TEMPERATURE = 0.75  # <1 => slightly sharper; helps when similarities are close

labels = []
out_files = []

with torch.no_grad():
    for fls, imgs in test_loader:
        b, ncrops, c, h, w = imgs.shape
        imgs = imgs.view(b * ncrops, c, h, w).to(device, non_blocking=True)

        if custom_weights_loaded:
            logits = model(imgs)  # [B*5, 5]
        else:
            feats = model(imgs)  # [B*5, 2048]
            feats = F.normalize(feats, dim=1)
            logits = feats @ prototypes.T  # [B*5, 5]

        logits = logits.view(b, ncrops, -1).mean(dim=1)  # [B, 5]
        logits = logits / LOGIT_TEMPERATURE

        predicted_label = (
            torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        )
        labels.extend(predicted_label)
        out_files.extend(list(fls))

submission_pred = pd.DataFrame({"image_id": out_files, "label": labels})



## === cell 7
SAMPLE_SUB_PATH_CANDIDATES = [
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
]
SAMPLE_SUB_PATH = None
for p in SAMPLE_SUB_PATH_CANDIDATES:
    if os.path.isfile(p):
        SAMPLE_SUB_PATH = p
        break
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {SAMPLE_SUB_PATH_CANDIDATES}"
    )

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

submission = sample_sub[["image_id"]].merge(submission_pred, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(
    f"Wrote submission.csv with shape={submission.shape} to {os.path.abspath('submission.csv')}"
)
print(f"Custom model loaded: {custom_weights_loaded}")
print(f"Prototype TTA used: {not custom_weights_loaded}")
print(f"Logit temperature: {LOGIT_TEMPERATURE}")
