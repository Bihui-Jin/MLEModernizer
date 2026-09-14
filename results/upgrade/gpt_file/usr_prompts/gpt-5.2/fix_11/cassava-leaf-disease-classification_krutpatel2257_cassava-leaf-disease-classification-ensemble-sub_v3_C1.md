# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
                feat_e = effnet_model.avgpool(
                    feat_e
                )  # AdaptiveAvgPool2d((1,1)) in torchvision
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

            aug_pil = Image.fromarray(aug_img, mode="RGB")

            tta_r.append(resnet_preprocess(aug_pil))
            tta_e.append(effnet_preprocess(aug_pil))

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
pred_labels = softmax(predictions, axis=1).argmax(axis=1).astype(int)

sub_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
