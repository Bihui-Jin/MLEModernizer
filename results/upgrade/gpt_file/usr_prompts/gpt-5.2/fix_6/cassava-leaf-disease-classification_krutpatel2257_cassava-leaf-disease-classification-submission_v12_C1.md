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

0.8729223330311272

# 6. Current score

0.70852

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23281) has done: 'I fix the missing pretrained weight path by falling back to a torchvision ImageNet-pretrained ResNeXt50 when the external `.pth` file isn’t available, so inference can run end-to-end. I also update the Albumentations pipeline to the v2 API (notably `RandomResizedCrop` signature changes and removal of deprecated `Cutout`) so `sub_aug` is defined without validation errors. Finally, I make inference stable and correct by using `torch.no_grad()`, consistent device handling, RGB conversion, and softmax-averaged TTA logits before argmax, then write a properly formatted `submission.csv`.'
- What this solution (achieved 0.20927) has done: 'Your current 0.23281 is far below the 0.8729 target, and the biggest reason is that you’re doing *strong training-time augmentations at test-time* (RandomResizedCrop/HSV/CoarseDropout/etc.), which destroys signal and tanks accuracy. I keep the same model and inference loop, but switch the test-time pipeline to a standard deterministic resize/center-crop + ImageNet normalization, and keep only safe TTA (horizontal flip) while removing color/dropout/geometric randomness. I also fix a subtle tensor conversion issue by replacing `transforms.ToTensor()` (which can mishandle already-float normalized arrays) with a direct NumPy→Torch CHW conversion to preserve your normalization exactly. These minimal changes should substantially increase accuracy and move the score toward your target without changing the core model logic.'
- What this solution (achieved 0.10127) has done: 'Your score is far below the target, so we should safely *increase* accuracy without changing the model or training logic. The main issue now is that if the external `.pth` is missing, you’re using an ImageNet-pretrained backbone but a randomly initialized 5-class `fc`, which predict near-random and caps accuracy. I keep the exact same architecture/inference loop, but when `model_path` is missing I replace the `fc` with the ImageNet classifier and output the Cassava 5 classes by mapping from the 1000 ImageNet logits using a fixed “nearest prototype in fc-weight space” (computed once from the pretrained head), which is a lightweight, deterministic improvement that stays within the same forward pass semantics. I also speed up and stabilize inference (no score semantics change) by batching instead of per-row model calls, so it finishes reliably under the time limit while producing the same submission format.'
- What this solution (achieved 0.74925) has done: 'Your current score is far below the target, so we should safely increase accuracy with the smallest change possible while preserving your inference/model setup. The main failure is the ImageNet fallback path: the 1000→5 “nearest-prototype” mapping is essentially arbitrary and yields near-random labels, which matches the ~0.10 accuracy. I keep the same ResNeXt50 backbone, same deterministic test transforms, same TTA averaging, and same softmax→argmax semantics, but replace the fallback post-processing with a simple, competition-legit pseudo-calibration: fit a 5-class linear head on top of the frozen ImageNet backbone using the provided `train.csv` + `train_images` (one short epoch), then use that head for test predictions. This does not change the core model architecture/training approach in spirit (still a single linear head on ResNeXt features with cross-entropy), but fixes the fallback so predictions become meaningful and should move the score much closer to your target.'
- What this solution (achieved 0.70852) has done: 'Your current score (0.74925) is still well below the target (0.87292), so we should make a small, safe accuracy improvement without changing the core model/inference logic. The biggest low-risk gain is to make the fallback linear-head training better matched to the real evaluation by using a stratified train/val split, class-balanced loss weights (cassava is imbalanced), and selecting the best head checkpoint by validation accuracy (no early stopping; fixed steps/epochs). I also switch the fallback-head optimizer to SGD+momentum (commonly better for linear probing) and add light, standard *training-only* augmentation (RandomResizedCrop + HFlip) while keeping test preprocessing unchanged. These changes only affect the fallback training path (when the external `.pth` isn’t present) and should move accuracy upward toward your target while preserving architecture and prediction semantics.'

# 9. Code solution

## === cell 0
import os

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models



## === cell 1
model_path = "../input/rnwcnwcttacalr/model(7).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
if not os.path.exists(test_images_path):
    test_images_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"

if not os.path.exists(train_csv_path):
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
if not os.path.exists(train_images_path):
    train_images_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True



## === cell 2
using_imagenet_fallback = False

model = models.resnext50_32x4d(weights=None)
model.fc = nn.Linear(2048, 5)

if os.path.exists(model_path):
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
else:
    using_imagenet_fallback = True
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

model.to(device)
model.eval()



## === cell 3
sub_aug = albumentations.Compose(
    [
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

sub_aug_hflip = albumentations.Compose(
    [
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

train_aug_fallback = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(
            size=(224, 224), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=1.0
        ),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 4
def hwc_float_to_chw_tensor(x: np.ndarray) -> torch.Tensor:
    """
    albumentations outputs HWC float32 (already normalized).
    Convert explicitly to avoid unintended scaling.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def _stratified_split_df(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    train_parts = []
    val_parts = []
    for lbl, g in df.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_parts.append(df.loc[val_idx])
        train_parts.append(df.loc[train_idx])
    train_df = (
        pd.concat(train_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return train_df, val_df


@torch.no_grad()
def _eval_head_accuracy(
    feature_extractor: nn.Module,
    head: nn.Module,
    df_val: pd.DataFrame,
    img_dir: str,
    aug,
    device: torch.device,
    batch_size: int = 128,
    max_batches: int = 10,
) -> float:
    head.eval()
    feature_extractor.eval()

    correct = 0
    total = 0
    batch_imgs = []
    batch_y = []
    batches_done = 0

    for _, row in df_val.iterrows():
        img_path = os.path.join(img_dir, row.image_id)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = aug(image=img)["image"]
        batch_imgs.append(hwc_float_to_chw_tensor(img))
        batch_y.append(int(row.label))

        if len(batch_imgs) >= batch_size:
            x = torch.stack(batch_imgs, dim=0).to(device)
            y = torch.tensor(batch_y, device=device, dtype=torch.long)
            feats = feature_extractor(x).flatten(1)
            logits = head(feats)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            batch_imgs, batch_y = [], []
            batches_done += 1
            if batches_done >= max_batches:
                break

    if total == 0:
        return 0.0
    return correct / total


def fit_fallback_linear_head(
    model_imagenet: nn.Module,
    train_csv: str,
    train_img_dir: str,
    train_aug,
    val_aug,
    device: torch.device,
    max_steps: int = 1200,
    batch_size: int = 64,
    val_frac: float = 0.1,
    val_check_every: int = 200,
) -> nn.Module:
    backbone = model_imagenet
    backbone.eval()

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(
        device
    )  # -> [B,2048,1,1]
    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()

    head = nn.Linear(2048, 5).to(device)

    df_all = pd.read_csv(train_csv)
    df_all["label"] = df_all["label"].astype(int)
    df_train, df_val = _stratified_split_df(df_all, val_frac=val_frac, seed=42)
    counts = df_train["label"].value_counts().sort_index()
    counts = counts.reindex(range(5), fill_value=1)
    inv = 1.0 / counts.to_numpy(dtype=np.float32)
    weights = inv / inv.mean()
    class_weights = torch.tensor(weights, device=device, dtype=torch.float32)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    opt = torch.optim.SGD(head.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, max_steps // 3), gamma=0.3
    )

    df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)

    step = 0
    idx = 0
    n = len(df_train)

    best_state = None
    best_acc = -1.0

    head.train()
    while step < max_steps:
        imgs = []
        ys = []
        while len(imgs) < batch_size:
            if idx >= n:
                idx = 0
            row = df_train.iloc[idx]
            idx += 1
            img_path = os.path.join(train_img_dir, row.image_id)
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img = train_aug(image=img)["image"]
            imgs.append(hwc_float_to_chw_tensor(img))
            ys.append(int(row.label))

        x = torch.stack(imgs, dim=0).to(device)
        y = torch.tensor(ys, device=device, dtype=torch.long)

        with torch.no_grad():
            feats = feature_extractor(x).flatten(1)  # [B,2048]

        logits = head(feats)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

        step += 1

        if (step % val_check_every) == 0:
            acc = _eval_head_accuracy(
                feature_extractor=feature_extractor,
                head=head,
                df_val=df_val,
                img_dir=train_img_dir,
                aug=val_aug,
                device=device,
                batch_size=128,
                max_batches=10,
            )
            if acc > best_acc:
                best_acc = acc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in head.state_dict().items()
                }

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    return head




## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

tta_augs = [sub_aug, sub_aug_hflip]
batch_size = 32

fallback_head = None
fallback_feature_extractor = None
if using_imagenet_fallback:
    fallback_head = fit_fallback_linear_head(
        model_imagenet=model,
        train_csv=train_csv_path,
        train_img_dir=train_images_path,
        train_aug=train_aug_fallback,  # training-only augmentation for better head
        val_aug=sub_aug,  # deterministic val aligned with test preprocessing
        device=device,
        max_steps=1200,  # small increase to better approach target; still lightweight
        batch_size=64,
        val_frac=0.1,
        val_check_every=200,
    )
    fallback_feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    fallback_feature_extractor.eval()

pred_image_ids = []
pred_labels = []

with torch.no_grad():
    batch_imgs = []
    batch_ids = []

    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        base_img = Image.open(img_path).convert("RGB")
        base_img = np.array(base_img)

        batch_imgs.append(base_img)
        batch_ids.append(sample_row.image_id)

        if len(batch_imgs) >= batch_size:
            probs_sum = None

            for aug in tta_augs:
                aug_tensors = []
                for img in batch_imgs:
                    aug_img = aug(image=img)["image"]
                    aug_tensors.append(hwc_float_to_chw_tensor(aug_img))
                x = torch.stack(aug_tensors, dim=0).to(device)

                if using_imagenet_fallback:
                    feats = fallback_feature_extractor(x).flatten(1)
                    logits5 = fallback_head(feats)
                    probs = torch.softmax(logits5, dim=1)
                else:
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)

                probs_sum = probs if probs_sum is None else (probs_sum + probs)

            probs_avg = probs_sum / len(tta_augs)
            batch_pred = (
                torch.argmax(probs_avg, dim=1).detach().cpu().numpy().astype(int)
            )

            pred_image_ids.extend(batch_ids)
            pred_labels.extend(batch_pred.tolist())

            batch_imgs = []
            batch_ids = []

    if len(batch_imgs) > 0:
        probs_sum = None
        for aug in tta_augs:
            aug_tensors = []
            for img in batch_imgs:
                aug_img = aug(image=img)["image"]
                aug_tensors.append(hwc_float_to_chw_tensor(aug_img))
            x = torch.stack(aug_tensors, dim=0).to(device)

            if using_imagenet_fallback:
                feats = fallback_feature_extractor(x).flatten(1)
                logits5 = fallback_head(feats)
                probs = torch.softmax(logits5, dim=1)
            else:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)

            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        probs_avg = probs_sum / len(tta_augs)
        batch_pred = torch.argmax(probs_avg, dim=1).detach().cpu().numpy().astype(int)

        pred_image_ids.extend(batch_ids)
        pred_labels.extend(batch_pred.tolist())

sub_df = pd.DataFrame({"image_id": pred_image_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print(
    "Fallback used (trained linear head on frozen ImageNet backbone):",
    using_imagenet_fallback,
)
