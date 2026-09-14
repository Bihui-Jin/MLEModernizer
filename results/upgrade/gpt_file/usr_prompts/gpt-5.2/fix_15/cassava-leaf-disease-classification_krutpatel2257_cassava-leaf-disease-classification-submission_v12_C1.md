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

0.7216

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23281) has done: 'I fix the missing pretrained weight path by falling back to a torchvision ImageNet-pretrained ResNeXt50 when the external `.pth` file isn’t available, so inference can run end-to-end. I also update the Albumentations pipeline to the v2 API (notably `RandomResizedCrop` signature changes and removal of deprecated `Cutout`) so `sub_aug` is defined without validation errors. Finally, I make inference stable and correct by using `torch.no_grad()`, consistent device handling, RGB conversion, and softmax-averaged TTA logits before argmax, then write a properly formatted `submission.csv`.'
- What this solution (achieved 0.20927) has done: 'Your current 0.23281 is far below the 0.8729 target, and the biggest reason is that you’re doing *strong training-time augmentations at test-time* (RandomResizedCrop/HSV/CoarseDropout/etc.), which destroys signal and tanks accuracy. I keep the same model and inference loop, but switch the test-time pipeline to a standard deterministic resize/center-crop + ImageNet normalization, and keep only safe TTA (horizontal flip) while removing color/dropout/geometric randomness. I also fix a subtle tensor conversion issue by replacing `transforms.ToTensor()` (which can mishandle already-float normalized arrays) with a direct NumPy→Torch CHW conversion to preserve your normalization exactly. These minimal changes should substantially increase accuracy and move the score toward your target without changing the core model logic.'
- What this solution (achieved 0.10127) has done: 'Your score is far below the target, so we should safely *increase* accuracy without changing the model or training logic. The main issue now is that if the external `.pth` is missing, you’re using an ImageNet-pretrained backbone but a randomly initialized 5-class `fc`, which predict near-random and caps accuracy. I keep the exact same architecture/inference loop, but when `model_path` is missing I replace the `fc` with the ImageNet classifier and output the Cassava 5 classes by mapping from the 1000 ImageNet logits using a fixed “nearest prototype in fc-weight space” (computed once from the pretrained head), which is a lightweight, deterministic improvement that stays within the same forward pass semantics. I also speed up and stabilize inference (no score semantics change) by batching instead of per-row model calls, so it finishes reliably under the time limit while producing the same submission format.'
- What this solution (achieved 0.74925) has done: 'Your current score is far below the target, so we should safely increase accuracy with the smallest change possible while preserving your inference/model setup. The main failure is the ImageNet fallback path: the 1000→5 “nearest-prototype” mapping is essentially arbitrary and yields near-random labels, which matches the ~0.10 accuracy. I keep the same ResNeXt50 backbone, same deterministic test transforms, same TTA averaging, and same softmax→argmax semantics, but replace the fallback post-processing with a simple, competition-legit pseudo-calibration: fit a 5-class linear head on top of the frozen ImageNet backbone using the provided `train.csv` + `train_images` (one short epoch), then use that head for test predictions. This does not change the core model architecture/training approach in spirit (still a single linear head on ResNeXt features with cross-entropy), but fixes the fallback so predictions become meaningful and should move the score much closer to your target.'
- What this solution (achieved 0.70852) has done: 'Your current score (0.74925) is still well below the target (0.87292), so we should make a small, safe accuracy improvement without changing the core model/inference logic. The biggest low-risk gain is to make the fallback linear-head training better matched to the real evaluation by using a stratified train/val split, class-balanced loss weights (cassava is imbalanced), and selecting the best head checkpoint by validation accuracy (no early stopping; fixed steps/epochs). I also switch the fallback-head optimizer to SGD+momentum (commonly better for linear probing) and add light, standard *training-only* augmentation (RandomResizedCrop + HFlip) while keeping test preprocessing unchanged. These changes only affect the fallback training path (when the external `.pth` isn’t present) and should move accuracy upward toward your target while preserving architecture and prediction semantics.'
- What this solution (achieved 0.74178) has done: 'Your current score (0.7085) is well below the target (0.8729), so we should make a small, legitimate accuracy improvement without changing the model architecture or inference semantics. The safest win is to improve the fallback linear-probing training by (1) using the full validation set for model selection (you currently cap validation to only ~1280 images), and (2) training a bit longer with a standard LR schedule that decays over the whole run (your current StepLR decays too aggressively and too early). These changes keep the same backbone, same linear head, same loss, and same test-time preprocessing/TTA, but should produce a better head and move accuracy upward toward the target. I also add a fast cached feature option for validation to make evaluating on the full val set cheap enough to stay within the 600s constraint.'
- What this solution (achieved 0.6917) has done: 'Your current score (0.74178) is well below the target (0.87292), so we should make a small, legitimate accuracy improvement without changing your model/inference semantics. The safest gain is to make the fallback linear-probe head train on a more stable and representative training stream by iterating through the full stratified-train set each epoch (instead of repeated sampling with replacement) while keeping the same backbone/head/loss. I also add a tiny amount of label smoothing in the fallback head’s CrossEntropyLoss to improve generalization on this noisy/imbalanced dataset without changing prediction format or test-time preprocessing. Finally, I keep your existing deterministic test transforms + HFlip TTA and still write `submission.csv` exactly as required.'
- What this solution (achieved 0.69843) has done: 'Your score gap to the target is still large (0.6917 vs 0.8729), so we should *increase* accuracy with small, low-risk changes that keep your current backbone + frozen-feature linear-probe fallback and the same inference/TTA semantics. The biggest issue is that your fallback head is trained for a fixed number of **steps** with a relatively high LR and label smoothing; this tends to underfit/overfit inconsistently and often lands well below what a simple linear probe can achieve on Cassava. I keep the exact same model and training loop, but make the fallback training more stable by (1) switching to a fixed small number of **epochs over the full stratified train set** (same SGD+CE), (2) using a more standard LR for linear probing plus a cosine schedule over epochs, and (3) removing label smoothing (often hurts top-1 on this dataset). These are minimal changes localized to the fallback head training path and should move accuracy upward toward your target.'
- What this solution (achieved 0.65172) has done: 'Your current score (0.698) is far below the target (0.873), so we should cautiously increase accuracy with minimal changes limited to the fallback linear-probe training (used when the external .pth is missing). The biggest low-risk gap is that the linear probe is trained on heavily augmented crops but validated/inferred on center-crops, creating a train/test preprocessing mismatch; I align fallback training augmentation to be closer to inference (resize+random crop with mild scale jitter) while keeping the same backbone/head/loss/optimizer/training loop structure. I also add a tiny amount of dropout to the linear head (still a linear probe) to improve generalization and reduce overfitting without changing inference semantics (still argmax of softmax-averaged TTA). Finally, I slightly extend the fixed epoch count (still epoch-based over the full stratified set) to reduce underfitting, while keeping runtime under the 600s limit.'
- What this solution (achieved 0.72795) has done: 'Your current score (0.6517) is far below the target (0.8729), so we should make a small, legitimate accuracy improvement while keeping the same backbone + frozen-feature linear-probe fallback and the same test-time preprocessing/TTA semantics. The lowest-risk gain is that the fallback head is currently trained on a more “random crop” view than what you use at inference (center-crop), which can hurt top-1; I align the fallback training transform closer to inference (keep resize/center-crop, with only horizontal flip) to reduce the train/test distribution shift. I also remove dropout from the head (still a linear probe) because on linear probing it often hurts raw accuracy and adds noise without changing the core approach. Finally, I modestly increase the fixed epoch count a bit to reduce underfitting, keeping runtime reasonable with the existing cached validation features.'
- What this solution (achieved 0.71712) has done: 'Your current score (0.72795) is far below the target (0.87292), so we should make a small, legitimate improvement that keeps the same ResNeXt50 backbone + frozen-feature linear-probe fallback and the same test-time preprocessing/TTA and argmax semantics. The most impactful minimal fix is to make the fallback head train on features extracted with the backbone in true inference mode (disable BatchNorm updates) and to use a more standard linear-probe learning rate for this dataset while keeping the same optimizer/loss/training loop structure. I also remove the per-epoch resampling/shuffling that can accidentally reduce effective coverage and replace it with a deterministic DataLoader-style iteration over the full stratified train set each epoch (still the same “iterate through all train images each epoch” approach, just safer and less error-prone). These changes are confined to the fallback training path (used because your external .pth likely doesn’t exist) and should move accuracy upward toward the target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.7216) has done: 'The timeout is dominated by the ImageNet-fallback path: it repeatedly loads/decodes images with PIL inside Python loops, applies Albumentations per-image, and then runs the backbone over an entire epoch cache; this multiplies CPU overhead and GPU idle time. To keep identical training/eval semantics while speeding up, I (1) avoid `iterrows`/`iloc` overhead, (2) use a multi-worker `DataLoader` with pinned memory to overlap CPU decode/augment with GPU compute, (3) keep the same per-epoch shuffled order and augmentation randomness (seeded) by generating per-sample seeds and using them to set `np.random.seed` right before each Albumentations call, and (4) for inference, avoid storing raw images and instead build augmented tensors directly in a batched loader. All model architecture, loss, optimizer, scheduler, number of epochs, TTA, and argmax decision logic remain unchanged.'

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

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
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



## === cell 2
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
        albumentations.Resize(256, 256, p=1.0),
        albumentations.CenterCrop(224, 224, p=1.0),
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




## === cell 3
def hwc_float_to_chw_tensor(x: np.ndarray) -> torch.Tensor:
    """
    albumentations outputs HWC float32 (already normalized).
    Convert explicitly to avoid unintended scaling.
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


from torch.utils.data import Dataset, DataLoader


class _SeededImageDataset(Dataset):
    def __init__(self, image_ids, labels, img_dir: str, aug, seeds=None):
        self.image_ids = np.asarray(image_ids)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.img_dir = img_dir
        self.aug = aug
        self.seeds = None if seeds is None else np.asarray(seeds, dtype=np.int64)

    def __len__(self):
        return int(self.image_ids.shape[0])

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.seeds is not None:
            np.random.seed(int(self.seeds[idx]))

        img = self.aug(image=img)["image"]
        x = hwc_float_to_chw_tensor(img)

        if self.labels is None:
            return x, img_id
        return x, int(self.labels[idx])


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )


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
def _build_val_feature_cache(
    feature_extractor: nn.Module,
    df_val: pd.DataFrame,
    img_dir: str,
    aug,
    device: torch.device,
    batch_size: int = 128,
):
    feature_extractor.eval()

    image_ids = df_val["image_id"].to_numpy()
    labels = df_val["label"].astype(np.int64).to_numpy()

    num_workers = min(4, os.cpu_count() or 1)
    ds = _SeededImageDataset(
        image_ids=image_ids, labels=labels, img_dir=img_dir, aug=aug
    )
    loader = _make_loader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    feats_all = []
    y_all = []
    for xb_cpu, yb in loader:
        xb = xb_cpu.to(device, non_blocking=True)
        feats = feature_extractor(xb).flatten(1).detach().cpu()
        feats_all.append(feats)
        y_all.append(torch.as_tensor(yb, dtype=torch.long))

    if len(feats_all) == 0:
        return None, None
    return torch.cat(feats_all, dim=0), torch.cat(y_all, dim=0)


@torch.no_grad()
def _eval_head_accuracy_cached(
    head: nn.Module,
    val_feats_cpu: torch.Tensor,
    val_y_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    head.eval()
    if val_feats_cpu is None or val_y_cpu is None or val_feats_cpu.numel() == 0:
        return 0.0

    correct = 0
    total = int(val_y_cpu.numel())

    head = head.to(device)
    for i in range(0, total, batch_size):
        xb = val_feats_cpu[i : i + batch_size].to(device, non_blocking=True)
        yb = val_y_cpu[i : i + batch_size].to(device, non_blocking=True)
        logits = head(xb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())

    return correct / max(1, total)


def fit_fallback_linear_head(
    model_imagenet: nn.Module,
    train_csv: str,
    train_img_dir: str,
    train_aug,
    val_aug,
    device: torch.device,
    max_steps: int = 1200,  # kept for API compatibility; unused in epoch-based loop below
    batch_size: int = 64,
    val_frac: float = 0.1,
    val_check_every: int = 200,  # kept for API compatibility; replaced by per-epoch checks
) -> nn.Module:
    backbone = model_imagenet
    backbone.eval()

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device)
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

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    opt = torch.optim.SGD(head.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)

    steps_per_epoch = max(1, int(np.ceil(len(df_train) / batch_size)))
    epochs = 12
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=0.001
    )

    val_feats_cpu, val_y_cpu = _build_val_feature_cache(
        feature_extractor=feature_extractor,
        df_val=df_val,
        img_dir=train_img_dir,
        aug=val_aug,
        device=device,
        batch_size=128,
    )

    best_state = None
    best_acc = -1.0

    num_workers = min(4, os.cpu_count() or 1)

    head.train()
    for ep in range(epochs):
        df_epoch = df_train.sample(frac=1.0, random_state=42 + ep).reset_index(
            drop=True
        )

        image_ids = df_epoch["image_id"].to_numpy()
        labels = df_epoch["label"].astype(np.int64).to_numpy()

        rng = np.random.RandomState(42 + ep)
        seeds = rng.randint(0, 2**31 - 1, size=len(df_epoch), dtype=np.int64)

        ds = _SeededImageDataset(
            image_ids=image_ids,
            labels=labels,
            img_dir=train_img_dir,
            aug=train_aug,
            seeds=seeds,
        )
        loader = _make_loader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        for xb_cpu, yb_cpu in loader:
            xb = xb_cpu.to(device, non_blocking=True)
            y = yb_cpu.to(device, non_blocking=True)

            feats = feature_extractor(xb).flatten(1)
            logits = head(feats)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

        acc = _eval_head_accuracy_cached(
            head=head,
            val_feats_cpu=val_feats_cpu,
            val_y_cpu=val_y_cpu,
            device=device,
            batch_size=4096,
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




## === cell 4
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
        train_aug=train_aug_fallback,
        val_aug=sub_aug,
        device=device,
        max_steps=1800,
        batch_size=64,
        val_frac=0.1,
        val_check_every=200,
    )
    fallback_feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    fallback_feature_extractor.eval()

test_image_ids = sample_sub["image_id"].to_numpy()
num_workers = min(4, os.cpu_count() or 1)

pred_image_ids = []
pred_labels = []

with torch.no_grad():
    probs_sum_cpu = None

    for aug in tta_augs:
        ds = _SeededImageDataset(
            image_ids=test_image_ids,
            labels=None,
            img_dir=test_images_path,
            aug=aug,
            seeds=None,  # deterministic aug; no RNG needed
        )
        loader = _make_loader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        probs_list = []
        ids_list = []
        for xb_cpu, id_batch in loader:
            x = xb_cpu.to(device, non_blocking=True)

            if using_imagenet_fallback:
                feats = fallback_feature_extractor(x).flatten(1)
                logits5 = fallback_head(feats)
                probs = torch.softmax(logits5, dim=1)
            else:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)

            probs_list.append(probs.detach().cpu())
            ids_list.append(np.asarray(id_batch))

        probs_cpu = torch.cat(probs_list, dim=0)
        ids_np = np.concatenate(ids_list, axis=0)

        if probs_sum_cpu is None:
            probs_sum_cpu = probs_cpu
            pred_image_ids = ids_np.tolist()
        else:
            probs_sum_cpu += probs_cpu

    probs_avg_cpu = probs_sum_cpu / len(tta_augs)
    pred_labels = torch.argmax(probs_avg_cpu, dim=1).numpy().astype(int).tolist()

sub_df = pd.DataFrame({"image_id": pred_image_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print(
    "Fallback used (trained linear head on frozen ImageNet backbone):",
    using_imagenet_fallback,
)
