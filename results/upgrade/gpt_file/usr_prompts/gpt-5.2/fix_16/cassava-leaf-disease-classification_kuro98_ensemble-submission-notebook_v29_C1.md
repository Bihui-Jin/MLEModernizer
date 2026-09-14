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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.8933212450891508

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10314) has done: 'You’re hitting a hard runtime error because `torchvision.models.vit_b_16` is configured for `image_size=224`, but the dataset resizes to 384, triggering the “Wrong image height” assertion. I fix this by switching the inference resize to 224 for both model inputs (score-neutral but required for correctness), and I also fix the ensemble averaging bug where you divide by 2.0 after applying weights (which unintentionally scales logits and can harm argmax stability). Finally, I keep the submission aligned to `sample_submission.csv` order and ensure `submission.csv` is always written with the correct columns and no missing predictions.'
- What this solution (achieved 0.75972) has done: 'Your score is very low because the current “linear_head” is randomly initialized and never trained, so predictions are essentially random even though the ViT backbone is pretrained. I keep the same ViT-B/16 backbone and the same inference loop, but add a minimal training step: replace the random linear head with a proper 5-class classification head and train only that head on `train.csv` (backbone frozen) for a few epochs. This preserves the core architecture (ViT feature extractor + linear classifier) while making outputs meaningful for the 5 cassava classes. I also keep submission ordering aligned to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.75187) has done: 'Your current gap to target is ~0.1336 (0.7597 → 0.8933), so we should improve accuracy without changing the core “ViT backbone frozen + linear head” approach. The biggest issue is you train the head on heavily augmented/random-cropped views but infer on a different distribution (center-cropped from a fixed 600×600 then resized), which hurts generalization; we make train-time preprocessing match inference (center-crop 600 then resize to 224) and keep the rest intact. We also use standard ImageNet augmentation (RandAugment + RandomErasing) on top of that matched crop/resize to improve robustness with minimal changes. Finally, we compute ensemble logits directly (no softmax needed for argmax) to avoid any numerical quirks while preserving identical prediction semantics.'
- What this solution (achieved 0.79671) has done: 'We need to move accuracy up from 0.75187 toward 0.89332 (higher-is-better), so the smallest safe improvements are in training stability/generalization while keeping the same frozen ViT + linear head setup. I (1) add a simple train/val split and report val accuracy so you can confirm the head is actually learning rather than overfitting, (2) apply label smoothing in CrossEntropyLoss (same loss family, improves generalization with minimal semantics change), and (3) use a cosine LR schedule with a short warmup-like effect (no early stopping, same optimizer/training loop) to get a better-trained linear head within the same 3 epochs. I also fix that your “ensemble” is currently identical (model_a == model_b), so I keep the same backbone but make model_b a separate copy (same architecture) so the weighted average is meaningful; this is minimal and doesn’t change the core approach.'
- What this solution (achieved 0.79335) has done: 'We need to move accuracy up from 0.79671 toward 0.89332, so the smallest safe gains are to (1) make the linear head train on features from the same ensemble you use at inference (currently you train only on model_a but infer on a weighted a+b mix), and (2) slightly strengthen the head training without changing the approach by training a bit longer with a proportionally smaller LR. I also keep preprocessing, backbone freezing, and the inference/CSV logic unchanged to preserve evaluation semantics. Finally, I enable AMP during feature extraction/inference to fit a slightly larger batch size for stabler gradients while staying within runtime and without altering the core method.'
- What this solution (achieved 0.79335) has done: 'The main timeout driver is repeatedly running two full ViT-B/16 backbones over images during every training epoch, even though both backbones are frozen; this is equivalent work that can be cached. I keep the exact same models, transforms, loss, optimizer, and training loop semantics, but precompute and store the frozen ensemble features for train/val/test once (under `torch.no_grad()` with the same AMP setting), then train/evaluate the same linear head on those cached features. I also enable persistent DataLoader workers and a prefetch factor to reduce input pipeline overhead, and make inference reuse cached test features to avoid a second expensive image pass. These changes are provably equivalent because the backbones are in `eval()` and `requires_grad=False`, so their outputs for a given transformed input are deterministic per sample and do not depend on the linear head’s training.'
- What this solution (achieved 0.7287) has done: 'We need to move accuracy up from 0.79335 toward 0.89332 (higher-is-better), so the smallest safe gains are to improve the linear head’s generalization without changing the frozen ViT ensemble feature extractor. I keep the exact same backbones, feature caching, and inference, but (1) normalize the cached feature vectors before the linear layer (a standard “linear probe” trick that often yields a noticeable boost for ViT features), and (2) add a very small weight decay to the head and slightly reduce label smoothing to avoid underfitting. These changes preserve the core approach (frozen pretrained ViT(s) + linear classifier trained with CrossEntropy) and keep runtime within the same envelope because feature caching is unchanged. Submission writing/order stays exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.66143) has done: 'We need to move accuracy up from 0.7287 toward 0.8933, so I make the smallest changes that improve the linear-probe training while keeping the same frozen ViT ensemble feature extractor, cached-feature workflow, and CrossEntropy training loop. The biggest likely regression in your current code is a train/test preprocessing mismatch: you apply RandAugment/RandomErasing on the resized tensor but do not apply the model’s expected ViT preprocessing (in particular the resize/crop normalization pipeline used by the pretrained weights), which can hurt a frozen-backbone probe a lot. I switch both train and test transforms to the official `weights.transforms()` for each backbone (applied after your existing center-crop+resize), and keep augmentations minimal and “safe” (flip only) to avoid distribution shift. I also add a tiny per-class weighting from the train split to counter label imbalance (still CrossEntropy), which often gives a noticeable accuracy lift with minimal semantic change.'
- What this solution (achieved 0.65994) has done: 'I fix the immediate crash by replacing the brittle `w.meta["mean"]/["std"]` access with a safe extraction that works across torchvision weight variants, while keeping the same normalization intent. Because cell 3 currently fails, downstream variables like `train_split_df` never get created, so I also ensure that once transforms are fixed, the split/dataloaders execute and cell 4 can run unchanged. I keep the model/backbone, frozen-feature caching, linear head training, and submission-writing logic the same to preserve core semantics. The output always write a valid `submission.csv` with the required `image_id,label` columns and sample-submission ordering.'
- What this solution (achieved 0.65994) has done: 'Your current accuracy (0.65994) is far below the target (0.89332), so we should increase performance with the smallest change that most directly improves the frozen-ViT linear-probe quality. The biggest likely issue is that you’re extracting features from `vit_b_16(...)` by calling the whole model after replacing the head with `Identity()`, which does **not** reliably give you the intended penultimate embedding for ViT and can yield poorly conditioned features for a linear probe. I keep the same backbones, same frozen-feature caching workflow, same linear head, same loss/optimizer/schedule, but switch feature extraction to `forward_features()` (with a safe fallback) so the head is trained on the correct ViT embeddings. This should materially raise accuracy while preserving your overall approach and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_DIR}/test_images/"
train_dir = f"{DATA_DIR}/train_images/"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"
train_csv_path = f"{DATA_DIR}/train.csv"

model_a_img_size = 224
model_b_img_size = 224

batch_size = 32
num_workers = 4
num_classes = 5
tta = False

import torchvision

weights_a = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
weights_b = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1

backbone_a = torchvision.models.vit_b_16(weights=weights_a).to(device)
backbone_b = torchvision.models.vit_b_16(weights=weights_b).to(device)

backbone_a.heads.head = torch.nn.Identity()
backbone_b.heads.head = torch.nn.Identity()

model_a = backbone_a
model_b = backbone_b

use_amp = torch.cuda.is_available()

ens_w_a, ens_w_b = 0.94, 0.06




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava data (train or test).

    For test: uses image_ids from sample_submission.csv to guarantee exact order.
    For train: uses image_ids from train.csv.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        image_ids,
        labels=None,
        transform_a=None,
        transform_b=None,
        ttas=None,
        train_mode: bool = False,
    ):
        super().__init__(root=data_dir)
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.image_ids = list(image_ids)
        self.labels = None if labels is None else list(labels)
        self.ttas = ttas
        self.train_mode = train_mode

        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

        self.hflip = v2.RandomHorizontalFlip(p=0.5)

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.train_mode:
            model_a_img = self.hflip(model_a_img)
            model_b_img = self.hflip(model_b_img)

        if (
            self.ttas is not None
            and (self.transform_a is not None)
            and (self.transform_b is not None)
        ):
            model_a_img = [self.transform_a(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform_b(t(model_b_img)) for t in self.ttas]
        else:
            if self.transform_a is not None:
                model_a_img = self.transform_a(model_a_img)
            if self.transform_b is not None:
                model_b_img = self.transform_b(model_b_img)

        if self.labels is None:
            return model_a_img, model_b_img, filename
        return model_a_img, model_b_img, int(self.labels[idx])

    def __len__(self):
        return len(self.image_ids)




## === cell 3
def _vit_norm_from_weights(w):
    """
    Bugfix: Some torchvision weight enums (notably SWAG variants) don't expose
    mean/std under w.meta['mean'/'std'] in newer versions; fall back to the
    weights' provided transforms() (which includes the correct normalization),
    and only extract mean/std when present.
    """
    mean = None
    std = None
    if hasattr(w, "meta") and isinstance(w.meta, dict):
        mean = w.meta.get("mean", None)
        std = w.meta.get("std", None)

    if mean is not None and std is not None:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    t = w.transforms()
    if not isinstance(t, torch.nn.Module):
        return v2.Lambda(lambda img: t(img))
    return t


test_transform_a = _vit_norm_from_weights(weights_a)
test_transform_b = _vit_norm_from_weights(weights_b)

train_transform_a = _vit_norm_from_weights(weights_a)
train_transform_b = _vit_norm_from_weights(weights_b)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].tolist()

train_df = pd.read_csv(train_csv_path)

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=3407)
idx = torch.arange(len(train_df)).numpy()
train_idx, val_idx = next(sss.split(idx, train_df["label"].values))

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

train_dataset = CassavaDataset(
    train_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=train_split_df["image_id"].tolist(),
    labels=train_split_df["label"].tolist(),
    transform_a=train_transform_a,
    transform_b=train_transform_b,
    ttas=None,
    train_mode=True,
)

val_dataset = CassavaDataset(
    train_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=val_split_df["image_id"].tolist(),
    labels=val_split_df["label"].tolist(),
    transform_a=test_transform_a,
    transform_b=test_transform_b,
    ttas=None,
    train_mode=False,
)

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=test_image_ids,
    labels=None,
    transform_a=test_transform_a,
    transform_b=test_transform_b,
    ttas=ttas,
    train_mode=False,
)

_loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)



## === cell 4
for p in model_a.parameters():
    p.requires_grad = False
for p in model_b.parameters():
    p.requires_grad = False

model_a.eval()
model_b.eval()

label_counts = train_split_df["label"].value_counts().sort_index()
counts = torch.zeros(num_classes, dtype=torch.float32)
for k, v in label_counts.items():
    counts[int(k)] = float(v)
class_weights = counts.sum() / torch.clamp(counts, min=1.0)
class_weights = class_weights / class_weights.mean()
class_weights = class_weights.to(device)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.02, weight=class_weights)

epochs = 6
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def _l2_normalize_cpu_feats(
    feats_cpu: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    denom = torch.clamp(feats_cpu.norm(p=2, dim=1, keepdim=True), min=eps)
    return feats_cpu / denom


@torch.no_grad()
def _vit_extract_features(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(m, "forward_features") and callable(getattr(m, "forward_features")):
        feats = m.forward_features(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        return feats
    feats = m(x)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim == 3:
        feats = feats[:, 0, :]
    return feats


with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
    dummy = torch.zeros(1, 3, model_a_img_size, model_a_img_size, device=device)
    dfa = _vit_extract_features(model_a, dummy)
    dfb = _vit_extract_features(model_b, dummy)
    dfe = ens_w_a * dfa + ens_w_b * dfb
    inferred_in_features = int(dfe.shape[-1])

linear_head = torch.nn.Linear(inferred_in_features, num_classes, bias=True).to(device)
linear_head.train()

optimizer = torch.optim.AdamW(linear_head.parameters(), lr=1.5e-3, weight_decay=5e-3)

steps_per_epoch = len(train_loader)
total_steps = max(1, epochs * steps_per_epoch)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


@torch.no_grad()
def _precompute_ens_features_for_loader(
    loader, with_labels: bool, with_filenames: bool
):
    model_a.eval()
    model_b.eval()
    feats_list = []
    y_list = [] if with_labels else None
    name_list = [] if with_filenames else None

    for batch in loader:
        if with_labels:
            model_a_inputs, model_b_inputs, y = batch
        else:
            model_a_inputs, model_b_inputs, filenames = batch

        xa = model_a_inputs.to(device, non_blocking=True)
        xb = model_b_inputs.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            feats_a = _vit_extract_features(model_a, xa)
            feats_b = _vit_extract_features(model_b, xb)
            feats = ens_w_a * feats_a + ens_w_b * feats_b

        feats_list.append(feats.detach().cpu())

        if with_labels:
            y_list.append(torch.as_tensor(y, dtype=torch.long).cpu())
        else:
            name_list.extend(list(filenames))

    feats_all = torch.cat(feats_list, dim=0)
    feats_all = _l2_normalize_cpu_feats(feats_all)

    if with_labels:
        y_all = torch.cat(y_list, dim=0)
        return feats_all, y_all
    return feats_all, name_list


train_feats_cpu, train_y_cpu = _precompute_ens_features_for_loader(
    train_loader, with_labels=True, with_filenames=False
)
val_feats_cpu, val_y_cpu = _precompute_ens_features_for_loader(
    val_loader, with_labels=True, with_filenames=False
)

if tta:
    pass
else:
    test_feats_cpu, test_names = _precompute_ens_features_for_loader(
        test_loader, with_labels=False, with_filenames=True
    )


class _FeatDataset(torch.utils.data.Dataset):
    def __init__(self, feats_cpu, y_cpu=None):
        self.feats = feats_cpu
        self.y = y_cpu

    def __len__(self):
        return self.feats.size(0)

    def __getitem__(self, idx):
        if self.y is None:
            return self.feats[idx]
        return self.feats[idx], self.y[idx]


feat_train_loader = DataLoader(
    _FeatDataset(train_feats_cpu, train_y_cpu),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

feat_val_loader = DataLoader(
    _FeatDataset(val_feats_cpu, val_y_cpu),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

global_step = 0
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for feats_cpu, y_cpu in feat_train_loader:
        feats = feats_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = linear_head(feats)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_step += 1

        running_loss += float(loss.item()) * feats.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += feats.size(0)

    linear_head.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for feats_cpu, y_cpu in feat_val_loader:
            feats = feats_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = linear_head(feats)
            pred = torch.argmax(logits, dim=1)
            val_correct += (pred == y).sum().item()
            val_total += feats.size(0)
    val_acc = val_correct / max(1, val_total)

    print(
        f"epoch {epoch+1}/{epochs} | loss={running_loss/total:.4f} | train_acc={correct/total:.4f} | val_acc={val_acc:.4f}"
    )
    linear_head.train()

linear_head.eval()
model_a.eval()
model_b.eval()

all_names = []
all_preds = []

with torch.no_grad():
    if tta:
        for model_a_inputs, model_b_inputs, filenames in test_loader:
            bs = len(filenames)

            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                model_a_feats = _vit_extract_features(model_a, model_a_inputs)
                model_b_feats = _vit_extract_features(model_b, model_b_inputs)

            model_a_batch_feats = torch.stack(torch.split(model_a_feats, bs), dim=0)
            model_b_batch_feats = torch.stack(torch.split(model_b_feats, bs), dim=0)

            model_a_mean = torch.mean(model_a_batch_feats, dim=0)
            model_b_mean = torch.mean(model_b_batch_feats, dim=0)

            model_a_mean = model_a_mean / torch.clamp(
                model_a_mean.norm(p=2, dim=1, keepdim=True), min=1e-12
            )
            model_b_mean = model_b_mean / torch.clamp(
                model_b_mean.norm(p=2, dim=1, keepdim=True), min=1e-12
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                model_a_out = linear_head(model_a_mean)
                model_b_out = linear_head(model_b_mean)

            outputs = ens_w_a * model_a_out + ens_w_b * model_b_out
            pred_labels = torch.argmax(outputs, dim=1).tolist()

            all_names.extend(list(filenames))
            all_preds.extend(pred_labels)
    else:
        feat_test_loader = DataLoader(
            _FeatDataset(test_feats_cpu, None),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        name_offset = 0
        for feats_cpu in feat_test_loader:
            feats = feats_cpu.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = linear_head(feats)
            pred_labels = torch.argmax(logits, dim=1).tolist()

            bs = len(pred_labels)
            batch_names = test_names[name_offset : name_offset + bs]
            name_offset += bs

            all_names.extend(batch_names)
            all_preds.extend(pred_labels)

assert len(all_names) == len(sample_df), (len(all_names), len(sample_df))
assert len(all_preds) == len(sample_df), (len(all_preds), len(sample_df))

my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})

my_submission = sample_df[["image_id"]].merge(my_submission, on="image_id", how="left")
assert (
    my_submission["label"].isna().sum() == 0
), "Missing predictions for some test images."
my_submission["label"] = my_submission["label"].astype(int)

my_submission.to_csv("submission.csv", index=False)
my_submission

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/1834515766.py in <cell line: 0>()
     52     dummy = torch.zeros(1, 3, model_a_img_size, model_a_img_size, device=device)
     53     dfa = _vit_extract_features(model_a, dummy)
---> 54     dfb = _vit_extract_features(model_b, dummy)
     55     dfe = ens_w_a * dfa + ens_w_b * dfb
     56     inferred_in_features = int(dfe.shape[-1])

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/1834515766.py in _vit_extract_features(m, x)
     39             feats = feats[:, 0, :]
     40         return feats
---> 41     feats = m(x)
     42     if isinstance(feats, (tuple, list)):
     43         feats = feats[0]

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in forward(self, x)
    289     def forward(self, x: torch.Tensor):
    290         # Reshape and permute the input tensor
--> 291         x = self._process_input(x)
    292         n = x.shape[0]
    293 

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _process_input(self, x)
    269         n, c, h, w = x.shape
    270         p = self.patch_size
--> 271         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    272         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    273         n_h = h // p

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _assert(condition, message)
   2130             _assert, (condition,), condition, message
   2131         )
-> 2132     assert condition, message
   2133 
   2134 

AssertionError: Wrong image height! Expected 384 but got 224!
