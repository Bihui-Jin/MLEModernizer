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

0.8986098519190088

# 6. Current score

0.79709

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I make the solution run end-to-end by (1) replacing the missing external model checkpoints with torchvision ImageNet backbones (keeping the same “two models + linear head ensemble” core idea), (2) fixing test dataset ordering and DataLoader shuffling so predictions align exactly with `sample_submission.csv`, and (3) making TTA deterministic and correctly averaged so it doesn’t change the submission length or misalign filenames. These changes directly address the FileNotFound/NameError crashes and the “wrong length” submission error. The code always write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.10987) has done: 'I fix the runtime crash by aligning the ViT input resolution to what `torchvision.models.vit_b_16` expects (224×224), while keeping your two-backbone + concatenation + linear head ensemble logic unchanged. I also correct the linear head input dimension to match the actual output sizes of ViT-B/16 (1000) and EfficientNet-B0 (1000) so the forward pass is consistent. Finally, I keep the deterministic TTA behavior and ensure the written `submission.csv` is correctly aligned to `sample_submission.csv` with integer labels and the correct row count.'
- What this solution (achieved 0.7784) has done: 'Your current score is low because the linear head is never trained (it’s randomly initialized), so predictions are essentially random; the smallest legitimate step toward the target is to train only that linear head on the provided `train.csv` while keeping both ImageNet backbones frozen and keeping the same “two models + concatenation + linear head” core logic. I add a minimal training dataset/loader and a short training loop (cross-entropy on 5 classes), then reuse your existing deterministic TTA inference and submission alignment. I also keep image preprocessing consistent between train and test, and ensure the script still finishes under the time limit by training only the small head for a few epochs. This should move accuracy much closer to the target without changing the architecture or inference semantics.'
- What this solution (achieved 0.05531) has done: 'I make two minimal, score-relevant fixes while keeping your “two frozen ImageNet backbones + concatenation + linear head” core logic unchanged. First, I switch the EfficientNet input size from 528 to its native 224 to remove the train/test feature distribution mismatch and reduce wasted resizing (this typically gives a noticeable accuracy gain without changing the model). Second, I align preprocessing to the official torchvision weights for each backbone (same normalization/statistics) and use autocast during inference/training to stay comfortably under the time limit without changing semantics. Everything else (frozen backbones, same linear head, same TTA averaging, same submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'The main crash is coming from `weights.transforms()` returning a `torchvision.transforms.v2._presets.ImageClassification` object in your torchvision version, which does not expose a `.transforms` attribute as your `_strip_resize_crop()` expects; fixing this allow the preprocessing/transforms and DataLoaders to be constructed so later cells (training/inference) can run. I replace the transform-stripping logic with a minimal, stable implementation that keeps the same intent: you do explicit crop/resize in the Dataset, and only apply `ToImage -> ToDtype(float, scale=True) -> Normalize(mean/std)` in the collate step, using the weights’ canonical mean/std. This is a correctness/runtime fix and is score-positive because it ensures the models receive properly normalized inputs consistent with their pretrained weights. No changes to the core “two frozen ImageNet backbones + concatenation + linear head, optional deterministic TTA, train only head” approach are made.'
- What this solution (achieved 0.77093) has done: 'The crash comes from a mismatch between what `CassavaDataset.__getitem__` returns under TTA and what `test_collate` assumes: right now `__getitem__` applies `t(vit_img)` before `self.transform`, but `self.transform` is `None`, so it returns a single PIL image (not a list), and then `len(vit_items[0])` fails. I fix this by making `CassavaDataset` always return a list of PIL images when `ttas` is provided (independent of `transform`), and letting the collate function apply the correct normalization transforms per backbone. This is a minimal runtime/correctness fix that preserves your model/ensemble/training logic and should also improve score by actually enabling the intended deterministic TTA averaging. The rest of the pipeline (head training, inference, and submission alignment to `sample_submission.csv`) remains unchanged.'
- What this solution (achieved 0.69768) has done: 'I make two minimal, score-positive adjustments while keeping your “two frozen ImageNet backbones + concatenation + linear head trained on train.csv + deterministic TTA inference” core logic unchanged. First, I switch the training crop from a hard CenterCrop(600) to a RandomResizedCrop around 600 (still resized to 224), which adds essential train-time augmentation and reduces overfitting of the small linear head. Second, I add label smoothing to CrossEntropyLoss (small, stable regularization) and ensure all randomness is properly seeded across DataLoader workers so the augmentation is deterministic/reproducible. Everything else (models, head dimension, TTA, inference averaging, submission alignment) remains the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.70703) has done: 'Your current score (0.69768) is well below the target (0.89861), so we should make a small, legitimate change that improves generalization without changing the “two frozen ImageNet backbones + concatenation + linear head” core logic. The biggest issue is that you train the head on all `train.csv` and never validate; this can overfit and also wastes the opportunity to pick the best epoch. I add a deterministic stratified train/val split, train the head exactly as before, and simply keep the best head weights by validation accuracy (no early stopping; still runs fixed `head_epochs`). This typically gives a solid accuracy lift for minimal code change and keeps inference/submission alignment unchanged.'
- What this solution (achieved 0.05531) has done: 'Your gap to the target is large (0.707 → 0.899), so we should improve generalization with minimal changes that keep your “two frozen ImageNet backbones + concatenation + linear head” approach intact. The biggest score limiter here is that you are training the head on raw backbone *logits* (1000-d) rather than the penultimate *features*, which are far more transferable; switching to feature extraction preserves the same architecture/training loop (still frozen backbones, still concat, still linear head + CE) but gives a substantial accuracy lift. I also use the canonical classifier input feature sizes (ViT-B/16: 768, EfficientNet-B0: 1280) and set the linear head input dim accordingly (2048). Everything else (dataset, augmentation, split/best-val selection, deterministic TTA, submission alignment) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.76383) has done: 'I fix the runtime crash by replacing the ViT feature extraction call with a torchvision-version-compatible method that returns the same 768-d CLS embedding (so the concatenated head input stays 768+1280). This is a minimal, architecture-preserving change: ViT and EfficientNet remain frozen, we still concatenate features and train only the same linear head with the same loop/epochs/loss/TTA. I also make the ViT feature function robust across torchvision variants by using `vit_model._process_input` + `vit_model.encoder` when `forward_features` is unavailable. This should both unblock end-to-end execution and restore the intended feature-based training/inference for a substantial score improvement toward the target.'
- What this solution (achieved 0.05531) has done: 'Main runtime bottlenecks are (1) repeatedly running two large frozen backbones for every epoch while also doing expensive PIL crop/resize/normalize in Python, and (2) an extremely slow test-time augmentation collate path with nested Python loops. To keep identical training semantics while cutting time, the optimized script caches frozen backbone features per image on-the-fly (train/val) so each image’s ViT/EfficientNet embedding is computed once instead of every epoch, and it switches training/validation loaders to lightweight index-based datasets. For inference, it replaces the nested-loop TTA collate with a vectorized “flatten TTA then normalize once” path that is equivalent but much faster. All changes preserve the same models, head training loop, loss, split, and TTA definitions; they only remove redundant computation and Python overhead.'
- What this solution (achieved 0.79709) has done: 'The timeout is dominated by repeatedly decoding large JPEGs and applying expensive PIL-based crops/resizes inside Python `__getitem__`, plus per-epoch recomputation of frozen backbone features for the entire train set. To keep core logic identical while reducing wall time, I (1) switch to torchvision’s faster `decode_jpeg` pipeline and apply crop/resize/tta directly on tensors (same ops/params), (2) precompute and cache frozen train features once (as you already do for val) and then train the linear head on those cached features for all epochs (same training loop semantics, same optimizer/scheduler/loss), and (3) streamline test-time TTA collation to avoid extra Python loops and reduce CPU overhead. These changes preserve model architecture, transforms, loss, epochs, and evaluation semantics, but remove redundant work and speed up I/O/augmentation substantially.'

# 9. Code solution

## === cell 0
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
random.seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_root = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(data_root, "train.csv")
train_dir = os.path.join(data_root, "train_images/")
test_dir = os.path.join(data_root, "test_images/")
sample_sub_path = os.path.join(data_root, "sample_submission.csv")

eff_img_size = 224
vit_img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

train_head = True
head_epochs = 10
head_lr = 2e-3

label_smoothing = 0.05

val_frac = 0.10
split_seed = 3407

import torchvision

vit_weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
eff_weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1

vit_model = torchvision.models.vit_b_16(weights=vit_weights).to(device)
eff_model = torchvision.models.efficientnet_b0(weights=eff_weights).to(device)

linear_head = torch.nn.Linear(768 + 1280, num_classes).to(device)

use_amp = torch.cuda.is_available()
amp_dtype = torch.float16


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(3407)

Image.MAX_IMAGE_PIXELS = None

try:
    torchvision.set_image_backend("accimage")
except Exception:
    pass

from torchvision.io import read_file, decode_jpeg




## === cell 1
def _norm_from_weights(weights):
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


vit_transforms = _norm_from_weights(vit_weights)
eff_transforms = _norm_from_weights(eff_weights)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
    ]
else:
    ttas = None




## === cell 2
class CassavaTrainDataset(VisionDataset):
    """Train dataset returning (vit_img_tensor, eff_img_tensor, label)."""

    def __init__(
        self, df, data_dir, vit_size, efficient_size, vit_transform, eff_transform
    ):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)

        self.vit_t = vit_transform
        self.eff_t = eff_transform

        self.train_crop = v2.RandomResizedCrop(
            size=(600, 600),
            scale=(0.70, 1.0),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
        )

        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])

        path = os.path.join(self.root, filename)
        img = decode_jpeg(
            read_file(path), mode=torchvision.io.ImageReadMode.RGB
        )  # uint8 [C,H,W]

        img = self.train_crop(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        vit_img = self.vit_t(vit_img)
        eff_img = self.eff_t(eff_img)

        return vit_img, eff_img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)


class CassavaValDataset(VisionDataset):
    """Validation dataset returning (vit_img_tensor, eff_img_tensor, label) with deterministic center-crop."""

    def __init__(
        self, df, data_dir, vit_size, efficient_size, vit_transform, eff_transform
    ):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)

        self.vit_t = vit_transform
        self.eff_t = eff_transform

        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])

        path = os.path.join(self.root, filename)
        img = decode_jpeg(read_file(path), mode=torchvision.io.ImageReadMode.RGB)

        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        vit_img = self.vit_t(vit_img)
        eff_img = self.eff_t(eff_img)

        return vit_img, eff_img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)


class CassavaDataset(VisionDataset):
    """Test dataset.

    Returns:
        vit_img: Tensor [3,H,W] or list[Tensor] if TTA enabled
        eff_img: Tensor [3,H,W] or list[Tensor] if TTA enabled
        filename: image filename
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        vit_transform,
        eff_transform,
        ttas=None,
    ):
        super().__init__(root=data_dir)

        self.vit_t = vit_transform
        self.eff_t = eff_transform

        imgs = []
        with os.scandir(data_dir) as it:
            for e in it:
                if e.is_file():
                    n = e.name
                    if n.lower().endswith(".jpg"):
                        imgs.append(n)
        self.images = sorted(imgs)

        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        path = os.path.join(self.root, filename)
        img = decode_jpeg(read_file(path), mode=torchvision.io.ImageReadMode.RGB)

        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.ttas is not None:
            vit_img = [self.vit_t(t(vit_img)) for t in self.ttas]
            eff_img = [self.eff_t(t(eff_img)) for t in self.ttas]
        else:
            vit_img = self.vit_t(vit_img)
            eff_img = self.eff_t(eff_img)

        return vit_img, eff_img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
train_df_full = pd.read_csv(train_csv_path)

train_parts = []
val_parts = []
for lbl, grp in train_df_full.groupby("label", sort=False):
    grp = grp.sample(frac=1.0, random_state=split_seed).reset_index(drop=True)
    n_val = max(1, int(round(len(grp) * val_frac)))
    val_parts.append(grp.iloc[:n_val])
    train_parts.append(grp.iloc[n_val:])
train_df = (
    pd.concat(train_parts, axis=0)
    .sample(frac=1.0, random_state=split_seed)
    .reset_index(drop=True)
)
val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)

print("train/val sizes:", len(train_df), len(val_df))
print(
    "train label dist:\n", train_df["label"].value_counts(normalize=True).sort_index()
)
print("val label dist:\n", val_df["label"].value_counts(normalize=True).sort_index())

train_dataset = CassavaTrainDataset(
    train_df,
    train_dir,
    vit_img_size,
    eff_img_size,
    vit_transform=vit_transforms,
    eff_transform=eff_transforms,
)
val_dataset = CassavaValDataset(
    val_df,
    train_dir,
    vit_img_size,
    eff_img_size,
    vit_transform=vit_transforms,
    eff_transform=eff_transforms,
)
test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    vit_transform=vit_transforms,
    eff_transform=eff_transforms,
    ttas=ttas,
)


def train_collate(batch):
    vit_imgs, eff_imgs, labels = zip(*batch)
    return torch.stack(vit_imgs, 0), torch.stack(eff_imgs, 0), torch.stack(labels, 0)


def test_collate(batch):
    vit_items, eff_items, fnames = zip(*batch)
    if tta:
        vit_by_t = list(zip(*vit_items))  # length T, each tuple length B
        eff_by_t = list(zip(*eff_items))
        vit_out = [torch.stack(vt, 0) for vt in vit_by_t]
        eff_out = [torch.stack(et, 0) for et in eff_by_t]
        return vit_out, eff_out, list(fnames)
    else:
        return torch.stack(vit_items, 0), torch.stack(eff_items, 0), list(fnames)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=train_collate,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=train_collate,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=test_collate,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 4
def vit_features(x: torch.Tensor) -> torch.Tensor:
    if hasattr(vit_model, "forward_features"):
        out = vit_model.forward_features(x)
        return out[:, 0] if out.ndim == 3 else out

    n = x.shape[0]
    x = vit_model._process_input(x)  # [B, seq_len, hidden_dim]
    cls = vit_model.class_token.expand(n, -1, -1)  # [B,1,hidden_dim]
    x = torch.cat([cls, x], dim=1)  # [B, 1+seq_len, hidden_dim]
    x = vit_model.encoder(x)  # [B, 1+seq_len, hidden_dim]
    return x[:, 0]  # CLS token [B,768]


def eff_features(x: torch.Tensor) -> torch.Tensor:
    return eff_model.features(x).mean(dim=(2, 3))


for p in vit_model.parameters():
    p.requires_grad = False
for p in eff_model.parameters():
    p.requires_grad = False

vit_model.eval()
eff_model.eval()

best_state = None
best_val_acc = -1.0


@torch.no_grad()
def precompute_features_cpu(loader):
    vit_list = []
    eff_list = []
    y_list = []
    for vit_inputs, eff_inputs, labels in loader:
        vit_inputs = vit_inputs.to(device, non_blocking=True)
        eff_inputs = eff_inputs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            v = vit_features(vit_inputs).float().cpu()
            e = eff_features(eff_inputs).float().cpu()
        vit_list.append(v)
        eff_list.append(e)
        y_list.append(labels.cpu())
    return torch.cat(vit_list, 0), torch.cat(eff_list, 0), torch.cat(y_list, 0)


val_vit_feat_cpu, val_eff_feat_cpu, val_y_cpu = precompute_features_cpu(val_loader)
train_vit_feat_cpu, train_eff_feat_cpu, train_y_cpu = precompute_features_cpu(
    train_loader
)

if train_head:
    linear_head.train()
    optimizer = torch.optim.AdamW(
        linear_head.parameters(), lr=head_lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4, min_lr=1e-5
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    feat_ds = torch.utils.data.TensorDataset(
        torch.cat([train_vit_feat_cpu, train_eff_feat_cpu], dim=1), train_y_cpu
    )
    feat_loader = DataLoader(
        feat_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=g,
    )

    for epoch in range(head_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for logit_inputs_cpu, labels_cpu in feat_loader:
            logit_inputs = logit_inputs_cpu.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)

            outputs = linear_head(logit_inputs)  # [B,5]
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        linear_head.eval()
        with torch.no_grad():
            logit_inputs = torch.cat([val_vit_feat_cpu, val_eff_feat_cpu], dim=1).to(
                device, non_blocking=True
            )
            outputs = linear_head(logit_inputs)
            preds = outputs.argmax(dim=1).cpu()
            val_correct = int((preds == val_y_cpu).sum().item())
            val_total = int(val_y_cpu.numel())

        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in linear_head.state_dict().items()
            }

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch+1}/{head_epochs} "
            f"lr={current_lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}"
        )
        linear_head.train()

linear_head.eval()
if best_state is not None:
    linear_head.load_state_dict(best_state)



## === cell 5
all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        base_batch_size = len(filenames)

        if tta:
            vit_inputs = torch.cat(vit_inputs, dim=0).to(device, non_blocking=True)
            eff_inputs = torch.cat(eff_inputs, dim=0).to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vit_out = vit_features(vit_inputs)  # [T*B,768]
                eff_out = eff_features(eff_inputs)  # [T*B,1280]

            vit_out = vit_out.float()
            eff_out = eff_out.float()

            T = 3 if ttas is not None else 1
            vit_batch = vit_out.view(T, base_batch_size, -1)  # [T,B,768]
            eff_batch = eff_out.view(T, base_batch_size, -1)  # [T,B,1280]
            vit_mean = vit_batch.mean(dim=0)  # [B,768]
            eff_mean = eff_batch.mean(dim=0)  # [B,1280]

            logit_inputs = torch.cat([vit_mean, eff_mean], dim=1)  # [B,2048]
            outputs = linear_head(logit_inputs)

            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vit_out = vit_features(vit_inputs)
                eff_out = eff_features(eff_inputs)

            logit_inputs = torch.cat([vit_out.float(), eff_out.float()], dim=1)
            outputs = linear_head(logit_inputs)

            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("Preds:", len(all_preds), "Names:", len(all_names))



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if merged["label"].isna().any():
    fill_label = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    merged["label"] = merged["label"].fillna(fill_label).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

merged.to_csv("submission.csv", index=False)
print(merged.head())
print("submission.csv rows:", len(merged))
print("label value counts:\n", merged["label"].value_counts(dropna=False).sort_index())
