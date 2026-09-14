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

0.8990631610758537

# 6. Current score

0.74626

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blocker by removing the hard dependency on missing `/kaggle/input/...` weight files and instead instantiate the same kind of inference pipeline with a torchvision backbone that is available in the Kaggle environment. I also fix submission correctness issues: `shuffle=True` causes misalignment/duplicates and the current TTA dataset/collation returns nested lists that don’t batch safely, which can lead to wrong row counts. To preserve the core “two-model ensemble + optional TTA + softmax + argmax” semantics, I keep the same structure but implement TTA as a fixed, deterministic set of transforms applied per-image and aggregate logits across TTA samples. Finally, I ensure the submission exactly matches `sample_submission.csv` ordering/length and always writes `submission.csv`.'
- What this solution (achieved 0.1136) has done: 'I fix the ViT input-size mismatch causing the runtime AssertionError by resizing ViT inputs to the model’s required `image_size` (224 for `vit_b_16`) instead of 384, while keeping the same two-model ensemble + optional TTA + softmax + argmax semantics. I also correct a small ensemble weighting bug where logits are divided by 2.0 after applying weights (this unintentionally shrinks logits and can hurt accuracy); the weighted sum be used directly. Finally, I keep `shuffle=False` and ensure the submission is aligned to `sample_submission.csv` and always written as `submission.csv`.'
- What this solution (achieved 0.76794) has done: 'Your current score is far below the target, and the main reason is that you are using ImageNet-pretrained backbones with randomly re-initialized 5-class heads (no training), so predictions are essentially arbitrary. To keep the same overall logic (two-model ensemble + optional TTA + softmax + argmax) while making the smallest score-relevant change, I add a lightweight training phase that fine-tunes only the classification heads on `train.csv` images (backbones frozen) and then runs the same inference/submission pipeline. I also keep your resizing/TTA behavior and ensure test order stays aligned with `sample_submission.csv`. This should move accuracy substantially upward toward your target without changing the architecture, loss, or inference semantics.'
- What this solution (achieved 0.13714) has done: 'You’re already training only the heads, so the biggest score lift with minimal logic change is to make the head-training better match the competition distribution without touching the backbone/training loop semantics. I (1) add standard ImageNet augmentation (RandomResizedCrop + HFlip) for the training dataset only (inference stays identical), (2) use a simple stratified holdout split to report a meaningful validation accuracy (no early stopping; still trains the same epochs), and (3) address class imbalance by using weighted CrossEntropyLoss (same loss family/semantics) computed from train labels. These changes typically move accuracy substantially upward from ~0.77 toward your ~0.90 target while keeping the two-model ensemble, head-only fine-tuning, TTA inference, and submission alignment unchanged.'
- What this solution (achieved 0.77765) has done: 'The crash is from a size mismatch: in the training dataset you apply `RandomResizedCrop(size=(600,600))` after already resizing to 224/528, which forces the ViT branch back to 600×600 and triggers the ViT assertion. I fix this by changing the training transform to operate at the already-resized resolution (i.e., do augmentation without changing spatial size), preserving the same “head-only fine-tune + two-model weighted ensemble + optional TTA” core logic. I also make the non-TTA inference path use the same ensemble weighting as the TTA path (this is a small correctness fix and should improve score toward the target). Finally, I keep submission alignment exactly matching `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.77541) has done: 'I fix the first runtime blocker by getting mean/std from the weights’ recommended transforms (TorchVision 0.21 no longer guarantees `weights.meta["mean"]`/`["std"]`). Next, I fix the cascading `NameError`s by ensuring the dataset classes are actually defined (cell 2 must execute successfully) and by renumbering cells so execution order is consistent with the provided “cell 0” content. These changes are score-neutral but make the notebook run end-to-end and reliably write `submission.csv` with the exact `sample_submission.csv` ordering/length.'
- What this solution (achieved 0.77541) has done: 'To move your score upward toward 0.899 (you’re currently 0.775), the smallest score-relevant fix is to make your training objective match your inference objective: right now you train on an ensemble of two frozen backbones, but you only update the two heads independently, so each head never learns to be good on its own at inference time. I keep the exact same models, loss (CrossEntropyLoss), optimizer (AdamW), epochs, TTA, and ensemble logic, but change training to optimize each head with its own CE loss (computed on its own logits) and sum them—this preserves semantics and typically yields a large accuracy gain without “changing the approach.” I also apply the appropriate pretrained transforms’ resizing/cropping consistently (using the weights’ own transforms for sizing, while keeping your center-crop-to-600 behavior) so the backbones see inputs closer to what they were pretrained on. Submission writing/ordering remains identical.'
- What this solution (achieved 0.74626) has done: 'The timeout is dominated by heavy PIL-based JPEG decoding plus running two large backbones (ViT-B/16 and EfficientNet-B0) over 18.7k training images (with augmentations) and again over test images with 6x TTA. To keep identical training/inference semantics while cutting overhead, I (1) switch dataset decoding to torchvision’s faster `read_image` path (still deterministic, same RGB content), (2) enable persistent DataLoader workers + prefetching to avoid worker respawn cost and improve input pipeline throughput, and (3) use `torch.inference_mode()` and channels-last layout for faster GPU execution without changing math. All model logic, losses, epochs, TTA variants, and weighting remain the same.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = False
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT = Path("/kaggle/input/cassava-leaf-disease-classification")
test_dir = str(DATA_ROOT / "test_images")
train_csv_path = str(DATA_ROOT / "train.csv")
train_dir = str(DATA_ROOT / "train_images")
sample_sub_path = str(DATA_ROOT / "sample_submission.csv")

eff_img_size = 528
vit_img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

train_head_epochs = 2
train_head_lr = 3e-3


def build_backbone(model_name: str):
    if model_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
        input_size = model.image_size  # 224
    elif model_name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        input_size = eff_img_size
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model.to(device), input_size


vit_model, vit_expected_size = build_backbone("vit_b_16")
eff_model, _ = build_backbone("efficientnet_b0")

assert (
    vit_expected_size == vit_img_size
), f"vit_img_size must match model.image_size ({vit_expected_size})"

vit_model = vit_model.to(memory_format=torch.channels_last)
eff_model = eff_model.to(memory_format=torch.channels_last)

normalizer = torch.nn.Softmax(dim=1)




## === cell 1
vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
eff_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

vit_preprocess = (
    vit_weights.transforms()
)  # includes resize/crop + normalize (we'll reuse its mean/std)
eff_preprocess = eff_weights.transforms()

vit_mean = list(vit_preprocess.mean)
vit_std = list(vit_preprocess.std)
eff_mean = list(eff_preprocess.mean)
eff_std = list(eff_preprocess.std)

vit_geom_eval = v2.Compose(
    [
        v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop((vit_img_size, vit_img_size)),
    ]
)
eff_geom_eval = v2.Compose(
    [
        v2.Resize(
            int(round(eff_img_size / 0.875)),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.CenterCrop((eff_img_size, eff_img_size)),
    ]
)

vit_base_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=vit_mean, std=vit_std),
    ]
)

eff_base_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=eff_mean, std=eff_std),
    ]
)

vit_train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply(
            [v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
            p=0.5,
        ),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=vit_mean, std=vit_std),
    ]
)

eff_train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply(
            [v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
            p=0.5,
        ),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=eff_mean, std=eff_std),
    ]
)


def tta_variants(img):
    variants = [img]
    variants.append(v2.functional.hflip(img))
    variants.append(v2.functional.vflip(img))
    variants.append(
        v2.functional.rotate(
            img, 90, interpolation=InterpolationMode.BICUBIC, expand=False
        )
    )
    variants.append(
        v2.functional.rotate(
            img, 180, interpolation=InterpolationMode.BICUBIC, expand=False
        )
    )
    variants.append(
        v2.functional.rotate(
            img, 270, interpolation=InterpolationMode.BICUBIC, expand=False
        )
    )
    return variants


from torchvision.io import read_image, ImageReadMode


def _read_rgb_tensor(path: str) -> torch.Tensor:
    return read_image(path, mode=ImageReadMode.RGB)


class CassavaDataset(VisionDataset):
    """Inference dataset returning tensors shaped for easy batching.

    Returns:
      vit_imgs: Tensor [T, 3, vit, vit] if TTA else [3, vit, vit]
      eff_imgs: Tensor [T, 3, eff, eff] if TTA else [3, eff, eff]
      filename: str
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        vit_transform=None,
        eff_transform=None,
        use_tta=False,
    ):
        super().__init__(root=data_dir)
        self.vit_transform = vit_transform
        self.eff_transform = eff_transform
        self.use_tta = use_tta

        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )

        self.vit_geom_eval = vit_geom_eval
        self.eff_geom_eval = eff_geom_eval

    def __getitem__(self, idx):
        filename = self.images[idx]
        path = os.path.join(self.root, filename)

        img = _read_rgb_tensor(path)  # uint8 [C,H,W]

        vit_img = self.vit_geom_eval(img)
        eff_img = self.eff_geom_eval(img)

        if self.use_tta:
            vit_list = tta_variants(vit_img)
            eff_list = tta_variants(eff_img)

            vit_t = torch.stack([self.vit_transform(x) for x in vit_list], dim=0)
            eff_t = torch.stack([self.eff_transform(x) for x in eff_list], dim=0)
            return vit_t, eff_t, filename

        vit_t = self.vit_transform(vit_img)
        eff_t = self.eff_transform(eff_img)
        return vit_t, eff_t, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Training dataset returning (vit_tensor, eff_tensor, label)."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        vit_size: int,
        eff_size: int,
        vit_transform=None,
        eff_transform=None,
        use_random_crop: bool = False,
    ):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.vit_transform = vit_transform
        self.eff_transform = eff_transform
        self.use_random_crop = use_random_crop

        self.vit_geom_eval = vit_geom_eval
        self.eff_geom_eval = eff_geom_eval

        self.vit_rc = v2.RandomResizedCrop(
            size=(vit_size, vit_size),
            scale=(0.80, 1.00),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.eff_rc = v2.RandomResizedCrop(
            size=(eff_size, eff_size),
            scale=(0.80, 1.00),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        path = os.path.join(self.root, filename)

        img = _read_rgb_tensor(path)  # uint8 [C,H,W]

        vit_img = self.vit_geom_eval(img)
        eff_img = self.eff_geom_eval(img)

        if self.use_random_crop:
            vit_img = self.vit_rc(vit_img)
            eff_img = self.eff_rc(eff_img)

        vit_t = self.vit_transform(vit_img)
        eff_t = self.eff_transform(eff_img)
        return vit_t, eff_t, label




## === cell 2
def freeze_all_but_head(model_name: str, model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    if model_name == "vit_b_16":
        for p in model.heads.head.parameters():
            p.requires_grad = True
        return model.heads.head.parameters()
    elif model_name == "efficientnet_b0":
        for p in model.classifier[1].parameters():
            p.requires_grad = True
        return model.classifier[1].parameters()
    else:
        raise ValueError(model_name)


def stratified_split_df(df: pd.DataFrame, val_frac: float = 0.10, seed: int = 3407):
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for _, g in df.groupby("label", sort=False):
        idx = np.array(g.index)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        val_parts.append(df.loc[val_idx])
        train_parts.append(df.loc[tr_idx])
    train_df = (
        pd.concat(train_parts)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )
    return train_df, val_df


train_df_full = pd.read_csv(train_csv_path)
train_df, val_df = stratified_split_df(train_df_full, val_frac=0.10, seed=SEED)
print("train size:", len(train_df), "val size:", len(val_df))
print(
    "train label dist:\n", train_df["label"].value_counts(normalize=True).sort_index()
)
print("val label dist:\n", val_df["label"].value_counts(normalize=True).sort_index())

vit_head_params = freeze_all_but_head("vit_b_16", vit_model)
eff_head_params = freeze_all_but_head("efficientnet_b0", eff_model)

trainable_params = list(vit_head_params) + list(eff_head_params)
optimizer = torch.optim.AdamW(trainable_params, lr=train_head_lr, weight_decay=1e-4)

counts = (
    train_df["label"]
    .value_counts()
    .reindex(range(num_classes), fill_value=0)
    .values.astype(np.float32)
)
counts = np.clip(counts, 1.0, None)
class_weights = counts.sum() / counts
class_weights = class_weights / class_weights.mean()
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
print("class_weights:", class_weights)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)

train_dataset = CassavaTrainDataset(
    train_df,
    train_dir,
    vit_img_size,
    eff_img_size,
    vit_transform=vit_train_transform,
    eff_transform=eff_train_transform,
    use_random_crop=True,
)
val_dataset = CassavaTrainDataset(
    val_df,
    train_dir,
    vit_img_size,
    eff_img_size,
    vit_transform=vit_base_transform,
    eff_transform=eff_base_transform,
    use_random_crop=False,
)

loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

vit_model.train()
eff_model.train()

vit_w = 0.45
eff_w = 0.55

for epoch in range(train_head_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for vit_x, eff_x, y in train_loader:
        vit_x = vit_x.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        eff_x = eff_x.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        vit_logits = vit_model(vit_x)
        eff_logits = eff_model(eff_x)

        loss_vit = criterion(vit_logits, y)
        loss_eff = criterion(eff_logits, y)
        loss = vit_w * loss_vit + eff_w * loss_eff

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * y.size(0)

        logits_ens = vit_w * vit_logits + eff_w * eff_logits
        preds = torch.argmax(logits_ens, dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    vit_model.eval()
    eff_model.eval()
    v_correct, v_total = 0, 0
    with torch.inference_mode():
        for vit_x, eff_x, y in val_loader:
            vit_x = vit_x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            eff_x = eff_x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            y = y.to(device, non_blocking=True)
            vit_logits = vit_model(vit_x)
            eff_logits = eff_model(eff_x)
            logits = vit_w * vit_logits + eff_w * eff_logits
            preds = torch.argmax(logits, dim=1)
            v_correct += int((preds == y).sum().item())
            v_total += int(y.size(0))
    vit_model.train()
    eff_model.train()

    print(
        f"epoch {epoch+1}/{train_head_epochs} "
        f"train_loss={running_loss/max(1,total):.4f} train_acc={correct/max(1,total):.4f} "
        f"val_acc={v_correct/max(1,v_total):.4f}"
    )

vit_model.eval()
eff_model.eval()




## === cell 3
test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    vit_transform=vit_base_transform,
    eff_transform=eff_base_transform,
    use_tta=tta,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

all_names = []
all_preds = []

vit_w = 0.45
eff_w = 0.55

with torch.inference_mode():
    for vit_inputs, eff_inputs, filenames in test_loader:
        if tta:
            B, T = vit_inputs.shape[0], vit_inputs.shape[1]

            vit_flat = (
                vit_inputs.view(B * T, *vit_inputs.shape[2:])
                .to(device, non_blocking=True)
                .contiguous(memory_format=torch.channels_last)
            )
            eff_flat = (
                eff_inputs.view(B * T, *eff_inputs.shape[2:])
                .to(device, non_blocking=True)
                .contiguous(memory_format=torch.channels_last)
            )

            vit_logits = vit_model(vit_flat).view(B, T, num_classes).mean(dim=1)
            eff_logits = eff_model(eff_flat).view(B, T, num_classes).mean(dim=1)

            outputs = vit_w * vit_logits + eff_w * eff_logits
            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, dim=1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            eff_inputs = eff_inputs.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )

            vit_logits = vit_model(vit_inputs)
            eff_logits = eff_model(eff_inputs)

            outputs = vit_w * vit_logits + eff_w * eff_logits
            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("predictions:", len(all_preds), "filenames:", len(all_names))
print("unique filenames:", len(set(all_names)))




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(all_names, all_preds))

if len(pred_map) != len(sample_sub):
    if len(all_preds) > 0:
        default_label = int(pd.Series(all_preds).value_counts().idxmax())
    else:
        default_label = 0
    sample_sub["label"] = (
        sample_sub["image_id"].map(pred_map).fillna(default_label).astype(int)
    )
else:
    sample_sub["label"] = sample_sub["image_id"].map(pred_map).astype(int)

assert len(sample_sub) == 2676, f"Unexpected submission length: {len(sample_sub)}"
assert list(sample_sub.columns) == ["image_id", "label"]

out_path = "submission.csv"
sample_sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sample_sub.head())
