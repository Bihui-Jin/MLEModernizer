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

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8677848292535509

# 6. Current score

0.73879

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42265) has done: 'I fix the Albumentations v2 API break that prevents `tta_transform` from being created (causing the downstream `NameError`). I also make the code robust to missing external weight files under `/kaggle/input/...` by falling back to torchvision ImageNet weights for EfficientNetV2-S (same architecture) so the notebook can run end-to-end and generate a valid CSV. Finally, I correct the DataLoader/unpacking so `image_id` is collected as a plain string (not a tuple/list) and ensure the submission length/order matches `sample_submission.csv` exactly.'
- What this solution (achieved 0.13528) has done: 'Your current score is far below the target because the code is effectively doing inference with an ImageNet-pretrained EfficientNet whose classifier head is randomly initialized (since the external .pth files are not present), and it also uses a very strong “train-like” augmentation pipeline for TTA (random crops/rotations/noise) that destroys signal at inference. To move accuracy up toward the target while preserving your overall approach (single-model TTA + argmax), I (1) switch TTA to safe test-time transforms (resize/center-crop + flips only) and (2) if fine-tuned weights are missing, fall back to a proper 5-class cassava checkpoint from HuggingFace via `torch.hub` (same core idea: load a pretrained classifier and run TTA inference). These are minimal, execution-safe changes that should substantially improve accuracy without changing the fundamental inference semantics or adding training. The submission writing/order logic be kept the same.'
- What this solution (achieved 0.61099) has done: 'Your score is far below the target mainly because, when your fine-tuned `.pth` files aren’t present, the EfficientNetV2-S head remains randomly initialized, so predictions are near-random. To move accuracy sharply upward while preserving your core inference approach (single-model TTA + argmax), I (1) load ImageNet weights first and then swap the classifier to 5 classes (so at least the backbone is meaningful), and (2) add a local fallback that uses the provided `train.csv` to build a class-frequency prior head (a deterministic, lightweight calibration) when cassava fine-tuned weights aren’t available. I also remove the extra CLAHE transform from the unused pipeline to avoid confusion, keep TTA as safe geometric transforms, and ensure the submission aligns exactly to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.86778), so we should improve accuracy with minimal, low-risk changes while keeping the same inference-only + TTA + argmax core logic. The biggest gain here is eliminating randomness in TTA: because your TTA pipeline uses probabilistic flips, repeating it 5 times produces noisy averages; switching to a small deterministic TTA set (identity / hflip / vflip / hvflip) keeps the same semantics but yields more stable and typically higher accuracy. I also make the EfficientNet classifier dropout inactive at inference by setting it to `p=0.0` (dropout is already disabled by `eval()`, but keeping `p=0.8` can still be problematic if the model is ever not in eval or exported; this is a safe stability tweak). Finally, I fix the model-selection order bug so you actually prefer the third checkpoint when present (you were checking `eff6_loaded` but only after `eff1_loaded`), which can improve results if that checkpoint exists.'
- What this solution (achieved 0.77653) has done: 'Your score is far below the target (0.61099 vs 0.86778, higher-is-better), and the main remaining issue is that if the cassava fine-tuned checkpoints are missing, you effectively infer with a non-cassava classifier (class-prior head), which caps accuracy. I keep your exact inference core (single EfficientNetV2-S + deterministic 4-view TTA + argmax) but add an on-the-fly, lightweight fine-tuning fallback on `train.csv` images when no fine-tuned `.pth` is available, so the classifier becomes cassava-specific. This training is only activated when all external checkpoints are absent, uses the same model/loss (cross-entropy), and stays within Kaggle constraints/time by freezing the backbone and training only the classifier for a small number of epochs. I also add a small stratified validation split to ensure labels are read correctly and keep submission ordering aligned to `sample_submission.csv`.'
- What this solution (achieved 0.72048) has done: 'Your current score (0.77653) is below the target (0.86778), so we should improve accuracy with minimal, low-risk changes while keeping the same EfficientNetV2-S + deterministic 4-view TTA + argmax core. The biggest remaining weakness is the fallback head-only training: it uses weak/partly-inappropriate augmentations (center-crop only; plus random flips for train) and does not address class imbalance, which tends to cap validation/generalization. I keep the same fallback training loop and loss (CrossEntropy) but (1) switch fallback train preprocessing to a stronger yet safe “resize + random resized crop + flips” (still standard for this task) and (2) add class-weighted CrossEntropyLoss computed from train split to reduce bias toward frequent classes. I also make inference a bit more stable by running the model in channels_last + autocast during inference/training (same semantics; negligible FP differences) to allow a slightly larger effective throughput without changing the approach, staying within the time limit.'
- What this solution (achieved 0.73879) has done: 'Your gap to the target is ~0.1473 absolute (0.72048 → 0.86778), so we should improve accuracy with very small, safe changes while keeping your same EfficientNetV2-S + (optional) fallback head-only training + deterministic 4-view TTA + argmax pipeline. The biggest low-risk win is to use a stronger, standard ImageNet-style test preprocessing (Resize to a slightly larger size + CenterCrop to 384) and to match training preprocessing to the same crop size, which typically improves generalization without changing the approach. I also make the fallback head-only training a bit more effective but still minimal by (1) training for 3 epochs (still light) and (2) using label smoothing in CrossEntropy to reduce overconfidence/imbalance sensitivity while keeping the same loss family and semantics. Finally, I keep submission ordering exactly aligned to `sample_submission.csv` and ensure everything runs end-to-end.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.model_selection import StratifiedShuffleSplit



## === cell 1
num_tta = 4  # deterministic 4-view TTA (id/hflip/vflip/hvflip) for stable averaging

fallback_train_if_no_weights = True
fallback_epochs = 3
fallback_batch_size = 64
fallback_lr = 2e-3
fallback_num_workers = 2
fallback_val_fraction = 0.1
seed = 42




## === cell 2
def seed_everything(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed)



## === cell 3
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

assert os.path.isdir(test_image_dir), f"Missing test_image_dir: {test_image_dir}"
assert os.path.isdir(train_image_dir), f"Missing train_image_dir: {train_image_dir}"
assert os.path.isfile(sample_sub_path), f"Missing sample submission: {sample_sub_path}"
assert os.path.isfile(train_csv_path), f"Missing train.csv: {train_csv_path}"



## === cell 4
test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 5
base_transform = A.Compose(
    [
        A.Resize(448, 448, interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

fallback_train_transform = A.Compose(
    [
        A.Resize(448, 448, interpolation=cv2.INTER_LINEAR),
        A.RandomResizedCrop(
            size=(384, 384), scale=(0.70, 1.0), ratio=(0.85, 1.15), p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name


class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = self.transform(image=image)["image"]
        y = torch.tensor(label, dtype=torch.long)
        return x, y




## === cell 7
def tta_predict_single_model(model, image, base_transform, device, n_tta=4):
    """
    Deterministic TTA: id/hflip/vflip/hvflip, then average probs and argmax outside.
    """
    model.eval()
    tta_predictions = []

    if not (isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 3):
        raise ValueError(
            f"Image must be HxWx3 numpy array, got type={type(image)}, shape={getattr(image, 'shape', None)}"
        )

    views = [
        image,  # identity
        np.ascontiguousarray(image[:, ::-1, :]),  # hflip
        np.ascontiguousarray(image[::-1, :, :]),  # vflip
        np.ascontiguousarray(image[::-1, ::-1, :]),  # hvflip
    ]

    if n_tta is None:
        n_tta = 4
    n_tta = int(n_tta)
    if n_tta <= 0:
        raise ValueError(f"n_tta must be positive, got {n_tta}")

    chosen = [views[i % 4] for i in range(n_tta)]

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for v in chosen:
            augmented = base_transform(image=v)["image"].unsqueeze(0).to(device)
            augmented = augmented.contiguous(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(augmented)
                probs = F.softmax(output, dim=1)
            tta_predictions.append(probs.float())

    avg_probs = torch.mean(torch.stack(tta_predictions, dim=0), dim=0)
    return avg_probs




## === cell 8
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(test_df, test_image_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=identity_collate,
)



## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 10
def build_class_prior_head_from_train_csv(
    train_csv, in_features, num_classes=5, temperature=1.0
):
    train_df = pd.read_csv(train_csv)
    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values.astype(np.float64)
    )
    probs = counts / max(counts.sum(), 1.0)
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature)

    head = nn.Linear(in_features, num_classes, bias=True)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor(logits, dtype=torch.float32))
    return head


def load_efficientnet_v2_s_5class(device, weights_path=None, train_csv_for_prior=None):
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(in_features, 5))

    loaded_finetuned = False
    if weights_path is not None and os.path.isfile(weights_path):
        sd = torch.load(weights_path, map_location=device)
        model.load_state_dict(sd, strict=True)
        loaded_finetuned = True
    else:
        if train_csv_for_prior is not None:
            prior_head = build_class_prior_head_from_train_csv(
                train_csv_for_prior,
                in_features=in_features,
                num_classes=5,
                temperature=1.25,
            )
            model.classifier = nn.Sequential(nn.Dropout(p=0.0), prior_head)

    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, loaded_finetuned


eff1_path = "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth"
eff7_path = "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth"
eff6_path = "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth"

efficientnet_model_1, eff1_loaded = load_efficientnet_v2_s_5class(
    device, weights_path=eff1_path, train_csv_for_prior=train_csv_path
)
efficientnet_model_7, eff7_loaded = load_efficientnet_v2_s_5class(
    device, weights_path=eff7_path, train_csv_for_prior=train_csv_path
)
efficientnet_model_8, eff6_loaded = load_efficientnet_v2_s_5class(
    device, weights_path=eff6_path, train_csv_for_prior=train_csv_path
)

print(
    f"Loaded finetuned weights: eff1={eff1_loaded}, eff7={eff7_loaded}, eff8={eff6_loaded}"
)



## === cell 11
if eff7_loaded:
    inference_model = efficientnet_model_7
elif eff6_loaded:
    inference_model = efficientnet_model_8
elif eff1_loaded:
    inference_model = efficientnet_model_1
else:
    inference_model = efficientnet_model_7  # prior-backed in this case; any is fine.




## === cell 12
def fallback_train_classifier_head(model, train_csv, train_dir, device):
    train_df_all = pd.read_csv(train_csv)
    y = train_df_all["label"].astype(int).values

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=fallback_val_fraction, random_state=seed
    )
    tr_idx, va_idx = next(splitter.split(np.zeros_like(y), y))
    tr_df = train_df_all.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df_all.iloc[va_idx].reset_index(drop=True)

    tr_ds = CassavaTrainDataset(tr_df, train_dir, transform=fallback_train_transform)
    va_ds = CassavaTrainDataset(va_df, train_dir, transform=base_transform)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=fallback_batch_size,
        shuffle=True,
        num_workers=fallback_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=fallback_batch_size,
        shuffle=False,
        num_workers=fallback_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=fallback_lr, weight_decay=1e-4
    )

    counts = (
        tr_df["label"]
        .value_counts()
        .reindex(range(5), fill_value=0)
        .values.astype(np.float32)
    )
    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.clip(freq, 1e-6, None)
    weights = weights / weights.mean()  # keep loss scale stable
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model.train()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(fallback_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in tqdm(
            tr_loader,
            desc=f"Fallback train epoch {epoch+1}/{fallback_epochs}",
            leave=False,
        ):
            xb = xb.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())

        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb)
                pred = logits.argmax(dim=1)
                v_correct += int((pred == yb).sum().item())
                v_total += int(yb.numel())
        model.train()

        tr_acc = correct / max(total, 1)
        va_acc = v_correct / max(v_total, 1)
        print(
            f"Fallback epoch {epoch+1}: train_loss={running_loss/max(total,1):.4f} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}"
        )

    model.eval()
    return model


any_finetuned_loaded = bool(eff1_loaded or eff6_loaded or eff7_loaded)
if (not any_finetuned_loaded) and fallback_train_if_no_weights:
    inference_model = fallback_train_classifier_head(
        inference_model,
        train_csv=train_csv_path,
        train_dir=train_image_dir,
        device=device,
    )



## === cell 13
ensemble_predictions = []
image_names = []

with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader)):
        image, img_name = batch[0]
        if not isinstance(img_name, str):
            img_name = str(img_name)

        probs = tta_predict_single_model(
            inference_model, image, base_transform, device, n_tta=num_tta
        )

        final_pred = int(probs.argmax(dim=1).cpu().item())
        ensemble_predictions.append(final_pred)
        image_names.append(img_name)

len(image_names), len(ensemble_predictions), test_df.shape



## === cell 14
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df[["image_id"]].copy()
submission_df["label"] = submission_df["image_id"].map(pred_map)

missing = int(submission_df["label"].isna().sum())
if missing:
    train_df = pd.read_csv(train_csv_path)
    most_freq = int(train_df["label"].value_counts().idxmax())
    submission_df["label"] = submission_df["label"].fillna(most_freq).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(
    f"Saved {submission_path} with shape={submission_df.shape}, missing_filled={missing}"
)
submission_df.head()
