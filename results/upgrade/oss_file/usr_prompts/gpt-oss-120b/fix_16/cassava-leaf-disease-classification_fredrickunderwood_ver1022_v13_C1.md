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

3.12

# 3. Installed packages

albumentations==2.0.8
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

# 5. Code solution

## === cell 0
import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import timm
from albumentations import (
    Compose,
    Resize,
    Transpose,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    HueSaturationValue,
    RandomBrightnessContrast,
    Normalize,
    CoarseDropout,
    GridDropout,
    OneOf,
    CenterCrop,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

torch.backends.cudnn.benchmark = True

INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 64  # used for batch inference
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 10




## === cell 1
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p_t = y_true * y_hat + (1 - y_true) * (1 - y_hat)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




## === cell 2
def lr_tune(epoch, num_epochs=NUM_EPOCHS):
    lr_start = LR_START
    lr_max = LR_MAX
    lr_final = LR_FINAL
    lr_warmup_epoch = 4
    lr_sustain_epoch = 0
    lr_decay_epoch = num_epochs - lr_warmup_epoch - lr_sustain_epoch - 1

    if epoch <= lr_warmup_epoch:
        lr = lr_start + (lr_max - lr_start) * (epoch / lr_warmup_epoch) ** 2.5
    elif epoch < lr_warmup_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        epoch_diff = epoch - lr_warmup_epoch - lr_sustain_epoch
        decay_factor = (epoch_diff / lr_decay_epoch) * math.pi
        decay_factor = (torch.cos(torch.tensor(decay_factor)).numpy() + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr


x = list(range(NUM_EPOCHS))
y = [lr_tune(i) for i in x]
plt.plot(x, y)
plt.title("Learning‑rate schedule")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.close()  # avoid blocking in headless environment



## === cell 3
train_augs = Compose(
    [
        Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        CoarseDropout(p=0.5),
        GridDropout(p=0.5),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

valid_augs = Compose(
    [
        Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
)



## === cell 4
test_augs = Compose(
    [
        OneOf(
            [
                Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
                CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
                Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),  # fallback
            ],
            p=1.0,
        ),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 5
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)



## === cell 6
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)



## === cell 7
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)



## === cell 8
torch.cuda.empty_cache()




## === cell 9
def safe_load(model, path):
    try:
        model_param = torch.load(path, map_location=DEVICE)
        new_state = {
            k[7:] if k.startswith("module.") else k: v for k, v in model_param.items()
        }
        model.load_state_dict(new_state, strict=False)
        print(f"Loaded weights from {path}")
    except FileNotFoundError:
        print(
            f"Weight file {path} not found – using pretrained model without fine‑tuned weights."
        )
    return model


my_model_1 = safe_load(my_model_1, os.path.join(INPUT_PATH, RESNEXT_PATH))
my_model_2 = safe_load(my_model_2, os.path.join(INPUT_PATH, B4_PATH))

if torch.cuda.device_count() > 1:
    my_model_1 = nn.DataParallel(my_model_1)
    my_model_2 = nn.DataParallel(my_model_2)

my_model_1 = my_model_1.to(DEVICE)
my_model_2 = my_model_2.to(DEVICE)

test_image_list = sorted(
    [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")]
)


class ImageDataset(torch.utils.data.Dataset):
    """Loads images from disk once and keeps them in RAM."""

    def __init__(self, image_names, base_path, aug):
        self.image_names = image_names
        self.base_path = base_path
        self.aug = aug
        self.images = [
            np.array(Image.open(os.path.join(self.base_path, name)).convert("RGB"))
            for name in self.image_names
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = self.images[idx]
        transformed = self.aug(image=img)
        return transformed["image"]


def _batch_inference(model, image_names, aug, base_path):
    """
    Return averaged logits over TTA as a CPU tensor.
    """
    model.eval()
    dataset = ImageDataset(image_names, base_path, aug)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    all_logits = []
    for _ in range(TTA):
        tta_logits = []
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = model(batch)  # (B, OUT_FEATURES)
            tta_logits.append(out.cpu())
        tta_logits = torch.cat(tta_logits, dim=0)  # (N, OUT_FEATURES)
        all_logits.append(tta_logits)

    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # (N, OUT_FEATURES)
    return avg_logits


val_df = pd.read_csv(TRAIN_CSV_PATH).reset_index(drop=True)
val_image_names = val_df["image_id"].tolist()

logits_val_1 = _batch_inference(
    my_model_1, val_image_names, valid_augs, TRAIN_IMAGE_PATH
)
logits_val_2 = _batch_inference(
    my_model_2, val_image_names, valid_augs, TRAIN_IMAGE_PATH
)

labels_val = val_df["label"].values


def ensemble_accuracy_weighted(w):
    """Blend *probabilities* from each model, then compute accuracy."""
    probs1 = F.softmax(logits_val_1, dim=1)
    probs2 = F.softmax(logits_val_2, dim=1)
    blended = probs1 * w + probs2 * (1 - w)
    pred = torch.argmax(blended, dim=1).cpu().numpy()
    return (pred == labels_val).mean()


best_w, best_acc = 0.5, 0.0
for w in np.arange(0.0, 1.001, 0.01):
    acc = ensemble_accuracy_weighted(w)
    if acc > best_acc:
        best_acc, best_w = acc, w

print(
    f"Chosen ensemble weight for model1 (resnext): {best_w:.2f} (validation acc={best_acc:.4f})"
)



## === cell 10
logits_test_1 = _batch_inference(
    my_model_1, test_image_list, test_augs, TEST_IMAGE_PATH
)
logits_test_2 = _batch_inference(
    my_model_2, test_image_list, test_augs, TEST_IMAGE_PATH
)

probs_test_1 = F.softmax(logits_test_1, dim=1)
probs_test_2 = F.softmax(logits_test_2, dim=1)

blended_test_probs = probs_test_1 * best_w + probs_test_2 * (1 - best_w)
label = blended_test_probs.argmax(dim=1).cpu().numpy()

df_submission = pd.DataFrame(
    {
        "image_id": test_image_list,
        "label": label,
    }
)
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
