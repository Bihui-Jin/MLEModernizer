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

# 5. Target score

0.894983378664249

# 6. Current score

0.48954

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13453) has done: 'I fix the Albumentations API breakage by updating `RandomResizedCrop` calls to the v2 signature so augmentations construct correctly. I also fix the missing-weights crash by making the script robust to absent `../input/ensemble-1023/*.pth` files: if weights are not found, it fall back to using `timm` pretrained weights for the same architectures (preserving the same models and ensembling logic) so it can run end-to-end. Finally, I ensure the submission is aligned to `sample_submission.csv` ordering (and keep the filename `submission.csv`) to guarantee a valid Kaggle submission format.'
- What this solution (achieved 0.13453) has done: 'I fix the Albumentations v2 API break causing the crash by replacing the removed `A.Cutout` augmentation with its supported equivalent (`A.CoarseDropout`) while keeping the same augmentation intent. I also correct a logic bug in the focal loss (even if it’s not used for inference here) to prevent future silent misuse: `p_t` must be computed from probabilities, not logits. Finally, I keep the existing ensembling/inference logic intact and ensure the script always writes `submission.csv` with the exact `image_id,label` format aligned to `sample_submission.csv` ordering.'
- What this solution (achieved 0.11771) has done: 'Your very low score is consistent with an inference-time bug: you L2-normalize raw logits across classes, which distorts class probabilities and tends to ruin argmax decisions. To move toward the target accuracy with minimal disruption, I remove that normalization and instead ensemble using softmax probabilities (standard for accuracy metrics) while keeping the same two models, weights-loading fallback, TTA count, and blend weights. I also keep submission ordering aligned to `sample_submission.csv` and ensure the output `submission.csv` remains `image_id,label`. These changes preserve the core modeling logic but fix the post-processing to match the evaluation semantics.'
- What this solution (achieved 0.48954) has done: 'Your score is far below the target, and the biggest likely cause (without changing your core models) is that your fallback weight loading is unintentionally using *ImageNet-1000 heads* (because `pretrained=True` with `num_classes=5` does not reliably give a true 5-class pretrained head across timm models), leaving your 5-class head essentially random. I keep the same two architectures, same ensembling, same TTA, and same augmentations, but change the fallback to: load a proper ImageNet pretrained backbone, then **recreate and keep your existing 5-class randomly initialized head**. I also make the inference faster/cleaner (without changing semantics) by moving `DataParallel(...).to(device)` outside the per-model function loop and using `torch.inference_mode()`; this should not hurt accuracy and helps stability under the 600s limit. These are minimal changes that should substantially increase accuracy toward your target.'

# 9. Code solution

## === cell 0
import os
import math
import random
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm import tqdm
import timm

warnings.filterwarnings("ignore")



## === cell 1
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"
DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 3


def _resolve_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


TRAIN_CSV_PATH = (
    _resolve_first_existing(
        [
            TRAIN_CSV_PATH,
            "/kaggle/input/cassava-leaf-disease-classification/train.csv",
            "/kaggle/data/cassava-leaf-disease-classification/train.csv",
            "/kaggle/input/train.csv",
            "/kaggle/data/train.csv",
        ]
    )
    or TRAIN_CSV_PATH
)

TEST_IMAGE_PATH = (
    _resolve_first_existing(
        [
            TEST_IMAGE_PATH,
            "/kaggle/input/cassava-leaf-disease-classification/test_images/",
            "/kaggle/data/cassava-leaf-disease-classification/test_images/",
            "/kaggle/input/test_images/",
            "/kaggle/data/test_images/",
        ]
    )
    or TEST_IMAGE_PATH
)

SAMPLE_SUB_PATH = _resolve_first_existing(
    [
        "../input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

DEVICE0 = DEVICES[0] if len(DEVICES) > 0 else torch.device("cpu")




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    """
    Bug fix: focal term must be computed on probabilities (sigmoid(logits)),
    not directly from logits. This keeps semantics correct if used later.
    """

    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor)
        y = y + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor)

    prob = torch.sigmoid(y_hat)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p_t = y_true * prob + (1 - y_true) * (1 - prob)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




## === cell 3
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




## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(p=0.5),
        A.CoarseDropout(p=0.5),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)



## === cell 5
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop(
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    p=1.0,
                ),
            ],
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 6
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)



## === cell 7
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=False)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)




## === cell 8
def _strip_module_prefix(state_dict):
    if isinstance(state_dict, dict) and any(
        k.startswith("module.") for k in state_dict.keys()
    ):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_weights_or_fallback(model, weight_path, model_name_for_fallback, strict=True):
    """
    Change to improve accuracy toward target:
    - If custom weights are missing, load TRUE ImageNet pretrained BACKBONE weights,
      but KEEP the existing randomly initialized 5-class head (do not import a 1000-class head).
    This avoids the prior failure mode where the head remains effectively random/mismatched.
    """
    if weight_path is not None and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        state = _strip_module_prefix(state)
        model.load_state_dict(state, strict=strict)
        return "loaded_custom"

    pretrained_backbone = timm.create_model(model_name_for_fallback, pretrained=True)
    pretrained_state = pretrained_backbone.state_dict()

    drop_prefixes = (
        "fc.",  # resnext
        "classifier.",  # efficientnet
        "head.",  # some timm models
    )
    filtered_state = {
        k: v for k, v in pretrained_state.items() if not k.startswith(drop_prefixes)
    }

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    return (
        f"fallback_backbone_only(missing={len(missing)},unexpected={len(unexpected)})"
    )


resnext_weight_path = os.path.join(INPUT_PATH, RESNEXT_PATH)
b4_weight_path = os.path.join(INPUT_PATH, B4_PATH)

status1 = _load_weights_or_fallback(
    my_model_1, resnext_weight_path, model_name1, strict=False
)
status2 = _load_weights_or_fallback(
    my_model_2, b4_weight_path, model_name2, strict=False
)

print("Model1 weights:", status1)
print("Model2 weights:", status2)



## === cell 9
torch.cuda.empty_cache()

if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "sample_submission.csv not found in expected Kaggle input locations."
    )
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_list = sample_sub["image_id"].values


def _infer_model_proba(model, tta_n):
    model.eval()
    preds = []
    with torch.inference_mode():
        for single_image_name in tqdm(
            test_image_list, total=len(test_image_list), desc=f"Infer (TTA={tta_n})"
        ):
            proba = torch.zeros(OUT_FEATURES, device=DEVICE0)
            for _ in range(tta_n):
                image = Image.open(
                    os.path.join(TEST_IMAGE_PATH, single_image_name)
                ).convert("RGB")
                aug_image = test_augs(image=np.array(image))["image"]
                test_image = aug_image.unsqueeze(0).to(DEVICE0)
                logits = model(test_image).view(-1)  # (C,)
                proba += F.softmax(logits, dim=0)
            proba /= float(tta_n)
            preds.append(proba.detach().cpu())
    return torch.stack(preds, dim=0)  # (N, C)


my_model_1 = nn.DataParallel(my_model_1).to(DEVICE0)
my_model_2 = nn.DataParallel(my_model_2).to(DEVICE0)

pred_proba_1 = _infer_model_proba(my_model_1, tta_n=1)
torch.cuda.empty_cache()
pred_proba_2 = _infer_model_proba(my_model_2, tta_n=TTA)
torch.cuda.empty_cache()

final_proba = (pred_proba_1 * 0.43) + (pred_proba_2 * 0.57)
label = final_proba.argmax(dim=-1).numpy().astype(int)

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print("Wrote:", SUBMISSION_PATH, "rows:", len(df_submission))
print(df_submission.head())
