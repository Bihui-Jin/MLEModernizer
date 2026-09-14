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

0.8934723481414325

# 6. Current score

0.49066

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58969) has done: 'I make the code robust to missing external weight files by falling back to running the same architectures with timm pretrained ImageNet weights (so it can always produce a valid submission). I also fix the CUDA/CPU dtype mismatch by ensuring the loaded state dict is moved onto the same device as the model after loading. Finally, I guard inference so it always produces `normalize_pred_1/2` and writes `submission.csv` with the exact required columns and row count, even if one of the models can’t load custom weights.'
- What this solution (achieved 0.49327) has done: 'Your current score is far below the target, so the smallest safe way to move accuracy upward is to fix prediction post-processing rather than changing model/training. Right now you L2-normalize raw logits and then argmax, which can distort class ranking and hurt accuracy; switching to averaging softmax probabilities preserves the intended classification semantics. I also make test-time augmentation deterministic per image (seeded) to reduce variance without changing the TTA approach, and I keep your ensembling weights and file paths unchanged. The rest of the pipeline (models, weights loading, dataset I/O, submission schema) stays the same.'
- What this solution (achieved 0.49738) has done: 'Your score is far below the target, so the least invasive way to move accuracy upward is to fix two inference-time mismatches without touching your model architectures or any training logic. First, your current “deterministic per-TTA seed” still produces correlated augmentations because it resets global RNGs while albumentations also uses its own randomness; switching to `ReplayCompose` guarantees each TTA draw is actually different but still reproducible per image. Second, for classification, enabling test-time horizontal-flip ensembling (a standard, label-preserving transform) is a minimal semantic change that usually improves accuracy without altering the core pipeline. I keep your ensemble weights (0.35/0.65), file paths, and submission formatting unchanged and ensure the script still finishes within the time budget by batching inference.'
- What this solution (achieved 0.49028) has done: 'Your score is far below the target, so we should improve accuracy with minimal inference-only fixes (no architecture/training changes). The biggest issue is that your `ReplayCompose` is not actually being “replayed” for hflip, so the flip branch is using a *different* random crop/transpose/etc, which makes the hflip ensemble noisy and can hurt accuracy; we apply the exact same replayed transform params to the flipped image. Second, we make model1 use the same TTA count as model2 (still small) to reduce variance and typically lift accuracy without changing the modeling approach. Finally, we keep the same ensembling weights and submission schema, and we make inference deterministic without repeatedly toggling cuDNN determinism inside the hot loop.'
- What this solution (achieved 0.49776) has done: 'Your current score is far below the target, so the smallest high-impact change is to fix an inference-time preprocessing mismatch that can crater accuracy: `A.Transpose` swaps H/W but your horizontal flip logic currently flips the wrong axis for transposed images. I keep your exact models, weights-loading behavior, and TTA/ReplayCompose pipeline, but change the flip-ensemble to flip the correct “width” dimension **after** the sampled ReplayCompose transform is applied (on the transformed tensor), which preserves true horizontal-flip invariance under transpose. I also keep determinism but remove repeated global RNG reseeding inside the hot loop (ReplayCompose already samples once per call), reducing accidental coupling and improving stability without changing semantics. Submission format and paths stay identical.'
- What this solution (achieved 0.49066) has done: 'Your score (0.49776) is far below the target (0.89347), so we should make the smallest inference-only fix that plausibly improves accuracy without changing your models or training. Right now `predict_proba_batch` reseeds RNGs per TTA, which unintentionally makes every image in a batch receive the *same* sampled ReplayCompose parameters for a given `t`, reducing TTA diversity and hurting ensemble gains. I change seeding to be **per-image per-TTA** (deterministic, but different across images) while keeping the same TTA count, same augmentations, same hflip-on-tensor fix, and the same ensembling weights. This should move accuracy upward toward the target while preserving your core logic and semantics.'
- What this solution (achieved 0.49066) has done: 'I fix a bug that currently prevents your custom weights from loading for model1 (`my_model__1` typo), which is likely why your accuracy is stuck near random and far from the target. To keep core logic unchanged, I only adjust the weight-loading call so it correctly detects `DataParallel` and loads into the right module. I also add a small safety fallback so if custom loading fails for any reason, the script continues using the timm pretrained backbone (still producing a valid submission). This should materially increase accuracy toward the target without changing model architectures, training, or inference semantics.'

# 9. Code solution

## === cell 0
import os
import math
import random

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt



## === cell 1
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"

INPUT_PATH = (
    BASE_PATH  # weights expected to live alongside this dataset (fallback logic below)
)
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

TRAIN_IMAGE_PATH = os.path.join(BASE_PATH, "train_images")
TEST_IMAGE_PATH = os.path.join(BASE_PATH, "test_images")

SUBMISSION_PATH = "submission.csv"

RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()

print("DEVICE:", DEVICE, "N_GPU:", N_GPU)
print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("TEST_IMAGE_PATH exists:", os.path.exists(TEST_IMAGE_PATH))




## === cell 2
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


x = [i for i in range(NUM_EPOCHS)]
y = [lr_tune(i) for i in x]
plt.plot(x, y)
plt.title("LR schedule")
plt.show()



## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333),
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
test_augs = A.ReplayCompose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop(
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333),
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
my_model_1 = timm.create_model(model_name1, pretrained=True)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)

print("Models initialized.")




## === cell 8
def _find_weight_file(filename: str) -> str:
    """
    Search common Kaggle input locations. Returns path if found, else raises.
    """
    candidates = [
        os.path.join(INPUT_PATH, filename),
        os.path.join(BASE_PATH, filename),
        os.path.join("/kaggle/input", filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    for root, _, files in os.walk(BASE_PATH):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(
        f"Could not find weight file '{filename}'. Looked in: {candidates} and under {BASE_PATH}."
    )


def _load_state_dict_safely(model: nn.Module, weight_path: str, device: torch.device):
    obj = torch.load(weight_path, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    print(
        f"Loaded {os.path.basename(weight_path)}; missing={len(missing)} unexpected={len(unexpected)}"
    )




## === cell 9
torch.cuda.empty_cache()

sub_df = pd.read_csv(SAMPLE_SUB_PATH)
test_image_list = sub_df["image_id"].astype(str).tolist()
test_image_list = [
    img for img in test_image_list if os.path.isfile(os.path.join(TEST_IMAGE_PATH, img))
]
print("Num test images:", len(test_image_list))

use_custom_1, use_custom_2 = False, False
try:
    resnext_w = _find_weight_file(RESNEXT_PATH)
    use_custom_1 = True
except FileNotFoundError as e:
    print("WARNING:", e)
    print("Proceeding without custom weights for model1.")

try:
    b4_w = _find_weight_file(B4_PATH)
    use_custom_2 = True
except FileNotFoundError as e:
    print("WARNING:", e)
    print("Proceeding without custom weights for model2.")

if torch.cuda.is_available() and N_GPU >= 1:
    my_model_1 = nn.DataParallel(my_model_1).to(DEVICE)
    my_model_2 = nn.DataParallel(my_model_2).to(DEVICE)
else:
    my_model_1 = my_model_1.to(DEVICE)
    my_model_2 = my_model_2.to(DEVICE)

if use_custom_1:
    try:
        _load_state_dict_safely(
            (
                my_model_1.module
                if isinstance(my_model_1, nn.DataParallel)
                else my_model_1
            ),
            resnext_w,
            DEVICE,
        )
    except Exception as e:
        print(
            "WARNING: Failed to load custom weights for model1, using timm pretrained instead."
        )
        print("Reason:", repr(e))

if use_custom_2:
    try:
        _load_state_dict_safely(
            (
                my_model_2.module
                if isinstance(my_model_2, nn.DataParallel)
                else my_model_2
            ),
            b4_w,
            DEVICE,
        )
    except Exception as e:
        print(
            "WARNING: Failed to load custom weights for model2, using timm pretrained instead."
        )
        print("Reason:", repr(e))

my_model_1.eval()
my_model_2.eval()




## === cell 10
def _load_pil_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _seed_for_image_tta(seed_base: int, t: int, image_index_in_batch: int) -> int:
    return int(seed_base + 1000 * t + image_index_in_batch)


def predict_proba_batch(
    model: nn.Module,
    image_paths: list[str],
    tta: int,
    seed_base: int,
    do_hflip_ensemble: bool = True,
) -> torch.Tensor:
    """
    Returns: (B, OUT_FEATURES) probabilities averaged over TTA (and optional hflip).

    Keeps your ReplayCompose + flip-on-tensor fix, but makes RNG seeding per-image per-TTA
    to actually realize TTA diversity deterministically.
    """
    model.eval()
    B = len(image_paths)
    with torch.no_grad():
        prob_sum = torch.zeros((B, OUT_FEATURES), device=DEVICE)

        for t in range(tta):
            batch_imgs = []
            for j, p in enumerate(image_paths):
                s = _seed_for_image_tta(
                    seed_base=seed_base, t=t, image_index_in_batch=j
                )
                random.seed(s)
                np.random.seed(s)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(s)
                    torch.cuda.manual_seed_all(s)

                img = _load_pil_rgb(p)
                out = test_augs(image=img)
                batch_imgs.append(out["image"])

            x = torch.stack(batch_imgs, dim=0).to(DEVICE, dtype=torch.float)
            logits = model(x)
            prob = F.softmax(logits, dim=1)

            if do_hflip_ensemble:
                x_hf = torch.flip(x, dims=[3])  # flip width dimension (tensor space)
                logits_hf = model(x_hf)
                prob_hf = F.softmax(logits_hf, dim=1)
                prob = 0.5 * (prob + prob_hf)

            prob_sum += prob

        prob_mean = prob_sum / float(tta)
    return prob_mean.detach().cpu()


paths = [os.path.join(TEST_IMAGE_PATH, n) for n in test_image_list]

proba_1_chunks = []
for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="Predict model1 (batched)"):
    chunk = paths[start : start + BATCH_SIZE]
    proba_1_chunks.append(
        predict_proba_batch(
            my_model_1, chunk, tta=TTA, seed_base=SEED + start, do_hflip_ensemble=True
        )
    )
proba_1 = torch.cat(proba_1_chunks, dim=0)
torch.cuda.empty_cache()

proba_2_chunks = []
for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="Predict model2 (batched)"):
    chunk = paths[start : start + BATCH_SIZE]
    proba_2_chunks.append(
        predict_proba_batch(
            my_model_2,
            chunk,
            tta=TTA,
            seed_base=SEED + 10_000 + start,
            do_hflip_ensemble=True,
        )
    )
proba_2 = torch.cat(proba_2_chunks, dim=0)
torch.cuda.empty_cache()



## === cell 11
final_proba = (proba_1 * 0.35) + (proba_2 * 0.65)
label = final_proba.argmax(dim=-1).numpy().astype(int)

sub_out = pd.DataFrame({"image_id": test_image_list, "label": label})
sub_out.to_csv(SUBMISSION_PATH, index=False)

print("Wrote:", SUBMISSION_PATH)
print(sub_out.head())
print("Submission shape:", sub_out.shape)
assert SUBMISSION_PATH.endswith(".csv") and os.path.isfile(SUBMISSION_PATH)
assert list(sub_out.columns) == ["image_id", "label"]
assert len(sub_out) == len(test_image_list)
assert sub_out["label"].between(0, OUT_FEATURES - 1).all()
