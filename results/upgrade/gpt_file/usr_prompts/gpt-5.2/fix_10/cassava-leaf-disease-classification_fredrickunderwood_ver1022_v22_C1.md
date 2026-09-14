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

0.8981565427621638

# 6. Current score

0.15209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.26046) has done: 'I fix the Albumentations API breakage by updating `RandomResizedCrop` calls to the v2 signature so augmentations build correctly under albumentations==2.x. Then I make the model-weight loading robust by automatically falling back to the competition dataset path (and, if no `.pth` files exist, falling back to timm pretrained weights) so inference can run end-to-end and produce `submission.csv`. I also fix a couple of inference-time issues that can crash or silently misbehave in Kaggle (CPU-only environments, unsorted `os.listdir` order, and double-wrapping tensors already produced by `ToTensorV2`). These changes preserve the ensemble/inference core logic and only adjust weight source to ensure a valid run and reasonable accuracy.'
- What this solution (achieved 0.27803) has done: 'I fix the Albumentations v2 API error by replacing the removed `A.Cutout` augmentation with the supported `A.CoarseDropout`, keeping the augmentation intent (randomly masking regions) the same. I also add a tiny compatibility wrapper so the code won’t crash if other environments have slightly different Albumentations behavior. These changes are strictly to unblock execution and should improve accuracy substantially versus the current run that fails before producing proper predictions. The rest of the model/inference core logic is left unchanged, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.15508) has done: 'Your score is far below target, and the biggest likely cause is that you are not actually using the trained ensemble weights (falling back to ImageNet-pretrained backbones with randomly initialized 5-class heads yields near-random accuracy around your current score). I make the weight-path resolution robust to the actual available dataset directories in this environment (including `/kaggle/data` and the provided competition folder), and I prefer *any* `.pth` weights found if the exact filenames aren’t present, so the intended ensemble can load and inference stays identical. I also fix a subtle inference-time bug: L2-normalizing raw logits across samples distorts class scores and can severely hurt accuracy, so we ensemble by averaging logits directly (same core models, same argmax). These are minimal changes aimed specifically at moving accuracy toward your target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.15957) has done: 'Your current score strongly suggests the intended trained `.pth` weights still aren’t being loaded (or the wrong `.pth` is being picked), so the models behave like ImageNet backbones with a random 5-class head. I make weight discovery deterministic and model-specific by selecting `.pth` files that match each model name (resnext vs b4) when the exact filenames aren’t found, instead of blindly taking the first `.pth`. I also add a safe fallback to search the entire `/kaggle/input` and `/kaggle/data/input` trees (common Kaggle mount points) so the ensemble weights are found even if packaged differently. These changes keep the exact same models, ensembling, and argmax evaluation semantics, but should move accuracy much closer to your target by actually using the trained heads.'
- What this solution (achieved 0.14798) has done: 'Your current accuracy (0.15957) is far below the target (0.89816), and the most likely reason is still that the intended trained 5-class heads aren’t being loaded (or are being loaded into mismatched key names), so inference behaves like “ImageNet backbone + random 5-class head”. I make the weight loading stricter and architecture-aware by (1) selecting weights by actual key compatibility with each model (fc vs classifier shapes) instead of filename heuristics, and (2) handling common checkpoint formats (`model`, `state_dict`) and `module.` prefixes robustly. I also ensure we load weights into the correct underlying module when using DataParallel, without changing the ensemble logic or augmentations. These minimal changes directly target getting the correct trained weights into the same models, which should move accuracy strongly toward your target.'
- What this solution (achieved 0.15321) has done: 'The timeout is dominated by per-image PIL disk I/O plus running albumentations+model forward TTA=8 times for each image and for each model (≈ 2 * 2676 * 8 forwards) with batch size effectively 1. To keep identical core logic (same models, same TTA transforms, same averaging/ensemble), we batch the TTA pipeline: for each image we generate all TTA views once, stack them into a single tensor batch, and run one forward per model (instead of 8), then average. We also avoid reopening the same image repeatedly by loading it once per image, and we use DataLoader-style worker threading only for image list prep (no semantic changes). These changes are mathematically equivalent (same TTA samples, same logits averaged), but drastically cut Python overhead and GPU kernel launch overhead.'
- What this solution (achieved 0.15209) has done: 'I fix the runtime error by replacing the removed `albumentations.set_seed` call with a version-safe deterministic seeding approach that works under albumentations==2.x, so TTA runs without crashing. Then I ensure `label_list` is always defined by making submission creation depend on successful prediction, and I add a small safety fallback so that if anything goes wrong during inference, the script still writes a valid `submission.csv` (but normal runs use the model predictions). These changes keep the same models, TTA count, and ensembling logic, and are score-neutral except that they unblock the intended inference pipeline. The output be a properly formatted `submission.csv` with `image_id,label`.'

# 9. Code solution

## === cell 0
import os
import math
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from timm.data import resolve_data_config

from tqdm import tqdm

warnings.filterwarnings("ignore")




## === cell 1
def resolve_existing_path(candidates):
    for p in candidates:
        if p is None:
            continue
        if os.path.exists(p):
            return p
    return None


def list_pth_files(root_dir):
    if root_dir is None or not os.path.isdir(root_dir):
        return []
    out = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".pth"):
                out.append(os.path.join(r, f))
    return sorted(out)




## === cell 2
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
TTA = 8



## === cell 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_CSV_PATH = (
    resolve_existing_path(
        [
            TRAIN_CSV_PATH,
            "/kaggle/input/cassava-leaf-disease-classification/train.csv",
            "/kaggle/data/cassava-leaf-disease-classification/train.csv",
            "/kaggle/data/input/cassava-leaf-disease-classification/train.csv",
            "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
        ]
    )
    or TRAIN_CSV_PATH
)

TEST_IMAGE_PATH = (
    resolve_existing_path(
        [
            TEST_IMAGE_PATH,
            "/kaggle/input/cassava-leaf-disease-classification/test_images/",
            "/kaggle/data/cassava-leaf-disease-classification/test_images/",
            "/kaggle/data/input/cassava-leaf-disease-classification/test_images/",
            "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images/",
        ]
    )
    or TEST_IMAGE_PATH
)

INPUT_PATH = (
    resolve_existing_path(
        [
            INPUT_PATH,
            "/kaggle/input/ensemble-1023",
            "/kaggle/data/ensemble-1023",
            "/kaggle/data/input/ensemble-1023",
            "/kaggle/input/cassava-leaf-disease-classification",
            "/kaggle/data/cassava-leaf-disease-classification",
            "/kaggle/data/input/cassava-leaf-disease-classification",
        ]
    )
    or INPUT_PATH
)




## === cell 4
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




## === cell 5
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




## === cell 6
def RandomResizedCropFixed(h, w, **kwargs):
    return A.RandomResizedCrop(size=(h, w), **kwargs)




## === cell 7
def CutoutCompat(p=0.5, **kwargs):
    if hasattr(A, "Cutout"):
        return A.Cutout(p=p, **kwargs)
    return A.CoarseDropout(
        p=p,
        num_holes_range=(1, 8),
        hole_height_range=(8, 64),
        hole_width_range=(8, 64),
        fill=0,
    )


train_augs = A.Compose(
    [
        RandomResizedCropFixed(IMAGE_SIZE, IMAGE_SIZE),
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
        CutoutCompat(p=0.5),
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




## === cell 8
def build_timm_base_preproc(model, img_size_fallback=IMAGE_SIZE):
    cfg = resolve_data_config({}, model=model)
    if (
        "input_size" in cfg
        and isinstance(cfg["input_size"], (tuple, list))
        and len(cfg["input_size"]) == 3
    ):
        h = cfg["input_size"][1]
        w = cfg["input_size"][2]
    else:
        h = w = img_size_fallback

    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    return (h, w, mean, std, cfg)


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


def seed_for_tta(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


seed_everything(SEED)



## === cell 9
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




## === cell 10
def _unwrap_checkpoint_state(state):
    if not isinstance(state, dict):
        return state
    for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def _strip_module_prefix(sd):
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _compat_score_for_model(sd, model):
    if not isinstance(sd, dict):
        return -1

    msd = model.state_dict()
    shared = set(sd.keys()) & set(msd.keys())
    if not shared:
        return -1

    exact = 0
    shape_match = 0
    for k in shared:
        v = sd[k]
        mv = msd[k]
        if hasattr(v, "shape") and hasattr(mv, "shape"):
            if tuple(v.shape) == tuple(mv.shape):
                shape_match += 1
                if any(
                    h in k.lower() for h in ["fc.", "classifier.", "head.", "logits"]
                ):
                    exact += 5
                else:
                    exact += 1

    head_keys = [
        k
        for k in sd.keys()
        if any(h in k.lower() for h in ["fc.", "classifier.", "head.", "logits"])
    ]
    head_penalty = 0
    for k in head_keys:
        if k in msd and hasattr(sd[k], "shape") and hasattr(msd[k], "shape"):
            if tuple(sd[k].shape) != tuple(msd[k].shape):
                head_penalty += 10

    return exact + shape_match - head_penalty


def load_weights_if_available(model, weight_file, input_root, prefer_keywords=None):
    roots = [
        input_root,
        "/kaggle/input/ensemble-1023",
        "/kaggle/data/ensemble-1023",
        "/kaggle/data/input/ensemble-1023",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data/input/cassava-leaf-disease-classification",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
    ]
    roots = [r for r in roots if r is not None and os.path.exists(r)]

    direct_candidates = []
    for root in roots:
        cand = os.path.join(root, weight_file)
        if os.path.exists(cand):
            direct_candidates.append(cand)

    all_pths = []
    if direct_candidates:
        all_pths = direct_candidates
    else:
        for root in roots:
            all_pths.extend(list_pth_files(root))
        all_pths = sorted(set(all_pths))

    if not all_pths:
        return model, False, None

    candidates = all_pths
    if prefer_keywords:
        kws = [k.lower() for k in prefer_keywords if k]
        filtered = []
        for p in candidates:
            name = os.path.basename(p).lower()
            if any(k in name for k in kws):
                filtered.append(p)
        if filtered:
            candidates = filtered

    best_path = None
    best_score = -(10**18)
    for p in candidates:
        try:
            state = torch.load(p, map_location="cpu")
            sd = _unwrap_checkpoint_state(state)
            sd = _strip_module_prefix(sd)
            score = _compat_score_for_model(sd, model)
        except Exception:
            score = -(10**18)
        if score > best_score:
            best_score = score
            best_path = p

    if best_path is None or best_score < 0:
        return model, False, None

    state = torch.load(best_path, map_location="cpu")
    sd = _strip_module_prefix(_unwrap_checkpoint_state(state))
    model.load_state_dict(sd, strict=False)
    return model, True, best_path




## === cell 11
torch.cuda.empty_cache()

my_model_1, loaded1, w1 = load_weights_if_available(
    my_model_1,
    RESNEXT_PATH,
    INPUT_PATH,
    prefer_keywords=["res", "resnext", "res50", "next50", "r50", "1022"],
)
if not loaded1:
    my_model_1 = timm.create_model(model_name1, pretrained=True)
    my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
    nn.init.xavier_uniform_(my_model_1.fc.weight)
    if my_model_1.fc.bias is not None:
        nn.init.zeros_(my_model_1.fc.bias)

my_model_2, loaded2, w2 = load_weights_if_available(
    my_model_2,
    B4_PATH,
    INPUT_PATH,
    prefer_keywords=["b4", "efficientnet", "efn", "b4ns", "ns", "1022"],
)
if not loaded2:
    my_model_2 = timm.create_model(model_name2, pretrained=True)
    my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
    nn.init.xavier_uniform_(my_model_2.classifier.weight)
    if my_model_2.classifier.bias is not None:
        nn.init.zeros_(my_model_2.classifier.bias)

print(
    f"Loaded weights: model1={loaded1} path={w1}, model2={loaded2} path={w2}, INPUT_PATH={INPUT_PATH}"
)



## === cell 12
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    my_model_1 = nn.DataParallel(my_model_1)
    my_model_2 = nn.DataParallel(my_model_2)

my_model_1 = my_model_1.to(DEVICE)
my_model_2 = my_model_2.to(DEVICE)

my_model_1.eval()
my_model_2.eval()



## === cell 13
h1, w1_sz, mean1, std1, cfg1 = build_timm_base_preproc(
    my_model_1.module if isinstance(my_model_1, nn.DataParallel) else my_model_1
)
h2, w2_sz, mean2, std2, cfg2 = build_timm_base_preproc(
    my_model_2.module if isinstance(my_model_2, nn.DataParallel) else my_model_2
)

INF_H = int(max(h1, h2, IMAGE_SIZE))
INF_W = int(max(w1_sz, w2_sz, IMAGE_SIZE))

if tuple(mean1) == tuple(mean2) and tuple(std1) == tuple(std2):
    INF_MEAN, INF_STD = mean1, std1
else:
    INF_MEAN, INF_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(INF_H, INF_W, p=1.0),
                A.CenterCrop(INF_H, INF_W, p=1.0),
                RandomResizedCropFixed(INF_H, INF_W, p=1.0),
            ],
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(INF_H, INF_W),
        A.Normalize(
            mean=list(INF_MEAN),
            std=list(INF_STD),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)


def pil_to_uint8_hwc(img_pil: Image.Image) -> np.ndarray:
    arr = np.asarray(img_pil.convert("RGB"))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def make_tta_batch(
    img_arr_uint8: np.ndarray, tta: int, device: torch.device, seed_base: int
) -> torch.Tensor:
    views = []
    for i in range(tta):
        seed_for_tta(int(seed_base + i))
        aug_image = test_augs(image=img_arr_uint8)["image"]  # torch.Tensor CHW
        views.append(aug_image)
    batch = (
        torch.stack(views, dim=0).contiguous().to(device=device, dtype=torch.float32)
    )
    return batch




## === cell 14
test_image_list = np.asarray(
    sorted(
        [
            image_name
            for image_name in os.listdir(TEST_IMAGE_PATH)
            if image_name.lower().endswith(".jpg")
        ]
    )
)

printed_debug = False
preds_1 = []
preds_2 = []

label_list = None

try:
    with torch.no_grad():
        for single_image_name in tqdm(
            test_image_list, desc="Predict ensemble (batched TTA)"
        ):
            image = Image.open(os.path.join(TEST_IMAGE_PATH, single_image_name))
            img_arr = pil_to_uint8_hwc(image)

            seed_base = (abs(hash(single_image_name)) % 10_000_000) + SEED * 1000
            tta_batch = make_tta_batch(
                img_arr, TTA, DEVICE, seed_base=seed_base
            )  # (TTA,C,H,W)

            if not printed_debug:
                printed_debug = True
                print(
                    "Debug first batch:",
                    "img_arr dtype",
                    img_arr.dtype,
                    "tta_batch dtype",
                    tta_batch.dtype,
                    "tta_batch shape",
                    tuple(tta_batch.shape),
                    "tta_batch min/max",
                    float(tta_batch.min().cpu()),
                    float(tta_batch.max().cpu()),
                    "INF_H/W",
                    (INF_H, INF_W),
                    "mean/std",
                    INF_MEAN,
                    INF_STD,
                )

            out1 = my_model_1(tta_batch).view(TTA, OUT_FEATURES).mean(dim=0)
            out2 = my_model_2(tta_batch).view(TTA, OUT_FEATURES).mean(dim=0)

            preds_1.append(out1.detach().cpu())
            preds_2.append(out2.detach().cpu())

    predictions_1 = torch.stack(preds_1, dim=0)
    predictions_2 = torch.stack(preds_2, dim=0)

    final_pred = (predictions_1 * 0.46) + (predictions_2 * 0.54)
    label = final_pred.argmax(dim=-1).numpy()
    label_list = list(label)
except Exception as e:
    print("Inference failed with exception:", repr(e))
    label_list = [0] * len(test_image_list)



## === cell 15
sample_sub_path = resolve_existing_path(
    [
        "../input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]
)

if sample_sub_path is not None:
    df_submission = pd.read_csv(sample_sub_path)
    pred_map = dict(zip(test_image_list.tolist(), label_list))
    df_submission["label"] = df_submission["image_id"].map(pred_map).astype(int)
else:
    df_submission = pd.DataFrame({"image_id": test_image_list, "label": label_list})

df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Wrote submission to: {SUBMISSION_PATH} with shape {df_submission.shape}")
print(df_submission.head())
