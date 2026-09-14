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

0.895436687821094

# 6. Current score

0.13565

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13528) has done: 'I fix the Albumentations v2 API break by updating `RandomResizedCrop` calls to the new `size=(H, W)` signature so augmentations construct correctly. I also fix the missing weights path by switching `INPUT_PATH` to the actual Kaggle dataset directory already present in your environment, and add a safe fallback that uses pretrained timm weights if the `.pth` files are still not found (so a valid submission is always produced). To prevent runtime errors on CPU-only environments, I make device selection robust and replace hard-coded `.cuda()` calls with `.to(device)`. Finally, I ensure the submission uses the provided `sample_submission.csv` ordering so `image_id` aligns correctly with predictions and the output is a valid `submission.csv`.'
- What this solution (achieved 0.1222) has done: 'Your current low score is consistent with a major post-processing mismatch: you are L2-normalizing *logits* across classes before argmax, which can severely distort class ranking and collapse performance. I keep your two-model inference/ensemble logic intact and make the smallest change by removing the per-sample L2 normalization and instead ensemble on softmax probabilities (a standard, metric-aligned choice for accuracy). I also ensure inference runs in `torch.inference_mode()` (no semantic change, just safer/faster) and keep the sample-submission ordering to guarantee alignment. These changes should substantially increase accuracy toward your target without changing the model architectures or training approach.'
- What this solution (achieved 0.12332) has done: 'Your current score is far below the target, so we should make a small, metric-aligned fix that can materially improve accuracy without changing the models or training. The biggest likely issue is test-time augmentation: your `test_augs` includes random spatial transforms (including `RandomResizedCrop` and flips) plus extra `Resize`, which can inject too much randomness and blur signal; for accuracy, a deterministic resize/center-crop TTA is usually safer. I keep your exact two-model ensemble and softmax-probability averaging, but change TTA to use a small set of deterministic transforms (original + horizontal flip) and average probabilities across them (equivalent semantics for argmax accuracy, but more stable). I also batch inference (no architecture/training change) to reduce overhead and keep runtime within limits.'
- What this solution (achieved 0.12332) has done: 'Your score is far below the target (gap ≈ -0.77), so we should make a small, high-impact fix that keeps your ensemble/model code intact. The most likely cause of ~0.12 accuracy is that the provided `.pth` weights are not being found/loaded and you’re silently falling back to generic pretrained backbones, which perform poorly on this dataset without finetuning. I minimally extend the weights-path resolution to search the common Kaggle input dataset directory (including the competition subfolder), and I print an explicit confirmation of which weights were actually loaded to avoid silent fallback. This preserves your exact architectures, inference, and submission formatting; it only fixes weight discovery/loading to move accuracy toward the target.'
- What this solution (achieved 0.13565) has done: 'Your score is far below the target, so we should make a small, high-impact correction rather than tuning. The most likely cause is that the EfficientNet model’s classifier layer name is wrong for `tf_efficientnet_b4_ns` in timm (it typically uses `head.fc`), so your fine-tuned weights aren’t actually being applied to the correct layer even if the file is found—leading to near-random predictions. I keep your exact two-model ensemble and inference logic, but (1) replace the brittle manual head replacement with `reset_classifier(OUT_FEATURES)` (timm-native) for both models and (2) make weight loading strict about head mismatch by fixing common key prefixes (`model.`, `module.`) so the intended checkpoints load. This should materially increase accuracy toward the target while preserving your overall approach and still writing a valid `submission.csv`.'
- What this solution (achieved 0.13565) has done: 'Your current score is far below the target, so we need a high-impact fix without changing your models, training, or ensemble logic. The biggest remaining likely issue is that your checkpoints are still not being found because you only search for the bare filename; many Kaggle datasets store them under a subfolder (e.g., `weights/`, `models/`, `checkpoints/`) or nested deeper. I minimally extend the weight-file discovery to recursively search within the available input roots for the two `.pth` filenames and then load them the same way as before. This keeps architecture/inference identical, but should move accuracy substantially toward your target if the finetuned weights are present somewhere in the dataset.'

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

import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm import tqdm
import timm

warnings.filterwarnings("ignore")



## === cell 1
BASE_INPUT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


BASE_DATA_PATH = _first_existing(BASE_INPUT_CANDIDATES)

INPUT_PATH = os.environ.get("WEIGHTS_PATH", BASE_DATA_PATH)

TRAIN_CSV_PATH = os.path.join(BASE_DATA_PATH, "train.csv")
TRAIN_IMAGE_PATH = os.path.join(BASE_DATA_PATH, "train_images")
TEST_IMAGE_PATH = os.path.join(BASE_DATA_PATH, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_DATA_PATH, "sample_submission.csv")

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

TTA = 2  # will map to: [original, horizontal_flip]

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    DEVICE = torch.device("cuda:0")
    DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
else:
    DEVICE = torch.device("cpu")
    DEVICES = [DEVICE]

print("BASE_DATA_PATH:", BASE_DATA_PATH)
print("INPUT_PATH (weights root):", INPUT_PATH)
print("DEVICE:", DEVICE)




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
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
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
test_augs_base = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
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
my_model_1.reset_classifier(OUT_FEATURES)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.reset_classifier(OUT_FEATURES)




## === cell 8
def _load_state(model, path):
    if path is None or (not os.path.exists(path)):
        return False

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        for prefix in ("module.", "model."):
            if any(k.startswith(prefix) for k in state.keys()):
                state = {k.replace(prefix, "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print(
            f"Warning: unexpected keys when loading {os.path.basename(path)}: {len(unexpected)}"
        )
    if len(missing) > 0:
        print(
            f"Warning: missing keys when loading {os.path.basename(path)}: {len(missing)}"
        )
    return True


def _find_weight_file(weight_name: str, roots):
    candidates = []
    for r in roots:
        if r is None:
            continue
        candidates.append(os.path.join(r, weight_name))
        candidates.append(
            os.path.join(r, "cassava-leaf-disease-classification", weight_name)
        )
        for sub in ("weights", "models", "checkpoints", "ckpt", "output", "outputs"):
            candidates.append(os.path.join(r, sub, weight_name))
            candidates.append(
                os.path.join(r, "cassava-leaf-disease-classification", sub, weight_name)
            )

    for c in candidates:
        if os.path.exists(c):
            return c

    seen = set()
    for r in roots:
        if r is None or (not os.path.exists(r)) or (r in seen):
            continue
        seen.add(r)
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if weight_name in filenames:
                return os.path.join(dirpath, weight_name)
    return None


weight_roots = [
    INPUT_PATH,
    BASE_DATA_PATH,
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
    "../input",
]

resnext_weight_path = _find_weight_file(RESNEXT_PATH, weight_roots)
b4_weight_path = _find_weight_file(B4_PATH, weight_roots)

print("Resolved RESNEXT weight path:", resnext_weight_path)
print("Resolved B4 weight path:", b4_weight_path)

loaded_1 = _load_state(my_model_1, resnext_weight_path)
loaded_2 = _load_state(my_model_2, b4_weight_path)

if not loaded_1:
    print(
        f"Weights not found for model_1 ({RESNEXT_PATH}). Falling back to timm pretrained backbone."
    )
    my_model_1 = timm.create_model(
        model_name1, pretrained=True, num_classes=OUT_FEATURES
    )
else:
    print(f"Loaded fine-tuned weights for model_1 from: {resnext_weight_path}")

if not loaded_2:
    print(
        f"Weights not found for model_2 ({B4_PATH}). Falling back to timm pretrained backbone."
    )
    my_model_2 = timm.create_model(
        model_name2, pretrained=True, num_classes=OUT_FEATURES
    )
else:
    print(f"Loaded fine-tuned weights for model_2 from: {b4_weight_path}")

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    my_model_1 = nn.DataParallel(my_model_1).to(DEVICES[0])
    my_model_2 = nn.DataParallel(my_model_2).to(DEVICES[0])
else:
    my_model_1 = my_model_1.to(DEVICE)
    my_model_2 = my_model_2.to(DEVICE)

my_model_1.eval()
my_model_2.eval()
torch.cuda.empty_cache()



## === cell 9
if os.path.exists(SAMPLE_SUB_PATH):
    sub_df = pd.read_csv(SAMPLE_SUB_PATH)
    test_image_list = sub_df["image_id"].values
else:
    test_image_list = np.array(sorted(os.listdir(TEST_IMAGE_PATH)))


def _apply_tta_pil(img_pil, tta_index: int):
    if tta_index == 0:
        return img_pil
    return img_pil.transpose(Image.FLIP_LEFT_RIGHT)


def _predict_batch_probs(model, image_names, tta=1, batch_size=32):
    all_probs = []
    n = len(image_names)
    for start in tqdm(range(0, n, batch_size), desc="Batched predict", leave=False):
        batch_names = image_names[start : start + batch_size]
        batch_prob = torch.zeros((len(batch_names), OUT_FEATURES), device=DEVICE)

        for t in range(tta):
            imgs = []
            for nm in batch_names:
                img = Image.open(os.path.join(TEST_IMAGE_PATH, nm)).convert("RGB")
                img = _apply_tta_pil(img, t)
                ten = test_augs_base(image=np.array(img))["image"].float()
                imgs.append(ten)
            x = torch.stack(imgs, dim=0).to(DEVICE)

            with torch.inference_mode():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                batch_prob += probs

        batch_prob /= float(tta)
        all_probs.append(batch_prob.detach().cpu())

    return torch.cat(all_probs, dim=0)


probs_1 = _predict_batch_probs(
    my_model_1, test_image_list, tta=1, batch_size=BATCH_SIZE
)
torch.cuda.empty_cache()
probs_2 = _predict_batch_probs(
    my_model_2, test_image_list, tta=TTA, batch_size=BATCH_SIZE
)

final_prob = (probs_1 * 0.47) + (probs_2 * 0.53)
label = final_prob.argmax(dim=-1).numpy().astype(int)

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print("Wrote:", SUBMISSION_PATH, "rows:", len(df_submission))
print(df_submission.head())
