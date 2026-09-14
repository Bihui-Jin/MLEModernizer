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

No external packages required in the script and installed.

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

0.8878815352070112

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it can actually load model weights in this environment by searching common Kaggle input/data locations and falling back to EfficientNetV2-L if the ViT checkpoint is missing. I also fix the device/dtype mismatch that caused CUDA inputs to be fed into a CPU model by ensuring the loaded state dict and the whole model are moved to the selected device before inference. Finally, I make the inference loop robust (skip missing/corrupt images but keep row alignment by default) and always write a correctly formatted `submission.csv` with exactly the same `image_id` order/length as `sample_submission.csv`, preventing “Not yielded”.'
- What this solution (achieved 0.61099) has done: 'Your current crash comes from relying on external weight files that don’t exist in this Kaggle environment, so the script never reaches inference/submission. I fix this by removing the hard failure and instead using torchvision’s built-in pretrained weights (available offline in Kaggle) for EfficientNetV2-L and ViT-H14, while keeping your same preprocessing and inference loop structure. Because pretrained heads won’t match 5 cassava classes, I keep the model heads as-is and output a safe, accuracy-improving baseline by predicting the majority class from `train.csv` (this is a minimal, legitimate calibration step and typically scores far above ~0.055 on Cassava). The code still produce a valid `submission.csv` aligned exactly to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I remove the most expensive source of runtime: the per-image PIL-based Python transforms and resizing pipeline, replacing it with an equivalent `torchvision.io`/tensor-based preprocessing that runs in C++ and inside DataLoader workers. I keep the same model, weights loading, training logic (only head-only training when no custom weights), loss, and evaluation semantics, but make the data pipeline faster via vectorized tensor ops, `channels_last` where safe, and avoiding redundant `pin_memory()` calls. I also tune DataLoader settings (workers/prefetch/persistent workers) for better throughput without changing outputs, and avoid per-sample exception overhead by doing robust decode with a consistent fallback.'
- What this solution (achieved 0.61099) has done: 'Your current 0.61099 score comes from predicting the majority class for every test image (the model head is untrained / random when custom weights aren’t found), so the smallest legitimate way to move toward the 0.8879 target is to actually fine-tune the existing model head on the provided `train.csv` images. I keep your exact model choices, transforms, and “head-only training when no custom weights” logic, but make that head-only training effective by (1) shuffling the training pairs, (2) training for a small fixed number of epochs, and (3) using a slightly larger (but safe) training subset, all without changing the loss or architecture. I also keep the same robust inference/submission alignment, so it still always emits a valid `submission.csv`. These changes should materially improve accuracy over majority-class while staying within Kaggle time constraints.'
- What this solution (achieved 0.61099) has done: 'You’re currently far below the target (0.61099 vs 0.88788, higher-is-better), and the main limiter is that you’re only doing a tiny amount of head-only training and not using any validation-guided selection, so the fine-tuned head is likely underfit and/or unstable. I keep the same model, preprocessing, and “freeze backbone, train only the head” approach, but (1) train the head on the full training set (instead of truncating to 12k) and for a few more fixed epochs, and (2) add a small deterministic validation split to pick the best head epoch (no early stopping—still runs a fixed number of epochs). This should legitimately increase accuracy toward your target without changing architecture/loss/feature extraction, and still writes a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import glob
import random



## === cell 1
test_data_directory_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/data/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
]
test_data_directory = next(
    (p for p in test_data_directory_candidates if os.path.isdir(p)), None
)
if test_data_directory is None:
    raise FileNotFoundError(
        "Could not find test_images directory. Tried:\n"
        + "\n".join(test_data_directory_candidates)
    )

train_image_directory_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
]
train_image_directory = next(
    (p for p in train_image_directory_candidates if os.path.isdir(p)), None
)
if train_image_directory is None:
    raise FileNotFoundError(
        "Could not find train_images directory (needed for head-only training). Tried:\n"
        + "\n".join(train_image_directory_candidates)
    )

sample_sub_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_sub_path = next((p for p in sample_sub_candidates if os.path.isfile(p)), None)
if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv. Tried:\n"
        + "\n".join(sample_sub_candidates)
    )

train_csv_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
train_csv_path = next((p for p in train_csv_candidates if os.path.isfile(p)), None)
if train_csv_path is None:
    raise FileNotFoundError(
        "Could not find train.csv (needed for head-only training / fallback). Tried:\n"
        + "\n".join(train_csv_candidates)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/6/vit_h_14_518_8860_base.pth"
)
vit_image_size = 518

model_select = "vit"

if model_select == "vit":
    model_image_size = vit_image_size
if model_select == "en":
    model_image_size = en_image_size

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 2
def invert_square_pad(img):
    width, height = img.size

    center_width, center_height = width // 2, height // 2
    top_left = img.crop((0, 0, center_width, center_height))
    top_right = img.crop((center_width, 0, width, center_height))
    bottom_left = img.crop((0, center_height, center_width, height))
    bottom_right = img.crop((center_width, center_height, width, height))

    top_combined = Image.new("RGB", (width, center_height))
    top_combined.paste(bottom_right, (0, 0))
    top_combined.paste(bottom_left, (center_width, 0))

    bottom_combined = Image.new("RGB", (width, center_height))
    bottom_combined.paste(top_right, (0, 0))
    bottom_combined.paste(top_left, (center_width, 0))

    flipped_img = Image.new("RGB", (width, height))
    flipped_img.paste(top_combined, (0, 0))
    flipped_img.paste(bottom_combined, (0, center_height))

    img = flipped_img.copy()
    del top_combined, bottom_combined, flipped_img

    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )

    padded_img = transforms.functional.pad(img, padding, padding_mode="reflect")

    return padded_img


def resize_max_side(img, size):
    width, height = img.size

    if max(width, height) <= size:
        return img

    if width > height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)

    return transforms.functional.resize(img, (new_height, new_width))


def pad_to_square(img):
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,
        (max_side - height) // 2,
        (max_side - width) - (max_side - width) // 2,
        (max_side - height) - (max_side - height) // 2,
    )  # left, top, right, bottom

    padded_img = transforms.functional.pad(img, padding, padding_mode="reflect")

    return padded_img




## === cell 3
import torchvision
from torchvision.io import read_file, decode_jpeg
from torchvision.transforms import functional as F

val_transforms = transforms.Compose(
    [
        v2.Lambda(lambda img: resize_max_side(img, 800)),
        v2.Lambda(pad_to_square),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # outputs float in [0,1]
        v2.Resize((model_image_size, model_image_size)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # maps to [-1,1]
    ]
)


def _resize_max_side_tensor(img_chw_u8: torch.Tensor, size: int) -> torch.Tensor:
    h, w = int(img_chw_u8.shape[-2]), int(img_chw_u8.shape[-1])
    if max(h, w) <= size:
        return img_chw_u8
    if w > h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return F.resize(
        img_chw_u8,
        [new_h, new_w],
        interpolation=F.InterpolationMode.BILINEAR,
        antialias=True,
    )


def _pad_to_square_reflect_tensor(img_chw: torch.Tensor) -> torch.Tensor:
    h, w = int(img_chw.shape[-2]), int(img_chw.shape[-1])
    m = h if h > w else w
    pad_left = (m - w) // 2
    pad_top = (m - h) // 2
    pad_right = (m - w) - pad_left
    pad_bottom = (m - h) - pad_top
    if pad_left == pad_top == pad_right == pad_bottom == 0:
        return img_chw
    return F.pad(
        img_chw, [pad_left, pad_top, pad_right, pad_bottom], padding_mode="reflect"
    )


def fast_val_transform_from_path(img_path: str) -> torch.Tensor:
    data = read_file(img_path)
    img = decode_jpeg(data, mode=torchvision.io.ImageReadMode.RGB)  # HWC uint8
    img = img.permute(2, 0, 1)  # CHW uint8
    img = _resize_max_side_tensor(img, 800)
    img = _pad_to_square_reflect_tensor(img)
    img = F.resize(
        img,
        [model_image_size, model_image_size],
        interpolation=F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    img = img.to(dtype=torch.float32).div_(255.0)
    img = img.sub_(0.5).div_(0.5)  # normalize to [-1,1]
    return img




## === cell 4
def _find_first_existing_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _glob_first(patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        hits = [h for h in hits if os.path.isfile(h)]
        if hits:
            return sorted(hits)[0]
    return None


vit_candidates = [
    vit_model_path,
    _glob_first(
        [
            "/kaggle/input/**/vit*_*.pth",
            "/kaggle/input/**/vit*h*14*.pth",
            "/kaggle/data/**/vit*_*.pth",
            "/kaggle/data/**/vit*h*14*.pth",
        ]
    ),
]

en_candidates = [
    en_model_path,
    _glob_first(
        [
            "/kaggle/input/**/efficientnet*_v2*_l*.pth",
            "/kaggle/input/**/efficientnet*v2*l*.pth",
            "/kaggle/data/**/efficientnet*_v2*_l*.pth",
            "/kaggle/data/**/efficientnet*v2*l*.pth",
        ]
    ),
]

resolved_vit_path = _find_first_existing_file(vit_candidates)
resolved_en_path = _find_first_existing_file(en_candidates)

use_custom_weights = False
if model_select == "vit" and resolved_vit_path is not None:
    use_custom_weights = True
if model_select == "en" and resolved_en_path is not None:
    use_custom_weights = True

if model_select == "vit":
    try:
        vit_weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    except Exception:
        vit_weights = None

    vit_model = models.vit_h_14(weights=vit_weights, image_size=vit_image_size)
    if use_custom_weights:
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        state = torch.load(resolved_vit_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        vit_model.load_state_dict(state, strict=True)
    else:
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )

    vit_model.to(device)
    vit_model.eval()

if model_select == "en":
    try:
        en_weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
    except Exception:
        en_weights = None

    en_model = models.efficientnet_v2_l(weights=en_weights)
    if use_custom_weights:
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
        state = torch.load(resolved_en_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        en_model.load_state_dict(state, strict=True)
    else:
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )

    en_model.to(device)
    en_model.eval()

print(
    f"Using model_select={model_select}, device={device}, image_size={model_image_size}, use_custom_weights={use_custom_weights}"
)
print(f"Resolved weights: vit={resolved_vit_path}, en={resolved_en_path}")



## === cell 5
train_df = pd.read_csv(train_csv_path)
majority_label = int(train_df["label"].value_counts().idxmax())

if not use_custom_weights:
    max_train_images = len(train_df)  # use full available training set
    head_train_epochs = 5  # fixed (not early stopping); improves fit vs 2 epochs

    val_frac = 0.10

    train_pairs_all = list(
        zip(
            train_df["image_id"].astype(str).tolist(),
            train_df["label"].astype(int).tolist(),
        )
    )
    rng = random.Random(seed)
    rng.shuffle(train_pairs_all)
    train_pairs_all = train_pairs_all[: min(len(train_pairs_all), max_train_images)]

    val_n = int(len(train_pairs_all) * val_frac)
    val_pairs = train_pairs_all[:val_n]
    train_pairs = train_pairs_all[val_n:]

    if model_select == "vit":
        model = vit_model
        for p in model.parameters():
            p.requires_grad = False
        for p in model.heads.head.parameters():
            p.requires_grad = True
        head_params = list(model.heads.head.parameters())
        _get_head_state = lambda: {
            k: v.detach().cpu().clone()
            for k, v in model.heads.head.state_dict().items()
        }
        _load_head_state = lambda sd: model.heads.head.load_state_dict(sd, strict=True)
    else:
        model = en_model
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier[1].parameters():
            p.requires_grad = True
        head_params = list(model.classifier[1].parameters())
        _get_head_state = lambda: {
            k: v.detach().cpu().clone()
            for k, v in model.classifier[1].state_dict().items()
        }
        _load_head_state = lambda sd: model.classifier[1].load_state_dict(
            sd, strict=True
        )

    model.train()
    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 16 if device.type == "cuda" else 8
    epoch_losses = []

    best_val_acc = -1.0
    best_head_state = _get_head_state()

    def _eval_pairs_accuracy(pairs, max_batches=None):
        model.eval()
        correct = 0
        total = 0
        bs = batch_size
        with torch.inference_mode():
            idx2 = 0
            nb = 0
            while idx2 < len(pairs):
                batch = pairs[idx2 : idx2 + bs]
                idx2 += bs
                images = []
                labels = []
                for img_id, lab in batch:
                    img_path = os.path.join(train_image_directory, img_id)
                    try:
                        images.append(fast_val_transform_from_path(img_path))
                        labels.append(lab)
                    except Exception:
                        continue
                if not images:
                    continue
                x = torch.stack(images, dim=0).to(
                    device, non_blocking=(device.type == "cuda")
                )
                y = torch.tensor(labels, dtype=torch.long, device=device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().detach().cpu().item())
                total += int(y.numel())
                nb += 1
                if max_batches is not None and nb >= max_batches:
                    break
        model.train()
        return (correct / total) if total > 0 else 0.0

    for ep in range(head_train_epochs):
        idx = 0
        ep_losses = []
        while idx < len(train_pairs):
            batch = train_pairs[idx : idx + batch_size]
            idx += batch_size

            images = []
            labels = []
            for img_id, lab in batch:
                img_path = os.path.join(train_image_directory, img_id)
                try:
                    images.append(fast_val_transform_from_path(img_path))
                    labels.append(lab)
                except Exception:
                    continue

            if not images:
                continue

            x = torch.stack(images, dim=0)
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            ep_loss = float(loss.detach().cpu().item())
            ep_losses.append(ep_loss)
            epoch_losses.append(ep_loss)

        val_acc = _eval_pairs_accuracy(val_pairs, max_batches=None)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = _get_head_state()

        print(
            f"Head-only training epoch {ep+1}/{head_train_epochs} done. "
            f"mean_loss={np.mean(ep_losses) if ep_losses else float('nan'):.4f}, "
            f"val_acc={val_acc:.4f}, best_val_acc={best_val_acc:.4f}"
        )

    _load_head_state(best_head_state)
    model.eval()
    print(
        f"Head-only training done (no custom weights). Trained_on={len(train_pairs)} images, "
        f"val_on={len(val_pairs)} images, best_val_acc={best_val_acc:.4f}, "
        f"overall_mean_loss={np.mean(epoch_losses) if epoch_losses else float('nan'):.4f}"
    )



## === cell 6
sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].astype(str).tolist()

predictions = []
image_ids = []

default_label_on_error = majority_label

from torch.utils.data import Dataset, DataLoader


class TestImageDataset(Dataset):
    def __init__(self, image_ids, root_dir):
        self.image_ids = image_ids
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            x = fast_val_transform_from_path(img_path)
            ok = True
        except Exception:
            x = torch.zeros(
                (3, model_image_size, model_image_size), dtype=torch.float32
            )
            ok = False
        return img_name, x, ok


def _collate(batch):
    names, xs, oks = zip(*batch)
    return list(names), torch.stack(xs, 0), torch.tensor(oks, dtype=torch.bool)


cpu_cnt = os.cpu_count() or 2
num_workers = min(8, max(2, cpu_cnt // 2))
pin_memory = device.type == "cuda"
infer_bs = 16 if device.type == "cuda" else 8

test_ds = TestImageDataset(test_image_ids, test_data_directory)
test_loader = DataLoader(
    test_ds,
    batch_size=infer_bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=_collate,
)

model = vit_model if model_select == "vit" else en_model
model.eval()

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

with torch.inference_mode():
    for names, xb, okb in tqdm(
        test_loader, desc="Test", total=(len(test_ds) + infer_bs - 1) // infer_bs
    ):
        image_ids.extend(names)

        batch_preds = torch.full(
            (len(names),), int(default_label_on_error), dtype=torch.int64
        )

        if okb.any():
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device)

            logits = model(xb)
            pred = torch.argmax(logits, dim=1).to("cpu", dtype=torch.int64)
            batch_preds[okb] = pred[okb]

        predictions.extend(batch_preds.tolist())



## === cell 7
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

if submission_df.shape[0] != sample_df.shape[0]:
    raise ValueError(
        f"Submission row count {submission_df.shape[0]} != sample_submission row count {sample_df.shape[0]}"
    )
if list(submission_df.columns) != ["image_id", "label"]:
    raise ValueError(
        f"Submission columns are {list(submission_df.columns)}; expected ['image_id','label']"
    )

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
print(f"majority_label_used={majority_label}, use_custom_weights={use_custom_weights}")
