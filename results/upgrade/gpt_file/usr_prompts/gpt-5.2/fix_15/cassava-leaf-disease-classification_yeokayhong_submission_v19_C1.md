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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it can actually load model weights in this environment by searching common Kaggle input/data locations and falling back to EfficientNetV2-L if the ViT checkpoint is missing. I also fix the device/dtype mismatch that caused CUDA inputs to be fed into a CPU model by ensuring the loaded state dict and the whole model are moved to the selected device before inference. Finally, I make the inference loop robust (skip missing/corrupt images but keep row alignment by default) and always write a correctly formatted `submission.csv` with exactly the same `image_id` order/length as `sample_submission.csv`, preventing “Not yielded”.'
- What this solution (achieved 0.61099) has done: 'Your current crash comes from relying on external weight files that don’t exist in this Kaggle environment, so the script never reaches inference/submission. I fix this by removing the hard failure and instead using torchvision’s built-in pretrained weights (available offline in Kaggle) for EfficientNetV2-L and ViT-H14, while keeping your same preprocessing and inference loop structure. Because pretrained heads won’t match 5 cassava classes, I keep the model heads as-is and output a safe, accuracy-improving baseline by predicting the majority class from `train.csv` (this is a minimal, legitimate calibration step and typically scores far above ~0.055 on Cassava). The code still produce a valid `submission.csv` aligned exactly to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I remove the most expensive source of runtime: the per-image PIL-based Python transforms and resizing pipeline, replacing it with an equivalent `torchvision.io`/tensor-based preprocessing that runs in C++ and inside DataLoader workers. I keep the same model, weights loading, training logic (only head-only training when no custom weights), loss, and evaluation semantics, but make the data pipeline faster via vectorized tensor ops, `channels_last` where safe, and avoiding redundant `pin_memory()` calls. I also tune DataLoader settings (workers/prefetch/persistent workers) for better throughput without changing outputs, and avoid per-sample exception overhead by doing robust decode with a consistent fallback.'
- What this solution (achieved 0.61099) has done: 'Your current 0.61099 score comes from predicting the majority class for every test image (the model head is untrained / random when custom weights aren’t found), so the smallest legitimate way to move toward the 0.8879 target is to actually fine-tune the existing model head on the provided `train.csv` images. I keep your exact model choices, transforms, and “head-only training when no custom weights” logic, but make that head-only training effective by (1) shuffling the training pairs, (2) training for a small fixed number of epochs, and (3) using a slightly larger (but safe) training subset, all without changing the loss or architecture. I also keep the same robust inference/submission alignment, so it still always emits a valid `submission.csv`. These changes should materially improve accuracy over majority-class while staying within Kaggle time constraints.'
- What this solution (achieved 0.61099) has done: 'You’re currently far below the target (0.61099 vs 0.88788, higher-is-better), and the main limiter is that you’re only doing a tiny amount of head-only training and not using any validation-guided selection, so the fine-tuned head is likely underfit and/or unstable. I keep the same model, preprocessing, and “freeze backbone, train only the head” approach, but (1) train the head on the full training set (instead of truncating to 12k) and for a few more fixed epochs, and (2) add a small deterministic validation split to pick the best head epoch (no early stopping—still runs a fixed number of epochs). This should legitimately increase accuracy toward your target without changing architecture/loss/feature extraction, and still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your gap to target is large (0.61099 → 0.88788), and the bottleneck is that you’re only training the classification head on ImageNet features without addressing severe class imbalance, so the head learns a biased decision boundary and underperforms. I keep the exact same model choices, transforms, loss, and “freeze backbone, train only head” approach, but change the head-only training batches to be class-balanced via a `WeightedRandomSampler` (this is a minimal data-loading change that typically gives a big accuracy lift on Cassava). I also switch the head optimizer from AdamW to SGD+momentum (still standard, still head-only, same loss) because it often yields better calibrated linear-head convergence on fixed features; everything else (epochs fixed, no early stopping, same preprocessing) remains the same. Submission writing and row alignment are unchanged.'
- What this solution (achieved 0.61099) has done: 'You’re far below the target (0.61099 vs 0.88788, higher-is-better), so we should legitimately improve generalization without changing your model architecture or loss. The biggest low-risk gain here is to (1) train the head with label smoothing to reduce overconfidence/overfit on fixed ImageNet features, and (2) apply lightweight test-time augmentation (horizontal flip) and average logits, which keeps the same inference semantics (argmax of averaged logits) and typically boosts Cassava accuracy. I keep your head-only training loop structure, sampler, optimizer family (SGD), and preprocessing, and only add these minimal, directly score-relevant adjustments. Submission writing/order remains identical to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current gap to the target is large (0.61099 → 0.88788), so we should improve legitimate generalization without changing the core “freeze backbone, train only head” approach. The smallest high-impact change is to switch the head-only training to use the model’s native ImageNet preprocessing (mean/std) instead of the current `[-1, 1]` normalization, because torchvision’s pretrained backbones expect ImageNet normalization and a mismatch can severely cap accuracy. I keep your same architecture, same optimizer/loss/training loop structure, same sampler/TTA, and only adjust preprocessing (plus make the exact same preprocessing used for both train-head and test inference). This should move accuracy materially upward toward the target while remaining stable and within runtime.'

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


def _build_weights_and_transform(model_select: str):
    if model_select == "vit":
        try:
            w = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        except Exception:
            w = None
    else:
        try:
            w = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        except Exception:
            w = None

    if w is None:
        IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(
            3, 1, 1
        )
        IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(
            3, 1, 1
        )

        def _fallback_transform_from_path(img_path: str) -> torch.Tensor:
            data = read_file(img_path)
            img = decode_jpeg(data, mode=torchvision.io.ImageReadMode.RGB)  # HWC u8
            img = img.permute(2, 0, 1)  # CHW u8
            img = F.resize(
                img,
                [model_image_size, model_image_size],
                interpolation=F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            img = img.to(dtype=torch.float32).div_(255.0)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            return img

        return None, _fallback_transform_from_path

    tfm = w.transforms()

    def _weights_transform_from_path(img_path: str) -> torch.Tensor:
        data = read_file(img_path)
        img = decode_jpeg(data, mode=torchvision.io.ImageReadMode.RGB)  # HWC u8
        img = img.permute(2, 0, 1)  # CHW u8
        out = tfm(img)  # float tensor, normalized
        return out

    return w, _weights_transform_from_path


resolved_weights_obj, fast_val_transform_from_path = _build_weights_and_transform(
    model_select
)




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
    vit_weights = resolved_weights_obj if model_select == "vit" else None
    model = models.vit_h_14(weights=vit_weights, image_size=vit_image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    if use_custom_weights:
        state = torch.load(resolved_vit_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
else:
    en_weights = resolved_weights_obj if model_select == "en" else None
    model = models.efficientnet_v2_l(weights=en_weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    if use_custom_weights:
        state = torch.load(resolved_en_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

model.to(device)
model.eval()

print(
    f"Using model_select={model_select}, device={device}, image_size={model_image_size}, use_custom_weights={use_custom_weights}"
)
print(f"Resolved weights: vit={resolved_vit_path}, en={resolved_en_path}")

if device.type == "cuda":
    try:
        torch.backends.cudnn.benchmark = (
            True  # fixed-size ops; deterministic disabled above anyway
        )
    except Exception:
        pass
try:
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
except Exception:
    pass



## === cell 5
train_df = pd.read_csv(train_csv_path)
majority_label = int(train_df["label"].value_counts().idxmax())

do_head_training = not use_custom_weights



## === cell 6
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

cpu_cnt = os.cpu_count() or 2
num_workers = min(8, max(2, cpu_cnt // 2))
pin_memory = device.type == "cuda"

infer_bs = 16 if device.type == "cuda" else 8
train_bs = 16 if device.type == "cuda" else 8


class TrainImageDataset(Dataset):
    def __init__(self, df, root_dir):
        self.image_ids = df["image_id"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        y = int(self.labels[idx])
        img_path = os.path.join(self.root_dir, img_name)
        try:
            x = fast_val_transform_from_path(img_path)
            ok = True
        except Exception:
            x = torch.zeros(
                (3, model_image_size, model_image_size), dtype=torch.float32
            )
            ok = False
        return x, y, ok


def _seed_worker(worker_id: int):
    s = seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


g = torch.Generator()
g.manual_seed(seed)

train_ds = TrainImageDataset(train_df, train_image_directory)

labels = np.array(train_df["label"].astype(int).values)
class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
class_counts[class_counts == 0] = 1.0
class_weights = class_counts.sum() / class_counts
sample_weights = class_weights[labels]
sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
    generator=g,
)

train_loader = DataLoader(
    train_ds,
    batch_size=train_bs,
    shuffle=False,
    sampler=sampler if do_head_training else None,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)



## === cell 7
if do_head_training:
    for p in model.parameters():
        p.requires_grad = False

    head_params = []
    if model_select == "vit":
        for p in model.heads.head.parameters():
            p.requires_grad = True
        head_params = list(model.heads.head.parameters())
    else:
        for p in model.classifier[1].parameters():
            p.requires_grad = True
        head_params = list(model.classifier[1].parameters())

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.SGD(head_params, lr=0.05, momentum=0.9, weight_decay=1e-4)

    model.train()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    epochs = 3  # fixed small number to stay within time while improving score
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        seen = 0
        ok_seen = 0
        for xb, yb, okb in tqdm(
            train_loader, desc=f"Head train epoch {epoch}/{epochs}"
        ):
            if not okb.any():
                continue

            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
                yb = torch.as_tensor(yb, device=device, dtype=torch.long)
                okb = okb.to(device, non_blocking=True)
            else:
                xb = xb.to(device)
                yb = torch.as_tensor(yb, device=device, dtype=torch.long)
                okb = okb.to(device)

            xb = xb[okb]
            yb = yb[okb]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bsz = int(xb.shape[0])
            running_loss += float(loss.detach().cpu()) * bsz
            seen += bsz
            ok_seen += bsz

        avg_loss = running_loss / max(1, seen)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.5f}, samples={ok_seen}")

    model.eval()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 102) is killed by signal: Killed. 

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1014054179.py in <cell line: 0>()
     29         seen = 0
     30         ok_seen = 0
---> 31         for xb, yb, okb in tqdm(
     32             train_loader, desc=f"Head train epoch {epoch}/{epochs}"
     33         ):

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 102) exited unexpectedly

## === cell 8
sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].astype(str).tolist()

predictions = []
image_ids = []

default_label_on_error = majority_label


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
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=(g if num_workers == 0 else None),
)

model.eval()
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

tta_hflip = True

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

            if tta_hflip:
                xb_flip = torch.flip(xb, dims=[3])
                logits_flip = model(xb_flip)
                logits = (logits + logits_flip) * 0.5

            pred = torch.argmax(logits, dim=1).to("cpu", dtype=torch.int64)
            batch_preds[okb] = pred[okb]

        predictions.extend(batch_preds.tolist())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 130) is killed by signal: Killed. 

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1435493357.py in <cell line: 0>()
     57 
     58 with torch.inference_mode():
---> 59     for names, xb, okb in tqdm(
     60         test_loader, desc="Test", total=(len(test_ds) + infer_bs - 1) // infer_bs
     61     ):

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 130) exited unexpectedly

## === cell 9
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
print(
    f"majority_label_used={majority_label}, use_custom_weights={use_custom_weights}, do_head_training={do_head_training}"
)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/405001150.py in <cell line: 0>()
      2 
      3 if submission_df.shape[0] != sample_df.shape[0]:
----> 4     raise ValueError(
      5         f"Submission row count {submission_df.shape[0]} != sample_submission row count {sample_df.shape[0]}"
      6     )

ValueError: Submission row count 0 != sample_submission row count 2676
