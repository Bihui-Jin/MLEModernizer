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

3.9

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

0.8757932910244787

# 6. Current score

0.25635

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the notebook-style `cd`/`pip` cells that don’t run in a plain Python Kaggle script and instead rely on already-installed Kaggle packages (torch/torchvision/timm/cv2). To fix the crash, I replace `efficientnet_pytorch` with the equivalent EfficientNet-B5 from `timm` and load your provided checkpoint with `strict=False` so it won’t error if key prefixes differ. I also fix a logic bug in `crop_image` (it referenced `img` instead of the passed `image`) and make the softmax call use an explicit `dim=1` to avoid runtime errors on newer torch. Finally, I ensure the submission is aligned to `sample_submission.csv` order and always writes `submission.csv`.'
- What this solution (achieved 0.17601) has done: 'I fix the immediate runtime blocker by removing the unavailable external checkpoints and ensuring the model objects are always defined, so inference can run end-to-end and write `submission.csv`. Because your current score (0.05531) is far below the target (0.8758), the safest minimal score-improving change is to use ImageNet-pretrained weights for the same backbones (EfficientNet-B5 and SEResNeXt101) rather than random initialization. I also make the MiDaS `torch.hub.load` robust by using the local cache when available (no internet) and falling back to a no-crop path if MiDaS cannot be loaded, so the script never crashes. The rest of your prediction loop and submission alignment to `sample_submission.csv` be preserved.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the smallest safe way to move accuracy upward without changing the core model/loop is to (1) switch preprocessing to each timm model’s native `resolve_data_config` + `create_transform` (so normalization/resize match what pretrained weights expect) and (2) ensure the input color order is correct (timm expects RGB, while OpenCV loads BGR). I also keep your MiDaS cropping logic intact but apply it on the RGB image to avoid channel-mismatch artifacts that can hurt predictions. Finally, I keep the same ensemble averaging and submission alignment, only changing the input pipeline so the pretrained backbones are used correctly.'
- What this solution (achieved 0.25635) has done: 'I fix the runtime error by ensuring the `timm` transforms receive the expected input type (PIL Image), since the current transform pipeline is a torchvision Compose that doesn’t accept raw numpy arrays in this environment. This is a minimal change confined to the preprocessing function and does not alter the model architecture, ensemble logic, or prediction semantics. I also add a small safety fallback for empty/invalid crops to avoid rare downstream transform failures, while keeping the rest of the inference loop intact. The script then run end-to-end and write a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

EFF_CKPT = "/kaggle/input/ensemblev5/eff_best.pth"
SE_CKPT = "/kaggle/input/ensemblev5/seresnext_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(0)
np.random.seed(0)
if device.type == "cuda":
    torch.cuda.manual_seed_all(0)



## === cell 1
midas = None
transform = None
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform
    print("Loaded MiDaS from torch.hub cache.")
except Exception as e:
    print("MiDaS not available (offline/no cache). Will skip depth-based cropping.")
    print("MiDaS load error:", repr(e))
    midas = None
    transform = None

efficient = timm.create_model("efficientnet_b5", pretrained=True, num_classes=5)
seresnext = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)

if os.path.exists(EFF_CKPT):
    eff_state = torch.load(EFF_CKPT, map_location="cpu")
    missing, unexpected = efficient.load_state_dict(eff_state, strict=False)
    print(
        f"Loaded EfficientNet checkpoint. missing={len(missing)}, unexpected={len(unexpected)}"
    )
else:
    print(f"Checkpoint not found: {EFF_CKPT} (using pretrained EfficientNet weights)")

if os.path.exists(SE_CKPT):
    se_state = torch.load(SE_CKPT, map_location="cpu")
    missing, unexpected = seresnext.load_state_dict(se_state, strict=False)
    print(
        f"Loaded SEResNeXt checkpoint. missing={len(missing)}, unexpected={len(unexpected)}"
    )
else:
    print(f"Checkpoint not found: {SE_CKPT} (using pretrained SEResNeXt weights)")

efficient.to(device).eval()
seresnext.to(device).eval()

eff_cfg = resolve_data_config({}, model=efficient)
se_cfg = resolve_data_config({}, model=seresnext)
eff_tf = create_transform(**eff_cfg, is_training=False)
se_tf = create_transform(**se_cfg, is_training=False)

print("Models and timm transforms have been loaded...\n")




## === cell 2
def _to_pil_rgb(image_rgb: np.ndarray) -> Image.Image:
    if (
        image_rgb is None
        or not isinstance(image_rgb, np.ndarray)
        or image_rgb.size == 0
    ):
        image_rgb = np.zeros(
            (
                eff_cfg.get("input_size", (3, 456, 456))[1],
                eff_cfg.get("input_size", (3, 456, 456))[2],
                3,
            ),
            dtype=np.uint8,
        )
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(image_rgb, mode="RGB")


def processor(image_rgb: np.ndarray) -> torch.Tensor:
    pil = _to_pil_rgb(image_rgb)
    x_eff = eff_tf(pil)  # CHW float tensor
    x_se = se_tf(pil)
    return x_eff.unsqueeze(0).to(device), x_se.unsqueeze(0).to(device)


@torch.no_grad()
def get_depth(img_rgb: np.ndarray) -> np.ndarray:
    if midas is None or transform is None:
        h, w = img_rgb.shape[:2]
        return np.ones((h, w), dtype=bool)

    input_batch = transform(img_rgb).to(device)
    prediction = midas(input_batch)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    output = prediction.detach().cpu().numpy()
    img_min = float(np.min(output))
    img_max = float(np.max(output))
    return output > ((img_min + img_max) / 3.0)


def crop_image(image: np.ndarray, depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.uint8)
    mask_3d = np.stack((depth, depth, depth), axis=2)
    masked_arr = np.where(mask_3d == 1, image, 0).astype(np.uint8)

    coords = np.where(np.any(masked_arr != 0, axis=2))
    if coords[0].size == 0 or coords[1].size == 0:
        return image

    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())

    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(image.shape[0] - 1, y_max)
    x_max = min(image.shape[1] - 1, x_max)

    if y_max <= y_min or x_max <= x_min:
        return image

    cropped = masked_arr[y_min : y_max + 1, x_min : x_max + 1]
    if (
        cropped is None
        or cropped.size == 0
        or cropped.shape[0] < 2
        or cropped.shape[1] < 2
    ):
        return image
    return cropped




## === cell 3
files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(files) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

names, labels = [], []

with torch.no_grad():
    for file in files:
        img_bgr = cv2.imread(file)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth = get_depth(img_rgb)
        img_cropped = crop_image(img_rgb, depth)

        eff_t, se_t = processor(img_cropped)

        se_out = seresnext(se_t)
        eff_out = efficient(eff_t)
        total = (se_out + eff_out) / 2.0

        pred = int(torch.argmax(torch.softmax(total, dim=1), dim=1).item())

        names.append(os.path.basename(file))
        labels.append(pred)

print("Predictions:", len(labels))



## === cell 4
sub = pd.read_csv(SAMPLE_SUB_PATH)
pred_df = pd.DataFrame({"image_id": names, "label": labels})

sub = sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if sub["label"].isna().any():
    fallback = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    sub["label"] = sub["label"].fillna(fallback).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path)
print(sub.head())
