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

0.17601

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the notebook-style `cd`/`pip` cells that don’t run in a plain Python Kaggle script and instead rely on already-installed Kaggle packages (torch/torchvision/timm/cv2). To fix the crash, I replace `efficientnet_pytorch` with the equivalent EfficientNet-B5 from `timm` and load your provided checkpoint with `strict=False` so it won’t error if key prefixes differ. I also fix a logic bug in `crop_image` (it referenced `img` instead of the passed `image`) and make the softmax call use an explicit `dim=1` to avoid runtime errors on newer torch. Finally, I ensure the submission is aligned to `sample_submission.csv` order and always writes `submission.csv`.'
- What this solution (achieved 0.17601) has done: 'I fix the immediate runtime blocker by removing the unavailable external checkpoints and ensuring the model objects are always defined, so inference can run end-to-end and write `submission.csv`. Because your current score (0.05531) is far below the target (0.8758), the safest minimal score-improving change is to use ImageNet-pretrained weights for the same backbones (EfficientNet-B5 and SEResNeXt101) rather than random initialization. I also make the MiDaS `torch.hub.load` robust by using the local cache when available (no internet) and falling back to a no-crop path if MiDaS cannot be loaded, so the script never crashes. The rest of your prediction loop and submission alignment to `sample_submission.csv` be preserved.'

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

print("Models have been loaded...\n")




## === cell 2
def processor(image: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image, (512, 512))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    image_t = (
        torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    return image_t


@torch.no_grad()
def get_depth(img_bgr: np.ndarray) -> np.ndarray:
    if midas is None or transform is None:
        h, w = img_bgr.shape[:2]
        return np.ones((h, w), dtype=bool)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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

    return masked_arr[y_min : y_max + 1, x_min : x_max + 1]




## === cell 3
files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(files) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

names, labels = [], []

with torch.no_grad():
    for file in files:
        img = cv2.imread(file)
        if img is None:
            continue

        depth = get_depth(img)
        img_cropped = crop_image(img, depth)
        img_t = processor(img_cropped)

        se_out = seresnext(img_t)
        eff_out = efficient(img_t)
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
