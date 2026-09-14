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

0.19283

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13416) has done: 'I fixed the script by making model weight loading optional (skip if files are missing), guaranteeing that `use_midas` is always defined, and loading the test images using the official `sample_submission.csv` order so the output length matches the expected number of rows. The code now safely falls back to random‑initialized models and skips depth estimation when MiDaS cannot be loaded, then writes a correct `submission.csv`.'
- What this solution (achieved 0.19283) has done: 'I load the EfficientNet and SEResNeXt models with ImageNet‑pretrained weights, fix the image preprocessing to scale pixel values to [0, 1] before applying the ImageNet mean/std, and skip the optional depth‑based cropping (which can shrink or distort leaves). These minimal adjustments keep the overall pipeline intact while greatly improving predictive quality, moving the validation accuracy from 0.13 toward the target 0.875.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn.functional as F
import cv2
import glob
import numpy as np
import pandas as pd
import timm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from efficientnet_pytorch import EfficientNet

    efficient = EfficientNet.from_name("efficientnet-b5", num_classes=5)
except Exception:
    efficient = timm.create_model(
        "tf_efficientnet_b5_ns", pretrained=True, num_classes=5
    )
else:
    pass

eff_path = Path("../input/ensemblev5/eff_best.pth")
if eff_path.is_file():
    try:
        efficient.load_state_dict(torch.load(str(eff_path), map_location=device))
        print("EfficientNet fine‑tuned weights loaded.")
    except Exception as e:
        print(f"Failed to load EfficientNet fine‑tuned weights: {e}")
else:
    print(
        "EfficientNet fine‑tuned weight file not found – using pretrained ImageNet weights."
    )
efficient.eval().to(device)

seresnext = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)
ser_path = Path("../input/ensemblev5/seresnext_best.pth")
if ser_path.is_file():
    try:
        seresnext.load_state_dict(torch.load(str(ser_path), map_location=device))
        print("SEResNeXt fine‑tuned weights loaded.")
    except Exception as e:
        print(f"Failed to load SEResNeXt fine‑tuned weights: {e}")
else:
    print(
        "SEResNeXt fine‑tuned weight file not found – using pretrained ImageNet weights."
    )
seresnext.eval().to(device)

print("Models have been prepared.")

use_midas = False
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform
    use_midas = True
    print("MiDaS depth model loaded.")
except Exception:
    print("MiDaS could not be loaded; depth will be ignored.")




## === cell 1
def processor(image):
    """Resize, normalize and convert a BGR image to a torch tensor."""
    img = cv2.resize(image, (512, 512)).astype(np.float32) / 255.0  # scale to [0,1]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet std
    tensor = (
        torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0).to(device)
    )
    return tensor


def get_depth(img):
    """Return a binary mask from MiDaS depth; fallback to all‑True mask."""
    if not use_midas:
        return np.ones((img.shape[0], img.shape[1]), dtype=bool)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    img_min, img_max = np.min(output), np.max(output)
    return output > ((img_min + img_max) / 3)


def crop_image(image, depth):
    """Crop image to the region where depth mask is True."""
    depth = depth.astype(int)
    mask_3d = np.stack((depth, depth, depth), axis=2)
    masked_arr = np.where(mask_3d == 1, image, mask_3d).astype(np.uint8)
    coords = np.where(masked_arr != [0, 0, 0])
    if coords[0].size == 0:  # safety check – return original if mask empty
        return image
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return masked_arr[y_min:y_max, x_min:x_max]




## === cell 2
sample_path = Path("../input/cassava-leaf-disease-classification/sample_submission.csv")
submission_template = pd.read_csv(sample_path)
test_img_dir = Path("../input/cassava-leaf-disease-classification/test_images")

names, labels = [], []

for img_name in submission_template["image_id"]:
    img_path = test_img_dir / img_name
    if not img_path.is_file():
        print(f"Warning: image {img_path} not found.")
        names.append(img_name)
        labels.append(0)  # dummy class
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: failed to read {img_path}.")
        names.append(img_name)
        labels.append(0)
        continue

    img_cropped = img  # use original image

    img_tensor = processor(img_cropped)

    with torch.no_grad():
        se_out = seresnext(img_tensor)
        eff_out = efficient(img_tensor)
        total = (se_out + eff_out) / 2.0
        probs = F.softmax(total, dim=1)
        pred_label = int(torch.argmax(probs, dim=1).cpu().item())

    names.append(img_name)
    labels.append(pred_label)



## === cell 3
submission_df = pd.DataFrame({"image_id": names, "label": labels})
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv (rows:", len(submission_df), ")")
