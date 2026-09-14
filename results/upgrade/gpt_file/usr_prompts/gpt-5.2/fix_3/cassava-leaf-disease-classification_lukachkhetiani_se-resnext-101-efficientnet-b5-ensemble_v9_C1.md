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

0.8038682381384104

# 6. Current score

0.14761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.14761) has done: 'I fix the runtime failure caused by missing private checkpoint files by switching to a safe fallback: use pretrained timm weights when the expected .pth files are not present, keeping the same two-model ensemble and inference flow. I also correct the input preprocessing to match each backbone’s expected normalization (via timm’s default config) while preserving the same resize/CLAHE core image processing and averaging of logits. Finally, I ensure the submission length and ordering exactly match `sample_submission.csv` (no missing/None paths, no fallback length mismatch) and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os, sys, glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import timm
import tqdm

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at: {TEST_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at: {SAMPLE_SUB_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 1
def _safe_load_state_dict(model, ckpt_path):
    """
    Bugfix: original notebook hard-failed when private checkpoints weren't present.
    We keep the same model definitions but only load checkpoints if they exist.
    """
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return False

    sd = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"Loaded {os.path.basename(ckpt_path)} | missing={len(missing)} unexpected={len(unexpected)}"
    )
    return True


EFF_CKPT = "../input/ensemble-2/eff_model_last.pth"
SERES_CKPT = "../input/ensemble-2/seresnext_model_last.pth"

efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)
_loaded_eff = _safe_load_state_dict(efficient, EFF_CKPT)
efficient.to(device).eval()

seres = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)
_loaded_seres = _safe_load_state_dict(seres, SERES_CKPT)
seres.to(device).eval()

print(
    f"Models ready. ckpt_loaded: efficient={_loaded_eff}, seresnext={_loaded_seres}\n"
)


def _get_norm_from_model(m):
    cfg = getattr(m, "default_cfg", {}) or {}
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


eff_mean, eff_std = _get_norm_from_model(efficient)
ser_mean, ser_std = _get_norm_from_model(seres)



## === cell 2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def process(image_bgr):
    """
    Preserve core logic (resize + CLAHE), but ensure tensor is float in [0,1].
    Normalization is applied later per-model to match timm defaults.
    """
    img = cv2.resize(image_bgr, (299, 299))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
    x = x.unsqueeze(0).to(device)
    return x




## === cell 3
test_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
assert len(test_files) > 0, f"No test images found in: {TEST_IMG_DIR}"

sample = pd.read_csv(SAMPLE_SUB_PATH)
sample_ids = sample["image_id"].astype(str).tolist()
file_map = {os.path.basename(p): p for p in test_files}

ordered_files = [file_map.get(img_id, None) for img_id in sample_ids]
missing = sum(p is None for p in ordered_files)
if missing > 0:
    print(
        f"Warning: {missing} image_ids from sample_submission not found in test_images; predicting label=0 for those."
    )

names, labels = [], []

for p, img_id in tqdm.tqdm(list(zip(ordered_files, sample_ids)), total=len(sample_ids)):
    names.append(img_id)

    if p is None:
        labels.append(0)
        continue

    img = cv2.imread(p)
    if img is None:
        labels.append(0)
        continue

    x = process(img)

    x_eff = (x - eff_mean) / eff_std
    x_ser = (x - ser_mean) / ser_std

    eff_out = efficient(x_eff)
    ser_out = seres(x_ser)

    total = (ser_out + eff_out) / 2.0
    pred = int(torch.argmax(total, dim=1).detach().cpu().item())
    labels.append(pred)

assert (
    len(names) == len(sample_ids) == len(labels)
), "Internal length mismatch; submission would be invalid."



## === cell 4
sub = pd.DataFrame({"image_id": names, "label": labels})

sub["image_id"] = sub["image_id"].astype(str)
sub["label"] = sub["label"].astype(int)

sample = pd.read_csv(SAMPLE_SUB_PATH)
assert (
    sub.shape[0] == sample.shape[0]
), f"Invalid submission length: got {sub.shape[0]}, expected {sample.shape[0]}"
assert list(sub.columns) == [
    "image_id",
    "label",
], f"Invalid submission columns: {sub.columns.tolist()}"

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Label distribution:\n", sub["label"].value_counts().sort_index())
