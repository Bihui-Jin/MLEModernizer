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

0.0994

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14761) has done: 'I fix the runtime failure caused by missing private checkpoint files by switching to a safe fallback: use pretrained timm weights when the expected .pth files are not present, keeping the same two-model ensemble and inference flow. I also correct the input preprocessing to match each backbone’s expected normalization (via timm’s default config) while preserving the same resize/CLAHE core image processing and averaging of logits. Finally, I ensure the submission length and ordering exactly match `sample_submission.csv` (no missing/None paths, no fallback length mismatch) and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.23393) has done: 'You’re hitting an inference-time preprocessing mismatch: `timm.data.create_transform` returns a torchvision pipeline that expects a PIL Image (or tensor), but you pass a NumPy array, causing the `TypeError` and preventing `names/labels` from being defined for submission. I fix this by converting the CLAHE-processed RGB NumPy array to a PIL image before applying the timm transforms (keeping the same models, resize+CLAHE, and 2-model + hflip-TTA averaging logic). I also make the resize size follow each model’s required `input_size` from its timm config (still resize+CLAHE, just correct dimensions), which should improve accuracy vs forcing 299 for both. Finally, I keep the sample_submission ordering and ensure `submission.csv` is always written.'
- What this solution (achieved 0.07623) has done: 'Your current score is far below the target, so the safest way to move accuracy upward (without changing the ensemble/architecture/inference semantics) is to fix a remaining preprocessing mismatch: `create_transform(..., is_training=False)` already includes resize/center-crop, but you also pre-resize with OpenCV, which can double-resize and harm accuracy. I keep your resize+CLAHE core logic, but I change the timm transforms to *skip* internal resizing/cropping so the model sees exactly your intended preprocessed image at the model’s native input size. I also ensure we apply the correct interpolation during the OpenCV resize (matching each model’s timm config) and keep everything else (two-model average + hflip TTA + argmax, submission ordering) identical.'
- What this solution (achieved 0.0994) has done: 'Your current accuracy (0.07623) is far below the target (0.8039), so we should make the smallest change that plausibly fixes a major correctness issue rather than tuning for marginal gains. The biggest problem here is that `timm.data.create_transform(..., is_training=False)` still applies its own resize/center-crop and expects to control input sizing, but you also resize beforehand; this double-resizing/center-cropping can destroy signal and crater accuracy. I keep your exact core logic (two-model ensemble + CLAHE + hflip-TTA + argmax), but replace the timm transforms with a minimal “no-resize/no-crop” transform that only does ToTensor + Normalize using each model’s own mean/std. This preserves your intended OpenCV resize+CLAHE as the sole geometric preprocessing while aligning normalization to the backbones, which should move the score substantially upward toward the target.'

# 9. Code solution

## === cell 0
import os, sys, glob
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import tqdm
from PIL import Image

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at: {TEST_IMG_DIR}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 1
def _safe_load_state_dict(model, ckpt_path):
    """
    Keep behavior: load checkpoints if present; otherwise fall back to timm pretrained weights.
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

eff_cfg = timm.data.resolve_data_config(
    getattr(efficient, "pretrained_cfg", {}), model=efficient
)
ser_cfg = timm.data.resolve_data_config(
    getattr(seres, "pretrained_cfg", {}), model=seres
)

eff_input_hw = tuple(eff_cfg["input_size"][-2:])  # (H, W)
ser_input_hw = tuple(ser_cfg["input_size"][-2:])  # (H, W)

print(
    "EfficientNet data cfg:",
    {
        k: eff_cfg[k]
        for k in ["input_size", "interpolation", "mean", "std", "crop_pct"]
        if k in eff_cfg
    },
)
print(
    "SEResNeXt data cfg:",
    {
        k: ser_cfg[k]
        for k in ["input_size", "interpolation", "mean", "std", "crop_pct"]
        if k in ser_cfg
    },
)


def _interp_from_timm_cfg(cfg):
    interp = str(cfg.get("interpolation", "bilinear")).lower()
    if "bicubic" in interp:
        return cv2.INTER_CUBIC
    if "lanczos" in interp:
        return cv2.INTER_LANCZOS4
    if "area" in interp:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR


eff_cv2_interp = _interp_from_timm_cfg(eff_cfg)
ser_cv2_interp = _interp_from_timm_cfg(ser_cfg)

from torchvision import transforms

eff_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=eff_cfg["mean"], std=eff_cfg["std"]),
    ]
)
ser_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=ser_cfg["mean"], std=ser_cfg["std"]),
    ]
)



## === cell 2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _clahe_bgr_to_rgb_uint8(image_bgr, out_size_hw, interpolation):
    """
    Preserve core logic (resize + CLAHE). Output RGB uint8.
    out_size_hw: (H, W)
    """
    h, w = int(out_size_hw[0]), int(out_size_hw[1])
    img = cv2.resize(image_bgr, (w, h), interpolation=interpolation)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def _timm_preprocess_from_rgb_uint8(img_rgb_uint8, transform):
    """
    timm/torchvision transform expects PIL image or Tensor.
    Convert numpy RGB uint8 -> PIL, then apply transform.
    """
    pil_img = Image.fromarray(img_rgb_uint8)
    x = transform(pil_img)  # CHW float tensor normalized
    x = x.unsqueeze(0).to(device)
    return x


def _predict_logits_from_bgr(image_bgr):
    """
    Keep behavior: two-model average, plus horizontal flip TTA averaged with original logits.
    """
    img_eff_rgb = _clahe_bgr_to_rgb_uint8(
        image_bgr, out_size_hw=eff_input_hw, interpolation=eff_cv2_interp
    )
    img_ser_rgb = _clahe_bgr_to_rgb_uint8(
        image_bgr, out_size_hw=ser_input_hw, interpolation=ser_cv2_interp
    )

    x_eff = _timm_preprocess_from_rgb_uint8(img_eff_rgb, eff_transform)
    x_ser = _timm_preprocess_from_rgb_uint8(img_ser_rgb, ser_transform)
    eff_out = efficient(x_eff)
    ser_out = seres(x_ser)
    logits = (ser_out + eff_out) / 2.0

    img_eff_rgb_fl = np.ascontiguousarray(img_eff_rgb[:, ::-1, :])
    img_ser_rgb_fl = np.ascontiguousarray(img_ser_rgb[:, ::-1, :])
    x_eff_fl = _timm_preprocess_from_rgb_uint8(img_eff_rgb_fl, eff_transform)
    x_ser_fl = _timm_preprocess_from_rgb_uint8(img_ser_rgb_fl, ser_transform)
    eff_out_fl = efficient(x_eff_fl)
    ser_out_fl = seres(x_ser_fl)
    logits_fl = (ser_out_fl + eff_out_fl) / 2.0

    return (logits + logits_fl) / 2.0




## === cell 3
sample = pd.read_csv(SAMPLE_SUB_PATH)
sample_ids = sample["image_id"].astype(str).tolist()

test_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
assert len(test_files) > 0, f"No test images found in: {TEST_IMG_DIR}"
file_map = {os.path.basename(p): p for p in test_files}

pred_map = {}
for image_id in tqdm.tqdm(sample_ids, total=len(sample_ids)):
    p = file_map.get(image_id, None)
    if p is None:
        pred_map[image_id] = 0
        continue
    img = cv2.imread(p)
    if img is None:
        pred_map[image_id] = 0
        continue
    logits = _predict_logits_from_bgr(img)
    pred_map[image_id] = int(torch.argmax(logits, dim=1).detach().cpu().item())

names, labels = [], []
missing = 0
for img_id in sample_ids:
    names.append(img_id)
    if img_id not in pred_map:
        labels.append(0)
        missing += 1
    else:
        labels.append(int(pred_map[img_id]))

if missing > 0:
    print(
        f"Warning: {missing} image_ids from sample_submission not predicted; set to 0."
    )

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
