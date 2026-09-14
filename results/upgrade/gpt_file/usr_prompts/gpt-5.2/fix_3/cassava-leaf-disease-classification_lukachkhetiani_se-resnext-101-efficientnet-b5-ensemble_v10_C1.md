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

0.8159564823209429

# 6. Current score

0.1278

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the missing `efficientnet_pytorch` dependency by removing the manual wheel installs and instead using only libraries that are already available in Kaggle (notably `timm`). I keep the ensemble logic the same (two models averaged) by loading both models via `timm` and then loading the provided `.pth` weights with `map_location='cpu'` for robustness before moving to GPU. I also fix the softmax/argmax call (`dim=1` is required) and add safe handling for CPU-only environments so the notebook always completes. Finally, I ensure the submission is written as `submission.csv` with exactly `image_id,label` and the correct test set order (sorted by filename, matching sample submission ordering expectations).'
- What this solution (achieved 0.1278) has done: 'I fix the immediate runtime failure by removing the hard dependency on missing `../input/ensemblev3/*.pth` files and instead loading robust pretrained `timm` weights (same two-model averaging ensemble logic) so the notebook runs end-to-end. I also correct the input preprocessing to match what these `timm` models expect (RGB + ImageNet normalization at the correct resolution) to move accuracy up substantially toward the target. Finally, I keep the exact submission schema/order by merging with `sample_submission.csv` and always writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import cv2
import torch
import timm
import pandas as pd
import tqdm

torch.backends.cudnn.benchmark = True

DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("DEVICE:", DEVICE)
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print(
    "TEST_IMG_DIR exists:",
    os.path.exists(TEST_IMG_DIR),
    "num_files:",
    len(glob.glob(os.path.join(TEST_IMG_DIR, "*"))),
)




## === cell 1
def load_state_dict_flexible(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(unexpected) > 0:
        print(
            f"Warning: unexpected keys while loading {os.path.basename(path)} (showing up to 5): {unexpected[:5]}"
        )
    if len(missing) > 0:
        print(
            f"Warning: missing keys while loading {os.path.basename(path)} (showing up to 5): {missing[:5]}"
        )
    return model


efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)
hrnet = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)

efficient.to(DEVICE).eval()
hrnet.to(DEVICE).eval()

print("Models have been loaded...\n")
print("efficient:", efficient.__class__.__name__)
print("hrnet:", hrnet.__class__.__name__)



## === cell 2
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    1, 3, 1, 1
)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

eff_size = efficient.default_cfg.get("input_size", (3, 456, 456))[1]
hr_size = hrnet.default_cfg.get("input_size", (3, 224, 224))[1]


def _clahe_bgr(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


def process_for_size(image_bgr, size):
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = _clahe_bgr(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = (
        torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
    )  # 3xHxW in [0,1]
    x = x.unsqueeze(0)  # 1x3xHxW
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(DEVICE)


print("Using input sizes eff_size=", eff_size, "hr_size=", hr_size)



## === cell 3
files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*")))
names, labels = [], []

if len(files) == 0:
    raise RuntimeError(f"No test images found in {TEST_IMG_DIR}")

with torch.no_grad():
    for file in tqdm.tqdm(files, total=len(files)):
        img = cv2.imread(file)
        if img is None:
            continue

        x_hr = process_for_size(img, hr_size)
        x_eff = process_for_size(img, eff_size)

        hr_out = hrnet(x_hr)
        eff_out = efficient(x_eff)

        total = (hr_out + eff_out) / 2.0
        pred = int(torch.argmax(total, dim=1).item())

        names.append(os.path.basename(file))
        labels.append(pred)

print("Predictions made for:", len(names), "images")



## === cell 4
df = pd.DataFrame({"image_id": names, "label": labels}, columns=["image_id", "label"])

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    df = sample[["image_id"]].merge(df, on="image_id", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {df.shape}")
print(df.head())
print("Label value counts:\n", df["label"].value_counts(dropna=False).sort_index())
