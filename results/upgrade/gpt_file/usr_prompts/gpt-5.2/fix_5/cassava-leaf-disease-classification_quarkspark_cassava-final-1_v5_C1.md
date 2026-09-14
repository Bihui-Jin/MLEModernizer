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

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.828649138712602

# 6. Current score

0.11622

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix three execution blockers with minimal changes: (1) make weight loading robust by checking for the expected files and falling back to random-weight inference (still produces a valid submission) instead of crashing; (2) fix test image enumeration to skip nested directories like `test_images/test_images` and only process `.jpg` files in a stable sorted order; and (3) align the submission to `sample_submission.csv` so length/order match exactly, preventing the “same length as the answers” error. I also ensure inference runs on the available device and wrap it in `torch.no_grad()` to avoid unnecessary memory use, without changing the model architecture or prediction logic.'
- What this solution (achieved 0.05531) has done: 'Your low score is consistent with running inference on essentially random weights because the code points to a non-existent `../input/cassava-models` directory and also sets `num_classes=4` even though the competition has 5 classes. To move accuracy toward your target with minimal logic change, I (1) fix `num_classes` to 5, (2) load `minority_weights.pth` and `binary_weights.pth` from the actual dataset folder you already have (`/kaggle/input/cassava-leaf-disease-classification/`) if present, and (3) make weight loading tolerant to `module.` prefixes / nested `state_dict` so your intended trained weights actually get applied. I not change the model architecture, transforms, or the existing two-model + threshold decision logic; this only ensures the intended weights/classes are used so predictions aren’t random.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so we should increase accuracy with the smallest changes that keep your two-model + threshold core logic intact. The biggest likely issue is that the intended `.pth` weights are still not being found/loaded, so the models effectively run with random weights; I make weight discovery actually locate those files anywhere under `proj_dir` (including nested folders) and load them with safe key normalization. I also fix `prediction_logic()` to apply softmax along the class dimension explicitly (axis=1), avoiding accidental normalization over the wrong axis which can severely damage predictions without changing the model/decision logic. Finally, I keep submission alignment to `sample_submission.csv` exactly as you already do.'
- What this solution (achieved 0.11622) has done: 'I fix the runtime KeyError by making the `minority_idx` mapping consistent with a 5-class minority model output (0–4), without changing your two-model + threshold decision logic. I also make the argmax explicitly over the class dimension to avoid accidental flattening behavior, keeping evaluation semantics the same. Finally, I keep the submission aligned to `sample_submission.csv` and ensure the script completes end-to-end and writes `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys, cv2
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

sz = 224
num_classes = 5
proj_dir = "/kaggle/input/cassava-leaf-disease-classification/"




## === cell 1
def softmax(X, theta=1.0, axis=None):
    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p




## === cell 2
def light_model(num_classes):
    squeezenet_custom = torchvision.models.squeezenet1_0(pretrained=False)

    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(1, 1),
        ),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )

    squeezenet_custom.classifier = classifier
    return squeezenet_custom


squeezenet_custom_4 = light_model(5)
squeezenet_custom_2 = light_model(2)



## === cell 3
leaf_transform = transforms.Compose(
    [
        transforms.CenterCrop(400),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 4
minority_idx = {
    0: 0,
    1: 1,
    2: 2,
    3: 4,  # historically remapped to class 4
    4: 4,  # prevent KeyError; map to a valid final label without changing decision structure
}

binary_idx = {
    0: "0_1_2_4",
    1: 3,
}



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

candidate_weight_dirs = [
    os.path.join(proj_dir, "cassava-models"),
    os.path.join(proj_dir, "models"),
    proj_dir,  # sometimes weights are placed at the dataset root
    "/kaggle/input/cassava-models",
    "../input/cassava-models",
]


def _find_weight_file(filename: str):
    for d in candidate_weight_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p

    if os.path.isdir(proj_dir):
        for root, _, files in os.walk(proj_dir):
            if filename in files:
                return os.path.join(root, filename)

    return None


model_1_path = _find_weight_file("minority_weights.pth")
model_2_path = _find_weight_file("binary_weights.pth")


def _normalize_state_dict_keys(state):
    if not isinstance(state, dict):
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(k, str) for k in obj.keys()) and any(
            k.startswith("classifier")
            or k.startswith("features")
            or k.startswith("module.")
            for k in obj.keys()
        ):
            return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def _load_weights_if_present(model, path, device):
    if path is None:
        print("Warning: weights file not found. Using random init.")
        return False
    if os.path.exists(path):
        try:
            obj = torch.load(path, map_location=device)
            state = _extract_state_dict(obj)
            state = _normalize_state_dict_keys(state)
            model.load_state_dict(state, strict=True)
            print(f"Loaded weights: {path}")
            return True
        except Exception as e:
            print(
                f"Warning: failed to load weights from {path}: {e}. Using random init."
            )
            return False
    else:
        print(f"Warning: weights not found at {path}. Using random init.")
        return False


squeezenet_custom_4 = squeezenet_custom_4.to(device).eval()
squeezenet_custom_2 = squeezenet_custom_2.to(device).eval()

_loaded_1 = _load_weights_if_present(squeezenet_custom_4, model_1_path, device)
_loaded_2 = _load_weights_if_present(squeezenet_custom_2, model_2_path, device)




## === cell 6
def prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2, thresh_3=0.65):
    preds_minority = softmax(squeezenet_custom_4(img).cpu().detach().numpy(), axis=1)
    preds_binary = softmax(squeezenet_custom_2(img).cpu().detach().numpy(), axis=1)

    binary_cls = int(np.argmax(preds_binary, axis=1)[0])
    minority_cls = int(np.argmax(preds_minority, axis=1)[0])

    cls_3 = float(preds_binary[0][1])

    if cls_3 < thresh_3:
        return minority_idx.get(minority_cls, 0)
    else:
        return binary_idx.get(binary_cls, 0)




## === cell 7
def img_transform(img_path):
    img = Image.open(img_path).convert("RGB")
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))

    img = leaf_transform(img).float()
    img = img.unsqueeze(0)
    return img




## === cell 8
train_dir = os.path.join(proj_dir, "train_images")
test_dir = os.path.join(proj_dir, "test_images")

df = pd.read_csv(os.path.join(proj_dir, "train.csv"))
sample_df = pd.read_csv(os.path.join(proj_dir, "sample_submission.csv"))

print("train.csv shape:", df.shape)
print("sample_submission.csv shape:", sample_df.shape)
print("test_dir exists:", os.path.isdir(test_dir))
print("Using weights:", {"minority": model_1_path, "binary": model_2_path})
print("Loaded flags:", {"minority_loaded": _loaded_1, "binary_loaded": _loaded_2})



## === cell 9
test_image_ids = sample_df["image_id"].tolist()

test_preds = []
missing = 0

with torch.no_grad():
    for i, image_id in enumerate(test_image_ids):
        if i % 200 == 0:
            print("Predicting", i, "/", len(test_image_ids))
        img_path = os.path.join(test_dir, image_id)
        if not os.path.isfile(img_path):
            found = None
            for root, _, files in os.walk(test_dir):
                if image_id in files:
                    found = os.path.join(root, image_id)
                    break
            if found is None:
                missing += 1
                test_preds.append([image_id, 0])
                continue
            img_path = found

        img = img_transform(img_path).to(device)
        final_pred = prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2)
        if isinstance(final_pred, str):
            final_pred = 0
        test_preds.append([image_id, int(final_pred)])

print("Done. Missing files:", missing)
print("Preds:", len(test_preds), "Expected:", len(sample_df))



## === cell 10
pred_df = pd.DataFrame.from_records(test_preds, columns=["image_id", "label"])

pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")
sub = sample_df[["image_id"]].merge(pred_df, on="image_id", how="left")

sub["label"] = sub["label"].fillna(0).astype(int)

print(sub.head())
print("Submission shape:", sub.shape)



## === cell 11
os.chdir("/kaggle/working/")
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv")
print("Columns:", sub.columns.tolist())
print("Unique labels:", sorted(sub["label"].unique().tolist()))
