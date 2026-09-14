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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
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

0.8942278634028408

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
from collections import Counter

possible_model_dirs = [
    "../input/eb7slseed70/efficientnet-b7sl_SEED70.best",
    "/kaggle/input/eb7slseed70/efficientnet-b7sl_SEED70.best",
    "./input/eb7slseed70/efficientnet-b7sl_SEED70.best",
]
pretrained_models = []
for d in possible_model_dirs:
    if os.path.isdir(d):
        pretrained_models.extend(glob.glob(os.path.join(d, "*.pth")))
pretrained_models = sorted(pretrained_models)

print(f"{len(pretrained_models)} pretrained model files found.")
if pretrained_models:
    print("\n".join(pretrained_models))
else:
    print("No pretrained checkpoints detected – will use a simple baseline.")




## === cell 1
possible_dirs = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "data/cassava-leaf-disease-classification",
    "input/cassava-leaf-disease-classification",
]
BASE_DIR = None
for d in possible_dirs:
    if os.path.isdir(d):
        BASE_DIR = d
        break

if BASE_DIR is None:
    sample_path = "sample_submission.csv"
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(
            "sample_submission.csv not found in the current working directory."
        )
    df_sample = pd.read_csv(sample_path)
    test_files = df_sample["image_id"].tolist()
else:
    TEST_PATH = os.path.join(BASE_DIR, "test_images")
    if os.path.isdir(os.path.join(TEST_PATH, "test_images")):
        TEST_PATH = os.path.join(TEST_PATH, "test_images")

    if os.path.isdir(TEST_PATH):
        test_files = sorted(os.listdir(TEST_PATH))
        test_files = [
            f for f in test_files if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    else:
        sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
        df_sample = pd.read_csv(sample_path)
        test_files = df_sample["image_id"].tolist()

print(f"Using base directory: {BASE_DIR if BASE_DIR else 'fallback (CSV)'}")
print(f"Number of test images/IDs detected: {len(test_files)}")

train_csv_path = os.path.join(BASE_DIR, "train.csv") if BASE_DIR else "train.csv"
if not os.path.isfile(train_csv_path):
    raise FileNotFoundError(
        f"train.csv not found at expected location: {train_csv_path}"
    )

df_train = pd.read_csv(train_csv_path)
most_common_label = Counter(df_train["label"]).most_common(1)[0][0]
print(
    f"Baseline prediction will use the most common training label: {most_common_label}"
)




## === cell 2
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image


def load_pretrained_model(checkpoint_paths):
    if not checkpoint_paths:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoint_paths[-1]
    try:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
        model = torchvision.models.efficientnet_b7(num_classes=5)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Loaded pretrained model from {ckpt_path}")
        return model, device
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return None


model_device_pair = load_pretrained_model(pretrained_models)
if model_device_pair is not None:
    model, device = model_device_pair
    preprocess = T.Compose(
        [
            T.Resize(600),
            T.CenterCrop(600),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    predictions = []
    for img_name in test_files:
        img_path = os.path.join(TEST_PATH if BASE_DIR else "", img_name)
        if not os.path.isfile(img_path):
            img_path = img_name  # assume image file is in cwd
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    pred_label = logits.argmax(dim=1).item()
        except Exception as e:
            print(f"Error processing {img_name}: {e}; using baseline label.")
            pred_label = most_common_label
        predictions.append(pred_label)

    df_test = pd.DataFrame({"image_id": test_files, "label": predictions})
else:
    df_test = pd.DataFrame({"image_id": test_files, "label": most_common_label})

output_path = "submission.csv"
df_test.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with {len(df_test)} rows.")
