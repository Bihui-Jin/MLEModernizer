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

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
timm==1.0.19
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

0.8977032336053188

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58445) has done: 'The fix removes the unavailable `Cutout` transform from the augmentation pipeline, filters the test directory to include only image files (avoiding a nested folder that caused an `IsADirectoryError`), and builds the submission DataFrame directly from the test filenames and predicted labels. These changes enable the script to run end‑to‑end and generate a valid `submission.csv` while preserving the original model logic.'
- What this solution (achieved 0.49439) has done: 'I enhance the ensemble prediction by applying test‑time augmentation to both models, converting raw logits to class probabilities with softmax, and using a modestly adjusted weight (0.4 / 0.6) when averaging the two models’ probabilities. This keeps the original architecture and training untouched while expectedly raising accuracy toward the target.'
- What this solution (achieved 0.50972) has done: 'The inference loop was the main bottleneck: the code opened each test image 12 times per model and performed a forward pass for every augmentation separately, resulting in thousands of costly GPU calls.  
We now load each image only once, generate all TTA augmented tensors, batch them, and run a single forward pass per model per image. This removes repeated disk I/O and reduces the number of model calls by a factor of 12 while keeping the exact augmentation logic and averaging behavior, so the predictions remain unchanged.'
- What this solution (achieved 0.05531) has done: 'The inference loop was doing 16 identical Test‑Time‑Augmentations per image and running both models on two separate copies of the same data, which caused thousands of unnecessary forward passes. Since the augmentation pipeline only normalizes the image, all TTA copies are identical, so we can safely set `TTA = 1` and share the single augmented tensor between the two models. This keeps the exact same predictions while cutting the total number of forward passes by a factor of 32, bringing the runtime well under the 600 s limit.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"
OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

my_model_1 = timm.create_model(
    "resnext50_32x4d", pretrained=True, num_classes=OUT_FEATURES
)
my_model_2 = timm.create_model(
    "tf_efficientnet_b4_ns", pretrained=True, num_classes=OUT_FEATURES
)

resnext_path_full = os.path.join(INPUT_PATH, RESNEXT_PATH)
if not os.path.isfile(resnext_path_full):
    fallback_path = os.path.join(
        "../input/cassava-leaf-disease-classification/", RESNEXT_PATH
    )
    if os.path.isfile(fallback_path):
        resnext_path_full = fallback_path
if os.path.isfile(resnext_path_full):
    model_param = torch.load(resnext_path_full, map_location=device)
    new_model_param = {
        k[7:]: v for k, v in model_param.items() if k.startswith("module.")
    }
    my_model_1.load_state_dict(new_model_param)
else:
    print(f"Warning: {resnext_path_full} not found – using pretrained ResNeXt weights.")

b4_path_full = os.path.join(INPUT_PATH, B4_PATH)
if not os.path.isfile(b4_path_full):
    fallback_path = os.path.join(
        "../input/cassava-leaf-disease-classification/", B4_PATH
    )
    if os.path.isfile(fallback_path):
        b4_path_full = fallback_path
if os.path.isfile(b4_path_full):
    model_param = torch.load(b4_path_full, map_location=device)
    new_model_param = {
        k[7:]: v for k, v in model_param.items() if k.startswith("module.")
    }
    my_model_2.load_state_dict(new_model_param)
else:
    print(
        f"Warning: {b4_path_full} not found – using pretrained EfficientNet‑B4 weights."
    )

my_model_1 = nn.DataParallel(my_model_1).to(device)
my_model_2 = nn.DataParallel(my_model_2).to(device)

test_augs = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()],
    p=1.0,
)



## === cell 1
preds_1 = []
preds_2 = []

my_model_1.eval()
my_model_2.eval()

test_image_list = np.asarray(
    sorted([f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")])
)

for single_image_name in tqdm(test_image_list, desc="Inference"):
    img_path = os.path.join(TEST_IMAGE_PATH, single_image_name)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    aug_imgs = [test_augs(image=img_np)["image"] for _ in range(TTA)]
    batch = torch.stack(aug_imgs).to(device)  # shape (1, 3, H, W)

    with torch.no_grad():
        logits_1 = my_model_1(batch)  # (1, OUT_FEATURES)
        logits_2 = my_model_2(batch)  # (1, OUT_FEATURES)

    ans_1 = logits_1.squeeze(0)  # (OUT_FEATURES)
    ans_2 = logits_2.squeeze(0)

    preds_1.append(ans_1)
    preds_2.append(ans_2)

predictions_1 = torch.stack(preds_1, dim=0).cpu()
predictions_2 = torch.stack(preds_2, dim=0).cpu()

w1, w2 = 0.5, 0.5
prob_1 = F.softmax(predictions_1, dim=1)
prob_2 = F.softmax(predictions_2, dim=1)
final_prob = prob_1 * w1 + prob_2 * w2

label = final_prob.argmax(dim=1).numpy()

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
