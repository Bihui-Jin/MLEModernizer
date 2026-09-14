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

0.8732245391356905

# 6. Current score

0.10389

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.10389) has done: 'I fix the immediate runtime blocker by making the model-loading robust to missing Kaggle input folders: it search for the expected `.pth` under `/kaggle/input`, and if none is found it fall back to a torchvision pretrained model with the same architecture so inference can proceed. I also guard against the downstream `NameError` by ensuring a model object always exists for the selected branch, and keep the rest of the inference/submission logic unchanged. Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and aligned to `sample_submission.csv`.'

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



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_root = "/kaggle/input/efficientnetv2-large-test"
en_image_size = 480

vit_model_root = "/kaggle/input/vit_l_cassava"
vit_image_size = 518

model_select = "vit"

if model_select == "vit":
    model_image_size = vit_image_size
elif model_select == "en":
    model_image_size = en_image_size
else:
    raise ValueError(f"Unknown model_select={model_select}")




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




## === cell 3
val_transforms = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((model_image_size, model_image_size)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 4
def find_single_pth(root_dir: str) -> str:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Model root directory not found: {root_dir}")
    candidates = sorted(
        glob.glob(os.path.join(root_dir, "**", "*.pth"), recursive=True)
    )
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .pth files found under: {root_dir}")
    candidates = sorted(candidates, key=lambda p: (len(p), p))
    return candidates[0]


def find_single_pth_fallback(preferred_root: str, pattern_hint: str) -> str | None:
    try:
        return find_single_pth(preferred_root)
    except FileNotFoundError:
        pass
    search_root = "/kaggle/input"
    if os.path.isdir(search_root):
        cands = sorted(
            glob.glob(os.path.join(search_root, "**", "*.pth"), recursive=True)
        )
        hinted = [p for p in cands if pattern_hint.lower() in p.lower()]
        if len(hinted) > 0:
            hinted = sorted(hinted, key=lambda p: (len(p), p))
            return hinted[0]
        if len(cands) > 0:
            cands = sorted(cands, key=lambda p: (len(p), p))
            return cands[0]
    return None


if model_select == "vit":
    vit_model_path = find_single_pth_fallback(vit_model_root, pattern_hint="vit")
    vit_model = models.vit_h_14(weights=None, image_size=vit_image_size)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )

    if vit_model_path is not None:
        state = torch.load(vit_model_path, map_location="cpu")
        vit_model.load_state_dict(state, strict=True)
        print(f"Loaded ViT weights from: {vit_model_path}")
    else:
        try:
            vit_pre = models.vit_h_14(
                weights=models.ViT_H_14_Weights.DEFAULT, image_size=vit_image_size
            )
            vit_pre.heads.head = torch.nn.Linear(
                vit_pre.heads.head.in_features, num_classes
            )
            vit_model = vit_pre
            print(
                "WARNING: No .pth found. Using torchvision pretrained ViT-H/14 backbone with fresh 5-class head."
            )
        except Exception as e:
            print(
                f"WARNING: Could not load torchvision pretrained ViT weights ({e}). Using random init model."
            )

    vit_model.to(device)
    vit_model.eval()

elif model_select == "en":
    en_model_path = find_single_pth_fallback(en_model_root, pattern_hint="efficientnet")
    en_model = models.efficientnet_v2_l(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )

    if en_model_path is not None:
        state = torch.load(en_model_path, map_location="cpu")
        en_model.load_state_dict(state, strict=True)
        print(f"Loaded EfficientNetV2-L weights from: {en_model_path}")
    else:
        try:
            en_pre = models.efficientnet_v2_l(
                weights=models.EfficientNet_V2_L_Weights.DEFAULT
            )
            en_pre.classifier[1] = torch.nn.Linear(
                en_pre.classifier[1].in_features, num_classes
            )
            en_model = en_pre
            print(
                "WARNING: No .pth found. Using torchvision pretrained EfficientNetV2-L backbone with fresh 5-class head."
            )
        except Exception as e:
            print(
                f"WARNING: Could not load torchvision pretrained EfficientNet weights ({e}). Using random init model."
            )

    en_model.to(device)
    en_model.eval()



## === cell 5
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].tolist()

if not os.path.isdir(test_data_directory):
    raise FileNotFoundError(f"Test directory not found: {test_data_directory}")

predictions = []
image_ids = []

for image_name in tqdm(test_image_ids, desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Missing test image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transformed_image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_select == "vit":
            vit_output = vit_model(transformed_image)
            _, predicted_class = torch.max(vit_output, 1)
        elif model_select == "en":
            en_output = en_model(transformed_image)
            _, predicted_class = torch.max(en_output, 1)
        else:
            raise ValueError(f"Unknown model_select={model_select}")

    predictions.append(int(predicted_class.item()))
    image_ids.append(image_name)



## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

if len(submission_df) != len(sample_df):
    raise ValueError(
        f"Submission row count {len(submission_df)} != sample_submission row count {len(sample_df)}"
    )

submission_df = sample_df[["image_id"]].merge(submission_df, on="image_id", how="left")
if submission_df["label"].isna().any():
    missing = submission_df[submission_df["label"].isna()]["image_id"].head(5).tolist()
    raise ValueError(f"Missing predictions for some image_ids, e.g.: {missing}")
submission_df["label"] = submission_df["label"].astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
