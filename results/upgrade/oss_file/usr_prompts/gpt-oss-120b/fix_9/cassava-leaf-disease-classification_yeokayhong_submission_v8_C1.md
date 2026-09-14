# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import json
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageOps
from tqdm.auto import tqdm

possible_base_dirs = [
    "data/cassava-leaf-disease-classification",
    "input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
]
base_dir = None
for p in possible_base_dirs:
    if os.path.isdir(p):
        base_dir = p
        break
if base_dir is None:
    raise FileNotFoundError(
        "Could not locate the cassava-leaf-disease-classification dataset. "
        "Checked paths: " + ", ".join(possible_base_dirs)
    )


def _find_checkpoint(pattern):
    """Return the first file matching *pattern* under base_dir, else empty string."""
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".pth") and pattern in f.lower():
                return os.path.join(root, f)
    return ""


model_select = "vit"  # choose "vit" or "en"
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_submission_path = os.path.join(base_dir, "sample_submission.csv")
test_data_directory = os.path.join(base_dir, "test_images")

vit_model_path = _find_checkpoint("vit")  # e.g., .../vit_checkpoint.pth
en_model_path = _find_checkpoint("en")  # e.g., .../en_checkpoint.pth

val_transforms = T.Compose(
    [
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 1
def _get_weight(enum_class, preferred_name):
    """Return the enum member if it exists, else None."""
    try:
        return getattr(enum_class, preferred_name)
    except AttributeError:
        return None


if model_select == "vit":
    vit_weights_enum = getattr(models, "ViT_H_14_Weights", None)
    vit_pretrained_weight = None
    if vit_weights_enum is not None:
        vit_pretrained_weight = _get_weight(vit_weights_enum, "IMAGENET1K_V1")
        if vit_pretrained_weight is None:
            vit_pretrained_weight = _get_weight(vit_weights_enum, "DEFAULT")
    vit_model = models.vit_h_14(weights=None, image_size=518)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )
    if vit_model_path and os.path.exists(vit_model_path):
        vit_model.load_state_dict(
            torch.load(vit_model_path, map_location=device, weights_only=True)
        )
    else:
        print("Warning: ViT checkpoint not found – using ImageNet pretrained weights.")
        if vit_pretrained_weight is not None:
            vit_model = models.vit_h_14(weights=vit_pretrained_weight, image_size=518)
            vit_model.heads.head = torch.nn.Linear(
                vit_model.heads.head.in_features, num_classes
            )
        else:
            print("No pretrained weight enum available; proceeding with random init.")
    vit_model = vit_model.to(device)
    vit_model.eval()
elif model_select == "en":
    en_weights_enum = getattr(models, "EfficientNet_V2_L_Weights", None)
    en_pretrained_weight = None
    if en_weights_enum is not None:
        en_pretrained_weight = _get_weight(en_weights_enum, "IMAGENET1K_V1")
        if en_pretrained_weight is None:
            en_pretrained_weight = _get_weight(en_weights_enum, "DEFAULT")
    en_model = models.efficientnet_v2_l(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
    if en_model_path and os.path.exists(en_model_path):
        en_model.load_state_dict(
            torch.load(en_model_path, map_location=device, weights_only=True)
        )
    else:
        print(
            "Warning: EfficientNet checkpoint not found – using ImageNet pretrained weights."
        )
        if en_pretrained_weight is not None:
            en_model = models.efficientnet_v2_l(weights=en_pretrained_weight)
            en_model.classifier[1] = torch.nn.Linear(
                en_model.classifier[1].in_features, num_classes
            )
        else:
            print("No pretrained weight enum available; proceeding with random init.")
    en_model = en_model.to(device)
    en_model.eval()
else:
    raise ValueError(f"Unsupported model_select: {model_select}")



## === cell 2
sample_df = pd.read_csv(sample_submission_path)
image_ids = sample_df["image_id"].tolist()
predictions = []

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

for image_name in tqdm(image_ids, desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)
    if (
        not os.path.isfile(image_path)
        or os.path.splitext(image_name)[1].lower() not in valid_extensions
    ):
        predictions.append(0)
        continue

    image = Image.open(image_path).convert("RGB")
    image_flipped = ImageOps.mirror(image)

    tensor_orig = val_transforms(image).unsqueeze(0).to(device)
    tensor_flip = val_transforms(image_flipped).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_select == "vit":
            logits_orig = vit_model(tensor_orig)
            logits_flip = vit_model(tensor_flip)
        elif model_select == "en":
            logits_orig = en_model(tensor_orig)
            logits_flip = en_model(tensor_flip)
        else:
            raise ValueError(f"Unsupported model_select: {model_select}")

        avg_logits = (logits_orig + logits_flip) / 2.0
        _, predicted_class = torch.max(avg_logits, 1)
        predictions.append(predicted_class.item())



## === cell 3
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
