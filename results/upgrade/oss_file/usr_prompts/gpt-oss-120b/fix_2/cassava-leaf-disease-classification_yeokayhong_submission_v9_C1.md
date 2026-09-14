# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8301601692354186

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms, v2
from torchvision.transforms import functional as F



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2341888816.py in <cell line: 0>()
      4 from tqdm import tqdm
      5 from PIL import Image
----> 6 from torchvision import models, transforms, v2
      7 from torchvision.transforms import functional as F
      8 

ImportError: cannot import name 'v2' from 'torchvision' (/usr/local/lib/python3.11/dist-packages/torchvision/__init__.py)

## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_data_directory):
    test_data_directory = os.path.join(os.getcwd(), "test_images")
    if not os.path.isdir(test_data_directory):
        raise FileNotFoundError(
            f"Test images directory not found at '{test_data_directory}'"
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/3/efficientnetv2_l_480_8450.pth"
vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/3/vit_h_14_518_8927.pth"

en_image_size = 480
vit_image_size = 518

model_select = "en"  # choose "en" or "vit"

model_image_size = en_image_size if model_select == "en" else vit_image_size




## === cell 2
def invert_square_pad(img: Image.Image) -> Image.Image:
    """Invert quadrants and pad to a square using reflection."""
    width, height = img.size
    cx, cy = width // 2, height // 2

    top_left = img.crop((0, 0, cx, cy))
    top_right = img.crop((cx, 0, width, cy))
    bottom_left = img.crop((0, cy, cx, height))
    bottom_right = img.crop((cx, cy, width, height))

    top_combined = Image.new("RGB", (width, cy))
    top_combined.paste(bottom_right, (0, 0))
    top_combined.paste(bottom_left, (cx, 0))

    bottom_combined = Image.new("RGB", (width, cy))
    bottom_combined.paste(top_right, (0, 0))
    bottom_combined.paste(top_left, (cx, 0))

    flipped = Image.new("RGB", (width, height))
    flipped.paste(top_combined, (0, 0))
    flipped.paste(bottom_combined, (0, cy))

    max_side = max(width, height)
    pad = (
        (max_side - width) // 2,
        (max_side - height) // 2,
        (max_side - width) - (max_side - width) // 2,
        (max_side - height) - (max_side - height) // 2,
    )
    padded = F.pad(flipped, pad, padding_mode="reflect")
    return padded




## === cell 3
val_transforms = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((model_image_size, model_image_size)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1178956088.py in <cell line: 0>()
      1 val_transforms = transforms.Compose(
      2     [
----> 3         v2.Lambda(invert_square_pad),
      4         v2.ToImage(),
      5         v2.ToDtype(torch.float32, scale=True),

NameError: name 'v2' is not defined

## === cell 4
if model_select == "vit":
    try:
        vit_model = models.vit_h_14(weights=None, image_size=518)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        vit_model.load_state_dict(
            torch.load(vit_model_path, map_location=device, weights_only=True)
        )
    except Exception as e:
        print(
            f"Custom ViT checkpoint not loaded ({e}); using torchvision pretrained weights."
        )
        vit_model = models.vit_h_14(
            weights=models.ViT_H_14_Weights.IMAGENET1K_V1, image_size=518
        )
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
    vit_model.to(device)
    vit_model.eval()

if model_select == "en":
    try:
        en_model = models.efficientnet_v2_l(weights=None)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
        en_model.load_state_dict(
            torch.load(en_model_path, map_location=device, weights_only=True)
        )
    except Exception as e:
        print(
            f"Custom EfficientNet checkpoint not loaded ({e}); using torchvision pretrained weights."
        )
        en_model = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        )
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
    en_model.to(device)
    en_model.eval()



## === cell 5
predictions = []
image_ids = []

for image_name in tqdm(sorted(os.listdir(test_data_directory)), desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)
    if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue

    image = Image.open(image_path).convert("RGB")
    transformed = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_select == "vit":
            output = vit_model(transformed)
        else:  # "en"
            output = en_model(transformed)
        _, pred = torch.max(output, dim=1)

    predictions.append(int(pred.item()))
    image_ids.append(image_name)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/536867318.py in <cell line: 0>()
      9 
     10     image = Image.open(image_path).convert("RGB")
---> 11     transformed = val_transforms(image).unsqueeze(0).to(device)
     12 
     13     with torch.no_grad():

NameError: name 'val_transforms' is not defined

## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

output_path = os.path.join(os.getcwd(), "submission.csv")
submission_df.to_csv(output_path, index=False)
print(f"Submission file created: {output_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
