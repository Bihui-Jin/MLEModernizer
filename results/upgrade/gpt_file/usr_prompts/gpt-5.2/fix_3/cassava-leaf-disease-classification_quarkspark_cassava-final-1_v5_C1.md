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

0.14873

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix three execution blockers without changing the core model/prediction logic: (1) make weight loading robust by searching the actual Kaggle input directory and loading state_dicts correctly on CPU/GPU, (2) iterate over the test images safely by filtering only `.jpg/.png` files and ignoring the nested `test_images/` directory that causes `IsADirectoryError`, and (3) ensure the submission rows exactly match `sample_submission.csv` (same length and order) to prevent the “same length as the answers” error. These changes are score-neutral aside from making inference actually run and align with the required submission format. The rest of the architecture and the two-stage threshold logic is preserved.'
- What this solution (achieved 0.14873) has done: 'I make the pipeline robust when the external weight files are missing by falling back to the same SqueezeNet architecture with ImageNet pretrained weights and a single 5-class head (no change to inference loop structure or transforms, just a safer model init). I also fix the CUDA/CPU dtype mismatch by ensuring both models and input tensors are on the same device with the same dtype, which eliminates the runtime error. Finally, I keep the submission aligned exactly to `sample_submission.csv` and always write `/kaggle/working/submission.csv` so Kaggle accepts it. These changes should substantially increase accuracy versus outputting mostly default labels, moving the score toward the target band.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys, cv2
import torch
import torchvision
import torch.nn as nn
import time
from PIL import Image
import torchvision.transforms as transforms

sz = 224
num_classes = 4
proj_dir = "/kaggle/input/cassava-leaf-disease-classification/"

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




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


squeezenet_custom_4 = light_model(4)
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
    3: 4,
}

binary_idx = {
    0: "0_1_2_4",
    1: 3,
}




## === cell 5
def find_weight_file(filename, search_roots):
    for root in search_roots:
        candidate = os.path.join(root, filename)
        if os.path.isfile(candidate):
            return candidate
    for root in search_roots:
        for dirpath, _, files in os.walk(root):
            if filename in files:
                return os.path.join(dirpath, filename)
    return None


def load_weights_into(model, weight_path, device):
    obj = torch.load(weight_path, map_location=device)
    if isinstance(obj, nn.Module):
        state = obj.state_dict()
    elif (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise TypeError(f"Unrecognized checkpoint type: {type(obj)}")

    if len(state) > 0:
        first_key = next(iter(state.keys()))
        if first_key.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"[WARN] Missing keys when loading {os.path.basename(weight_path)}: {missing[:5]}{'...' if len(missing)>5 else ''}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading {os.path.basename(weight_path)}: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
        )
    model.to(device)
    model.eval()
    return model


def make_squeezenet5_imagenet(device):
    weights = torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1
    model = torchvision.models.squeezenet1_0(weights=weights)
    model.classifier[1] = nn.Conv2d(
        in_channels=512, out_channels=5, kernel_size=1, stride=1, padding=0
    )
    model.to(device)
    model.eval()
    return model


search_roots = [
    "/kaggle/input",
    proj_dir,
]

model_1_path = find_weight_file("minority_weights.pth", search_roots)
model_2_path = find_weight_file("binary_weights.pth", search_roots)

print("model_1_path:", model_1_path)
print("model_2_path:", model_2_path)

use_two_stage = (model_1_path is not None) and (model_2_path is not None)

if use_two_stage:
    squeezenet_custom_4 = load_weights_into(squeezenet_custom_4, model_1_path, device)
    squeezenet_custom_2 = load_weights_into(squeezenet_custom_2, model_2_path, device)
    model_5 = None
    print("Using provided two-stage weights.")
else:
    model_5 = make_squeezenet5_imagenet(device)
    print(
        "[WARN] Two-stage weights not found; using 5-class ImageNet-pretrained SqueezeNet fallback."
    )




## === cell 6
def prediction_logic(
    img, squeezenet_custom_4, squeezenet_custom_2, thresh_3=0.65, model_5=None
):
    with torch.no_grad():
        if model_5 is not None:
            logits = model_5(img)
            pred = int(torch.argmax(logits, dim=1).item())
            return pred

        preds_minority = softmax(
            squeezenet_custom_4(img).detach().float().cpu().numpy()
        )
        preds_binary = softmax(squeezenet_custom_2(img).detach().float().cpu().numpy())
    binary_cls = np.argmax(preds_binary)
    minority_cls = np.argmax(preds_minority)

    cls_3 = preds_binary[0][1]

    if cls_3 < thresh_3:
        return minority_idx[minority_cls]
    else:
        return binary_idx[binary_cls]




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

print("train:", df.shape, "sample:", sample_df.shape)
sample_df.head()



## === cell 9
test_preds = []
missing_files = 0

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

for i, image_id in enumerate(sample_df["image_id"].tolist()):
    img_path = os.path.join(test_dir, image_id)
    if not (os.path.isfile(img_path) and image_id.lower().endswith(valid_ext)):
        missing_files += 1
        test_preds.append([image_id, 0])
        continue

    if i < 5:
        print("Predicting", i, image_id)

    img = img_transform(img_path).to(device=device, dtype=torch.float32)
    final_pred = prediction_logic(
        img,
        squeezenet_custom_4,
        squeezenet_custom_2,
        model_5=model_5,
    )
    if isinstance(final_pred, str):
        final_pred = 0
    test_preds.append([image_id, int(final_pred)])

print("done. missing_files:", missing_files, "num_preds:", len(test_preds))



## === cell 10
sub = pd.DataFrame.from_records(test_preds, columns=["image_id", "label"])

sub = sample_df[["image_id"]].merge(sub, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)

print("sub shape:", sub.shape)
sub.head()



## === cell 11
os.chdir("/kaggle/working/")
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub["label"].value_counts(dropna=False).sort_index())
