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

0.8325778180719251

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the two execution blockers that prevent a valid submission: the missing model weight paths (by locating available `.pth` files or falling back safely without crashing) and the test image iteration error caused by nested `test_images/` directories (by filtering only actual image files and using the sample submission order). I also ensure inference runs on the right device consistently and that the model output tensor shape is handled correctly for softmax/argmax. Finally, I build the submission by merging predictions onto `sample_submission.csv` so the row count and ordering exactly match Kaggle’s expected answers length, producing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, so we should improve accuracy with the smallest changes that don’t alter your model architectures or training logic. The biggest likely accuracy bug is that you’re swapping RGB↔BGR in `img_transform`, which mismatches the ImageNet normalization expected by the torchvision models and heavily degrade predictions; removing that channel swap should materially raise the score. Additionally, your binary head returns strings (`"0_1_2_4"` / `3`) and relies on a threshold; fixing `binary_idx` to return proper integer labels (and keeping the same thresholding logic) avoids unintended casting/edge behavior and stabilizes the gating decision. These two minimal fixes keep your pipeline intact while moving the score upward toward the target.'
- What this solution (achieved 0.61099) has done: 'Your score gap to the target is large (0.61099 → 0.8326), so we should apply a minimal, high-impact fix that preserves your two-model gating logic and architectures. The most likely accuracy killer here is a mismatch between your custom SqueezeNet classifier head and the saved weights: your `Conv2d` uses `padding=(1,1)` with `kernel_size=1`, which changes tensor shapes and typically prevents correct weight loading (silently, because you use `strict=False`)—so you end up with partially random heads. I change that padding to `(0,0)` (the standard for 1×1 conv) so the checkpoint keys/shapes match and weights load correctly, and I also make the loader explicitly verify that the classifier conv weights actually loaded (otherwise fail loudly so you don’t unknowingly submit random predictions). Everything else (transforms, softmax/argmax, threshold gating, submission alignment) remains the same.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.83258), so we should apply a small, high-impact change that doesn’t alter your architecture or training logic. The most likely issue is that both models are being run in full FP32 on GPU without automatic mixed precision, which can slightly change numerical behavior and sometimes worsen calibration for a threshold-gated ensemble; we keep FP32 but fix a bigger accuracy lever: ensure the checkpoint weights are loaded from the *correct* files by tightening the search to prefer the exact expected filenames when present and only then fall back to substring matches. Additionally, we align preprocessing more closely to ImageNet inference by adding `transforms.InterpolationMode.BICUBIC` for resize (common for torchvision ImageNet models) while keeping the same crop/resize pipeline. These are minimal changes that should improve accuracy materially without changing your core two-model gating logic, and the script still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, time
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

sz = 224
num_classes = 4
proj_dir = "/kaggle/input/cassava-leaf-disease-classification/"

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
            padding=(0, 0),
        ),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )

    squeezenet_custom.classifier = classifier
    return squeezenet_custom


squeezenet_custom_4 = light_model(4).to(device)
squeezenet_custom_2 = light_model(2).to(device)



## === cell 3
leaf_transform = transforms.Compose(
    [
        transforms.CenterCrop(400),
        transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 4
minority_idx = {0: 0, 1: 1, 2: 2, 3: 4}
binary_idx = {0: 0, 1: 3}




## === cell 5
def find_existing_weight_file(
    preferred_path, candidates_substrings, preferred_filenames=()
):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    preferred_filenames = [
        p.lower() for p in preferred_filenames if isinstance(p, str) and len(p) > 0
    ]
    if preferred_filenames:
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                fl = f.lower()
                if fl in preferred_filenames and (
                    fl.endswith(".pth") or fl.endswith(".pt")
                ):
                    return os.path.join(root, f)

    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            fl = f.lower()
            if fl.endswith(".pth") or fl.endswith(".pt"):
                full = os.path.join(root, f)
                for s in candidates_substrings:
                    if s in fl:
                        return full
    return None


weight_dir = "../input/cassava-models"
model_1_path = os.path.join(weight_dir, "minority_weights.pth")
model_2_path = os.path.join(weight_dir, "binary_weights.pth")

model_1_path = find_existing_weight_file(
    model_1_path,
    ["minority", "minor"],
    preferred_filenames=(
        "minority_weights.pth",
        "minority.pth",
        "minority_weights.pt",
        "minority.pt",
    ),
)
model_2_path = find_existing_weight_file(
    model_2_path,
    ["binary", "bin"],
    preferred_filenames=(
        "binary_weights.pth",
        "binary.pth",
        "binary_weights.pt",
        "binary.pt",
    ),
)

model_1_path, model_2_path




## === cell 6
def load_checkpoint_into_model(model, ckpt_path, device, required_tensor_names=()):
    if ckpt_path is None:
        return False

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        return False

    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()

    for tname in required_tensor_names:
        if tname not in new_state:
            raise RuntimeError(
                f"Checkpoint did not contain required tensor '{tname}'. "
                f"Likely wrong weights file selected: {ckpt_path}"
            )
        model_tensor = dict(model.named_parameters()).get(tname, None)
        if model_tensor is None:
            raise RuntimeError(f"Model does not have required parameter '{tname}'.")
        if tuple(model_tensor.shape) != tuple(new_state[tname].shape):
            raise RuntimeError(
                f"Shape mismatch for '{tname}': model {tuple(model_tensor.shape)} vs "
                f"ckpt {tuple(new_state[tname].shape)}. Check classifier definition/padding."
            )

    return True


loaded_1 = load_checkpoint_into_model(
    squeezenet_custom_4,
    model_1_path,
    device,
    required_tensor_names=("classifier.1.weight", "classifier.1.bias"),
)
loaded_2 = load_checkpoint_into_model(
    squeezenet_custom_2,
    model_2_path,
    device,
    required_tensor_names=("classifier.1.weight", "classifier.1.bias"),
)

loaded_1, loaded_2




## === cell 7
def prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2, thresh_3=0.50):
    img = img.to(device)

    with torch.no_grad():
        out4 = squeezenet_custom_4(img)
        out2 = squeezenet_custom_2(img)

    out4 = out4.view(out4.size(0), -1)
    out2 = out2.view(out2.size(0), -1)

    preds_minority = softmax(out4.detach().cpu().numpy())
    preds_binary = softmax(out2.detach().cpu().numpy())
    binary_cls = int(np.argmax(preds_binary, axis=1)[0])
    minority_cls = int(np.argmax(preds_minority, axis=1)[0])

    cls_3 = float(preds_binary[0][1])

    if cls_3 <= thresh_3:
        return int(minority_idx[minority_cls])
    else:
        return int(binary_idx[binary_cls])




## === cell 8
def img_transform(img_path):
    img = Image.open(img_path).convert("RGB")
    img = leaf_transform(img).float()
    img = img.unsqueeze(0)
    return img




## === cell 9
train_dir = os.path.join(proj_dir, "train_images")
test_dir = os.path.join(proj_dir, "test_images")

df = pd.read_csv(os.path.join(proj_dir, "train.csv"))
sample_df = pd.read_csv(os.path.join(proj_dir, "sample_submission.csv"))

test_dir, train_dir, df.shape, sample_df.shape




## === cell 10
def resolve_test_image_path(image_id):
    candidates = [
        os.path.join(test_dir, image_id),
        os.path.join(test_dir, "test_images", image_id),
        os.path.join(
            proj_dir, "cassava-leaf-disease-classification", "test_images", image_id
        ),
        os.path.join(
            proj_dir,
            "cassava-leaf-disease-classification",
            "test_images",
            "test_images",
            image_id,
        ),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


test_preds = {}
missing = 0

for i, image_id in enumerate(sample_df["image_id"].tolist()):
    img_path = resolve_test_image_path(image_id)
    if img_path is None:
        test_preds[image_id] = 0
        missing += 1
        continue

    img = img_transform(img_path)
    final_pred = prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2)
    test_preds[image_id] = final_pred

missing, len(test_preds)



## === cell 11
sub = sample_df.copy()
sub["label"] = sub["image_id"].map(test_preds).astype(int)

assert sub.shape[0] == sample_df.shape[0]
assert list(sub.columns) == ["image_id", "label"]
assert sub["label"].between(0, 4).all()

sub.head()



## === cell 12
os.chdir("/kaggle/working/")
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote:", "/kaggle/working/submission.csv", "rows:", len(sub))
print(sub["label"].value_counts().sort_index())
