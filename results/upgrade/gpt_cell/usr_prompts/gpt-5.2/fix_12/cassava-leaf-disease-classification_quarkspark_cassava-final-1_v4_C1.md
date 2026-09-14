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

0.8286

# 6. Current score

0.53176

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Diagnosis: Cell 10 iterates over `os.listdir(test_dir)` which includes a nested directory entry (`test_images`) inside the `test_images` folder. When that directory name is joined into `img_path`, `Image.open()` in `img_transform()` receives a directory path and raises `IsADirectoryError`.  
Patch summary: In cell 10, filter the directory listing to include only files (and optionally only common image extensions) before attempting to open them; keep the same prediction logic and output structure. This preserves `test_preds` as a list of `[image_id, final_pred]` pairs for cell 11.  
Updated cells: Only cell 10 is modified.  
Compatibility notes for cell k+1: `test_preds` remains defined and has the same structure, so `pd.DataFrame.from_records(test_preds, columns=['image_id','label'])` in cell 11 works unchanged.  
Assumptions: Test images are regular files located directly under `test_dir`, and any nested directories should be ignored.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.8286), so we should make the smallest changes that legitimately improve accuracy without changing the model architecture or training. The biggest likely issue is that `num_classes` is set to 4 (wrong for this competition, which has 5 classes), and the custom classifier uses `padding=(1,1)` with a 1x1 kernel (shape mismatch vs the saved weights and/or incorrect logits). I (1) fix the classifier conv padding to `(0,0)` (standard for 1x1 conv) and (2) set the 4-way model to output 5 classes (and update the mapping dict) while keeping your two-stage “binary + multiclass” inference logic unchanged. I also make test prediction ordering match `sample_submission.csv` to avoid any accidental row misalignment in submission scoring.'
- What this solution (achieved 0.10762) has done: 'Your score is far below the target, and the most likely reason (given your code) is that the “else” branch in `prediction_logic()` returns a string class (`"0_1_2_4"` or `3`) instead of an integer label 0–4, which breaks predictions and effectively collapses accuracy. I make the smallest change to keep your two-model inference exactly the same, but ensure the binary model’s output maps to valid Kaggle labels (return `3` if predicted class-3, otherwise fall back to the 5-class model’s argmax). I also run inference under `torch.no_grad()` to avoid any accidental autograd overhead and keep behavior deterministic, without changing architecture or training. The submission ordering already follows `sample_submission.csv`, so we keep that intact.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the smallest high-impact fix is to ensure inference runs on GPU (if available) and that both the models and input tensors are on the same device; right now everything is on CPU, which often leads people to shorten/skip inference or silently run mismatched tensors in other variants. I keep your exact two-model + threshold logic unchanged, but (1) move models to `cuda` when present, (2) move the transformed image tensor to the same device, and (3) set `torch.inference_mode()` for safer/faster deterministic inference without changing outputs. I also add a tiny safety check to fail loudly if weights weren’t found (instead of producing near-random predictions), because missing weights would explain the very low accuracy and doesn’t change core modeling logic.'
- What this solution (achieved 0.06614) has done: 'Diagnosis: Cell 6 crashes because it hard-requires external weight files (`minority_weights.pth`, `binary_weights.pth`) that are not present anywhere in the provided filesystem, so `_find_weights_dir(...)` returns `None` and a `FileNotFoundError` is raised. This prevents the notebook from running at all even though the rest of the pipeline (model definitions and prediction logic) can execute with randomly initialized weights. The root cause is an environment/path mismatch: `../input/cassava-models` and the other candidate roots do not exist in this runtime.

Patch summary: Modify only cell 6 to stop raising when weights are missing; instead, keep the already-constructed models with their default initialization, move them to `device`, and set them to `eval()`. If weights are found, keep the exact same loading behavior as before (including the strict/nostrict fallback), so semantics are unchanged when files exist.

Updated cells: Only cell 6 is updated below.

Compatibility notes for cell k+1: Cell 7 expects `squeezenet_custom_5` and `squeezenet_custom_2` to exist, be on the correct device, and be in eval mode; this patch guarantees that regardless of whether weights are found.

Assumptions: It is acceptable to run without weights to unblock execution (even if resulting accuracy is low), because the immediate goal is to fix the crash without redesigning the solution or adding new modeling logic.'
- What this solution (achieved 0.18274) has done: 'Your score is extremely low because the custom models are running with random weights (the expected `.pth` files aren’t present), so the smallest legitimate way to move toward the target is to stop relying on missing external weights. I keep your exact SqueezeNet-based architecture and single-pass inference logic, but switch to using an ImageNet-pretrained SqueezeNet backbone (still 5-class and 2-class heads) so predictions are no longer random in this environment. I also make the “binary head” a deterministic heuristic derived from the 5-class logits (still preserving your two-stage gating semantics) so the gating is meaningful without needing a second trained weight file. Submission ordering remains tied to `sample_submission.csv` exactly as before.'
- What this solution (achieved 0.41405) has done: 'Your score is far below the target, so we should make a small, legitimate accuracy improvement without changing your model architecture or training loop. The biggest likely issue left is preprocessing: `img_transform()` swaps RGB→BGR (via channel swap), which mismatches the ImageNet normalization you’re using and can severely harm a pretrained backbone’s predictions. I remove that channel swap so the pretrained SqueezeNet sees standard RGB with the intended ImageNet mean/std, keeping everything else (models, gating logic, threshold, and submission alignment) identical. This should move accuracy upward toward the target while staying minimal and safe.'
- What this solution (achieved 0.08931) has done: 'Your current score (0.41405) is far below the target (0.8286), so we should make the smallest change that legitimately improves accuracy without changing your model architecture or inference logic. The most likely remaining issue is that, when external weights are missing, you intended to rely on an ImageNet-pretrained backbone—but in your current code, the “pretrained” flag is only applied in cell 6 (fallback) while cell 2 initially creates non-pretrained models; additionally, the ImageNet-pretrained path is only used when weights are missing, so we keep that behavior but make preprocessing match SqueezeNet’s expected input exactly. Concretely, we replace the custom CenterCrop/Resize pipeline with the official `SqueezeNet1_0_Weights.IMAGENET1K_V1.transforms()` when running without external weights (same semantic: produce a normalized 224x224 tensor), while keeping your original transform when external weights exist. This is a minimal, metric-aligned preprocessing fix that should move accuracy upward toward the target band while preserving the two-model gating logic and submission alignment.'
- What this solution (achieved 0.53176) has done: 'Your score is far below the target, so we should make a small, legitimate accuracy improvement without changing your model or inference logic. The biggest likely issue now is that `softmax()` is being applied to a 4D logits tensor (N,C,1,1), and the current implementation picks `axis=1` (the singleton spatial dim) instead of the class dim, producing near-uniform probabilities and effectively random argmaxes. I minimally fix `softmax()` so it always normalizes over the last dimension after flattening per-sample logits to shape (N, C), preserving the same two-model gating semantics and threshold. Everything else (models, transforms, submission alignment) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys, cv2
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os, time, cv2, sys
from PIL import Image
import torchvision.transforms as transforms

sz = 224
num_classes = 5
proj_dir = "/kaggle/input/cassava-leaf-disease-classification/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 1
def softmax(X, theta=1.0, axis=None):
    y = np.asarray(X)

    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    elif y.ndim == 1:
        y = y.reshape(1, -1)

    axis = 1 if axis is None else axis

    y = y * float(theta)
    y = y - np.max(y, axis=axis, keepdims=True)
    y = np.exp(y)
    y = y / np.sum(y, axis=axis, keepdims=True)
    return y




## === cell 2
def light_model(num_classes, pretrained_backbone=False):
    if pretrained_backbone:
        try:
            from torchvision.models import SqueezeNet1_0_Weights

            squeezenet_custom = torchvision.models.squeezenet1_0(
                weights=SqueezeNet1_0_Weights.IMAGENET1K_V1
            )
        except Exception:
            squeezenet_custom = torchvision.models.squeezenet1_0(pretrained=True)
    else:
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


squeezenet_custom_5 = light_model(5, pretrained_backbone=False)
squeezenet_custom_2 = light_model(2, pretrained_backbone=False)



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
minority_idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

binary_idx = {0: None, 1: 3}



## === cell 5
weight_dir = "../input/cassava-models"
model_1_path = os.path.join(weight_dir, "minority_weights.pth")
model_2_path = os.path.join(weight_dir, "binary_weights.pth")



## === cell 6
import os


def _find_weights_dir(model_files, candidate_roots):
    for root in candidate_roots:
        if not root or not os.path.isdir(root):
            continue
        if all(os.path.isfile(os.path.join(root, f)) for f in model_files):
            return root
        try:
            for name in os.listdir(root):
                d = os.path.join(root, name)
                if os.path.isdir(d) and all(
                    os.path.isfile(os.path.join(d, f)) for f in model_files
                ):
                    return d
        except Exception:
            pass
    return None


model_files = ["minority_weights.pth", "binary_weights.pth"]
candidate_roots = [
    weight_dir,  # from cell 6
    "/kaggle/input/cassava-models",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/input",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
]

weights_dir = _find_weights_dir(model_files, candidate_roots)

if weights_dir is None:
    squeezenet_custom_5 = light_model(5, pretrained_backbone=True).to(device).eval()
    squeezenet_custom_2 = light_model(2, pretrained_backbone=True).to(device).eval()
    HAVE_EXTERNAL_WEIGHTS = False
else:
    model_1_path = os.path.join(weights_dir, "minority_weights.pth")
    model_2_path = os.path.join(weights_dir, "binary_weights.pth")

    model_1 = torch.load(
        model_1_path, map_location=torch.device("cpu"), weights_only=False
    )
    weights_1 = model_1.state_dict() if hasattr(model_1, "state_dict") else model_1

    try:
        squeezenet_custom_5.load_state_dict(weights_1, strict=True)
    except Exception:
        squeezenet_custom_5.load_state_dict(weights_1, strict=False)

    model_2 = torch.load(
        model_2_path, map_location=torch.device("cpu"), weights_only=False
    )
    weights_2 = model_2.state_dict() if hasattr(model_2, "state_dict") else model_2
    try:
        squeezenet_custom_2.load_state_dict(weights_2, strict=True)
    except Exception:
        squeezenet_custom_2.load_state_dict(weights_2, strict=False)

    squeezenet_custom_5.to(device).eval()
    squeezenet_custom_2.to(device).eval()
    HAVE_EXTERNAL_WEIGHTS = True

print("Found external weights:", HAVE_EXTERNAL_WEIGHTS)

if not HAVE_EXTERNAL_WEIGHTS:
    try:
        from torchvision.models import SqueezeNet1_0_Weights

        leaf_transform = SqueezeNet1_0_Weights.IMAGENET1K_V1.transforms()
        print("Using SqueezeNet1_0_Weights IMAGENET1K_V1 transforms for inference.")
    except Exception:
        print(
            "Could not load official weights transforms; using existing leaf_transform."
        )




## === cell 7
def prediction_logic(img, squeezenet_custom_5, squeezenet_custom_2, thresh_3=0.65):
    preds_multi = softmax(squeezenet_custom_5(img).cpu().detach().numpy())
    multi_cls = int(np.argmax(preds_multi))

    if HAVE_EXTERNAL_WEIGHTS:
        preds_binary = softmax(squeezenet_custom_2(img).cpu().detach().numpy())
        binary_cls = int(np.argmax(preds_binary))
        cls_3_prob = float(preds_binary[0][1])
        if cls_3_prob < thresh_3:
            return int(minority_idx[multi_cls])
        else:
            mapped = binary_idx[binary_cls]
            return int(mapped) if mapped is not None else int(minority_idx[multi_cls])
    else:
        cls_3_prob = float(preds_multi[0][3])
        if cls_3_prob >= thresh_3:
            return 3
        return int(minority_idx[multi_cls])




## === cell 8
def img_transform(img_path):
    img = Image.open(img_path).convert("RGB")

    img = leaf_transform(img).float()
    img = img.unsqueeze(0)
    img = img.to(device)
    return img




## === cell 9
train_dir = os.path.join(proj_dir, "train_images")
test_dir = os.path.join(proj_dir, "test_images")

df = pd.read_csv(os.path.join(proj_dir, "train.csv"))
sample_df = pd.read_csv(os.path.join(proj_dir, "sample_submission.csv"))
sample_df.head()



## === cell 10
test_preds = []

with torch.inference_mode():
    for i, j in enumerate(sample_df["image_id"].tolist()):
        img_path = os.path.join(test_dir, j)
        if not os.path.isfile(img_path):
            continue
        if i < 5:
            print(i, j)
        img = img_transform(img_path)
        final_pred = prediction_logic(img, squeezenet_custom_5, squeezenet_custom_2)
        test_preds.append([j, int(final_pred)])

print("num_preds:", len(test_preds))



## === cell 11
sub = pd.DataFrame.from_records(test_preds, columns=["image_id", "label"])

sub = sample_df[["image_id"]].merge(sub, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)
sub.head()



## === cell 12
os.chdir("/kaggle/working/")
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote:", "/kaggle/working/submission.csv", "rows:", len(sub))
