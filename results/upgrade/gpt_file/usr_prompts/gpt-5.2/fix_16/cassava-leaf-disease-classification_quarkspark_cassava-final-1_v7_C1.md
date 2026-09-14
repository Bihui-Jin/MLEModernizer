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
- What this solution (achieved 0.61099) has done: 'Your gap to the target is large (0.61099 → 0.83258), so we focus on one minimal, high-impact inference-time fix while preserving your two-model gating and architectures. The main likely score killer is a preprocessing mismatch: you apply a hard `CenterCrop(400)` before resizing, which can crop away disease cues and also fails on smaller images; replacing it with a standard ImageNet-style `Resize(256) + CenterCrop(224)` keeps semantics but typically boosts accuracy. Additionally, we switch your NumPy softmax/argmax to pure Torch `softmax`/`argmax` to avoid any subtle axis/shape issues and keep outputs consistent with the model. Everything else (models, weights loading, threshold gating, and submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.83258), so we should make a small, inference-only change that improves accuracy without touching model architectures or training. The biggest controllable lever in your pipeline is the class-3 gating threshold (`thresh_3=0.50`): if it’s miscalibrated, you route many images to the wrong head and accuracy collapses. I tune this threshold using a small validation split from `train.csv` (no training, just running your existing models) by selecting the threshold that maximizes validation accuracy, then use that single best threshold for test inference. This preserves your two-model gating logic and semantics, only calibrates the decision boundary to move the score upward toward the target.'
- What this solution (achieved 0.61099) has done: 'Your current score is well below the target, so the smallest likely accuracy win is to fix an inference mismatch: your SqueezeNet heads include a `ReLU` in the classifier, but SqueezeNet’s standard classifier does not apply ReLU after the final 1×1 conv; that can severely distort logits/softmax and gating. I remove that `ReLU` (keeping the same model family, training/inference approach, and checkpoint loading) so outputs behave like proper logits, which should move accuracy upward toward the target. I also make the validation threshold tuning slightly more stable by increasing the number of sampled validation images (still fast enough) without changing the tuning method. Everything else (two-model gating logic, transforms, submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your gap to the target is large (0.61099 → 0.83258), so we should apply a minimal, inference-only change that typically improves accuracy without changing model architectures or any training: use test-time augmentation (horizontal flip) and average the two predictions. This preserves your two-model gating logic (still computes the same out4/out2, same softmax, same threshold routing), but makes it more robust to left/right leaf orientation, which is common in Cassava and often yields a meaningful accuracy lift. I also batch the two TTA views together (batch size 2) to keep runtime within limits and keep the rest of the pipeline, including threshold tuning and submission alignment, unchanged. The output remains a valid `submission.csv` with the required schema.'
- What this solution (achieved 0.61099) has done: 'Your score gap to the target is large (0.61099 → 0.83258), so we should apply one minimal, inference-only fix that is very likely harming accuracy without changing your model architectures or training logic. Right now the 4-class model predicts among {0,1,2,3} but you map its class-3 output to label 4 via `minority_idx`, which systematically turns “CBB (label 3)” predictions into “healthy (label 4)” and can severely damage accuracy. I change `minority_idx` to the identity mapping `{0:0,1:1,2:2,3:3}` so the 4-class model’s argmax is interpreted correctly, while keeping your binary gating logic and threshold tuning exactly the same. Everything else (weights loading checks, transforms, TTA flip, tuned threshold selection, and submission alignment with `sample_submission.csv`) stays intact and still writes a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.83258), so we should make the smallest inference-only change that plausibly yields a sizable accuracy lift without touching your models, training, or gating semantics. The most likely remaining mismatch is that your 2-class “binary” head is being interpreted as if its class-1 probability corresponds to label 3, but the checkpoint may have been trained with the opposite class order; this would systematically invert the gating decision and collapse accuracy. I add a tiny calibration step on a small validation subset to choose whether the binary head should be treated as “class1=label3” or “class0=label3”, and use that choice for both threshold tuning and test inference. Everything else (architectures, transforms, TTA, threshold tuning method, and submission writing) remains the same and still produces `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.83258), so we should make a small inference-only fix that’s very likely suppressing accuracy without changing your models, training, or gating logic. The biggest issue is that your 2-class “binary” model is being used to output labels {0,3} via `binary_idx`, which can never predict labels 1/2/4 when the gate routes to the binary head; for this competition the binary head should only decide whether the image is class 3 or “not 3”, and the final label should come from the 4-class model when “not 3”. I keep the same two-model gating semantics, but change the post-processing so that when `cls_3 > thresh_3` we output label 3, otherwise we output the 4-class model’s argmax (0–3) directly. I also tune the threshold on a validation subset using the corrected decision rule (still no training), which should move accuracy substantially toward the target while staying within your core logic.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, so we should apply a small, high-impact inference-only fix without changing your two-model gating design. The most likely remaining accuracy drag is a preprocessing mismatch: Cassava images are not ImageNet-photographic and your current pipeline lacks any inference-time color robustness, so the binary “is class 3” gate can be brittle. I add a very lightweight, standard test-time augmentation set (original + horizontal flip + mild brightness/contrast jitter) and average probabilities before the same threshold routing; this keeps the exact same models, loss semantics, and gating rule, but makes the gate and 4-class head more stable. I also compute the tuned threshold using the same TTA averaging (so calibration matches inference), which typically improves accuracy versus tuning on a different distribution.'
- What this solution (achieved 0.61099) has done: 'Your score (0.61099) is far below the target (0.83258), so we should make a small inference-only fix that’s very likely hurting accuracy without changing architectures or training. The key issue is in the gating: when the binary “is class 3” probability is low, you currently output the 4-class model’s argmax directly, but that model was trained as a “minority” head that does **not** include class 3; therefore you must map its predicted index into the true label set (e.g., {0,1,2,4}) using `minority_idx`, otherwise you systematically mislabel many non-3 images. I change `minority_idx` back to the proper mapping and keep everything else (models, transforms, TTA, calibration, threshold tuning, submission alignment) identical. This should materially increase accuracy toward the target while preserving your two-model gating semantics.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.83258), so we should make one small, high-impact correctness fix rather than broad tuning. The biggest likely remaining accuracy bug is an inconsistent label mapping for the 4-class “minority” head: the code currently maps its class-3 output to label 4, which is only correct if that head was trained on labels {0,1,2,4} (no class 3) and class-3 corresponds to label 4; if instead it was trained on {0,1,2,3}, this mapping systematically corrupt many predictions. To preserve your two-model gating core logic, we add a tiny calibration step on a small validation subset to choose between the two plausible mappings (identity {0,1,2,3} vs {0,1,2,4}) and then use the best mapping consistently in threshold tuning and test inference. Everything else (models, weights loading, preprocessing, TTA, threshold tuning method, and submission writing) remains unchanged.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below target, so we should make one small but high-impact correctness fix while keeping your two-model gating design intact. The largest likely remaining issue is that your calibration and tuning paths use `img_transform_tta` with `ColorJitter`, which introduces randomness (no fixed seed per image) and makes threshold/mapping calibration noisy and potentially wrong, hurting accuracy. I make the TTA deterministic by removing the stochastic `ColorJitter` view and replacing it with a deterministic brightness/contrast adjustment, preserving the same “3-view” averaging semantics (base + flip + color-variant). Everything else (models, weights loading, gating rule, threshold tuning, submission alignment) stays unchanged and still writes a valid `submission.csv`.'

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
from torchvision.transforms import functional as TF

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
        nn.AdaptiveAvgPool2d((1, 1)),
    )

    squeezenet_custom.classifier = classifier
    return squeezenet_custom


squeezenet_custom_4 = light_model(4).to(device)
squeezenet_custom_2 = light_model(2).to(device)



## === cell 3
leaf_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
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
def prediction_logic(
    img,
    squeezenet_custom_4,
    squeezenet_custom_2,
    thresh_3=0.50,
    binary_pos_index=1,  # which p2 index represents probability of "label 3"
    minority_idx_map=None,
):
    if minority_idx_map is None:
        minority_idx_map = minority_idx

    img = img.to(device)

    with torch.no_grad():
        out4 = squeezenet_custom_4(img)
        out2 = squeezenet_custom_2(img)

    out4 = out4.view(out4.size(0), -1)
    out2 = out2.view(out2.size(0), -1)

    p4 = torch.softmax(out4, dim=1)
    p2 = torch.softmax(out2, dim=1)

    if p4.size(0) > 1:
        p4 = p4.mean(dim=0, keepdim=True)
        p2 = p2.mean(dim=0, keepdim=True)

    minority_cls = int(torch.argmax(p4, dim=1).item())
    cls_3 = float(p2[0, int(binary_pos_index)].item())

    if cls_3 > thresh_3:
        return 3
    else:
        return int(minority_idx_map[minority_cls])




## === cell 8
def img_transform(img_path):
    img = Image.open(img_path).convert("RGB")
    img = leaf_transform(img).float()
    img = img.unsqueeze(0)
    return img


def img_transform_tta(img_path):
    img_pil = Image.open(img_path).convert("RGB")

    base = leaf_transform(img_pil).float().unsqueeze(0)
    flip = torch.flip(base, dims=[3])

    det = TF.adjust_contrast(img_pil, 1.12)
    det = TF.adjust_brightness(det, 1.08)
    det_t = leaf_transform(det).float().unsqueeze(0)

    return torch.cat([base, flip, det_t], dim=0)




## === cell 9
train_dir = os.path.join(proj_dir, "train_images")
test_dir = os.path.join(proj_dir, "test_images")

df = pd.read_csv(os.path.join(proj_dir, "train.csv"))
sample_df = pd.read_csv(os.path.join(proj_dir, "sample_submission.csv"))

test_dir, train_dir, df.shape, sample_df.shape




## === cell 10
def resolve_train_image_path(image_id):
    candidates = [
        os.path.join(train_dir, image_id),
        os.path.join(train_dir, "train_images", image_id),
        os.path.join(
            proj_dir, "cassava-leaf-disease-classification", "train_images", image_id
        ),
        os.path.join(
            proj_dir,
            "cassava-leaf-disease-classification",
            "train_images",
            "train_images",
            image_id,
        ),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def calibrate_binary_pos_index(
    df_train,
    max_calib_images=800,
    seed=0,
):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df_train))
    rng.shuffle(idx)

    calib_n = min(max_calib_images, len(df_train))
    calib_df = df_train.iloc[idx[:calib_n]].copy()

    records = []
    for _, row in calib_df.iterrows():
        image_id = row["image_id"]
        y = int(row["label"])
        p = resolve_train_image_path(image_id)
        if p is None:
            continue
        records.append((p, y))

    if len(records) == 0:
        return 1, {"calib_images": 0, "acc_pos1": 0.0, "acc_pos0": 0.0}

    correct_pos1 = 0
    correct_pos0 = 0

    for path, y in records:
        img = img_transform(path).to(device)
        with torch.no_grad():
            out2 = squeezenet_custom_2(img).view(1, -1)
            p2 = torch.softmax(out2, dim=1)[0].detach().cpu().numpy()

        y_is_3 = int(y == 3)

        pred_is_3_pos1 = int(p2[1] >= 0.5)
        pred_is_3_pos0 = int(p2[0] >= 0.5)

        correct_pos1 += int(pred_is_3_pos1 == y_is_3)
        correct_pos0 += int(pred_is_3_pos0 == y_is_3)

    acc_pos1 = correct_pos1 / len(records)
    acc_pos0 = correct_pos0 / len(records)

    best_pos_index = 1 if acc_pos1 >= acc_pos0 else 0
    return int(best_pos_index), {
        "calib_images": int(len(records)),
        "acc_pos1": float(acc_pos1),
        "acc_pos0": float(acc_pos0),
    }


def tune_thresh_on_val(
    df_train,
    max_val_images=1200,
    seed=0,
    thresh_grid=None,
    binary_pos_index=1,
    minority_idx_map=None,
):
    if minority_idx_map is None:
        minority_idx_map = minority_idx

    if thresh_grid is None:
        thresh_grid = np.round(np.linspace(0.15, 0.85, 29), 3).tolist()

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df_train))
    rng.shuffle(idx)

    val_n = min(max_val_images, len(df_train))
    val_df = df_train.iloc[idx[:val_n]].copy()

    records = []
    for _, row in val_df.iterrows():
        image_id = row["image_id"]
        y = int(row["label"])
        p = resolve_train_image_path(image_id)
        if p is None:
            continue
        records.append((image_id, p, y))

    if len(records) == 0:
        return 0.50, {"val_images": 0, "best_acc": 0.0}

    best_t = 0.50
    best_acc = -1.0

    cache = []
    for _, path, y in records:
        img = img_transform_tta(path).to(device)
        with torch.no_grad():
            out4 = squeezenet_custom_4(img).view(img.size(0), -1)
            out2 = squeezenet_custom_2(img).view(img.size(0), -1)
            p4 = torch.softmax(out4, dim=1).mean(dim=0, keepdim=True)
            p2 = torch.softmax(out2, dim=1).mean(dim=0, keepdim=True)

        minority_cls = int(torch.argmax(p4, dim=1).item())
        cls_3 = float(p2[0, int(binary_pos_index)].item())
        cache.append((cls_3, minority_cls, y))

    for t in thresh_grid:
        correct = 0
        for cls_3, minority_cls, y in cache:
            pred = 3 if cls_3 > t else int(minority_idx_map[minority_cls])
            correct += int(pred == y)
        acc = correct / len(cache)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    return best_t, {"val_images": len(cache), "best_acc": float(best_acc)}


def calibrate_minority_mapping(
    df_train,
    max_map_images=1200,
    seed=1,
    binary_pos_index=1,
):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df_train))
    rng.shuffle(idx)

    val_n = min(max_map_images, len(df_train))
    val_df = df_train.iloc[idx[:val_n]].copy()

    records = []
    for _, row in val_df.iterrows():
        image_id = row["image_id"]
        y = int(row["label"])
        p = resolve_train_image_path(image_id)
        if p is None:
            continue
        records.append((p, y))

    if len(records) == 0:
        return {0: 0, 1: 1, 2: 2, 3: 4}, {
            "map_images": 0,
            "acc_identity": 0.0,
            "acc_minority": 0.0,
        }

    identity_map = {0: 0, 1: 1, 2: 2, 3: 3}
    minority_map = {0: 0, 1: 1, 2: 2, 3: 4}

    correct_id = 0
    correct_min = 0

    for path, y in records:
        img = img_transform_tta(path).to(device)
        with torch.no_grad():
            out4 = squeezenet_custom_4(img).view(img.size(0), -1)
            out2 = squeezenet_custom_2(img).view(img.size(0), -1)
            p4 = torch.softmax(out4, dim=1).mean(dim=0, keepdim=True)
            p2 = torch.softmax(out2, dim=1).mean(dim=0, keepdim=True)

        minority_cls = int(torch.argmax(p4, dim=1).item())
        cls_3 = float(p2[0, int(binary_pos_index)].item())

        pred_id = 3 if cls_3 > 0.5 else int(identity_map[minority_cls])
        pred_min = 3 if cls_3 > 0.5 else int(minority_map[minority_cls])

        correct_id += int(pred_id == y)
        correct_min += int(pred_min == y)

    acc_id = correct_id / len(records)
    acc_min = correct_min / len(records)

    best_map = identity_map if acc_id >= acc_min else minority_map
    return best_map, {
        "map_images": int(len(records)),
        "acc_identity": float(acc_id),
        "acc_minority": float(acc_min),
    }


binary_pos_index, calib_info = calibrate_binary_pos_index(
    df, max_calib_images=800, seed=0
)

minority_idx_map, map_info = calibrate_minority_mapping(
    df, max_map_images=1200, seed=1, binary_pos_index=binary_pos_index
)

best_thresh_3, tune_info = tune_thresh_on_val(
    df,
    max_val_images=1200,
    seed=0,
    binary_pos_index=binary_pos_index,
    minority_idx_map=minority_idx_map,
)

binary_pos_index, calib_info, minority_idx_map, map_info, best_thresh_3, tune_info




## === cell 11
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

    img_tta = img_transform_tta(img_path)

    final_pred = prediction_logic(
        img_tta,
        squeezenet_custom_4,
        squeezenet_custom_2,
        thresh_3=best_thresh_3,
        binary_pos_index=binary_pos_index,
        minority_idx_map=minority_idx_map,
    )
    test_preds[image_id] = final_pred

missing, len(test_preds)



## === cell 12
sub = sample_df.copy()
sub["label"] = sub["image_id"].map(test_preds).astype(int)

assert sub.shape[0] == sample_df.shape[0]
assert list(sub.columns) == ["image_id", "label"]
assert sub["label"].between(0, 4).all()

sub.head()



## === cell 13
os.chdir("/kaggle/working/")
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote:", "/kaggle/working/submission.csv", "rows:", len(sub))
print(
    "Using calibrated binary_pos_index =", binary_pos_index, "calib_info =", calib_info
)
print("Using calibrated minority_idx_map =", minority_idx_map, "map_info =", map_info)
print("Using tuned thresh_3 =", best_thresh_3, "tune_info =", tune_info)
print(sub["label"].value_counts().sort_index())
