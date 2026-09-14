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

0.30269

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix three execution blockers with minimal changes: (1) make weight loading robust by checking for the expected files and falling back to random-weight inference (still produces a valid submission) instead of crashing; (2) fix test image enumeration to skip nested directories like `test_images/test_images` and only process `.jpg` files in a stable sorted order; and (3) align the submission to `sample_submission.csv` so length/order match exactly, preventing the “same length as the answers” error. I also ensure inference runs on the available device and wrap it in `torch.no_grad()` to avoid unnecessary memory use, without changing the model architecture or prediction logic.'
- What this solution (achieved 0.05531) has done: 'Your low score is consistent with running inference on essentially random weights because the code points to a non-existent `../input/cassava-models` directory and also sets `num_classes=4` even though the competition has 5 classes. To move accuracy toward your target with minimal logic change, I (1) fix `num_classes` to 5, (2) load `minority_weights.pth` and `binary_weights.pth` from the actual dataset folder you already have (`/kaggle/input/cassava-leaf-disease-classification/`) if present, and (3) make weight loading tolerant to `module.` prefixes / nested `state_dict` so your intended trained weights actually get applied. I not change the model architecture, transforms, or the existing two-model + threshold decision logic; this only ensures the intended weights/classes are used so predictions aren’t random.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so we should increase accuracy with the smallest changes that keep your two-model + threshold core logic intact. The biggest likely issue is that the intended `.pth` weights are still not being found/loaded, so the models effectively run with random weights; I make weight discovery actually locate those files anywhere under `proj_dir` (including nested folders) and load them with safe key normalization. I also fix `prediction_logic()` to apply softmax along the class dimension explicitly (axis=1), avoiding accidental normalization over the wrong axis which can severely damage predictions without changing the model/decision logic. Finally, I keep submission alignment to `sample_submission.csv` exactly as you already do.'
- What this solution (achieved 0.11622) has done: 'I fix the runtime KeyError by making the `minority_idx` mapping consistent with a 5-class minority model output (0–4), without changing your two-model + threshold decision logic. I also make the argmax explicitly over the class dimension to avoid accidental flattening behavior, keeping evaluation semantics the same. Finally, I keep the submission aligned to `sample_submission.csv` and ensure the script completes end-to-end and writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.11697) has done: 'Your score is far below the target, so the most likely cause is that your “two-model + threshold” logic is effectively broken at inference time (either due to label mapping inconsistencies or the binary gating returning a non-integer label). I make two minimal fixes that preserve your core two-model architecture and threshold decision structure: (1) make the binary branch always return a valid integer class (3 vs “not-3”) instead of sometimes returning a string, and (2) simplify `minority_idx` to an identity mapping for 5 classes so predictions don’t get unintentionally collapsed into class 4. These changes keep the same models, transforms, weights, and thresholding, but remove a major correctness bug that can destroy accuracy. The script still run end-to-end and write `/kaggle/working/submission.csv` with the exact required columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.11024) has done: 'Your current score suggests the models are still effectively behaving like random/untrained at inference, so the smallest change with the biggest expected gain is to reliably load the intended weights even when the checkpoint uses different key names/shapes. I keep your exact two-model + threshold logic and architecture, but make weight loading tolerant to common checkpoint variants (nested dicts, `module.` prefixes, and especially conv padding shape mismatches) by using `strict=False` and reporting missing/unexpected keys. I also replace the custom NumPy softmax with `torch.softmax` to avoid any subtle axis/shape issues and reduce CPU↔GPU overhead, without changing the semantics (still softmax over class dimension). These changes should move accuracy upward toward the target while keeping everything else (transforms, models, thresholding, submission alignment) the same.'
- What this solution (achieved 0.1065) has done: 'Your score is extremely low for this competition, which strongly suggests the binary gating branch is routing most images to the wrong label (it currently returns class 0 or 3 based on the *argmax* of a 2-way model, instead of returning class 3 when “is_3” is confident and otherwise deferring to the 5-class model). I make the smallest possible fix inside `prediction_logic` to preserve your two-model + threshold core logic: ignore `binary_cls` and only use the binary model’s probability for class-3 (`preds_binary[:,1]`) to decide whether to output label 3, otherwise output the 5-class model’s prediction. I also remove the unnecessary RGB channel swap in `img_transform`, which currently corrupts colors for PIL-loaded images and can severely hurt accuracy, while keeping the same transforms and model architecture. The rest of the pipeline (weight loading, test ordering aligned to `sample_submission.csv`, and CSV writing) stays unchanged to maintain stability and ensure a valid submission.'
- What this solution (achieved 0.10762) has done: 'Your current score (0.1065) is far below the target (0.8286), so we should increase accuracy with minimal, low-risk changes that keep your two-model + threshold core logic intact. The biggest likely blocker is that you are not loading the intended trained weights at all (or they load with large mismatches), so inference behaves close to random; I (1) expand weight-file discovery to search `/kaggle/input/**` as well (still no new data, just finding files), and (2) load checkpoints in a more compatible way by fixing the SqueezeNet classifier padding to the standard `(0,0)` so checkpoint shapes match (same architecture intent; only fixes an incompatibility). Finally, I ensure the gating threshold is only applied when the binary model weights actually loaded; otherwise we defer to the 5-class model to avoid systematic misrouting when the binary branch is random.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so we should increase accuracy with the smallest, lowest-risk corrections that keep your two-model + threshold core logic intact. The most likely remaining issue is that your SqueezeNet forward outputs `(N,C,1,1)` and you currently apply `softmax(dim=1)` on that 4D tensor; while `argmax(dim=1)` is fine, the binary gating probability `preds_binary[0,1]` is taken from a tensor that still has spatial dims, which can lead to unintended behavior and unstable gating. I minimally fix this by flattening logits to `(N,C)` inside `prediction_logic` before softmax/argmax (no change to architecture/training, just correct tensor shaping for classification). I also ensure the binary gating only activates when the binary model is both loaded and produces the expected 2-class output, otherwise safely fall back to the 5-class model.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so we should increase accuracy with minimal, low-risk fixes that keep your two-model + threshold gating core logic unchanged. The biggest likely issue is that the binary model’s weights may be partially/incorrectly loading because your `light_model(2)` architecture differs from common Cassava “binary head” checkpoints (often a modified 5-class head with only the last conv changed). I keep the same SqueezeNet backbone and gating logic, but (1) load weights more strictly when possible (and automatically fall back to the closest compatible load), and (2) add a safe “binary head adaptation” that can remap a 5-class classifier conv weight to a 2-class conv when the checkpoint contains 5-class weights—this preserves the intended binary-vs-not-3 behavior without changing inference semantics. Everything else (transforms, thresholding, submission alignment/format, and output path) stays the same and still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.30269) has done: 'Your score is far below the target, so the priority is to get your inference to use *trained* weights instead of effectively-random weights. The code currently searches for `minority_weights.pth` / `binary_weights.pth`, but those files typically don’t exist in the base Cassava dataset; I add a minimal fallback that loads standard torchvision ImageNet weights for the SqueezeNet backbone (only for the `features` part) when your custom `.pth` files aren’t found, keeping your exact classifier head sizes (5 and 2) and your same two-model + threshold logic. I also ensure the same backbone init is applied deterministically and only when custom weights are missing, which should materially improve accuracy toward your target without changing architecture or prediction semantics. Submission creation and alignment to `sample_submission.csv` stays the same.'

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

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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
minority_idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

binary_idx = {0: 0, 1: 3}



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

candidate_weight_dirs = [
    os.path.join(proj_dir, "cassava-models"),
    os.path.join(proj_dir, "models"),
    proj_dir,  # sometimes weights are placed at the dataset root
    "/kaggle/input/cassava-models",
    "../input/cassava-models",
    "/kaggle/input",
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

    broad_root = "/kaggle/input"
    if os.path.isdir(broad_root):
        for root, _, files in os.walk(broad_root):
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


def _maybe_adapt_binary_head_from_5class(state, model):
    """
    Change rationale (score improvement): many 'binary' checkpoints in this competition
    were saved with a 5-class SqueezeNet classifier conv and only used as a binary gate.
    If we try to load them into a 2-class head, the final conv mismatches and the binary
    model becomes effectively random -> poor routing -> very low accuracy. This adapts
    the final conv weights/biases to 2 outputs by mapping:
      out0 = "not-3" = average of classes [0,1,2,4]
      out1 = "is-3"  = class [3]
    This keeps the same two-model + threshold decision semantics.
    """
    try:
        k_w = "classifier.1.weight"
        k_b = "classifier.1.bias"
        if k_w not in state:
            return state, False

        w = state[k_w]
        b = state.get(k_b, None)

        if not (isinstance(w, torch.Tensor) and w.ndim == 4):
            return state, False

        if w.shape[0] == 5 and getattr(model.classifier[1], "out_channels", None) == 2:
            not3_idx = [0, 1, 2, 4]
            w_not3 = w[not3_idx].mean(dim=0, keepdim=True)
            w_is3 = w[3:4]
            new_w = torch.cat([w_not3, w_is3], dim=0)

            state = dict(state)
            state[k_w] = new_w

            if isinstance(b, torch.Tensor) and b.shape[0] == 5:
                b_not3 = b[not3_idx].mean(dim=0, keepdim=True)
                b_is3 = b[3:4]
                state[k_b] = torch.cat([b_not3, b_is3], dim=0)
            return state, True
    except Exception:
        pass
    return state, False


def _load_weights_if_present(
    model, path, device, prefer_strict=True, allow_binary_adapt=False
):
    if path is None:
        print("Warning: weights file not found. Using random init.")
        return False

    if not os.path.exists(path):
        print(f"Warning: weights not found at {path}. Using random init.")
        return False

    try:
        obj = torch.load(path, map_location=device)
        state = _extract_state_dict(obj)
        state = _normalize_state_dict_keys(state)

        adapted = False
        if allow_binary_adapt:
            state, adapted = _maybe_adapt_binary_head_from_5class(state, model)
            if adapted:
                print(
                    "Info: adapted binary checkpoint 5-class head -> 2-class head for gating."
                )

        if prefer_strict:
            try:
                model.load_state_dict(state, strict=True)
                print(f"Loaded weights (strict): {path}")
                return True
            except Exception as e_strict:
                print(f"Info: strict load failed ({e_strict}); retrying non-strict...")

        incompat = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompat, "missing_keys", []))
        unexpected = list(getattr(incompat, "unexpected_keys", []))

        print(f"Loaded weights (non-strict): {path}")
        if len(missing) > 0:
            print(f"  missing_keys ({len(missing)}): {missing[:20]}")
        if len(unexpected) > 0:
            print(f"  unexpected_keys ({len(unexpected)}): {unexpected[:20]}")

        if any(k.startswith("classifier.1") for k in missing):
            print(
                "Warning: classifier conv weights missing -> predictions may be poor."
            )
        return True
    except Exception as e:
        print(f"Warning: failed to load weights from {path}: {e}. Using random init.")
        return False


def _init_features_from_imagenet_if_needed(model, enabled=True):
    """
    Change rationale (score improvement): your current low accuracy strongly indicates the
    intended trained .pth weights are not available/loaded, leaving random weights.
    Without changing architecture or prediction logic, we initialize the *backbone features*
    from torchvision's ImageNet-pretrained SqueezeNet to get meaningful representations.
    We do NOT change the classifier head sizes (still 5-class and 2-class), so core logic
    remains identical; this only replaces random backbone init when custom weights are missing.
    """
    if not enabled:
        return False
    try:
        weights = torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1
        base = torchvision.models.squeezenet1_0(weights=weights)
        model.features.load_state_dict(base.features.state_dict(), strict=True)
        return True
    except Exception as e:
        print(f"Warning: failed to init features from ImageNet weights: {e}")
        return False


squeezenet_custom_4 = squeezenet_custom_4.to(device).eval()
squeezenet_custom_2 = squeezenet_custom_2.to(device).eval()

_loaded_1 = _load_weights_if_present(
    squeezenet_custom_4,
    model_1_path,
    device,
    prefer_strict=True,
    allow_binary_adapt=False,
)
_loaded_2 = _load_weights_if_present(
    squeezenet_custom_2,
    model_2_path,
    device,
    prefer_strict=True,
    allow_binary_adapt=True,
)

_imagenet_init_4 = False
_imagenet_init_2 = False
if not _loaded_1:
    _imagenet_init_4 = _init_features_from_imagenet_if_needed(
        squeezenet_custom_4, enabled=True
    )
    print("Info: minority model ImageNet feature init applied:", _imagenet_init_4)
if not _loaded_2:
    _imagenet_init_2 = _init_features_from_imagenet_if_needed(
        squeezenet_custom_2, enabled=True
    )
    print("Info: binary model ImageNet feature init applied:", _imagenet_init_2)




## === cell 6
def prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2, thresh_3=0.65):
    logits_minority = squeezenet_custom_4(img)
    if logits_minority.ndim == 4:
        logits_minority = torch.flatten(logits_minority, 1)
    preds_minority = torch.softmax(logits_minority, dim=1)

    if _loaded_2:
        logits_binary = squeezenet_custom_2(img)
        if logits_binary.ndim == 4:
            logits_binary = torch.flatten(logits_binary, 1)

        if logits_binary.shape[1] == 2:
            preds_binary = torch.softmax(logits_binary, dim=1)
            p_is_3 = float(preds_binary[0, 1].item())
            if p_is_3 >= thresh_3:
                return 3

    minority_cls = int(torch.argmax(preds_minority, dim=1).item())
    return int(minority_idx.get(minority_cls, 0))




## === cell 7
def img_transform(img_path):
    img = Image.open(img_path).convert("RGB")
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
print(
    "ImageNet init flags:",
    {
        "minority_imagenet_features": _imagenet_init_4,
        "binary_imagenet_features": _imagenet_init_2,
    },
)



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
