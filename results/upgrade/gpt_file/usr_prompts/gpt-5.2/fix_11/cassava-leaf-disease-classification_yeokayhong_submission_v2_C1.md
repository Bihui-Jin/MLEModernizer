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

geopandas==0.14.4
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

0.8845572680568148

# 6. Current score

0.0568

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25673) has done: 'I make the pipeline run end-to-end without requiring an external weights dataset by falling back to a torchvision pretrained ViT when the requested `.pth` file isn’t available, keeping the same ViT family and inference flow. I also fix the device/dtype mismatch that caused `Input type ... and weight type ... should be the same` by ensuring the model is moved to `device` after any potential weight loading and that inputs use the same device. Finally, I keep the submission formatting aligned to `sample_submission.csv` and always write a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.25673) is far below the target (0.88456), and the biggest issue is that your fallback path uses an ImageNet-pretrained ViT with a freshly initialized 5-class head, so predictions are essentially near-random. To move the score sharply toward the target while keeping the same core ViT inference pipeline, the minimal fix is to *require* a valid cassava fine-tuned `.pth/.pt` and load it correctly (including common checkpoint key-prefix patterns like `module.`). I also switch test-time preprocessing to the official ViT_H_14 weights’ transforms when using torchvision weights (kept as a last-resort fallback), and I add a lightweight batch inference (no change to semantics) to reduce overhead and stay within runtime. The output submission format/path stays identical and still always write `submission.csv`.'
- What this solution (achieved 0.05792) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end and always writes a valid `submission.csv`. The main bug is that you currently hard-fail when the external cassava weights dataset isn’t present, leaving `weights_path=None` and causing the later `torch.load(None)` crash; I restore a safe fallback to torchvision ViT weights while still attempting to auto-discover a 5-class cassava checkpoint if one exists. I also fix the CUDA dtype/device mismatch by explicitly moving the model to `device` *after* loading weights and ensuring inputs are on the same device. Finally, I align the preprocessing with the selected torchvision ViT weights transforms when falling back, which should improve score compared to the generic normalization while keeping the same inference semantics.'
- What this solution (achieved 0.15658) has done: 'Your current score is far below the target because the code almost certainly falls back to an ImageNet-pretrained ViT with a randomly initialized 5-class head (no cassava fine-tuning weights found), which yields near-random predictions. The smallest change that moves accuracy sharply upward (toward your 0.8846 target) while keeping the same ViT inference core is to reliably find and load *any* cassava 5-class checkpoint if present, by widening the search to include `/kaggle/data` and `/kaggle/working` (not just `/kaggle/input`) and prioritizing likely cassava checkpoints. I also make the checkpoint filtering slightly more permissive for common head key names so the 5-class head weights actually load when available, while keeping the architecture, transforms, and inference loop intact. If no suitable checkpoint exists anywhere, the fallback behavior remains unchanged and still produces a valid `submission.csv`.'
- What this solution (achieved 0.13453) has done: 'Your score is far below the target, and the dominant reason is that the script is almost certainly *not* loading a cassava-fine-tuned checkpoint (so the 5-class head stays random and predictions are near-random). The minimal, core-logic-preserving improvement is to (1) broaden and strengthen checkpoint discovery specifically for cassava ViT-H/14 weights and (2) make state-dict loading more permissive for common naming differences (not just exact key matches), while still keeping the same architecture, transforms choice, and argmax inference. If a real cassava checkpoint exists anywhere under the provided Kaggle filesystem, this should move accuracy sharply upward toward the target; if not, behavior remains unchanged and still produces a valid `submission.csv`. I also add a deterministic key-remapping step (e.g., `heads.head.*` ↔ `head.*`) so the 5-class head weights actually load when present.'
- What this solution (achieved 0.08146) has done: 'Your current score is far below the target, and the biggest likely cause is that the script is still not loading any cassava fine-tuned weights, leaving the 5-class head effectively random. I keep the exact same ViT-H/14 + argmax inference core, but (1) explicitly search for and prefer a 5-class checkpoint under the *cassava competition folder* first, (2) broaden head-key detection to include more common naming patterns so we correctly recognize/choose cassava checkpoints, and (3) add a deterministic remap for a few additional common classifier key variants so the head weights actually load when present. If no real 5-class checkpoint exists in the environment, behavior remains a stable torchvision fallback and still produces a valid `submission.csv`.'
- What this solution (achieved 0.11996) has done: 'Your score is far below the target, so we should aggressively move it upward while keeping the same ViT-H/14 + argmax inference core. The dominant issue is that you are almost certainly *not* loading a cassava-fine-tuned checkpoint, so the 5-class head stays random and accuracy is near chance; the minimal high-impact fix is to (1) correctly adapt to common checkpoint formats where the classifier head is stored under different key names (especially `heads.*` vs `heads.head.*`) and (2) allow loading from checkpoints whose head shape doesn’t match by *dropping only the head weights* while still loading the backbone (so if a nearby cassava checkpoint exists with different naming, you still benefit). I also add a deterministic sanity print of detected head out_features and loaded fraction so you can verify whether you’re actually using a 5-class fine-tuned model (without changing semantics). Submission formatting/path remains unchanged and a valid `submission.csv` is always written.'
- What this solution (achieved 0.0568) has done: 'Your score is far below the target because the code is almost certainly still not using a cassava-finetuned checkpoint, so the 5-class head is effectively random and accuracy stays near chance. The smallest high-impact fix (without changing the ViT-H/14 + argmax inference core) is to broaden checkpoint discovery to include the common Kaggle dataset locations and common filename patterns (including `.bin`), and to correctly extract a `state_dict` from more checkpoint wrapper formats. I also make the state-dict key remapping slightly more complete for torchvision ViT naming (`heads.head.*` vs `heads.*` vs `head.*`) and ensure we only drop the head when its shape mismatches, while still loading the backbone. These changes should substantially increase accuracy toward your target if any real cassava checkpoint exists in the environment; otherwise the fallback behavior remains unchanged and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models
from torchvision.models import ViT_H_14_Weights
from tqdm import tqdm
from PIL import Image

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 518
num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def torch_load_state_dict(path: str, map_location):
    """
    Compatibility: environments differ on torch.load(weights_only=...).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(state):
    """
    Score-relevant: handle more checkpoint wrapper conventions so we actually reach the
    underlying model weights (this is often why finetuned weights aren't loaded).
    """
    if isinstance(state, dict):
        for key in (
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "network",
            "weights",
        ):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if isinstance(state, dict):
        new_sd = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("net."):
                nk = nk[len("net.") :]
            new_sd[nk] = v
        state = new_sd

    return state


def state_dict_head_out_features(sd: dict) -> int | None:
    """
    Detect the classifier head out_features so we can prefer true 5-class cassava checkpoints.
    """
    for k in (
        "heads.head.weight",
        "heads.weight",
        "head.weight",
        "classifier.weight",
        "fc.weight",
        "final.weight",
        "logits.weight",
        "last_linear.weight",
        "linear.weight",
        "cls_head.weight",
        "classification_head.weight",
    ):
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 2:
            return int(sd[k].shape[0])

    for k in (
        "heads.head.bias",
        "heads.bias",
        "head.bias",
        "classifier.bias",
        "fc.bias",
        "final.bias",
        "logits.bias",
        "last_linear.bias",
        "linear.bias",
        "cls_head.bias",
        "classification_head.bias",
    ):
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 1:
            return int(sd[k].shape[0])

    return None


def find_best_cassava_weights_path(preferred_path: str) -> str | None:
    """
    Score-relevant: broaden checkpoint discovery (incl. /kaggle/input root and .bin),
    and prioritize checkpoints that clearly have a 5-class head.
    """
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    prioritized_roots = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/working/cassava-leaf-disease-classification",
    ]
    fallback_roots = ["/kaggle/input", "/kaggle/data", "/kaggle/working"]

    def collect_candidates(roots):
        candidates_local = []
        for root in roots:
            for pattern in [
                f"{root}/**/*.pth",
                f"{root}/**/*.pt",
                f"{root}/**/*.ckpt",
                f"{root}/**/*.bin",  # NEW: common HF/torch checkpoint extension in some datasets
            ]:
                candidates_local.extend(glob.glob(pattern, recursive=True))
        return sorted(set(candidates_local))

    candidates = collect_candidates(prioritized_roots)
    if len(candidates) == 0:
        candidates = collect_candidates(fallback_roots)

    def quick_text_score(p: str) -> int:
        base = os.path.basename(p).lower()
        full = p.lower()
        s = 0
        if "cassava" in full:
            s += 200
        if "leaf" in full:
            s += 40
        if "disease" in full:
            s += 30
        if "finetune" in full or "fine-tune" in full:
            s += 30
        if "best" in base:
            s += 10
        if "fold" in full:
            s += 3
        if "swa" in full:
            s += 3

        if (
            "vit_h" in full
            or "vit-h" in full
            or "vit_h_14" in full
            or "vit-h-14" in full
        ):
            s += 40
        elif "vit" in full:
            s += 20

        if "imagenet" in full:
            s -= 80
        if "pretrain" in full or "pretrained" in full:
            s -= 10

        if "weight" in base or "ckpt" in base or "checkpoint" in base:
            s += 5

        return s

    prefiltered = sorted(
        candidates, key=lambda p: (quick_text_score(p), p), reverse=True
    )[
        :400
    ]  # slight widening to avoid missing the true ckpt

    best = None
    best_score = -(10**9)

    for p in prefiltered:
        try:
            state = torch_load_state_dict(p, map_location="cpu")
            sd = unwrap_state_dict(state)
            if not isinstance(sd, dict) or len(sd) == 0:
                continue

            out_features = state_dict_head_out_features(sd)
            s = quick_text_score(p)

            if out_features == num_classes:
                s += 5000
            elif out_features is None:
                s -= 300
            else:
                s -= 1500

            if s > best_score:
                best_score = s
                best = p
        except Exception:
            continue

    return best


weights_path = find_best_cassava_weights_path(model_path)
if weights_path is None:
    print(
        "Warning: No cassava fine-tuned checkpoint found under /kaggle/input,/kaggle/data,/kaggle/working. "
        "Falling back to torchvision ImageNet-pretrained ViT-H/14 weights."
    )
else:
    print(f"Using weights: {weights_path}")



## === cell 1
tv_weights = ViT_H_14_Weights.DEFAULT

if weights_path is None:
    val_transforms = tv_weights.transforms()
else:
    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

if weights_path is None:
    model = models.vit_h_14(weights=tv_weights, image_size=image_size)
else:
    model = models.vit_h_14(weights=None, image_size=image_size)

model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)


def remap_common_vit_keys(sd: dict) -> dict:
    """
    Score-relevant: complete deterministic remaps across common ViT classifier key names
    so the 5-class head is actually detected/loaded when present.
    """
    if not isinstance(sd, dict):
        return sd
    out = dict(sd)

    if "head.weight" in out and "heads.head.weight" not in out:
        out["heads.head.weight"] = out["head.weight"]
    if "head.bias" in out and "heads.head.bias" not in out:
        out["heads.head.bias"] = out["head.bias"]

    if "heads.head.weight" in out and "head.weight" not in out:
        out["head.weight"] = out["heads.head.weight"]
    if "heads.head.bias" in out and "head.bias" not in out:
        out["head.bias"] = out["heads.head.bias"]

    if "heads.weight" in out and "heads.head.weight" not in out:
        out["heads.head.weight"] = out["heads.weight"]
    if "heads.bias" in out and "heads.head.bias" not in out:
        out["heads.head.bias"] = out["heads.bias"]
    if "heads.head.weight" in out and "heads.weight" not in out:
        out["heads.weight"] = out["heads.head.weight"]
    if "heads.head.bias" in out and "heads.bias" not in out:
        out["heads.bias"] = out["heads.head.bias"]

    if "classifier.weight" in out and "heads.head.weight" not in out:
        out["heads.head.weight"] = out["classifier.weight"]
    if "classifier.bias" in out and "heads.head.bias" not in out:
        out["heads.head.bias"] = out["classifier.bias"]

    if "fc.weight" in out and "heads.head.weight" not in out:
        out["heads.head.weight"] = out["fc.weight"]
    if "fc.bias" in out and "heads.head.bias" not in out:
        out["heads.head.bias"] = out["fc.bias"]

    if "classification_head.weight" in out and "heads.head.weight" not in out:
        out["heads.head.weight"] = out["classification_head.weight"]
    if "classification_head.bias" in out and "heads.head.bias" not in out:
        out["heads.head.bias"] = out["classification_head.bias"]

    return out


if weights_path is not None:
    state = torch_load_state_dict(weights_path, map_location="cpu")
    state_dict = unwrap_state_dict(state)
    state_dict = remap_common_vit_keys(state_dict)

    model_sd = model.state_dict()

    for hk in [
        "heads.head.weight",
        "heads.head.bias",
        "heads.weight",
        "heads.bias",
        "head.weight",
        "head.bias",
    ]:
        if hk in state_dict and hk in model_sd:
            if hasattr(state_dict[hk], "shape") and hasattr(model_sd[hk], "shape"):
                if tuple(state_dict[hk].shape) != tuple(model_sd[hk].shape):
                    del state_dict[hk]

    filtered_sd = {}
    matched = 0
    total = 0
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            total += 1
            if tuple(v.shape) == tuple(model_sd[k].shape):
                filtered_sd[k] = v
                matched += 1

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    ckpt_head = (
        state_dict_head_out_features(state_dict)
        if isinstance(state_dict, dict)
        else None
    )
    print(f"Checkpoint head out_features detected (post-remap/post-drop): {ckpt_head}")
    print(
        f"Loaded tensors (shape-matched): {matched} / {total}. Missing: {len(missing)}, unexpected: {len(unexpected)}"
    )

model = model.to(device)
model.eval()



## === cell 2
sample_sub = pd.read_csv(sample_submission_path)
image_ids = sample_sub["image_id"].tolist()

batch_size = 16 if device.type == "cuda" else 8

test_predictions = []
batch_imgs = []

with torch.no_grad():
    for image_name in tqdm(image_ids, desc="Test"):
        image_path = os.path.join(test_data_directory, image_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Test image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_t = val_transforms(img)
        batch_imgs.append(img_t)

        if len(batch_imgs) == batch_size:
            x = torch.stack(batch_imgs, dim=0).to(device)
            logits = model(x)
            preds = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
            )
            test_predictions.extend(preds)
            batch_imgs = []

    if batch_imgs:
        x = torch.stack(batch_imgs, dim=0).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        test_predictions.extend(preds)

if len(test_predictions) != len(image_ids):
    raise RuntimeError(
        f"Prediction length mismatch: {len(test_predictions)} vs {len(image_ids)}"
    )

submission_df = pd.DataFrame({"image_id": image_ids, "label": test_predictions})
submission_df = submission_df[["image_id", "label"]]
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df))
