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

albumentations==2.0.8
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

0.7243880326382592

# 6. Current score

0.28027

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.44993) has done: 'I fix the Albumentations tensor conversion bug by replacing the removed `ToTensor`/`A.pytorch.ToTensor` usage with `ToTensorV2`, and adjust the dataset to return the already-tensor output without re-wrapping it (which currently breaks shapes/types). I also make the file paths robust for this Kaggle layout (so the notebook finds `sample_submission.csv`, `test_images`, and the checkpoint if it exists) while keeping the same model and inference loop logic. Finally, I ensure inference always completes and writes a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.44993) has done: 'Your current score (0.44993) is far below the target (0.72439), so we should increase accuracy with minimal, metric-aligned changes while preserving the same model/inference structure. The biggest likely issue is that the model is receiving unnormalized integer tensors (ToTensorV2 without float conversion), which breaks the expected ImageNet-style preprocessing and can tank accuracy even with correct weights. I adjust the Albumentations pipeline to explicitly convert to float in `[0,1]` (via `A.ToFloat`) before `Normalize`, keeping the same transforms/components otherwise. I also load checkpoints more robustly (handle `state_dict` keys) without changing architecture, and keep submission writing identical.'
- What this solution (achieved 0.33184) has done: 'Your current score (0.44993) is far below the target (0.72439), so we should increase accuracy with minimal risk while keeping the same model and inference loop. The most likely remaining issue is input sizing: the pretrained ResNeXt expects a fixed-size crop/resize (commonly 224×224), but your pipeline feeds original-resolution images, which typically hurts accuracy badly even with correct normalization. I add a simple `Resize(224,224)` before normalization and tensor conversion (no change to model, loss, or loop), and I also fix determinism settings (`deterministic=True` conflicts with `benchmark=True`) to avoid unpredictable behavior. Everything else (paths, checkpoint loading logic, submission writing) stays the same.'
- What this solution (achieved 0.33221) has done: 'I fix the DataLoader crash by only applying `channels_last` memory format after batching (rank-4 tensors), not inside `Dataset.__getitem__` where images are rank-3. I also make the inference loop robust by ensuring tensors are `float32` and using the same transforms, so prediction completes and `submission.csv` is always written with the required `image_id,label` columns. These changes are execution blockers only and do not alter the core model or scoring semantics beyond enabling the intended preprocessing/inference to run end-to-end.'
- What this solution (achieved 0.108) has done: 'Your score gap to the target is large (0.33221 vs 0.72439, higher-is-better), so we should improve accuracy with the smallest change that corrects a likely inference mismatch. The main issue is your model is always created with `pretrained=False`; if the checkpoint is missing or partially incompatible, you effectively run near-random weights and score collapses. I keep the same architecture and inference loop, but (1) load ImageNet pretrained weights when no checkpoint is found (or when it’s clearly incompatible), and (2) make checkpoint loading strict by default so we don’t silently run with mostly-uninitialized weights. This should move the score substantially upward toward the target without changing the core modeling approach.'
- What this solution (achieved 0.09492) has done: 'Your current score (0.108) is far below the target (0.7244), so we should increase accuracy with the smallest, most reliable fix that keeps the same model and inference flow. The most likely cause is that you are not actually using the Cassava-trained checkpoint (it isn’t found in this environment), so you end up running an ImageNet backbone with a randomly initialized 5-class head, which produces near-random predictions. I keep the same architecture and inference loop, but (1) automatically locate a checkpoint if it exists anywhere under `/kaggle/input` or `/kaggle/data` and (2) if no checkpoint is found, I still initialize the new final layer from the ImageNet classifier weights (simple mean-over-groups mapping) instead of random init, which typically gives a large accuracy jump while preserving semantics. Everything else (transforms, resizing, normalization, argmax submission writing) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.08221) has done: 'Your score (0.09492) is far below the target (0.72439), so we should increase accuracy by fixing the most likely inference mismatch while keeping your model and inference loop intact. The biggest issue is that you always replace the classifier with a new random 5-class `Linear` after (or instead of) loading weights, so unless a perfect Cassava checkpoint exists and matches, you effectively predict near-random. I (1) switch to the current torchvision weights API (avoids silently ignoring pretrained in newer versions), and (2) if no compatible Cassava checkpoint is found, keep the original pretrained classifier backbone and build a deterministic 5-class head from the pretrained 1000-class weights (and **do not overwrite it afterward**). This preserves the same architecture family (ResNeXt101), same preprocessing, same argmax submission semantics, but should move accuracy much closer to your target.'
- What this solution (achieved 0.08221) has done: 'Your current score (0.08221) is far below the target (0.72439), so we should increase accuracy by fixing a likely “near-random head” bug while keeping the same ResNeXt101 inference pipeline. Right now, in the no-checkpoint path you correctly build an ImageNet-derived 5-class head, but then you immediately overwrite it with a new randomly initialized `Linear`, destroying performance. I remove that overwrite so the deterministic ImageNet-derived 5-class head is preserved, and keep everything else (transforms, resize/normalize, model family, argmax submission) unchanged. This is the smallest change that plausibly moves accuracy substantially toward the target without changing evaluation semantics.'
- What this solution (achieved 0.07997) has done: 'Your score is far below the target (0.08221 vs 0.72439, higher-is-better), which strongly suggests an inference mismatch rather than a small modeling deficiency. The most likely remaining core issue is that the model’s final classification layer is not being replaced correctly for ResNeXt in torchvision (the last module is `avgpool`, not the classifier), so your 5-class head initialization/loading logic is silently targeting the wrong layer and the model outputs 1000 logits (or an incompatible head), leading to near-random 5-class predictions. I minimally fix head detection/replacement to specifically use `model.fc` for ResNet/ResNeXt, and ensure checkpoint loading maps to that head correctly, while keeping the same architecture family, transforms, and argmax submission semantics. This should move accuracy substantially upward toward the target without changing the training/inference approach.'
- What this solution (achieved 0.1136) has done: 'Your current score (0.07997) is far below the target (0.72439), so we need a minimal change that plausibly fixes a major inference mismatch without changing the model family or loops. The biggest remaining issue is that your code is doing “ImageNet-derived 5-class head” inference when no Cassava checkpoint is found, which is essentially unrelated to the Cassava labels and leads to near-random accuracy. I keep the same architecture and inference semantics, but change the fallback to use the ImageNet pretrained backbone with a standard 5-class head (random init) and add simple test-time augmentation (horizontal flip) averaging, which is a small, safe inference-only change that should move accuracy substantially upward toward the target band if the backbone features are useful. I also make `batchsize=16` to stabilize predictions and speed up inference while keeping the same logic.'
- What this solution (achieved 0.28027) has done: 'Your current score (0.1136) is far below the target (0.7244), so we should increase accuracy by fixing the most likely major inference issue while keeping your architecture and inference loop intact. Right now you replace the ImageNet classifier with a fresh random 5-class head even when no Cassava checkpoint is found, which makes predictions essentially random; instead, we build a deterministic 5-class head from the pretrained 1000-class ImageNet head (same backbone, same single forward pass, same argmax submission semantics). We keep your resize/normalize and your horizontal-flip TTA averaging exactly as-is, and only change the no-checkpoint initialization so it produces meaningful logits. This is a minimal, inference-only change that should move the score strongly upward toward the target band without altering the core loop or loss/training (there is none).'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.auto import tqdm




## === cell 1
def _resolve_existing_path(candidates):
    for p in candidates:
        if p is None:
            continue
        if os.path.exists(p):
            return p
    return candidates[0] if candidates else None


DATA_ROOT = _resolve_existing_path(
    [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
)



## === cell 2
pass




## === cell 3
def _find_checkpoint():
    candidates = [
        "../input/cassava/weights.pt",
        "/kaggle/input/cassava/weights.pt",
        "/kaggle/input/weights/weights.pt",
        "/kaggle/input/cassava-leaf-disease-classification/weights.pt",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/weights.pt",
        "/kaggle/input/cassava-leaf-disease-classification/best.pth",
        "/kaggle/input/cassava-leaf-disease-classification/model.pth",
        "/kaggle/input/cassava-leaf-disease-classification/model.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    roots = ["/kaggle/input", "/kaggle/data"]
    patterns = {
        "weights.pt",
        "best.pth",
        "best.pt",
        "model.pth",
        "model.pt",
        "checkpoint.pth",
        "checkpoint.pt",
    }
    for r in roots:
        if not os.path.exists(r):
            continue
        for dirpath, _, filenames in os.walk(r):
            for fn in filenames:
                if fn in patterns:
                    return os.path.join(dirpath, fn)
    return None




## === cell 4
class cfg:
    """Main config."""

    NUMCLASSES = 5  # CONST
    seed = 42  # random seed

    pathtoimgs = os.path.join(DATA_ROOT, "test_images")
    pathtocsv = os.path.join(DATA_ROOT, "sample_submission.csv")

    chk = _find_checkpoint()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device
    modelname = "resnext101_32x8d"  # PyTorch model

    batchsize = 16
    numworkers = 4  # Number of workers

    transforms = [
        dict(name="Resize", params=dict(height=224, width=224, p=1.0)),
        dict(name="ToFloat", params=dict(max_value=255.0)),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,
                p=1.0,
            ),
        ),
        dict(name="/custom/totensor", params=dict()),
    ]




## === cell 5
print(cfg.device)
print("DATA_ROOT:", DATA_ROOT)
print("Images path exists:", os.path.exists(cfg.pathtoimgs), cfg.pathtoimgs)
print("CSV path exists:", os.path.exists(cfg.pathtocsv), cfg.pathtocsv)
print("Checkpoint:", cfg.chk)



## === cell 6
assert os.path.exists(
    cfg.pathtocsv
), f"sample_submission.csv not found at: {cfg.pathtocsv}"
assert os.path.exists(
    cfg.pathtoimgs
), f"test_images folder not found at: {cfg.pathtoimgs}"




## === cell 7
def totensor():
    return ToTensorV2()




## === cell 8
def _bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




## === cell 9
def fullseed(seed=42):
    """Sets the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fullseed(cfg.seed)



## === cell 10
cv2.setNumThreads(0)




## === cell 11
def _clean_state_dict_keys(sd):
    if not isinstance(sd, dict):
        return sd
    if not sd:
        return sd

    k0 = next(iter(sd.keys()))
    if isinstance(k0, str) and k0.startswith("module."):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _extract_state_dict(cp):
    """Minimal robustness: handle common checkpoint wrappers without changing model logic."""
    if isinstance(cp, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in cp and isinstance(cp[key], dict):
                return cp[key]
    if isinstance(cp, dict):
        tensor_vals = [v for v in cp.values() if torch.is_tensor(v)]
        if len(tensor_vals) > 0:
            return cp
    return cp


def _get_imagenet_weights_for_modelname(modelname: str):
    weights_map = {
        "resnext101_32x8d": getattr(models, "ResNeXt101_32X8D_Weights", None),
        "resnet50": getattr(models, "ResNet50_Weights", None),
        "resnet101": getattr(models, "ResNet101_Weights", None),
        "efficientnet_b0": getattr(models, "EfficientNet_B0_Weights", None),
    }
    w_enum = weights_map.get(modelname, None)
    if w_enum is None:
        return None
    return w_enum.DEFAULT


def _get_classifier_attr(model: nn.Module):
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        return "fc"
    if hasattr(model, "classifier"):
        head = getattr(model, "classifier")
        if isinstance(head, nn.Linear):
            return "classifier"
        if (
            isinstance(head, nn.Sequential)
            and len(head) > 0
            and isinstance(head[-1], nn.Linear)
        ):
            return "classifier"
    raise AttributeError(
        "Could not locate classifier head (expected .fc or .classifier)."
    )


def _get_head_linear(model: nn.Module, head_attr: str) -> nn.Linear:
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):
        return head
    if isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
        return head[-1]
    raise TypeError(f"Unsupported head type for {head_attr}: {type(head)}")


def _set_head_linear(model: nn.Module, head_attr: str, new_linear: nn.Linear):
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):
        setattr(model, head_attr, new_linear)
        return
    if (
        isinstance(head, nn.Sequential)
        and len(head) > 0
        and isinstance(head[-1], nn.Linear)
    ):
        head = list(head)
        head[-1] = new_linear
        setattr(model, head_attr, nn.Sequential(*head))
        return
    raise TypeError(f"Unsupported head type for {head_attr}: {type(head)}")


def _init_5class_head_from_imagenet(
    old_head: nn.Linear, num_classes: int, seed: int = 0
):
    """
    Change rationale (score-improving, minimal): if no Cassava checkpoint exists, a random 5-class
    head makes predictions near-random. Instead, derive a deterministic 5-class linear head from
    the pretrained 1000-class ImageNet head by averaging weight rows into 5 buckets.
    This keeps the same backbone/forward/argmax semantics; only improves the fallback init.
    """
    assert isinstance(old_head, nn.Linear)
    w = old_head.weight.detach().clone()  # [1000, in_features]
    b = old_head.bias.detach().clone() if old_head.bias is not None else None  # [1000]

    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.randperm(w.shape[0], generator=g)
    groups = torch.chunk(perm, num_classes)

    new = nn.Linear(w.shape[1], num_classes, bias=(b is not None))
    with torch.no_grad():
        for i, idx in enumerate(groups):
            new.weight[i].copy_(w[idx].mean(dim=0))
            if b is not None:
                new.bias[i].copy_(b[idx].mean())
    return new


def get_model(cfg):
    """Get PyTorch model."""
    weights = _get_imagenet_weights_for_modelname(cfg.modelname)
    try:
        if weights is not None:
            model = getattr(models, cfg.modelname)(weights=weights)
        else:
            model = getattr(models, cfg.modelname)(pretrained=True)
    except TypeError:
        model = getattr(models, cfg.modelname)(pretrained=True)

    head_attr = _get_classifier_attr(model)
    head_linear = _get_head_linear(model, head_attr)

    if cfg.chk is None or not os.path.exists(cfg.chk):
        new_head = _init_5class_head_from_imagenet(
            head_linear, cfg.NUMCLASSES, seed=cfg.seed
        )
        _set_head_linear(model, head_attr, new_head)
        print(
            "No checkpoint found; using ImageNet pretrained backbone with a deterministic "
            "ImageNet-derived 5-class head (non-random fallback)."
        )
    else:
        _set_head_linear(
            model,
            head_attr,
            nn.Linear(
                in_features=head_linear.in_features,
                out_features=cfg.NUMCLASSES,
                bias=True,
            ),
        )

        cp = torch.load(cfg.chk, map_location="cpu")
        sd = _extract_state_dict(cp)
        sd = _clean_state_dict_keys(sd)

        try:
            model.load_state_dict(sd, strict=True)
            print("Checkpoint load: strict=True (fully matched).")
        except RuntimeError as e:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            loaded_frac = 1.0 - (len(missing) / max(1, len(model.state_dict())))
            print(
                f"Checkpoint load: strict=False fallback after strict failure: {e}\n"
                f"loaded_frac={loaded_frac:.3f}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
            if loaded_frac < 0.8:
                raise RuntimeError(
                    "Checkpoint appears incompatible with the model (too many missing keys). "
                    "Refusing to run with mostly-random weights."
                )

        if isinstance(cp, dict):
            if "lr" in cp:
                cfg.lr = cp["lr"]
            if "stopflag" in cp:
                cfg.stopflag = cp["stopflag"]

    model = model.to(cfg.device)

    if cfg.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=cfg.device)
        if cfg.device.type == "cuda":
            dummy = dummy.contiguous(memory_format=torch.channels_last)
        out = model(dummy)
        assert (
            out.shape[-1] == cfg.NUMCLASSES
        ), f"Model output dim {out.shape[-1]} != {cfg.NUMCLASSES}"

    return model


def get_transforms(cfg):
    """Get test augmentations."""
    transforms = [
        (
            globals()[item["name"][8:]](**item["params"])
            if item["name"].startswith("/custom/")
            else getattr(A, item["name"])(**item["params"])
        )
        for item in cfg.transforms
    ]
    return A.Compose(transforms)




## === cell 12
class CassavaDataset(torch.utils.data.Dataset):
    """Cassava Dataset for uploading images and targets."""

    def __init__(self, cfg, images, transforms):
        self.images = images
        self.transforms = transforms
        self.cfg = cfg

    def __getitem__(self, idx):
        img_path = os.path.join(self.cfg.pathtoimgs, self.images[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = _bgr2rgb(img)
        img = self.transforms(image=img)["image"]
        return img

    def __len__(self):
        return len(self.images)




## === cell 13
def get_loader(cfg):
    """Getting dataloader for test."""
    data = pd.read_csv(cfg.pathtocsv)
    imgs = list(data["image_id"])
    transforms = get_transforms(cfg)
    dataset = CassavaDataset(cfg, imgs, transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batchsize,
        pin_memory=torch.cuda.is_available(),
        num_workers=cfg.numworkers,
    )
    return dataloader




## === cell 14
dataloader = get_loader(cfg)
batch0 = next(iter(dataloader))
print(
    "Batch type:",
    type(batch0),
    "dtype:",
    batch0.dtype,
    "shape:",
    tuple(batch0.shape),
    "min/max:",
    float(batch0.min()),
    float(batch0.max()),
)



## === cell 15
torch.cuda.empty_cache()
model = get_model(cfg)
model.eval()

preds = []
with torch.no_grad():
    for img in tqdm(dataloader, total=len(dataloader)):
        img = img.to(cfg.device, non_blocking=True)
        if img.dtype != torch.float32:
            img = img.float()
        if cfg.device.type == "cuda":
            img = img.contiguous(memory_format=torch.channels_last)

        logits = model(img)

        img_flip = torch.flip(img, dims=[3])
        logits_flip = model(img_flip)

        logits_avg = (logits + logits_flip) / 2.0
        pred = (
            torch.argmax(logits_avg, dim=1).detach().cpu().numpy().astype(int).tolist()
        )
        preds.extend(pred)

print("Preds:", len(preds), "first5:", preds[:5])



## === cell 16
df = pd.read_csv(cfg.pathtocsv)
assert len(df) == len(preds), f"Prediction length mismatch: {len(preds)} vs {len(df)}"

df["label"] = preds
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
