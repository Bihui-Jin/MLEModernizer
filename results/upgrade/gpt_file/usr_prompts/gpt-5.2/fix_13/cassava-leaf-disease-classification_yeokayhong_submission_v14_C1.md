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

0.8555454820187368

# 6. Current score

0.0781

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.20777) has done: 'I fix the missing-weight-path crash by making the script automatically fall back to torchvision’s built-in pretrained weights when the Kaggle input checkpoints aren’t present, while keeping the same model choices (EfficientNet-V2-L or ViT-H/14) and inference flow. I also fix the CUDA/CPU dtype mismatch by ensuring the model is moved to the same device as the input and that no inference runs if model loading failed. Finally, I prevent partial/empty submissions by only writing the CSV after successfully predicting the full sample_submission list, and I add a deterministic, safe ordering that matches the sample submission exactly.'
- What this solution (achieved 0.09978) has done: 'Your current score is low because the script is effectively doing ImageNet feature extraction with a randomly initialized 5-class head (you replace the classifier head after loading pretrained weights), so predictions are near-random. I keep the same EfficientNet-V2-L inference core, but change the checkpoint loading so it can correctly load a cassava-finetuned checkpoint even if keys are prefixed (e.g., `module.`) and so the classifier head is only replaced when needed, preserving trained weights. I also make the normalization match torchvision’s EfficientNet-V2-L pretrained weights when falling back to ImageNet weights (this improves accuracy without changing the overall approach). These are minimal, targeted fixes aimed at moving accuracy upward toward your 0.8555 target.'
- What this solution (achieved 0.15919) has done: 'Your score is far below the target, so we should improve accuracy while keeping the same inference-only core logic (EfficientNet-V2-L / ViT-H forward pass + argmax). The main likely cause is that your local checkpoint is either not being found (so you fall back to ImageNet with a fresh random 5-class head) or is being loaded but the head weights aren’t actually applied due to key/name mismatches, leaving the classifier effectively random. I make checkpoint loading more robust by (1) forcing the model definition to match the checkpoint’s classifier shape before loading (so head weights can load), (2) supporting common key patterns for EfficientNet heads (`classifier.1.*` vs `classifier.*`) and removing additional wrappers like `model_state_dict`, and (3) verifying that classifier weights were loaded (otherwise we clearly warn and still produce a valid submission). These are minimal, targeted changes that should move accuracy up toward your 0.8555 target without changing the model architecture or prediction semantics.'
- What this solution (achieved 0.13939) has done: 'Your score is far below the target, so the smallest meaningful improvement is to fix the preprocessing mismatch that can make even a correctly loaded cassava checkpoint perform poorly. I keep your exact model/inference logic (single forward pass + argmax), but change transforms to match the model family’s expected input: EfficientNet gets a standard pad-to-square + resize (no quadrant inversion), and ViT gets ImageNet mean/std instead of 0.5/0.5/0.5. I also set `torch.inference_mode()` and use the model’s weights’ recommended normalization when falling back to torchvision pretrained weights, which improves stability without changing semantics. These changes are directly aimed at improving accuracy toward your 0.8555 target while keeping the core approach intact.'
- What this solution (achieved 0.08857) has done: 'Your score is far below the target, so we should improve accuracy while keeping your exact inference core (single forward pass + argmax on EfficientNet-V2-L). The most likely cause of near-random performance is that the cassava checkpoint isn’t actually being applied (common when checkpoints store weights under nested keys or use different classifier key names), leaving you with an ImageNet backbone and a random 5-class head. I make checkpoint loading more robust by (1) extracting the correct state_dict from more checkpoint formats, (2) handling additional common prefixes, and (3) remapping EfficientNet classifier keys so the trained head can load. I also align preprocessing to torchvision EfficientNet-V2-L defaults by using bicubic resize/crop (instead of reflect-pad), which often matters a lot for fine-tuned weights and should move accuracy upward toward your target without changing evaluation semantics.'
- What this solution (achieved 0.29709) has done: 'Your score is far below the target, so we should make the smallest changes that plausibly increase accuracy without changing your inference core (single forward pass + argmax on EfficientNet-V2-L). The biggest likely issue is that your cassava checkpoint path doesn’t exist in this environment, so you silently fall back to ImageNet weights with a fresh random 5-class head, producing near-random predictions. I keep your model choice and prediction semantics identical, but (1) auto-discover a `.pth/.pt` checkpoint under `/kaggle/input` and load it if found, and (2) if no cassava checkpoint is available, avoid the random head by using the ImageNet head and map 1000 ImageNet logits to 5 classes via a fixed random projection (deterministic), which is a minimal post-head mapping that typically beats random-5-class guessing while preserving the single-pass argmax flow. These changes should move accuracy upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.08595) has done: 'Your current score is far below the target, so the smallest meaningful move upward is to stop using the deterministic random 1000→5 projection when no cassava-finetuned checkpoint is found, because that mapping is essentially arbitrary and often performs poorly. I keep your exact inference core (single EfficientNet-V2-L forward + argmax) and instead map ImageNet logits to the 5 cassava classes by using ImageNet-superclass groups (a fixed semantic mapping) so predictions become much less random without changing the model or training approach. I also switch preprocessing to the weights-recommended EfficientNet-V2-L eval transforms only in the fallback-ImageNet case, keeping your existing transforms unchanged when a cassava checkpoint is successfully loaded. This should increase accuracy substantially toward your 0.8555 target while preserving your overall pipeline and producing the same valid `submission.csv`.'
- What this solution (achieved 0.08595) has done: 'Your score is extremely far below the target, which strongly suggests you are still running in the “fallback ImageNet mode” (no cassava-finetuned checkpoint actually loaded), making predictions essentially unrelated to the cassava labels. The smallest legitimate improvement without changing your core inference logic is to ensure you actually load the cassava-finetuned weights by correctly (a) autodiscovering checkpoints, (b) extracting the right nested `state_dict`, and (c) stripping prefixes without accidentally dropping most keys. I also fix a subtle bug in `_strip_prefix_if_present` that can discard all non-prefixed keys (hurting partial loads) and add a strict check to prefer checkpoints that contain a 5-class classifier head, which directly moves accuracy toward your target. If no 5-class checkpoint exists, the script still produces a valid submission, but it clearly report that it’s in fallback mode.'
- What this solution (achieved 0.08595) has done: 'Your score is far below the target, and the biggest lever with minimal code change is to ensure you actually load a cassava-finetuned 5-class checkpoint instead of silently running in the ImageNet-fallback mode (which is near-random for this task). I (1) fix checkpoint discovery so it prefers files that truly look like cassava 5-class EfficientNet checkpoints (not just any `.pth`), (2) broaden/repair state_dict key normalization (including `ema.` and `model_ema.` prefixes which are common), and (3) stop applying the destructive `invert_square_pad`-style preprocessing by default and instead use the model’s expected eval preprocessing when a real cassava checkpoint is loaded (keeping the same single-pass argmax inference semantics). These changes keep your model/inference core intact (EfficientNet-V2-L forward + argmax) but should move accuracy substantially upward toward the 0.855 target by preventing “random head / wrong preprocessing” behavior. The script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.08595) has done: 'Your current score strongly indicates you’re still running in the ImageNet-fallback path, where the 1000→5 “bucket mapping” is not aligned to cassava labels and behaves near-random. To move accuracy up toward the 0.8555 target with minimal changes and identical inference semantics (single forward + argmax), I (1) make checkpoint auto-discovery actually prefer cassava-finetuned EfficientNetV2-L checkpoints by verifying the presence of a real 5-class classifier in the state_dict, and (2) avoid false-positive loads where only a few keys match (which effectively leaves a random/incorrect head). If a valid 5-class checkpoint cannot be found, the code still produce a valid submission, but it clearly stay in fallback mode; otherwise it should jump substantially toward the target because it finally use the finetuned weights.'
- What this solution (achieved 0.0781) has done: 'Your score is far below the target, which strongly suggests the code is still frequently falling back to the ImageNet-pretrained EfficientNet path (with an arbitrary 1000→5 bucket mapping), producing near-random cassava labels. The most direct minimal fix is to ensure we *actually use a real cassava-trained 5-class checkpoint* by (1) making checkpoint discovery prefer files that contain a 5-class EfficientNet head, and (2) not discarding partially compatible checkpoints unnecessarily (while still warning if the head isn’t present). If no such checkpoint exists, we keep the exact fallback behavior, but we make the bucket mapping slightly more stable by using max-over-bucket (often better than mean for “any leaf-like evidence”) without changing the single-forward + argmax inference semantics. These changes are narrowly targeted at increasing accuracy toward your 0.8555 target without changing the model architecture or overall pipeline.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
import os
import random

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/5/vit_h_14_518_8369_base.pth"
)
vit_image_size = 518

model_select = "en"

if model_select == "vit":
    model_image_size = vit_image_size
if model_select == "en":
    model_image_size = en_image_size




## === cell 1
def invert_square_pad(img: Image.Image) -> Image.Image:
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


def square_pad_reflect(img: Image.Image) -> Image.Image:
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )
    return transforms.functional.pad(img, padding, padding_mode="reflect")




## === cell 2
_mean, _std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

val_transforms = transforms.Compose(
    [
        transforms.Lambda(square_pad_reflect),
        transforms.Resize(
            (model_image_size, model_image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ]
)



## === cell 3
torch.set_grad_enabled(False)

vit_model = None
en_model = None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "student",
            "teacher",
            "ema",
            "model_ema",
            "weights",
            "params",
        ]:
            if k in obj and isinstance(obj[k], dict) and len(obj[k]) > 0:
                inner = obj[k]
                for kk in ["state_dict", "model_state_dict"]:
                    if (
                        kk in inner
                        and isinstance(inner[kk], dict)
                        and len(inner[kk]) > 0
                    ):
                        return inner[kk]
                return inner
    return obj


def _strip_prefix_if_present(state_dict, prefix: str):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith(prefix) for k in state_dict.keys()):
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_sd[k[len(prefix) :]] = v
            else:
                new_sd[k] = v
        return new_sd
    return state_dict


def _infer_num_classes_from_sd(sd: dict, model_type: str):
    if not isinstance(sd, dict):
        return None

    if model_type == "en":
        for k in ["classifier.1.weight", "classifier.weight", "classifier.2.weight"]:
            if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 2:
                return int(sd[k].shape[0])
        return None

    if model_type == "vit":
        for k in ["heads.head.weight", "head.weight"]:
            if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 2:
                return int(sd[k].shape[0])
        return None

    return None


def _remap_en_classifier_keys(sd: dict):
    if not isinstance(sd, dict):
        return sd
    sd = dict(sd)  # shallow copy

    if "classifier.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd.pop("classifier.weight")
    if "classifier.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd.pop("classifier.bias")

    if "classifier.2.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd.pop("classifier.2.weight")
    if "classifier.2.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd.pop("classifier.2.bias")

    return sd


def _safe_load_state_dict(model, ckpt_path: str, model_type: str):
    if not (ckpt_path and os.path.exists(ckpt_path)):
        return False, None

    try:
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            raw = torch.load(ckpt_path, map_location="cpu")

        sd = _extract_state_dict(raw)

        for pref in [
            "module.",
            "model.",
            "net.",
            "network.",
            "student.",
            "encoder.",
            "backbone.",
            "ema.",
            "model_ema.",
        ]:
            sd = _strip_prefix_if_present(sd, pref)

        if model_type == "en":
            sd = _remap_en_classifier_keys(sd)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                f"[WARN] Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)} keys."
            )
        return True, sd
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint {ckpt_path}: {e}")
        return False, None


def _en_classifier_loaded(sd: dict):
    if not isinstance(sd, dict):
        return False
    return (
        ("classifier.1.weight" in sd)
        or ("classifier.weight" in sd)
        or ("classifier.2.weight" in sd)
    )


def _vit_head_loaded(sd: dict):
    if not isinstance(sd, dict):
        return False
    return ("heads.head.weight" in sd) or ("head.weight" in sd)


def _autodiscover_checkpoint(
    prefer_keywords, search_root="/kaggle/input", exts=(".pth", ".pt"), model_type="en"
):
    """
    Change rationale (score-impacting, minimal):
    - Your low score implies we are still not consistently locating a real cassava 5-class checkpoint.
    - Prefer candidates that contain an actual 5-class head for the selected model_type.
    """
    hits = []
    for root, _, files in os.walk(search_root):
        for fn in files:
            low = fn.lower()
            if not low.endswith(exts):
                continue
            full = os.path.join(root, fn)
            full_low = full.lower()
            score = 0
            for kw in prefer_keywords:
                if kw in full_low:
                    score += 1
            hits.append((score, full))

    if not hits:
        return None

    hits.sort(key=lambda x: (x[0], x[1]), reverse=True)

    best = None
    best_score = None

    for base_score, cand in hits[:200]:
        try:
            try:
                raw = torch.load(cand, map_location="cpu", weights_only=True)
            except TypeError:
                raw = torch.load(cand, map_location="cpu")
            sd = _extract_state_dict(raw)
            for pref in [
                "module.",
                "model.",
                "net.",
                "network.",
                "student.",
                "encoder.",
                "backbone.",
                "ema.",
                "model_ema.",
            ]:
                sd = _strip_prefix_if_present(sd, pref)

            if model_type == "en":
                sd = _remap_en_classifier_keys(sd)
                n = _infer_num_classes_from_sd(sd, "en")
            else:
                n = _infer_num_classes_from_sd(sd, "vit")

            head_bonus = 0
            if n == 5:
                head_bonus = 100000
            elif n is not None:
                head_bonus = 200  # small preference for "some head exists"

            size_bonus = 0
            if isinstance(sd, dict):
                size_bonus = min(30, len(sd) // 200)

            cand_score = head_bonus + base_score * 10 + size_bonus

            if best is None or cand_score > best_score:
                best = cand
                best_score = cand_score
        except Exception:
            continue

    return best if best is not None else hits[0][1]


use_imagenet_projection_head = False
imagenet_to_5class = None
val_transforms_imagenet_en = None

if model_select == "vit":
    try:
        if not os.path.exists(vit_model_path):
            cand = _autodiscover_checkpoint(
                ["vit", "cassava", "leaf", "disease"], model_type="vit"
            )
            if cand is not None:
                print(
                    f"[INFO] vit_model_path not found; auto-discovered checkpoint: {cand}"
                )
                vit_model_path = cand

        tmp = models.vit_h_14(weights=None, image_size=518)
        ok, sd = _safe_load_state_dict(tmp, vit_model_path, model_type="vit")
        ckpt_classes = (
            _infer_num_classes_from_sd(sd, model_type="vit")
            if (ok and sd is not None)
            else None
        )

        vit_model = models.vit_h_14(weights=None, image_size=518)
        desired_classes = ckpt_classes if ckpt_classes is not None else num_classes
        if vit_model.heads.head.out_features != desired_classes:
            vit_model.heads.head = torch.nn.Linear(
                vit_model.heads.head.in_features, desired_classes
            )

        loaded, sd2 = _safe_load_state_dict(vit_model, vit_model_path, model_type="vit")
        if loaded and sd2 is not None and not _vit_head_loaded(sd2):
            print(
                "[WARN] ViT checkpoint loaded but head weights not detected; predictions may be poor."
            )

        if not loaded:
            vit_model = models.vit_h_14(
                weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
            )
            vit_model.heads.head = torch.nn.Linear(
                vit_model.heads.head.in_features, num_classes
            )
            print(
                "[WARN] vit_model_path not found/failed; using torchvision pretrained ViT-H/14 weights with a fresh 5-class head."
            )

        vit_model = vit_model.to(device)
        vit_model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize/load ViT model: {e}")

if model_select == "en":
    try:
        if not os.path.exists(en_model_path):
            cand = _autodiscover_checkpoint(
                [
                    "efficientnet",
                    "effnet",
                    "efficientnetv2",
                    "cassava",
                    "leaf",
                    "disease",
                    "v2",
                    "480",
                    "finetune",
                    "fold",
                ],
                model_type="en",
            )
            if cand is not None:
                print(
                    f"[INFO] en_model_path not found; auto-discovered checkpoint: {cand}"
                )
                en_model_path = cand

        tmp = models.efficientnet_v2_l(weights=None)
        ok, sd = _safe_load_state_dict(tmp, en_model_path, model_type="en")
        ckpt_classes = (
            _infer_num_classes_from_sd(sd, model_type="en")
            if (ok and sd is not None)
            else None
        )
        ckpt_has_5class_head = ckpt_classes == 5

        en_model = models.efficientnet_v2_l(weights=None)
        desired_classes = ckpt_classes if ckpt_classes is not None else num_classes
        if en_model.classifier[1].out_features != desired_classes:
            en_model.classifier[1] = torch.nn.Linear(
                en_model.classifier[1].in_features, desired_classes
            )

        loaded, sd2 = _safe_load_state_dict(en_model, en_model_path, model_type="en")

        if loaded and (sd2 is not None) and (not _en_classifier_loaded(sd2)):
            print(
                "[WARN] EfficientNet checkpoint loaded but classifier weights not detected; treating as unusable to avoid random-head predictions."
            )
            loaded = False

        if loaded and (not ckpt_has_5class_head):
            print(
                f"[WARN] EfficientNet checkpoint loaded but inferred ckpt_classes={ckpt_classes} (expected 5). Proceeding anyway because classifier weights are present."
            )

        if not loaded:
            w = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
            en_model = models.efficientnet_v2_l(weights=w)
            use_imagenet_projection_head = True
            val_transforms_imagenet_en = w.transforms()

            imagenet_to_5class = {
                0: [948, 949, 950, 951, 952, 953, 954, 955, 957, 958],
                1: [992, 993, 994, 995, 996, 997],
                2: [300, 301, 302, 303, 304, 305, 306, 307, 308, 309],
                3: [413, 414, 415, 416, 417, 418, 419, 420],
                4: [80, 81, 82, 83, 84, 85, 86, 87],
            }

            print(
                "[WARN] No usable cassava EfficientNet checkpoint found; using torchvision pretrained EfficientNetV2-L ImageNet head with fixed 1000->5 bucket mapping."
            )

        en_model = en_model.to(device)
        en_model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize/load EfficientNet model: {e}")



## === cell 4
sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].tolist()

predictions = []
image_ids = []

with torch.inference_mode():
    for image_name in tqdm(test_image_ids, desc="Test"):
        image_path = os.path.join(test_data_directory, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Test image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if (
            model_select == "en"
            and use_imagenet_projection_head
            and val_transforms_imagenet_en is not None
        ):
            transformed_image = (
                val_transforms_imagenet_en(image).unsqueeze(0).to(device)
            )
        else:
            transformed_image = val_transforms(image).unsqueeze(0).to(device)

        if model_select == "vit":
            output = vit_model(transformed_image)
            predicted_class = int(torch.argmax(output, dim=1).item())
        else:
            logits = en_model(transformed_image)

            if use_imagenet_projection_head:
                if imagenet_to_5class is None:
                    raise RuntimeError(
                        "ImageNet->5class mapping missing in fallback mode."
                    )
                scores = torch.full(
                    (logits.shape[0], num_classes),
                    -1e9,
                    device=logits.device,
                    dtype=logits.dtype,
                )

                for cls_idx in range(num_classes):
                    idxs = imagenet_to_5class.get(cls_idx, [])
                    if len(idxs) == 0:
                        continue
                    scores[:, cls_idx] = logits[:, idxs].max(dim=1).values

                predicted_class = int(torch.argmax(scores, dim=1).item())
            else:
                predicted_class = int(torch.argmax(logits, dim=1).item())

        predictions.append(predicted_class)
        image_ids.append(image_name)



## === cell 5
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(
    pd.read_csv(sample_sub_path)
), "Submission row count mismatch vs sample_submission.csv"
assert (
    submission_df["image_id"].tolist()
    == pd.read_csv(sample_sub_path)["image_id"].tolist()
), "Image_id order mismatch vs sample_submission.csv"

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
