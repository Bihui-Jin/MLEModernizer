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

0.8732245391356905

# 6. Current score

0.12332

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10389) has done: 'I fix the immediate runtime blocker by making the model-loading robust to missing Kaggle input folders: it search for the expected `.pth` under `/kaggle/input`, and if none is found it fall back to a torchvision pretrained model with the same architecture so inference can proceed. I also guard against the downstream `NameError` by ensuring a model object always exists for the selected branch, and keep the rest of the inference/submission logic unchanged. Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.12332) has done: 'Your current score is far below the target because the code is almost certainly not loading the intended trained `.pth` weights, so it falls back to an ImageNet-pretrained backbone with a randomly initialized 5-class head (near-random predictions). I make the weight-loading robust to common checkpoint formats (e.g., `{"state_dict": ...}`, `{"model": ...}`, `module.` prefixes) so the real cassava-trained weights load correctly without changing the model architecture or inference logic. I also add an automatic `model_select` fallback: if the selected model’s checkpoint isn’t found, it tries the other model root before resorting to torchvision defaults. These are minimal changes focused on moving accuracy upward toward the target while preserving your pipeline and submission format.'
- What this solution (achieved 0.12407) has done: 'Your score is far below the target because the checkpoint you’re loading is very likely not the intended cassava-trained weights (or it loads with many missing keys), leaving you effectively with a random 5-class head and near-random predictions. I keep your exact inference flow and architectures, but make the checkpoint discovery and loading stricter: (1) prefer checkpoints inside the provided model roots, (2) choose the “best match” candidate by testing compatibility against the instantiated model, and (3) avoid silently accepting poor matches by enforcing a higher matched-parameter threshold. I also fix a small but impactful preprocessing mismatch: you compute `invert_square_pad` but never apply it; applying it before resize is a minimal change that often aligns with how cassava models were trained and should lift accuracy. These changes are narrowly targeted to move accuracy upward toward your target while keeping your overall pipeline intact.'
- What this solution (achieved 0.12332) has done: 'Your score is far below the target, so the most likely issue is still that you’re not actually loading the intended cassava-trained checkpoint (or you’re loading it but with a preprocessing mismatch), yielding near-random predictions. I keep your model choices and pure-inference flow intact, but make checkpoint selection more reliable by (1) requiring a very strong match including the classifier/head shapes and (2) preferring “best/finetuned/cassava” checkpoint filenames when multiple candidates match. I also make normalization align with the default torchvision weights for ViT/EfficientNet when you fall back to pretrained backbones (otherwise the fallback is handicapped), while preserving your existing normalization for cassava-trained checkpoints. These are minimal changes aimed specifically at moving accuracy upward toward the target without changing your architecture or inference semantics.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import glob
import re



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_root = "/kaggle/input/efficientnetv2-large-test"
en_image_size = 480

vit_model_root = "/kaggle/input/vit_l_cassava"
vit_image_size = 518

model_select = "vit"

if model_select == "vit":
    model_image_size = vit_image_size
elif model_select == "en":
    model_image_size = en_image_size
else:
    raise ValueError(f"Unknown model_select={model_select}")




## === cell 2
def invert_square_pad(img):
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




## === cell 3
def build_val_transforms(image_size: int, normalize_mode: str):
    if normalize_mode == "cassava_05":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif normalize_mode == "imagenet_default":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown normalize_mode={normalize_mode}")

    return transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_size, image_size)),
            v2.Normalize(mean, std),
        ]
    )


normalize_mode = "cassava_05"
val_transforms = build_val_transforms(model_image_size, normalize_mode)




## === cell 4
def find_pth_candidates(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        return []
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.pth"), recursive=True))


def find_pth_candidates_global(pattern_hint: str | None = None) -> list[str]:
    search_root = "/kaggle/input"
    if not os.path.isdir(search_root):
        return []
    cands = sorted(glob.glob(os.path.join(search_root, "**", "*.pth"), recursive=True))
    if pattern_hint:
        hinted = [p for p in cands if pattern_hint.lower() in p.lower()]
        return hinted if len(hinted) > 0 else cands
    return cands


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if len(ckpt) > 0 and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def _clean_state_dict_keys(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _filename_preference_bonus(pth_path: str, pattern_hint: str) -> float:
    """
    Change rationale (score improvement): if multiple checkpoints match equally well by shape,
    prefer filenames that look like finetuned/best/cassava rather than generic/last snapshots.
    This helps pick the intended competition checkpoint when many .pth exist.
    """
    name = os.path.basename(pth_path).lower()
    bonus = 0.0
    for token, b in [
        ("best", 0.030),
        ("cassava", 0.030),
        ("finetune", 0.020),
        ("finetuned", 0.020),
        ("final", 0.015),
        ("fold", 0.010),
        ("epoch", 0.000),  # neutral
        ("last", -0.005),
        ("tmp", -0.010),
        ("debug", -0.020),
    ]:
        if token in name:
            bonus += b
    if pattern_hint and pattern_hint.lower() in pth_path.lower():
        bonus += 0.010
    return bonus


def score_checkpoint_match(
    model: torch.nn.Module, pth_path: str, important_keys: list[str] | None = None
) -> tuple[float, str]:
    """
    Change rationale (score improvement): ensure the checkpoint matches not just most layers
    but also the classifier/head layers; a mismatch there yields near-random predictions.
    """
    try:
        ckpt = torch.load(pth_path, map_location="cpu")
        sd = _clean_state_dict_keys(_extract_state_dict(ckpt))
        if not isinstance(sd, dict):
            return 0.0, "not_a_state_dict"

        model_sd = model.state_dict()
        matched = 0
        total = len(model_sd)
        for k, v in model_sd.items():
            if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].shape == v.shape:
                matched += 1
        ratio = matched / max(1, total)

        head_ok = 0
        head_total = 0
        if important_keys:
            for k in important_keys:
                head_total += 1
                if (
                    k in model_sd
                    and k in sd
                    and isinstance(sd[k], torch.Tensor)
                    and sd[k].shape == model_sd[k].shape
                ):
                    head_ok += 1
        head_ratio = (head_ok / head_total) if head_total > 0 else 1.0

        score = (
            (0.70 * ratio)
            + (0.30 * head_ratio)
            + _filename_preference_bonus(pth_path, pattern_hint="")
        )
        return (
            float(score),
            f"shape_ratio={ratio:.3f} head_ratio={head_ratio:.3f} head={head_ok}/{head_total}",
        )
    except Exception as e:
        return 0.0, f"error={e}"


def select_best_checkpoint(
    model: torch.nn.Module,
    preferred_root: str,
    pattern_hint: str,
    important_keys: list[str] | None = None,
) -> str | None:
    """
    Change rationale (score improvement): pick best compatible checkpoint with a mild filename
    preference toward likely 'best' weights; reduces chance of loading wrong/partial weights.
    """
    cands = []
    cands.extend(find_pth_candidates(preferred_root))
    if len(cands) == 0:
        cands.extend(find_pth_candidates_global(pattern_hint=pattern_hint))

    if len(cands) == 0:
        return None

    best_path = None
    best_score = -1.0
    best_msg = ""
    for p in cands[:80]:
        s, msg = score_checkpoint_match(model, p, important_keys=important_keys)
        s += _filename_preference_bonus(p, pattern_hint=pattern_hint)
        if s > best_score:
            best_score, best_path, best_msg = s, p, msg

    if best_path is not None:
        print(
            f"Best checkpoint candidate for hint='{pattern_hint}': {best_path} ({best_msg}, score={best_score:.3f})"
        )
    return best_path


def load_weights_robust(
    model: torch.nn.Module,
    pth_path: str,
    min_match_ratio: float = 0.98,
    important_keys: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Change rationale (score improvement): require very high match and require head layers to match;
    prevents silently running with an untrained/random classifier.
    """
    try:
        ckpt = torch.load(pth_path, map_location="cpu")
        sd = _clean_state_dict_keys(_extract_state_dict(ckpt))
        if not isinstance(sd, dict):
            return False, "Checkpoint does not contain a usable state_dict"

        model_sd = model.state_dict()
        matched = 0
        total = len(model_sd)
        for k, v in model_sd.items():
            if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].shape == v.shape:
                matched += 1
        ratio = matched / max(1, total)

        head_ok = 0
        head_total = 0
        if important_keys:
            for k in important_keys:
                head_total += 1
                if (
                    k in model_sd
                    and k in sd
                    and isinstance(sd[k], torch.Tensor)
                    and sd[k].shape == model_sd[k].shape
                ):
                    head_ok += 1
        head_ratio = (head_ok / head_total) if head_total > 0 else 1.0

        missing, unexpected = model.load_state_dict(sd, strict=False)
        msg = (
            f"Loaded weights (strict=False). shape_match_ratio={ratio:.3f} "
            f"head_match_ratio={head_ratio:.3f} missing={len(missing)} unexpected={len(unexpected)}"
        )
        ok = (ratio >= min_match_ratio) and (head_ratio >= 1.0)
        return ok, msg
    except Exception as e:
        return False, f"Failed to load weights: {e}"


vit_model = None
en_model = None
using_torchvision_fallback = False

if model_select == "vit":
    vit_model = models.vit_h_14(weights=None, image_size=vit_image_size)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )

    vit_important_keys = ["heads.head.weight", "heads.head.bias"]

    vit_model_path = select_best_checkpoint(
        vit_model, vit_model_root, pattern_hint="vit", important_keys=vit_important_keys
    )
    if vit_model_path is not None:
        ok, msg = load_weights_robust(
            vit_model,
            vit_model_path,
            min_match_ratio=0.98,
            important_keys=vit_important_keys,
        )
        print(f"ViT checkpoint selected: {vit_model_path}")
        print(msg)
        if not ok:
            print(
                "WARNING: ViT checkpoint match too low or head mismatch; will try EfficientNet checkpoint/model before torchvision fallback."
            )
            vit_model_path = None

    if vit_model_path is None:
        en_tmp = models.efficientnet_v2_l(weights=None)
        en_tmp.classifier[1] = torch.nn.Linear(
            en_tmp.classifier[1].in_features, num_classes
        )
        en_important_keys = ["classifier.1.weight", "classifier.1.bias"]
        en_model_path = select_best_checkpoint(
            en_tmp,
            en_model_root,
            pattern_hint="efficientnet",
            important_keys=en_important_keys,
        )
        if en_model_path is not None:
            ok2, msg2 = load_weights_robust(
                en_tmp,
                en_model_path,
                min_match_ratio=0.98,
                important_keys=en_important_keys,
            )
            print(f"EfficientNet checkpoint selected: {en_model_path}")
            print(msg2)
            if ok2:
                model_select = "en"
                model_image_size = en_image_size
                en_model = en_tmp
                vit_model = None

    if model_select == "vit":
        if vit_model_path is None:
            try:
                vit_pre = models.vit_h_14(
                    weights=models.ViT_H_14_Weights.DEFAULT, image_size=vit_image_size
                )
                vit_pre.heads.head = torch.nn.Linear(
                    vit_pre.heads.head.in_features, num_classes
                )
                vit_model = vit_pre
                using_torchvision_fallback = True
                print(
                    "WARNING: No usable cassava-trained .pth found. Using torchvision pretrained ViT-H/14 backbone with fresh 5-class head."
                )
            except Exception as e:
                print(
                    f"WARNING: Could not load torchvision pretrained ViT weights ({e}). Using random init model."
                )
                using_torchvision_fallback = True

        vit_model.to(device)
        vit_model.eval()
    else:
        en_model.to(device)
        en_model.eval()

elif model_select == "en":
    en_model = models.efficientnet_v2_l(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )

    en_important_keys = ["classifier.1.weight", "classifier.1.bias"]

    en_model_path = select_best_checkpoint(
        en_model,
        en_model_root,
        pattern_hint="efficientnet",
        important_keys=en_important_keys,
    )
    if en_model_path is not None:
        ok, msg = load_weights_robust(
            en_model,
            en_model_path,
            min_match_ratio=0.98,
            important_keys=en_important_keys,
        )
        print(f"EfficientNet checkpoint selected: {en_model_path}")
        print(msg)
        if not ok:
            print(
                "WARNING: EfficientNet checkpoint match too low or head mismatch; will try ViT checkpoint/model before torchvision fallback."
            )
            en_model_path = None

    if en_model_path is None:
        vit_tmp = models.vit_h_14(weights=None, image_size=vit_image_size)
        vit_tmp.heads.head = torch.nn.Linear(
            vit_tmp.heads.head.in_features, num_classes
        )
        vit_important_keys = ["heads.head.weight", "heads.head.bias"]
        vit_model_path = select_best_checkpoint(
            vit_tmp,
            vit_model_root,
            pattern_hint="vit",
            important_keys=vit_important_keys,
        )
        if vit_model_path is not None:
            ok2, msg2 = load_weights_robust(
                vit_tmp,
                vit_model_path,
                min_match_ratio=0.98,
                important_keys=vit_important_keys,
            )
            print(f"ViT checkpoint selected: {vit_model_path}")
            print(msg2)
            if ok2:
                model_select = "vit"
                model_image_size = vit_image_size
                vit_model = vit_tmp
                en_model = None

    if model_select == "en":
        if en_model_path is None:
            try:
                en_pre = models.efficientnet_v2_l(
                    weights=models.EfficientNet_V2_L_Weights.DEFAULT
                )
                en_pre.classifier[1] = torch.nn.Linear(
                    en_pre.classifier[1].in_features, num_classes
                )
                en_model = en_pre
                using_torchvision_fallback = True
                print(
                    "WARNING: No usable cassava-trained .pth found. Using torchvision pretrained EfficientNetV2-L backbone with fresh 5-class head."
                )
            except Exception as e:
                print(
                    f"WARNING: Could not load torchvision pretrained EfficientNet weights ({e}). Using random init model."
                )
                using_torchvision_fallback = True

        en_model.to(device)
        en_model.eval()
    else:
        vit_model.to(device)
        vit_model.eval()
else:
    raise ValueError(f"Unknown model_select={model_select}")

normalize_mode = "imagenet_default" if using_torchvision_fallback else "cassava_05"
val_transforms = build_val_transforms(model_image_size, normalize_mode)
print(
    f"Using model_select={model_select}, image_size={model_image_size}, normalize_mode={normalize_mode}"
)



## === cell 5
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].tolist()

if not os.path.isdir(test_data_directory):
    raise FileNotFoundError(f"Test directory not found: {test_data_directory}")

predictions = []
image_ids = []

for image_name in tqdm(test_image_ids, desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Missing test image: {image_path}")

    image = Image.open(image_path).convert("RGB")

    image = invert_square_pad(image)

    transformed_image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_select == "vit":
            vit_output = vit_model(transformed_image)
            _, predicted_class = torch.max(vit_output, 1)
        elif model_select == "en":
            en_output = en_model(transformed_image)
            _, predicted_class = torch.max(en_output, 1)
        else:
            raise ValueError(f"Unknown model_select={model_select}")

    predictions.append(int(predicted_class.item()))
    image_ids.append(image_name)



## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

if len(submission_df) != len(sample_df):
    raise ValueError(
        f"Submission row count {len(submission_df)} != sample_submission row count {len(sample_df)}"
    )

submission_df = sample_df[["image_id"]].merge(submission_df, on="image_id", how="left")
if submission_df["label"].isna().any():
    missing = submission_df[submission_df["label"].isna()]["image_id"].head(5).tolist()
    raise ValueError(f"Missing predictions for some image_ids, e.g.: {missing}")
submission_df["label"] = submission_df["label"].astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
