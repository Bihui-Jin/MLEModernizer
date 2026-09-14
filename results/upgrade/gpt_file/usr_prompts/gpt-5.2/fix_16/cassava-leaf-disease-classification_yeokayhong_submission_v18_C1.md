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

0.40284

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10389) has done: 'I fix the immediate runtime blocker by making the model-loading robust to missing Kaggle input folders: it search for the expected `.pth` under `/kaggle/input`, and if none is found it fall back to a torchvision pretrained model with the same architecture so inference can proceed. I also guard against the downstream `NameError` by ensuring a model object always exists for the selected branch, and keep the rest of the inference/submission logic unchanged. Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.12332) has done: 'Your current score is far below the target because the code is almost certainly not loading the intended trained `.pth` weights, so it falls back to an ImageNet-pretrained backbone with a randomly initialized 5-class head (near-random predictions). I make the weight-loading robust to common checkpoint formats (e.g., `{"state_dict": ...}`, `{"model": ...}`, `module.` prefixes) so the real cassava-trained weights load correctly without changing the model architecture or inference logic. I also add an automatic `model_select` fallback: if the selected model’s checkpoint isn’t found, it tries the other model root before resorting to torchvision defaults. These are minimal changes focused on moving accuracy upward toward the target while preserving your pipeline and submission format.'
- What this solution (achieved 0.12407) has done: 'Your score is far below the target because the checkpoint you’re loading is very likely not the intended cassava-trained weights (or it loads with many missing keys), leaving you effectively with a random 5-class head and near-random predictions. I keep your exact inference flow and architectures, but make the checkpoint discovery and loading stricter: (1) prefer checkpoints inside the provided model roots, (2) choose the “best match” candidate by testing compatibility against the instantiated model, and (3) avoid silently accepting poor matches by enforcing a higher matched-parameter threshold. I also fix a small but impactful preprocessing mismatch: you compute `invert_square_pad` but never apply it; applying it before resize is a minimal change that often aligns with how cassava models were trained and should lift accuracy. These changes are narrowly targeted to move accuracy upward toward your target while keeping your overall pipeline intact.'
- What this solution (achieved 0.12332) has done: 'Your score is far below the target, so the most likely issue is still that you’re not actually loading the intended cassava-trained checkpoint (or you’re loading it but with a preprocessing mismatch), yielding near-random predictions. I keep your model choices and pure-inference flow intact, but make checkpoint selection more reliable by (1) requiring a very strong match including the classifier/head shapes and (2) preferring “best/finetuned/cassava” checkpoint filenames when multiple candidates match. I also make normalization align with the default torchvision weights for ViT/EfficientNet when you fall back to pretrained backbones (otherwise the fallback is handicapped), while preserving your existing normalization for cassava-trained checkpoints. These are minimal changes aimed specifically at moving accuracy upward toward the target without changing your architecture or inference semantics.'
- What this solution (achieved 0.14051) has done: 'I fix the ViT initialization crash by ensuring the image size is divisible by the ViT patch size (16), which currently prevents model construction and stops the whole pipeline. I also make the model selection consistent so that if ViT cannot be constructed/loaded and we fall back to EfficientNet (or torchvision weights), the corresponding model object is guaranteed to be non-None before inference. These changes are minimal and directly address the runtime errors that prevented any submission from being written. The rest of your inference loop, transforms (including `invert_square_pad`), and submission formatting are preserved.'
- What this solution (achieved 0.14163) has done: 'Your score is far below the target, so the smallest high-impact fix is to stop corrupting the input with `invert_square_pad`, which is an aggressive augmentation that should not be applied at test-time for accuracy. I keep your model selection, checkpoint loading logic, and inference loop intact, but add a simple `APPLY_INVERT_SQUARE_PAD=False` gate so we can disable it without changing core architecture or training semantics. I also make the resize interpolation explicit (bicubic) to better match common ViT/EfficientNet evaluation preprocessing while keeping the same normalization logic you already use. These minimal preprocessing corrections are the most likely to move accuracy upward toward the target without touching the model itself.'
- What this solution (achieved 0.10052) has done: 'Your current score (0.14163) is far below the target (0.8732), which strongly suggests you are still not loading the intended cassava-trained checkpoint and are effectively submitting near-random predictions from a fresh 5-class head. I make the checkpoint matching stricter specifically for the classifier/head (and stop “accepting” a checkpoint when the head doesn’t match), then automatically rebuild the model head to the checkpoint’s class count when the checkpoint indicates a different number of classes. Finally, I keep your exact inference loop but add a safe softmax+argmax path (same semantics) and ensure normalization stays aligned with whether we truly loaded a cassava checkpoint vs. torchvision fallback, which should move accuracy upward toward the target.'
- What this solution (achieved 0.10501) has done: 'Your score is far below the target, so the smallest likely high-impact fix is to stop silently using a random 5-class head when falling back to torchvision pretrained backbones. I keep your exact architectures and single-pass inference, but (1) load torchvision weights with the correct normalization and (2) replace the fresh head by using the pretrained 1000-class head and map its logits to 5 classes via an ImageNet→cassava prior computed from train labels (a minimal calibration layer outside the model). This preserves the core inference semantics (argmax over 5 logits) while making the fallback meaningfully predictive instead of near-random, which should move accuracy upward toward the target. I also ensure this mapping is used only when a true cassava-trained checkpoint wasn’t loaded, keeping your current behavior unchanged when weights are correctly found.'
- What this solution (achieved 0.48356) has done: 'Your score is far below the target, and the most likely reason is still that you are not loading any cassava-trained checkpoint and are falling back to ImageNet models whose predictions don’t correspond to cassava labels. I keep your exact model options and inference loop, but make the checkpoint discovery actually find weights that exist in this environment by additionally searching inside `/kaggle/input/cassava-leaf-disease-classification/**` (where Kaggle datasets typically live) and by broadening the filename hint so it doesn’t filter out good checkpoints. I also ensure we only accept a checkpoint as “cassava-trained” when the head matches 5 classes; otherwise we keep looking instead of silently using a mismatched head. These are minimal, execution-safe changes aimed at moving accuracy substantially upward toward your target without changing the core architecture or prediction semantics.'
- What this solution (achieved 0.28214) has done: 'Your current score (0.48356) is far below the target (0.8732), so we should improve accuracy; the most likely remaining issue is a preprocessing mismatch with the cassava-trained checkpoints (if one is found/loaded). I keep your exact model choices and inference loop, but change the cassava-checkpoint normalization from `[0.5,0.5,0.5]` to standard ImageNet mean/std (a common choice for EfficientNet/ViT cassava finetunes) while keeping the torchvision-fallback path unchanged. This is a minimal, safe change that often yields a large jump when the weights were trained with ImageNet normalization. I also add a tiny safety: if we are not using torchvision fallback but the loaded model outputs 1000 logits (indicating we accidentally ended up with an ImageNet head), we apply the same prior-mapping logic instead of producing nonsense 5-way softmax.'
- What this solution (achieved 0.08296) has done: 'Your score is far below the target, so we should push accuracy upward with minimal risk changes that preserve your inference-only core logic. The biggest remaining accuracy issue is that your “prior mapping” from 1000 ImageNet logits to 5 cassava classes is currently wrong (it just repeats the class prior for every ImageNet class), which makes predictions collapse to the majority class. I replace that mapping with a safe, deterministic fallback that uses only the cassava label prior directly when we’re stuck with a 1000-class head (i.e., predict the most likely cassava label), and only apply it in that fallback scenario. I also fix the small bug where `needs_mapping` is computed but `mapped_logits` isn’t actually mapped when `using_torchvision_fallback` is True, ensuring consistent behavior.'
- What this solution (achieved 0.40284) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest high-impact fixes that preserve your inference-only core flow. The biggest issue is that when you fall back to torchvision ImageNet models (1000-class head), you currently ignore those logits and predict only the majority cassava class via the prior, which is typically much worse than a simple learned mapping. I keep your same model choices and single-image inference loop, but replace the “prior-only” mapping with a deterministic, data-driven ImageNet→cassava mapping computed from the cassava training images using your same torchvision preprocessing and the same backbone you’re using at test-time. This mapping is only used in the 1000-class fallback case, so it won’t affect runs where a true 5-class cassava checkpoint is loaded, and it stays within time by using a capped number of training samples.'

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

VIT_PATCH_SIZE = 16
if vit_image_size % VIT_PATCH_SIZE != 0:
    vit_image_size = (vit_image_size // VIT_PATCH_SIZE) * VIT_PATCH_SIZE
    print(
        f"Adjusted vit_image_size to {vit_image_size} to be divisible by patch size {VIT_PATCH_SIZE}"
    )

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
        (max_side - width) // 2,
        (max_side - height) // 2,
        (max_side - width) - (max_side - width) // 2,
        (max_side - height) - (max_side - height) // 2,
    )

    padded_img = transforms.functional.pad(img, padding, padding_mode="reflect")
    return padded_img




## === cell 3
def build_val_transforms(image_size: int, normalize_mode: str):
    if normalize_mode == "cassava_05":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif normalize_mode == "imagenet_default":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown normalize_mode={normalize_mode}")

    return transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
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
    search_roots = [
        "/kaggle/input",
        "/kaggle/input/cassava-leaf-disease-classification",
    ]
    cands = []
    for sr in search_roots:
        if os.path.isdir(sr):
            cands.extend(glob.glob(os.path.join(sr, "**", "*.pth"), recursive=True))
    cands = sorted(set(cands))

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
    name = os.path.basename(pth_path).lower()
    bonus = 0.0
    for token, b in [
        ("best", 0.030),
        ("cassava", 0.030),
        ("finetune", 0.020),
        ("finetuned", 0.020),
        ("final", 0.015),
        ("fold", 0.010),
        ("epoch", 0.000),
        ("last", -0.005),
        ("tmp", -0.010),
        ("debug", -0.020),
    ]:
        if token in name:
            bonus += b
    if pattern_hint and pattern_hint.lower() in pth_path.lower():
        bonus += 0.010
    return bonus


def _infer_num_classes_from_state_dict(sd: dict, model_kind: str) -> int | None:
    try:
        if model_kind == "vit":
            w = sd.get("heads.head.weight", None)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                return int(w.shape[0])
        elif model_kind == "en":
            w = sd.get("classifier.1.weight", None)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                return int(w.shape[0])
    except Exception:
        return None
    return None


def score_checkpoint_match(
    model: torch.nn.Module,
    pth_path: str,
    important_keys: list[str] | None = None,
    pattern_hint: str = "",
) -> tuple[float, str]:
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
            (0.55 * ratio)
            + (0.45 * head_ratio)
            + _filename_preference_bonus(pth_path, pattern_hint=pattern_hint)
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
    cands = []
    cands.extend(find_pth_candidates(preferred_root))
    if len(cands) == 0:
        cands.extend(find_pth_candidates_global(pattern_hint=pattern_hint))

    if len(cands) == 0:
        return None

    best_path = None
    best_score = -1.0
    best_msg = ""
    for p in cands[:200]:
        s, msg = score_checkpoint_match(
            model, p, important_keys=important_keys, pattern_hint=pattern_hint
        )
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


def compute_cassava_prior_from_train(train_csv_path: str) -> np.ndarray:
    df = pd.read_csv(train_csv_path)
    counts = df["label"].value_counts().sort_index()
    prior = np.zeros(5, dtype=np.float64)
    for k, v in counts.items():
        if 0 <= int(k) < 5:
            prior[int(k)] = float(v)
    prior = prior / max(1.0, prior.sum())
    prior = np.clip(prior, 1e-12, 1.0)
    prior = prior / prior.sum()
    return prior


def cassava_probs_from_prior(
    prior5: np.ndarray, batch_size: int, device: torch.device
) -> torch.Tensor:
    p = torch.tensor(prior5, dtype=torch.float32, device=device)
    p = p / torch.clamp(p.sum(), min=1e-12)
    return p.unsqueeze(0).repeat(batch_size, 1)


def build_imagenet_to_cassava_mapping(
    model_1000: torch.nn.Module,
    train_csv_path: str,
    train_images_dir: str,
    image_size: int,
    device: torch.device,
    max_samples: int = 512,
) -> tuple[torch.Tensor | None, str]:
    if not os.path.isfile(train_csv_path):
        return None, f"train.csv not found: {train_csv_path}"
    if not os.path.isdir(train_images_dir):
        return None, f"train_images dir not found: {train_images_dir}"

    df = pd.read_csv(train_csv_path)
    if "image_id" not in df.columns or "label" not in df.columns:
        return None, "train.csv missing required columns"

    df = df.sort_values("image_id").reset_index(drop=True)
    if len(df) == 0:
        return None, "train.csv empty"

    n = min(int(max_samples), len(df))
    df = df.iloc[:n]

    local_tfms = build_val_transforms(image_size, normalize_mode="imagenet_default")

    model_1000 = model_1000.to(device)
    model_1000.eval()

    sums = torch.zeros((5, 1000), dtype=torch.float64, device="cpu")
    counts = torch.zeros((5,), dtype=torch.float64, device="cpu")

    with torch.no_grad():
        for _, row in df.iterrows():
            img_id = row["image_id"]
            y = int(row["label"])
            if y < 0 or y >= 5:
                continue
            pth = os.path.join(train_images_dir, img_id)
            if not os.path.isfile(pth):
                continue
            img = Image.open(pth).convert("RGB")
            x = local_tfms(img).unsqueeze(0).to(device)

            logits = model_1000(x)
            if logits.ndim != 2 or logits.shape[1] != 1000:
                return (
                    None,
                    f"unexpected logits shape from model_1000: {tuple(logits.shape)}",
                )

            probs = (
                torch.softmax(logits, dim=1)
                .detach()
                .to("cpu", dtype=torch.float64)
                .squeeze(0)
            )
            sums[y] += probs
            counts[y] += 1.0

    if float(counts.sum().item()) < 10:
        return (
            None,
            f"insufficient usable train images for mapping: total={int(counts.sum().item())}",
        )

    global_mean = sums.sum(dim=0) / torch.clamp(counts.sum(), min=1.0)
    for y in range(5):
        if counts[y] > 0:
            sums[y] = sums[y] / counts[y]
        else:
            sums[y] = global_mean

    prototypes = sums.to(dtype=torch.float32)  # (5,1000)
    return (
        prototypes,
        f"built mapping from {int(counts.sum().item())} train images (per-class counts={counts.tolist()})",
    )


train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
cassava_prior5 = compute_cassava_prior_from_train(train_csv_path)

vit_model = None
en_model = None
using_torchvision_fallback = False
imagenet_to_cassava_prototypes = None  # (5,1000) when available

if model_select == "vit":
    try:
        vit_model = models.vit_l_16(weights=None, image_size=vit_image_size)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
    except Exception as e:
        print(f"WARNING: Failed to build ViT model ({e}). Will try EfficientNet.")
        vit_model = None

    vit_model_path = None
    if vit_model is not None:
        vit_important_keys = ["heads.head.weight", "heads.head.bias"]

        vit_model_path = select_best_checkpoint(
            vit_model,
            vit_model_root,
            pattern_hint="vit",
            important_keys=vit_important_keys,
        )

        if vit_model_path is not None:
            ckpt = torch.load(vit_model_path, map_location="cpu")
            sd = _clean_state_dict_keys(_extract_state_dict(ckpt))
            inferred = _infer_num_classes_from_state_dict(sd, model_kind="vit")

            if inferred is not None and inferred != 5:
                print(
                    f"WARNING: ViT checkpoint head suggests num_classes={inferred} (expected 5). Rejecting this checkpoint and continuing search."
                )
                vit_model_path = None
            else:
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
        else:
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
            pattern_hint="efficient",
            important_keys=en_important_keys,
        )

        if en_model_path is not None:
            ckpt = torch.load(en_model_path, map_location="cpu")
            sd = _clean_state_dict_keys(_extract_state_dict(ckpt))
            inferred = _infer_num_classes_from_state_dict(sd, model_kind="en")

            if inferred is not None and inferred != 5:
                print(
                    f"WARNING: EfficientNet checkpoint head suggests num_classes={inferred} (expected 5). Rejecting this checkpoint."
                )
                en_model_path = None
            else:
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
                else:
                    en_model_path = None

    if model_select == "vit":
        if vit_model is None or vit_model_path is None:
            try:
                vit_pre = models.vit_l_16(
                    weights=models.ViT_L_16_Weights.DEFAULT, image_size=vit_image_size
                )
                vit_model = vit_pre
                using_torchvision_fallback = True
                print(
                    "WARNING: No usable cassava-trained .pth found. Using torchvision pretrained ViT-L/16 (1000-class head). Will use learned ImageNet->cassava mapping (or prior if mapping fails)."
                )
            except Exception as e:
                print(
                    f"WARNING: Could not load torchvision pretrained ViT weights ({e}). Using random init model."
                )
                if vit_model is None:
                    vit_model = models.vit_l_16(weights=None, image_size=vit_image_size)
                    vit_model.heads.head = torch.nn.Linear(
                        vit_model.heads.head.in_features, num_classes
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
        pattern_hint="efficient",
        important_keys=en_important_keys,
    )

    if en_model_path is not None:
        ckpt = torch.load(en_model_path, map_location="cpu")
        sd = _clean_state_dict_keys(_extract_state_dict(ckpt))
        inferred = _infer_num_classes_from_state_dict(sd, model_kind="en")

        if inferred is not None and inferred != 5:
            print(
                f"WARNING: EfficientNet checkpoint head suggests num_classes={inferred} (expected 5). Rejecting this checkpoint."
            )
            en_model_path = None
        else:
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
        vit_tmp = models.vit_l_16(weights=None, image_size=vit_image_size)
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
            ckpt = torch.load(vit_model_path, map_location="cpu")
            sd = _clean_state_dict_keys(_extract_state_dict(ckpt))
            inferred = _infer_num_classes_from_state_dict(sd, model_kind="vit")

            if inferred is not None and inferred != 5:
                print(
                    f"WARNING: ViT checkpoint head suggests num_classes={inferred} (expected 5). Rejecting this checkpoint."
                )
                vit_model_path = None
            else:
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
                en_model = en_pre
                using_torchvision_fallback = True
                print(
                    "WARNING: No usable cassava-trained .pth found. Using torchvision pretrained EfficientNetV2-L (1000-class head). Will use learned ImageNet->cassava mapping (or prior if mapping fails)."
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
    f"Using model_select={model_select}, image_size={model_image_size}, normalize_mode={normalize_mode}, num_classes={num_classes}"
)

if using_torchvision_fallback:
    model_1000 = vit_model if model_select == "vit" else en_model
    try:
        imagenet_to_cassava_prototypes, msg = build_imagenet_to_cassava_mapping(
            model_1000=model_1000,
            train_csv_path=train_csv_path,
            train_images_dir=train_images_dir,
            image_size=model_image_size,
            device=device,
            max_samples=512,
        )
        print(f"ImageNet->cassava mapping: {msg}")
    except Exception as e:
        imagenet_to_cassava_prototypes = None
        print(
            f"WARNING: Failed to build ImageNet->cassava mapping ({e}). Will use prior-only fallback."
        )



## === cell 5
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].tolist()

if not os.path.isdir(test_data_directory):
    raise FileNotFoundError(f"Test directory not found: {test_data_directory}")

if model_select == "vit" and vit_model is None:
    raise RuntimeError(
        "model_select='vit' but vit_model is None (model failed to build/load)."
    )
if model_select == "en" and en_model is None:
    raise RuntimeError(
        "model_select='en' but en_model is None (model failed to build/load)."
    )

APPLY_INVERT_SQUARE_PAD = False

predictions = []
image_ids = []

for image_name in tqdm(test_image_ids, desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Missing test image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    if APPLY_INVERT_SQUARE_PAD:
        image = invert_square_pad(image)

    transformed_image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_select == "vit":
            logits = vit_model(transformed_image)
        elif model_select == "en":
            logits = en_model(transformed_image)
        else:
            raise ValueError(f"Unknown model_select={model_select}")

        needs_mapping = logits.ndim == 2 and logits.shape[1] == 1000

        if needs_mapping:
            probs1000 = torch.softmax(logits, dim=1)  # (B,1000)
            if imagenet_to_cassava_prototypes is not None:
                proto = imagenet_to_cassava_prototypes.to(
                    device=device, dtype=probs1000.dtype
                )  # (5,1000)
                scores5 = probs1000 @ proto.t()  # (B,5)
                predicted_class = int(torch.argmax(scores5, dim=1).item())
            else:
                probs5 = cassava_probs_from_prior(
                    cassava_prior5, batch_size=logits.shape[0], device=device
                )
                predicted_class = int(torch.argmax(probs5, dim=1).item())
        else:
            probs = torch.softmax(logits, dim=1)
            predicted_class = int(torch.argmax(probs, dim=1).item())

    predictions.append(predicted_class)
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
