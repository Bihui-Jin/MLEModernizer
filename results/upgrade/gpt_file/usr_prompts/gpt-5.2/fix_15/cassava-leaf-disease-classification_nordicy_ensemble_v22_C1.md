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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8859171955273496

# 6. Current score

0.32848

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.32175) has done: 'I fix the end-to-end runtime blockers so a valid `submission.csv` is always produced: (1) make weight loading robust by automatically searching `/kaggle/input` for the expected `.pth` files and falling back to ImageNet weights if they aren’t present, (2) fix the CUDA/CPU dtype mismatch by ensuring models and inputs are on the same device/dtype, and (3) fix submission alignment/NaNs by storing plain strings for `image_id` and filling any missing predictions safely. These changes preserve your core inference/TTA/ensemble semantics, and also improve score vs. the current “no submission” situation by ensuring real model predictions are generated rather than crashing.'
- What this solution (achieved 0.33296) has done: 'The timeout is dominated by per-image Python overhead: each test image is decoded once, but then you run 4 models × 5 TTA = 20 separate forward passes with batch_size=1 and you also re-run Albumentations (resize/normalize) 20 times per image. To keep identical model logic and TTA semantics while cutting wall time, the main change is to batch the TTA augmentations into a single tensor per model (so 5 TTAs become one forward pass with batch size 5), and to use a DataLoader that actually batches and parallelizes image decode (num_workers + pin_memory + prefetch). We also remove the expensive `/kaggle/input` full-tree weight-file search by checking only the known preferred paths (preserves behavior when weights exist, and avoids multi-GB directory walks). These changes are provably equivalent in outputs up to negligible floating-point differences and typically reduce inference time by ~3–8×.'
- What this solution (achieved 0.3352) has done: 'Your current score (~0.33) strongly suggests your notebook is mostly falling back to ImageNet weights because the custom `.pth` files aren’t being found/loaded, so the minimal path to move toward the 0.886 target is to reliably locate and load those weights. I keep your exact ensemble + 5-TTA semantics, but make weight discovery robust by searching only for the specific filenames (fast) under `/kaggle/input` and then loading them. I also ensure the correct transform is used consistently (keep your existing TTA transform) and keep submission ordering identical to `sample_submission.csv`. These changes should substantially increase accuracy (closer to target) while preserving your core logic and runtime budget.'
- What this solution (achieved 0.32698) has done: 'Your score is far below the target (0.3352 vs 0.8859), and the most likely cause is still that at least some of the intended custom `.pth` weights are not actually being loaded (or are being loaded in a partially mismatched way), leaving the ensemble close to ImageNet-random for this task. I make weight loading more robust but still minimal: (1) search only for the exact expected filenames using a fast `os.walk` with pruning, (2) improve `safe_load_state_dict` to correctly handle common checkpoint formats (`state_dict`, `model`, nested keys) and enforce classifier/fc compatibility checks, and (3) print a clear per-model “custom vs imagenet” status so you can verify you’re not silently falling back. These changes keep your exact ensemble + 5-TTA inference semantics and should move accuracy substantially toward the target if the weights exist in `/kaggle/input`. The submission ordering/format remains identical to `sample_submission.csv`, and the script still always writes `submission.csv`.'
- What this solution (achieved 0.32661) has done: 'Your gap to the target is very large (0.32698 → 0.8859), and the most likely reason is still “not actually using your trained weights” (or loading them but with a mismatched classifier head), which yields near-random predictions. I make the smallest set of changes that (1) loads weights *after* constructing the final 5-class head (so the head weights can load), (2) expands checkpoint parsing to handle common nesting like `{'model': {'state_dict': ...}}`, and (3) explicitly detects “head not loaded” (based on whether fc/classifier weights were actually present in the checkpoint) to avoid silently accepting a bad load. This preserves your exact ensemble + 5-TTA inference semantics and submission formatting, but should move accuracy sharply upward if the custom `.pth` files exist in `/kaggle/input`.'
- What this solution (achieved 0.33408) has done: 'Your score is far below the target, so we should increase accuracy with the smallest changes that preserve your ensemble + 5-TTA inference semantics. The biggest likely issue is that the custom weights still aren’t being found/loaded, because your `safe_load_state_dict` currently rejects valid checkpoints when the head key names don’t match your exact `fc.*` / `classifier.1.*` expectations. I make head-detection robust by (a) remapping common EfficientNet head names like `classifier.weight/bias` to `classifier.1.*`, and (b) validating “head loaded” by shape-match against the model head rather than exact key strings. This keeps the same models, transforms, and averaging, but should flip you from “mostly ImageNet fallback / partial loads” to “actually using trained heads,” moving accuracy substantially toward the target.'
- What this solution (achieved 0.33744) has done: 'Your score is far below the target, so we need a real accuracy jump while keeping your ensemble + 5-TTA semantics intact. The most likely remaining issue is still weight loading: your current loader rejects many valid checkpoints due to head-key naming differences (e.g., `classifier.1.*/fc.*` vs `classifier.*` vs `head.*`) and due to overly strict “compatible head” detection before/without key normalization. I make `safe_load_state_dict` robust but still minimal by (1) normalizing common key patterns for ResNet/EfficientNet heads, (2) allowing “head compatible” if the checkpoint contains any tensor matching the expected head shapes even if the key name differs, and (3) only treating the load as failed if no compatible head tensors exist after remapping. This preserves your model definitions, transforms, TTA, and averaging, but should switch you from ImageNet-fallback/partial-load to actually using the intended trained weights when present, moving accuracy much closer to the target.'
- What this solution (achieved 0.3281) has done: 'Your score (~0.337) is far below the target (0.886), so we need a real accuracy jump; the most plausible blocker is still that your intended custom checkpoints are not being loaded correctly (so you’re effectively predicting with ImageNet heads). I keep your exact ensemble + 5-TTA inference semantics, but make weight loading reliably succeed by (1) using a fast exact-filename index of `/kaggle/input` once (instead of repeated `os.walk` per model), and (2) making `safe_load_state_dict` accept common checkpoint layouts and automatically align head key names by matching tensors to the model’s head shapes (not just key strings). This is a minimal change that directly targets the likely cause of near-random accuracy, without changing model architectures, transforms, TTA count, or ensembling logic. The submission ordering/format remains identical to `sample_submission.csv`, and it still always write `submission.csv`.'
- What this solution (achieved 0.33034) has done: 'Your score (0.3281) is far below the target (0.8859), so we need a real accuracy jump; the smallest likely lever (without changing your ensemble/TTA/model definitions) is ensuring the custom checkpoints actually load and are used. Right now `safe_load_state_dict` can successfully *load* weights but still return `False` because `has_compatible_head` is checked **before** `load_state_dict`, and because the “head present” heuristic is too broad (it can match non-head tensors by shape), which can lead to incorrect fallbacks to ImageNet weights (near-random). I (1) detect whether the *actual head parameters* were loaded by checking for head key presence with correct shapes after key-remap/shape-map, (2) only fall back when the checkpoint truly doesn’t provide a compatible 5-class head, and (3) add a tiny, non-invasive log to confirm per-model whether the head really loaded—this directly targets moving accuracy toward the target while preserving your inference semantics.'
- What this solution (achieved 0.34118) has done: 'Your score (~0.33) is far below the target (~0.886), which strongly suggests the custom checkpoints still aren’t being used correctly at inference time (i.e., you’re effectively ensembling mostly-ImageNet models). I make the smallest change that preserves your ensemble + 5-TTA logic but fixes a critical bug in `safe_load_state_dict`: you currently load weights and then sometimes *discard them* (return `False`) because the “compatible head” check is done on the pre-load state dict and is overly strict about key presence. The patch makes the loader: (1) load first, (2) then verify that the model’s head tensors actually match the checkpoint’s head (by inspecting checkpoint head tensors and load results), and (3) only fall back to ImageNet when the checkpoint truly cannot provide a 5-class head—this should move accuracy sharply toward your target if the `.pth` files exist. Submission formatting/ordering and the inference/TTA/averaging semantics remain unchanged.'
- What this solution (achieved 0.33296) has done: 'Your score is far below the target, so we should increase accuracy with the smallest changes that keep your ensemble + 5-TTA semantics identical. The biggest likely remaining issue is that checkpoints are being loaded but the model is still effectively using ImageNet heads because the loader is too strict: it currently *rejects* checkpoints unless it can prove a full head is present in the checkpoint, even though many valid training setups save only the backbone and expect the head to be randomly initialized or differently named. I make `safe_load_state_dict` prefer loading whatever it can (strict=False) and only fall back to ImageNet if the checkpoint can’t load any meaningful non-head weights (or loads almost nothing), while still keeping your exact model definitions and inference procedure. This should move your accuracy substantially toward the target when those `.pth` files exist, without changing TTA count, transforms, ensemble averaging, or submission format.'
- What this solution (achieved 0.32848) has done: 'Your score is far below the target, so we should increase accuracy with the smallest changes that preserve your ensemble + 5-TTA inference semantics. The most likely remaining cause of ~0.33 accuracy is still “custom weights not actually being used” due to key-mismatch between saved checkpoints and torchvision model definitions (especially EfficientNet-V2-S classifier keys and BN tracking buffers). I make `safe_load_state_dict` more permissive in a controlled way: normalize additional common key patterns (EfficientNet `classifier.1.*` vs `classifier.*`, `features.*`, and `num_batches_tracked`) and treat a load as “successful” if a meaningful fraction of tensors match (instead of failing due to lots of expected-but-absent buffers), which should move you much closer to the target if the `.pth` files exist. I also ensure the weight index is built only for the specific expected filenames (fast and avoids noise) while keeping the same paths and output submission format.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

printed = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if printed < 10:
            print(os.path.join(dirname, filename))
        printed += 1
    if printed >= 10:
        break
print(
    f"Found at least {printed} files under /kaggle/input (truncated listing for speed)."
)



## === cell 1
import os
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, ResNet50_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm



## === cell 2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



## === cell 3
num_tta = 5



## === cell 4
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 5
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()



## === cell 6
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 7
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, img_name




## === cell 8
tta_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 9
def tta_predict_single_model(model, image, tta_transform, device, n_tta=5):
    model.eval()

    if image.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (H, W, 3)")

    with torch.no_grad():
        augmented_list = [tta_transform(image=image)["image"] for _ in range(n_tta)]
        batch = torch.stack(augmented_list, dim=0).to(
            device=device, dtype=torch.float32
        )  # (n_tta, C, H, W)

        output = model(batch)  # (n_tta, num_classes)
        probs = F.softmax(output, dim=1)  # (n_tta, num_classes)
        avg_probs = probs.mean(dim=0, keepdim=True)  # (1, num_classes)

    return avg_probs




## === cell 10
def collate_images_and_names(batch):
    images, names = zip(*batch)
    return list(images), list(names)


test_dataset = CassavaTestDataset(test_df, test_image_dir)

num_workers = min(4, os.cpu_count() or 1)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    prefetch_factor=2 if num_workers > 0 else None,
    persistent_workers=True if num_workers > 0 else False,
    collate_fn=collate_images_and_names,
)



## === cell 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 12
def _iter_input_roots():
    roots = ["/kaggle/input"]
    for r in roots:
        if os.path.isdir(r):
            yield r


EXPECTED_WEIGHT_FILENAMES = {
    "cassava_leaf_best_model_fine_aug.pth",
    "Eff_best5.pth",
    "Eff_best10.pth",
    "Eff_best6.pth",
}


def build_weight_index(expected_filenames: set[str]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for base in _iter_input_roots():
        for root, dirs, files in os.walk(base):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", "__pycache__", ".ipynb_checkpoints"}
            ]
            for fn in files:
                if fn in expected_filenames and fn not in idx:
                    idx[fn] = os.path.join(root, fn)
            if len(idx) == len(expected_filenames):
                return idx
    return idx


WEIGHT_INDEX = build_weight_index(EXPECTED_WEIGHT_FILENAMES)
print(
    f"Indexed {len(WEIGHT_INDEX)} expected weight files under /kaggle/input: {WEIGHT_INDEX}"
)


def find_weight_file(preferred_path: str, filename_hint: str) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
    if filename_hint and filename_hint in WEIGHT_INDEX:
        return WEIGHT_INDEX[filename_hint]
    return None


def _extract_state_dict(ckpt):
    obj = ckpt
    for _ in range(10):  # bounded unwrapping depth; robust to nested wrappers
        if isinstance(obj, dict):
            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "module",
                "ema",
                "params",
                "weights",
            ):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                break
        else:
            break
    return obj


def _strip_prefixes(state):
    if not isinstance(state, dict):
        return state
    new_state = {}
    for k, v in state.items():
        nk = k
        for pfx in ("module.", "model.", "net.", "ema."):
            if nk.startswith(pfx):
                nk = nk[len(pfx) :]
        new_state[nk] = v
    return new_state


def _normalize_key_for_matching(k: str) -> str:
    if k.endswith("num_batches_tracked"):
        return ""  # we will ignore these buffers for "load quality" checks
    return k


def _remap_common_head_keys_for_torchvision(
    model: torch.nn.Module, state: dict
) -> dict:
    """
    Change rationale (score improvement): accept common head key names so trained heads load when present,
    avoiding silent partial-loads that behave like ImageNet.
    """
    if not isinstance(state, dict):
        return state

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        if len(model.classifier) >= 2 and isinstance(model.classifier[1], nn.Linear):
            if "classifier.weight" in state and "classifier.1.weight" not in state:
                state["classifier.1.weight"] = state["classifier.weight"]
            if "classifier.bias" in state and "classifier.1.bias" not in state:
                state["classifier.1.bias"] = state["classifier.bias"]

            if "head.weight" in state and "classifier.1.weight" not in state:
                state["classifier.1.weight"] = state["head.weight"]
            if "head.bias" in state and "classifier.1.bias" not in state:
                state["classifier.1.bias"] = state["head.bias"]

            if "classifier.0.weight" in state and "classifier.1.weight" not in state:
                state["classifier.1.weight"] = state["classifier.0.weight"]
            if "classifier.0.bias" in state and "classifier.1.bias" not in state:
                state["classifier.1.bias"] = state["classifier.0.bias"]

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if "classifier.weight" in state and "fc.weight" not in state:
            state["fc.weight"] = state["classifier.weight"]
        if "classifier.bias" in state and "fc.bias" not in state:
            state["fc.bias"] = state["classifier.bias"]
        if "head.weight" in state and "fc.weight" not in state:
            state["fc.weight"] = state["head.weight"]
        if "head.bias" in state and "fc.bias" not in state:
            state["fc.bias"] = state["head.bias"]

    return state


def _get_head_param_names(model: torch.nn.Module) -> set[str]:
    head = set()
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        head |= {"fc.weight", "fc.bias"}
    if hasattr(model, "classifier"):
        cls = getattr(model, "classifier")
        if (
            isinstance(cls, nn.Sequential)
            and len(cls) >= 2
            and isinstance(cls[1], nn.Linear)
        ):
            head |= {"classifier.1.weight", "classifier.1.bias"}
    return head


def _estimate_match_fraction(model: torch.nn.Module, state: dict) -> float:
    """
    Change rationale (score improvement): determine whether we loaded a meaningful amount of the checkpoint,
    without being overly penalized by BN tracking buffers or minor naming differences.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return 0.0
    model_sd = model.state_dict()
    matched = 0
    total = 0
    for mk, mv in model_sd.items():
        nk = _normalize_key_for_matching(mk)
        if nk == "":
            continue  # ignore BN num_batches_tracked
        total += 1
        if (
            mk in state
            and isinstance(state[mk], torch.Tensor)
            and state[mk].shape == mv.shape
        ):
            matched += 1
    return matched / max(1, total)


def safe_load_state_dict(
    model: torch.nn.Module, weight_path: str | None, device: torch.device
) -> bool:
    if not weight_path:
        return False
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        state = _strip_prefixes(state)

        if not isinstance(state, dict) or len(state) == 0:
            print(f"[WARN] Unrecognized/empty checkpoint format at {weight_path}")
            return False

        state = _remap_common_head_keys_for_torchvision(model, state)

        missing, unexpected = model.load_state_dict(state, strict=False)

        match_frac = _estimate_match_fraction(model, state)
        if match_frac < 0.15:
            print(
                f"[WARN] Very low tensor match fraction when loading {weight_path} "
                f"(match_frac={match_frac:.3f}). Treating as failed load."
            )
            return False

        head_names = _get_head_param_names(model)
        missing_set = set(missing) if isinstance(missing, (list, tuple)) else set()
        unexpected_set = (
            set(unexpected) if isinstance(unexpected, (list, tuple)) else set()
        )

        missing_set_filtered = {
            k for k in missing_set if not k.endswith("num_batches_tracked")
        }

        total_model_keys = len(
            [
                k
                for k in model.state_dict().keys()
                if not k.endswith("num_batches_tracked")
            ]
        )
        num_missing = len(missing_set_filtered)
        missing_ratio = num_missing / max(1, total_model_keys)

        non_head_missing = [k for k in missing_set_filtered if k not in head_names]
        non_head_missing_ratio = len(non_head_missing) / max(1, total_model_keys)

        if missing_ratio > 0.97 or non_head_missing_ratio > 0.95:
            print(
                f"[WARN] Too many missing keys (filtered) when loading {weight_path} "
                f"(missing_ratio={missing_ratio:.2f}, non_head_missing_ratio={non_head_missing_ratio:.2f}, "
                f"match_frac={match_frac:.3f}). Treating as failed load."
            )
            return False

        if missing_set_filtered:
            ms = list(missing_set_filtered)
            print(
                f"[WARN] Missing keys (filtered) from {weight_path}: {ms[:8]}{'...' if len(ms)>8 else ''}"
            )
        if unexpected_set:
            us = list(unexpected_set)
            print(
                f"[WARN] Unexpected keys from {weight_path}: {us[:8]}{'...' if len(us)>8 else ''}"
            )

        print(
            f"[INFO] Loaded custom weights (strict=False) from: {weight_path} (match_frac={match_frac:.3f})"
        )
        return True
    except Exception as e:
        print(f"[WARN] Failed to load weights from {weight_path}: {repr(e)}")
        return False




## === cell 13
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_weight_pref = (
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)
resnet_weight_path = find_weight_file(
    resnet_weight_pref, "cassava_leaf_best_model_fine_aug.pth"
)

loaded_resnet = safe_load_state_dict(resnet_model, resnet_weight_path, device)
if not loaded_resnet:
    print(
        "[INFO] ResNet custom weights not found/compatible; falling back to ImageNet pretrained weights."
    )
    resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_model = resnet_model.to(device=device, dtype=torch.float32)
resnet_model.eval()

print(
    f"ResNet weights: {'custom' if loaded_resnet else 'imagenet'}; path={resnet_weight_path}"
)



## === cell 14
efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff1_weight_pref = "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth"
eff1_weight_path = find_weight_file(eff1_weight_pref, "Eff_best5.pth")

loaded_eff1 = safe_load_state_dict(efficientnet_model_1, eff1_weight_path, device)
if not loaded_eff1:
    print(
        "[INFO] EfficientNet custom weights (Eff_best5) not found/compatible; falling back to ImageNet pretrained weights."
    )
    efficientnet_model_1 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
    efficientnet_model_1.classifier = nn.Sequential(
        nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
    )

efficientnet_model_1 = efficientnet_model_1.to(device=device, dtype=torch.float32)
efficientnet_model_1.eval()

print(
    f"EfficientNet_1 weights: {'custom' if loaded_eff1 else 'imagenet'}; path={eff1_weight_path}"
)



## === cell 15
efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff7_weight_pref = "/kaggle/input/effbest10temp/pytorch/default/1/Eff_best10.pth"
eff7_weight_path = find_weight_file(eff7_weight_pref, "Eff_best10.pth")

loaded_eff7 = safe_load_state_dict(efficientnet_model_7, eff7_weight_path, device)
if not loaded_eff7:
    print(
        "[INFO] EfficientNet custom weights (Eff_best10) not found/compatible; falling back to ImageNet pretrained weights."
    )
    efficientnet_model_7 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
    efficientnet_model_7.classifier = nn.Sequential(
        nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
    )

efficientnet_model_7 = efficientnet_model_7.to(device=device, dtype=torch.float32)
efficientnet_model_7.eval()

print(
    f"EfficientNet_7 weights: {'custom' if loaded_eff7 else 'imagenet'}; path={eff7_weight_path}"
)



## === cell 16
efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff8_weight_pref = "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth"
eff8_weight_path = find_weight_file(eff8_weight_pref, "Eff_best6.pth")

loaded_eff8 = safe_load_state_dict(efficientnet_model_8, eff8_weight_path, device)
if not loaded_eff8:
    print(
        "[INFO] EfficientNet custom weights (Eff_best6) not found/compatible; falling back to ImageNet pretrained weights."
    )
    efficientnet_model_8 = models.efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
    efficientnet_model_8.classifier = nn.Sequential(
        nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
    )

efficientnet_model_8 = efficientnet_model_8.to(device=device, dtype=torch.float32)
efficientnet_model_8.eval()

print(
    f"EfficientNet_8 weights: {'custom' if loaded_eff8 else 'imagenet'}; path={eff8_weight_path}"
)



## === cell 17
models_for_ensemble = [
    ("eff7", efficientnet_model_7),
    ("eff1", efficientnet_model_1),
    ("eff8", efficientnet_model_8),
    ("resnet", resnet_model),
]

ensemble_predictions = []
image_names = []

with torch.no_grad():
    for images, img_names in tqdm(test_loader, total=len(test_loader)):
        for image, img_name in zip(images, img_names):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            img_name = str(img_name)

            probs_sum = None
            for _, m in models_for_ensemble:
                p = tta_predict_single_model(
                    m, image, tta_transform, device, n_tta=num_tta
                )
                probs_sum = p if probs_sum is None else (probs_sum + p)

            combined_probs = probs_sum / float(len(models_for_ensemble))
            final_pred = combined_probs.argmax(dim=1).cpu().item()

            ensemble_predictions.append(int(final_pred))
            image_names.append(img_name)

len(ensemble_predictions), len(image_names), image_names[:3]



## === cell 18
pred_map = dict(zip(image_names, ensemble_predictions))

ordered_preds = []
missing = 0
for img_id in test_df["image_id"].tolist():
    if img_id in pred_map:
        ordered_preds.append(int(pred_map[img_id]))
    else:
        ordered_preds.append(0)
        missing += 1

if missing > 0:
    print(
        f"[WARN] Missing predictions for {missing} images after ordering; filled with 0."
    )

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].tolist(), "label": ordered_preds}
)
submission_df["label"] = submission_df["label"].astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(submission_df.head())
print(f"Submission file saved as '{submission_path}' with {len(submission_df)} rows.")
