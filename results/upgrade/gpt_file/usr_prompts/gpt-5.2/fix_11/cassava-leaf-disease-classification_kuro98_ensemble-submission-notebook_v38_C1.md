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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.8987609549712904

# 6. Current score

0.3935

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23543) has done: 'The crash comes from how the DataLoader collates TTA outputs: it turns your per-sample list-of-T tensors into a list (length T) of batch tensors, so the stacking logic in cell 3 is incorrect. I fix cell 3 to handle the actual collated structure (list of T tensors shaped `[bs, C, H, W]`), flatten to `[T*bs, C, H, W]`, run the ensemble, then average over T. I also add a small safety fallback so if a filename is missing in the batch predictions (shouldn’t happen), it defaults to 0 rather than crashing, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.32997) has done: 'Your current score suggests the model is producing essentially random predictions, which is consistent with your inference-time TTA being *stochastic* and far too destructive (e.g., `RandomPerspective(p=1)` and full 180° affine/rotation) for cassava leaves; this wash out signal and collapse ensemble agreement. I keep the same models, averaging logic, and overall pipeline, but replace those extreme TTAs with a standard, label-preserving set (horizontal flip + small rotate + mild color jitter) while keeping deterministic per-image seeding. This should move accuracy upward substantially toward the 0.8988 target without changing the core architecture or training. I also keep the submission ordering and CSV writing unchanged to ensure a valid `submission.csv`.'
- What this solution (achieved 0.32997) has done: 'Your current gap to the target is large (0.32997 → 0.89876), and the most likely cause is that the model checkpoints are not actually being loaded (those `/kaggle/input/...` paths often don’t exist), so you’re effectively running untrained ImageNet backbones with random-ish heads. I keep the exact same ensemble, TTA averaging, and inference loop, but (1) automatically search `/kaggle/input` for the intended `.pt/.pth` checkpoint files and load them when present, and (2) fix ViT head replacement to match torchvision’s actual `heads` structure so state_dict loading works reliably. These are minimal changes that directly affect correctness of inference and should move accuracy substantially upward toward the target while preserving core logic and semantics. The submission writing/order stays identical.'
- What this solution (achieved 0.28326) has done: 'Your current score (0.32997) is far below the target (0.89876), so we should improve accuracy with the smallest change that preserves your ensemble and inference logic. The most likely remaining issue is a distribution mismatch: you’re normalizing with ImageNet stats but not using each model’s exact recommended preprocessing (resize/crop strategy and normalization), which can heavily degrade predictions even with correct checkpoints. I keep the same three models, averaging, and TTA structure, but switch to each torchvision weight’s `.transforms()` (which includes the correct resize/crop/normalization) and apply it per-model after your existing center-crop/resize step. I also make the TTA transforms deterministic and explicitly “safe” by removing color jitter (often harmful at test time) while keeping identity + horizontal flip + small rotation.'
- What this solution (achieved 0.33146) has done: 'Your current score (0.283) is far below the target (0.899), and the most likely remaining issue is that your inference-time resizing/cropping is not aligned with the pretrained weights’ expected preprocessing: you center-crop to 600 and then resize, but the weights’ `.transforms()` already include their own resize/center-crop logic and are being applied on top, causing a distribution mismatch. I make the smallest change that preserves your ensemble/TTA/inference loop: remove the manual `CenterCrop(600)` and manual per-model `Resize(...)`, and instead apply each model’s weights `.transforms()` directly to the original PIL image (with TTA applied before those transforms). I also make the rotation TTA deterministic and truly “small” by using a fixed ±10° rotation via `v2.functional.rotate` rather than `RandomRotation` (still the same TTA idea, but avoids RNG/seed quirks inside workers). This should materially increase accuracy toward the target while keeping the core logic (three-model ensemble + mean over TTAs) unchanged.'
- What this solution (achieved 0.33184) has done: 'Your score is far below the target, so we should make the smallest inference-time fixes that plausibly recover accuracy without changing the ensemble/core loop. The most likely remaining problem is a label-index mismatch between your checkpoints and the competition labels (very common with Cassava due to differing class order), which can yield near-random accuracy even when checkpoints load correctly. I add an optional, lightweight remapping step using the provided `label_num_to_disease_map.json` plus a canonical Cassava disease order; if the mapping can’t be inferred safely, it becomes a no-op (so it won’t break your current behavior). This keeps your models, TTA, averaging, and submission writing unchanged, but can substantially move accuracy upward toward the target when the checkpoint was trained with a different class order.'
- What this solution (achieved 0.3935) has done: 'I fix the crash by handling the fact that `weights.transforms()` returns a preset object (not a v2 `Compose` with a `.transforms` list), so we can’t iterate `base.transforms`. I implement a minimal normalization-override wrapper that leaves the preset’s resize/crop logic intact and only swaps the `Normalize` step by applying it after the preset output tensor. I also add a tiny robustness fix: ensure we always create a v2 `Normalize` compatible with tensor outputs, and keep the rest of your ensemble/TTA/inference logic unchanged so it runs end-to-end and writes `submission.csv`. This change is score-relevant only in the intended way (choosing between ImageNet vs 0.5/0.5 normalization) and removes the runtime error.'

# 9. Code solution

## === cell 0
import os
import random
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)
random.seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_DIR}/test_images/"
sample_path = f"{DATA_DIR}/sample_submission.csv"
label_map_path = f"{DATA_DIR}/label_num_to_disease_map.json"

model_a_img_size = 224  # kept for compatibility; actual sizing handled by transforms
model_b_img_size = 528
model_c_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True


def _find_checkpoint_path(preferred_path: str) -> str | None:
    p = Path(preferred_path)
    if p.exists():
        return str(p)

    fname = p.name
    search_root = Path("/kaggle/input")
    try:
        matches = list(search_root.rglob(fname))
    except Exception:
        matches = []

    if matches:
        matches = sorted(matches, key=lambda x: (len(str(x)), str(x)))
        return str(matches[0])

    stem = p.stem
    for ext in (".pt", ".pth", ".bin"):
        if ext == p.suffix:
            continue
        try:
            matches = list(search_root.rglob(stem + ext))
        except Exception:
            matches = []
        if matches:
            matches = sorted(matches, key=lambda x: (len(str(x)), str(x)))
            return str(matches[0])

    return None


def _load_model_or_fallback(
    model_path: str, arch: str, num_classes: int, device: torch.device
):
    """
    Keep same model ensemble logic; ensure checkpoints load robustly when present.
    """
    resolved_path = _find_checkpoint_path(model_path)
    if resolved_path is None:
        resolved_path = model_path
    p = Path(resolved_path)

    def _build_arch():
        if arch == "vit_b_16":
            m = torchvision.models.vit_b_16(
                weights=torchvision.models.ViT_B_16_Weights.DEFAULT
            )
            if hasattr(m, "heads") and isinstance(m.heads, torch.nn.Module):
                if (
                    hasattr(m.heads, "__len__")
                    and len(m.heads) > 0
                    and isinstance(m.heads[-1], torch.nn.Linear)
                ):
                    in_f = m.heads[-1].in_features
                    m.heads[-1] = torch.nn.Linear(in_f, num_classes)
                elif hasattr(m.heads, "head") and isinstance(
                    m.heads.head, torch.nn.Linear
                ):
                    in_f = m.heads.head.in_features
                    m.heads.head = torch.nn.Linear(in_f, num_classes)
                else:
                    in_f = getattr(m, "hidden_dim", 768)
                    m.heads = torch.nn.Sequential(torch.nn.Linear(in_f, num_classes))
            else:
                in_f = getattr(m, "hidden_dim", 768)
                m.heads = torch.nn.Sequential(torch.nn.Linear(in_f, num_classes))
        elif arch == "efficientnet_b4":
            m = torchvision.models.efficientnet_b4(
                weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT
            )
            m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
        else:
            m = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            )
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m

    if p.exists():
        ckpt = torch.load(str(p), map_location="cpu")
        if isinstance(ckpt, torch.nn.Module):
            m = ckpt
        elif isinstance(ckpt, dict):
            state = None
            for k in ("state_dict", "model_state_dict", "model"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]
                    break
            if state is None:
                state = ckpt
            m = _build_arch()
            cleaned = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                cleaned[nk] = v
            m.load_state_dict(cleaned, strict=False)
        else:
            m = _build_arch()

        m.to(device).eval()
        print(f"Loaded checkpoint: {p}")
        return m

    m = _build_arch()
    m.to(device).eval()
    print(
        f"WARNING: checkpoint not found, using ImageNet weights + random head: {model_path}"
    )
    return m


model_a = _load_model_or_fallback(
    "/kaggle/input/vit-v1/vit_v1.pt", "vit_b_16", num_classes, device
)
model_b = _load_model_or_fallback(
    "/kaggle/input/efficient-net/efficient_net.pt",
    "efficientnet_b4",
    num_classes,
    device,
)
model_c = _load_model_or_fallback(
    "/kaggle/input/vit-v6/vit_v6.pt", "resnet50", num_classes, device
)

sample_df = pd.read_csv(sample_path)
sample_image_ids = sample_df["image_id"].tolist()
sample_set = set(sample_image_ids)


def _build_optional_label_remap(label_map_json_path: str) -> list[int] | None:
    """
    Optional predicted index -> competition label remap.
    No-op unless we can infer confidently.
    """
    p = Path(label_map_json_path)
    if not p.exists():
        return None

    try:
        m = json.loads(p.read_text())
    except Exception:
        return None

    canonical = [
        "cassava bacterial blight",
        "cassava brown streak disease",
        "cassava green mottle",
        "cassava mosaic disease",
        "healthy",
    ]

    try:
        inv = {}
        for k, v in m.items():
            if not isinstance(k, str):
                k = str(k)
            if not isinstance(v, str):
                return None
            inv[int(k)] = v.strip().lower()
    except Exception:
        return None

    if set(inv.keys()) != {0, 1, 2, 3, 4}:
        return None

    disease_to_comp = {name: i for i, name in enumerate(canonical)}

    json_names = set(inv.values())
    if not json_names.issuperset(set(canonical)):
        synonyms = {
            "cbb": "cassava bacterial blight",
            "cbsd": "cassava brown streak disease",
            "cgm": "cassava green mottle",
            "cmd": "cassava mosaic disease",
        }
        normalized = set()
        inv2 = {}
        for k, name in inv.items():
            n = synonyms.get(name, name)
            inv2[k] = n
            normalized.add(n)
        if not normalized.issuperset(set(canonical)):
            return None
        inv = inv2

    remap = [0, 1, 2, 3, 4]
    for pred_idx in range(5):
        disease_name = inv.get(pred_idx, None)
        if disease_name is None or disease_name not in disease_to_comp:
            return None
        remap[pred_idx] = int(disease_to_comp[disease_name])

    if remap == [0, 1, 2, 3, 4]:
        return None
    return remap


LABEL_REMAP = _build_optional_label_remap(label_map_path)
if LABEL_REMAP is not None:
    print(
        "Applying predicted-label remap (pred_idx -> competition_label):", LABEL_REMAP
    )
else:
    print("No label remap applied (using raw argmax indices).")




## === cell 1
class CassavaDataset(VisionDataset):
    """Cassava test dataset with deterministic, label-preserving TTA.

    Core logic unchanged: return (A_inputs, B_inputs, C_inputs, filename), where inputs
    are either tensors (no TTA) or lists of tensors (TTA).
    """

    def __init__(
        self,
        data_dir,
        transform_a=None,
        transform_b=None,
        transform_c=None,
        ttas=None,
        image_ids=None,
        base_seed=3407,
    ):
        super().__init__(root=data_dir)
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.transform_c = transform_c
        self.ttas = ttas
        self.base_seed = int(base_seed)

        if image_ids is None:
            self.images = sorted([p.name for p in Path(data_dir).glob("*.jpg")])
        else:
            self.images = list(image_ids)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        if self.ttas is not None:
            model_a_list, model_b_list, model_c_list = [], [], []
            for j, t in enumerate(self.ttas):
                torch.manual_seed(self.base_seed + idx * 1000 + j)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(self.base_seed + idx * 1000 + j)

                img_t = t(img)

                a_img = (
                    self.transform_a(img_t) if self.transform_a is not None else img_t
                )
                b_img = (
                    self.transform_b(img_t) if self.transform_b is not None else img_t
                )
                c_img = (
                    self.transform_c(img_t) if self.transform_c is not None else img_t
                )

                model_a_list.append(a_img)
                model_b_list.append(b_img)
                model_c_list.append(c_img)

            return model_a_list, model_b_list, model_c_list, filename

        a_img = self.transform_a(img) if self.transform_a is not None else img
        b_img = self.transform_b(img) if self.transform_b is not None else img
        c_img = self.transform_c(img) if self.transform_c is not None else img
        return a_img, b_img, c_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
w_a = torchvision.models.ViT_B_16_Weights.DEFAULT
w_b = torchvision.models.EfficientNet_B4_Weights.DEFAULT
w_c = torchvision.models.ResNet50_Weights.DEFAULT


def _tta_identity(img):
    return img


def _tta_hflip(img):
    return v2.functional.horizontal_flip(img)


def _tta_rot_plus10(img):
    return v2.functional.rotate(
        img, angle=10.0, interpolation=InterpolationMode.BILINEAR, expand=False
    )


if tta:
    ttas = [_tta_identity, _tta_hflip, _tta_rot_plus10]
else:
    ttas = None


class _PresetWithPostNormalize(torch.nn.Module):
    def __init__(self, preset, normalize_mode: str):
        super().__init__()
        self.preset = preset
        if normalize_mode == "imagenet":
            self.post_norm = None
        elif normalize_mode == "half":
            self.post_norm = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    def forward(self, img):
        x = self.preset(img)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


def _weights_transforms_with_normalization_override(weights, normalize_mode: str):
    """
    Keep the same resize/crop logic from weights.transforms(); optionally override normalization.
    This fixes the AttributeError and preserves the core inference semantics.
    """
    preset = weights.transforms()
    if normalize_mode == "imagenet":
        return preset
    return _PresetWithPostNormalize(preset, normalize_mode)


def _run_inference_get_preds_and_confidence(
    transform_a, transform_b, transform_c, use_subset_n=None
):
    test_dataset = CassavaDataset(
        test_dir,
        transform_a=transform_a,
        transform_b=transform_b,
        transform_c=transform_c,
        ttas=ttas,
        image_ids=(
            sample_image_ids
            if use_subset_n is None
            else sample_image_ids[:use_subset_n]
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    normalizer = torch.nn.Softmax(dim=1)

    all_names = []
    all_preds = []
    conf_sum = 0.0
    n_seen = 0

    model_a.eval()
    model_b.eval()
    model_c.eval()

    with torch.no_grad():
        for model_a_inputs, model_b_inputs, model_c_inputs, filenames in test_loader:
            bs = len(filenames)

            if tta:
                if not isinstance(model_a_inputs, (list, tuple)):
                    raise TypeError(
                        f"Expected list/tuple for TTA inputs, got {type(model_a_inputs)}"
                    )
                T = len(model_a_inputs)

                a = torch.cat([x for x in model_a_inputs], dim=0).to(
                    device, non_blocking=True
                )  # [T*bs, C, H, W]
                b = torch.cat([x for x in model_b_inputs], dim=0).to(
                    device, non_blocking=True
                )
                c = torch.cat([x for x in model_c_inputs], dim=0).to(
                    device, non_blocking=True
                )

                out_a = model_a(a).view(T, bs, -1).mean(dim=0)
                out_b = model_b(b).view(T, bs, -1).mean(dim=0)
                out_c = model_c(c).view(T, bs, -1).mean(dim=0)

                outputs = (out_a + out_b + out_c) / 3.0
            else:
                a = model_a_inputs.to(device, non_blocking=True)
                b = model_b_inputs.to(device, non_blocking=True)
                c = model_c_inputs.to(device, non_blocking=True)

                out_a = model_a(a)
                out_b = model_b(b)
                out_c = model_c(c)
                outputs = (out_a + out_b + out_c) / 3.0

            probs = normalizer(outputs)
            max_probs, pred_idx = torch.max(probs, dim=1)
            pred_labels = pred_idx.tolist()

            if LABEL_REMAP is not None:
                pred_labels = [int(LABEL_REMAP[int(p)]) for p in pred_labels]

            all_names.extend(list(filenames))
            all_preds.extend(pred_labels)

            conf_sum += float(max_probs.sum().item())
            n_seen += int(max_probs.numel())

    avg_conf = conf_sum / max(1, n_seen)
    pred_map = dict(zip(all_names, all_preds))
    return pred_map, avg_conf


subset_n = 256  # small to keep runtime low; enough to detect gross mismatch
t_imagenet_a = _weights_transforms_with_normalization_override(w_a, "imagenet")
t_imagenet_b = _weights_transforms_with_normalization_override(w_b, "imagenet")
t_imagenet_c = _weights_transforms_with_normalization_override(w_c, "imagenet")

t_half_a = _weights_transforms_with_normalization_override(w_a, "half")
t_half_b = _weights_transforms_with_normalization_override(w_b, "half")
t_half_c = _weights_transforms_with_normalization_override(w_c, "half")

_, conf_imagenet = _run_inference_get_preds_and_confidence(
    t_imagenet_a, t_imagenet_b, t_imagenet_c, use_subset_n=subset_n
)
_, conf_half = _run_inference_get_preds_and_confidence(
    t_half_a, t_half_b, t_half_c, use_subset_n=subset_n
)

print(
    f"Normalization confidence on subset (n={subset_n}): imagenet={conf_imagenet:.6f}, half={conf_half:.6f}"
)

if conf_half > conf_imagenet:
    print("Chose normalization: mean=0.5,std=0.5 (timm-style)")
    transform_a, transform_b, transform_c = t_half_a, t_half_b, t_half_c
else:
    print("Chose normalization: ImageNet (torchvision weights default)")
    transform_a, transform_b, transform_c = t_imagenet_a, t_imagenet_b, t_imagenet_c

pred_map, _ = _run_inference_get_preds_and_confidence(
    transform_a, transform_b, transform_c, use_subset_n=None
)

ordered_preds = [int(pred_map.get(iid, 0)) for iid in sample_image_ids]

my_submission = pd.DataFrame({"image_id": sample_image_ids, "label": ordered_preds})
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
