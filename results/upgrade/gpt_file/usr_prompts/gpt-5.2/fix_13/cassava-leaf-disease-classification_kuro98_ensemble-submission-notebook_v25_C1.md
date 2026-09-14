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

0.8931701420368692

# 6. Current score

0.63416

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14312) has done: 'I fix the dataset bug that accidentally includes the nested `test_images/` directory as an “image”, which causes `IsADirectoryError` and prevents `pred_map` (and thus the submission) from being created. I make the file listing robust by filtering to actual image files (and ignoring directories/hidden files), preserving ordering so predictions align with `sample_submission.csv`. I also add a safe fallback path for `test_dir` in case the directory structure differs across Kaggle mounts, and ensure the script always writes `submission.csv` with the required columns.'
- What this solution (achieved 0.60015) has done: 'We fix the centroid-fallback feature extractor to match the current torchvision ViT implementation: `vit_model.heads` is a `Sequential` without a `pre_logits` attribute, so we safely run the encoder and then apply `heads` (or identity) to get a consistent feature vector. This unblocks both centroid building and inference so `pred_map` is created and the submission write step can run. We also make the centroid sums/counts live on the same device to avoid hidden CPU/GPU mismatches. These are runtime/stability fixes and preserve the existing approach (centroid fallback when pretrained checkpoints aren’t available), enabling a valid `submission.csv`.'
- What this solution (achieved 0.60015) has done: 'Your score is far below the target (0.60015 vs 0.89317), and the biggest issue is that you’re currently using the centroid-fallback path because the pretrained competition checkpoints aren’t being found at the hardcoded `/kaggle/input/...` locations. I keep your inference logic intact, but add a minimal, robust checkpoint discovery that searches the provided dataset tree under `/kaggle/input/` (and `/kaggle/data/` as a fallback) for `vit_v1.pt`, `vit_boosted.pt`, and `linear_cls.pt`, then loads them if present so you use your intended ensemble instead of centroids. Additionally, to avoid accidental silent mismatch, I ensure the loaded models are real `nn.Module`s and move them to the right device, while keeping the same weighting/softmax/argmax semantics. This should increase accuracy substantially toward your target without changing the architecture or training approach.'
- What this solution (achieved 0.63191) has done: 'Your current score (0.60015) is far below the target (0.89317), so we should cautiously improve accuracy without changing your core approach. The main low-score driver is that your fallback path only uses ~8000 training images to build centroids and applies the ViT classifier head to features, which makes the “feature” space less suitable for cosine-similarity centroids. I (1) extract pre-head CLS embeddings for centroid building/inference (keeping the same ViT backbone and centroid logic), and (2) build centroids from the full training set (still just a single pass, no training loop changes) to reduce centroid noise. These are minimal, metric-aligned fixes that should move the score materially upward toward your target while preserving the solution’s semantics.'
- What this solution (achieved 0.63453) has done: 'Your score (0.63191) is far below the target (0.89317), so we should improve accuracy while keeping your overall centroid-fallback approach unchanged. The biggest likely issue is that the fallback uses ImageNet ViT features without any domain adaptation; a minimal, high-impact fix is to compute a simple “centroid-only calibration” on a held-out split of the training set and apply it at inference time (still nearest-centroid, just temperature scaling of cosine similarities). This preserves your model/backbone, feature extraction, and prediction semantics (argmax over class scores), but better aligns similarity scores to accuracy. I also make the train/val split deterministic and keep all file/path behavior unchanged, so the script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.63453) has done: 'Your score gap to the target is large (0.63453 vs 0.89317), and the dominant limiter is that you’re still in the centroid-fallback path, which ignores your intended ensemble and the `linear_head` you load. With minimal semantic change, I (1) robustly load checkpoints whether they were saved as full `nn.Module` objects or as `state_dict` dicts, and (2) if `linear_head` exists, apply it on top of ViT CLS embeddings (same backbone/features) during inference to produce class logits (instead of raw cosine similarities), which should move accuracy substantially upward toward the target. The fallback centroid calibration remains as a backup if `linear_head` cannot be used, and submission ordering/format stays identical.'
- What this solution (achieved 0.63453) has done: 'Your current score (0.63453) is far below the target (0.89317), so we should improve accuracy with minimal semantic changes while keeping your centroid/ensemble logic intact. The biggest likely issue is that `linear_head` is being instantiated inside the inference loop and (when it’s a dict) repeatedly reloaded and moved to device, which is error-prone and can silently fall back to weaker paths; we construct it once, correctly, and reuse it. We also ensure the linear head input dimension matches the CLS embedding size and only apply it when it’s valid, preserving your existing fallback behavior. These changes should move performance upward without changing the model architectures, training loops, feature extraction method, or loss.'
- What this solution (achieved 0.63416) has done: 'Your gap to the target is large (0.63453 → 0.89317), and the most likely blocker is that the “linear head” is either not being loaded into a real `nn.Module`, or it’s being applied on features that don’t match what it was trained on. I keep your ensemble and inference flow the same, but make the linear head loading robust (support full module / raw state_dict / state_dict with prefixes) and infer its expected input dimension from the checkpoint so we don’t silently use the wrong features. I also make `_vit_features` produce the same “CLS embedding” as torchvision’s ViT forward path (including `encoder.ln`) to better match typical training, without changing the backbone/architecture. These minimal fixes should move accuracy upward toward your target while keeping your overall approach intact and still producing a valid `submission.csv`.'
- What this solution (achieved 0.63416) has done: 'To move your score upward toward the 0.893 target without changing the core model/loop logic, the most likely missing piece is that your loaded ViT checkpoints are being used with a mismatched feature path: you extract CLS embeddings via `_vit_features(..., apply_head=False)`, but if the checkpoints were trained end-to-end with their own heads, the most reliable inference is to call `model(x)` directly (same architecture, same weights) and only use the linear head when you truly intend to replace the model head. I make a minimal change so that when pretrained `model_a/model_b` are available we use their native forward logits (and only fall back to CLS+linear_head when it’s explicitly present and compatible), while keeping the centroid fallback unchanged. I also ensure the linear head is only applied when its input dimension matches the extracted features, avoiding silent misapplication that can depress accuracy. Submission writing/order stays identical.'
- What this solution (achieved 0.63416) has done: 'I make two minimal, score-relevant fixes that should increase accuracy toward your 0.893 target without changing your overall architecture or training approach. First, your `_as_module` loader currently doesn’t strip common prefixes (like `module.`) before `load_state_dict`, which can cause silent partial loading and keep you effectively near the weak fallback behavior; I apply the same prefix-stripping you already use elsewhere when loading ViT checkpoints. Second, when a compatible `linear_head_module` exists, you currently (a) probe features with a relatively expensive forward each batch and (b) apply the same head to both model_a/model_b features even though their feature spaces may differ; I keep the same decision semantics but make it stable by probing once and then only using the external head on `model_a` features (and using native forward for `model_b`) to avoid mismatched logits depressing ensemble accuracy.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images/",
    "/kaggle/input/cassava-leaf-disease-classification/test_images/test_images/",
    "/kaggle/input/test_images/",
    "/kaggle/input/test_images/test_images/",
    "/kaggle/data/cassava-leaf-disease-classification/test_images/",
    "/kaggle/data/cassava-leaf-disease-classification/test_images/test_images/",
    "/kaggle/data/test_images/",
    "/kaggle/data/test_images/test_images/",
]
test_dir = None
for p in test_dir_candidates:
    if os.path.isdir(p):
        try:
            has_jpg = any(
                os.path.isfile(os.path.join(p, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(p)
            )
        except Exception:
            has_jpg = False
        if has_jpg:
            test_dir = p
            break
if test_dir is None:
    test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

sample_sub_path_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_sub_path = None
for p in sample_sub_path_candidates:
    if os.path.exists(p):
        sample_sub_path = p
        break
if sample_sub_path is None:
    sample_sub_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )

train_csv_path_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/train.csv",
]
train_csv_path = None
for p in train_csv_path_candidates:
    if os.path.exists(p):
        train_csv_path = p
        break

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _safe_torch_load(path, map_location):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location=map_location)
        return None
    except Exception:
        return None


def _find_checkpoint(filename, roots=("/kaggle/input", "/kaggle/data")):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".") and d not in ("__pycache__",)
            ]
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


vit_v1_path = "/kaggle/input/vit-v1/vit_v1.pt"
vit_boosted_path = "/kaggle/input/vit-boosted/vit_boosted.pt"
linear_head_path = "/kaggle/input/linear-head/linear_cls.pt"

if not os.path.exists(vit_v1_path):
    found = _find_checkpoint("vit_v1.pt")
    if found is not None:
        vit_v1_path = found
if not os.path.exists(vit_boosted_path):
    found = _find_checkpoint("vit_boosted.pt")
    if found is not None:
        vit_boosted_path = found
if not os.path.exists(linear_head_path):
    found = _find_checkpoint("linear_cls.pt")
    if found is not None:
        linear_head_path = found

print("Checkpoint paths:")
print(" vit_v1_path     =", vit_v1_path, "| exists:", os.path.exists(vit_v1_path))
print(
    " vit_boosted_path=",
    vit_boosted_path,
    "| exists:",
    os.path.exists(vit_boosted_path),
)
print(
    " linear_head_path=",
    linear_head_path,
    "| exists:",
    os.path.exists(linear_head_path),
)

model_a = _safe_torch_load(vit_v1_path, map_location=device)
model_b = _safe_torch_load(vit_boosted_path, map_location=device)
linear_head = _safe_torch_load(linear_head_path, map_location=device)

use_centroid_fallback = False
centroids = None  # shape [num_classes, D]
feat_dim = None

centroid_temp = 1.0

train_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images/",
    "/kaggle/input/cassava-leaf-disease-classification/train_images/train_images/",
    "/kaggle/input/train_images/",
    "/kaggle/input/train_images/train_images/",
    "/kaggle/data/cassava-leaf-disease-classification/train_images/",
    "/kaggle/data/cassava-leaf-disease-classification/train_images/train_images/",
    "/kaggle/data/train_images/",
    "/kaggle/data/train_images/train_images/",
]
train_dir = None
for p in train_dir_candidates:
    if os.path.isdir(p):
        try:
            has_jpg = any(
                os.path.isfile(os.path.join(p, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(p)
            )
        except Exception:
            has_jpg = False
        if has_jpg:
            train_dir = p
            break

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1


def _strip_state_dict_prefixes(sd: dict):
    if not isinstance(sd, dict):
        return sd
    prefixes = ("module.", "model.", "net.", "backbone.")
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _as_module(obj, arch_builder):
    if obj is None:
        return None
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        if isinstance(sd, dict):
            sd = _strip_state_dict_prefixes(sd)
            m = arch_builder()
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if len(unexpected) > 0:
                print(
                    "Warning: unexpected keys when loading state_dict:", unexpected[:5]
                )
            if len(missing) > 0:
                print("Warning: missing keys when loading state_dict:", missing[:5])
            return m
    return None


def _build_vit5():
    m = vit_b_16(weights=None)
    if hasattr(m, "heads") and m.heads is not None:
        if (
            isinstance(m.heads, torch.nn.Sequential)
            and len(m.heads) > 0
            and isinstance(m.heads[-1], torch.nn.Linear)
        ):
            in_f = m.heads[-1].in_features
            m.heads[-1] = torch.nn.Linear(in_f, num_classes)
        else:
            hidden = getattr(m, "hidden_dim", 768)
            m.heads = torch.nn.Linear(hidden, num_classes)
    return m


model_a = _as_module(model_a, _build_vit5)
model_b = _as_module(model_b, _build_vit5)

if (model_a is None) or (model_b is None):
    backbone = vit_b_16(weights=weights).to(device)
    backbone.eval()
    use_centroid_fallback = True
else:
    backbone = None

if not use_centroid_fallback:
    model_a = model_a.to(device)
    model_b = model_b.to(device)
    model_a.eval()
    model_b.eval()

    if linear_head is not None and isinstance(linear_head, dict):
        pass
    elif linear_head is not None and hasattr(linear_head, "to"):
        linear_head = linear_head.to(device)
        linear_head.eval()

print("use_centroid_fallback =", use_centroid_fallback)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data.

    Args:
        data_dir: base directory to the images.
        model_a_size/model_b_size: per-model input sizes.
        transform: transforms to be used (post resize/crop).
        ttas: optional list of transforms for test-time augmentation.
        image_list: optional explicit image_id ordering (used to match sample_submission exactly).
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        image_list=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        exts = (".jpg", ".jpeg", ".png")

        if image_list is not None:
            files = []
            for f in image_list:
                if not isinstance(f, str):
                    continue
                if f.startswith("."):
                    continue
                fp = os.path.join(data_dir, f)
                if os.path.isfile(fp) and f.lower().endswith(exts):
                    files.append(f)
            self.images = files
        else:
            files = []
            for f in os.listdir(data_dir):
                if f.startswith("."):
                    continue
                fp = os.path.join(data_dir, f)
                if os.path.isfile(fp) and f.lower().endswith(exts):
                    files.append(f)
            self.images = sorted(files)

        if len(self.images) == 0:
            raise RuntimeError(f"No image files found in test_dir={data_dir}")

        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_image_list = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_list=test_image_list,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)


def _vit_features(vit_model, x, apply_head: bool = True):
    vit_model.eval()
    with torch.no_grad():
        x = vit_model._process_input(x)
        n = x.shape[0]
        batch_class_token = vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = vit_model.encoder(x)
        if hasattr(vit_model.encoder, "ln") and vit_model.encoder.ln is not None:
            x = vit_model.encoder.ln(x)
        x = x[:, 0]  # CLS token, [B, hidden_dim]

        if apply_head and hasattr(vit_model, "heads") and vit_model.heads is not None:
            x = vit_model.heads(x)
    return x


def _infer_linear_in_dim_from_sd(sd: dict):
    if not isinstance(sd, dict):
        return None
    sd = _strip_state_dict_prefixes(sd)
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith("weight") and v.ndim == 2 and v.shape[0] == num_classes:
            return int(v.shape[1])
    return None


def _ensure_linear_head(linear_head_obj, in_dim: int):
    if linear_head_obj is None:
        return None
    if isinstance(linear_head_obj, torch.nn.Module):
        return linear_head_obj
    if isinstance(linear_head_obj, dict):
        sd = linear_head_obj.get("state_dict", linear_head_obj)
        if isinstance(sd, dict):
            sd = _strip_state_dict_prefixes(sd)
            m = torch.nn.Linear(in_dim, num_classes)
            m.load_state_dict(sd, strict=False)
            return m
    return None


if "use_centroid_fallback" in globals() and use_centroid_fallback:
    if train_csv_path is None or train_dir is None:
        raise RuntimeError(
            "Centroid fallback enabled but train.csv or train_images directory not found."
        )

    train_df_full = pd.read_csv(train_csv_path)

    g = torch.Generator().manual_seed(3407)
    labels_t = torch.tensor(train_df_full["label"].values, dtype=torch.long)
    calib_mask = torch.zeros(len(train_df_full), dtype=torch.bool)
    for c in range(num_classes):
        idx = (labels_t == c).nonzero(as_tuple=False).view(-1)
        if len(idx) == 0:
            continue
        perm = idx[torch.randperm(len(idx), generator=g)]
        take = max(1, int(0.1 * len(perm)))  # 10% per class for calibration
        calib_mask[perm[:take]] = True

    calib_df = train_df_full.loc[calib_mask.numpy()].reset_index(drop=True)
    build_df = train_df_full.loc[(~calib_mask).numpy()].reset_index(drop=True)

    class_sums = None
    class_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    train_ds = CassavaDataset(
        train_dir,
        model_a_img_size,
        model_b_img_size,
        transform=test_transforms,
        ttas=None,
        image_list=build_df["image_id"].tolist(),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    labels_map = dict(zip(build_df["image_id"].tolist(), build_df["label"].tolist()))

    backbone.eval()
    for model_a_inputs, _, filenames in train_loader:
        model_a_inputs = model_a_inputs.to(device, non_blocking=True)

        feats = _vit_features(backbone, model_a_inputs, apply_head=False)  # [B,D]
        if class_sums is None:
            feat_dim = feats.shape[1]
            class_sums = torch.zeros(num_classes, feat_dim, device=device)

        for i, fn in enumerate(filenames):
            y = int(labels_map[str(fn)])
            class_sums[y] += feats[i]
            class_counts[y] += 1

    class_counts_safe = class_counts.clamp(min=1).unsqueeze(1)
    centroids = class_sums / class_counts_safe
    centroids = torch.nn.functional.normalize(centroids, dim=1)
    print("Built centroids with counts:", class_counts.detach().cpu().tolist())

    calib_ds = CassavaDataset(
        train_dir,
        model_a_img_size,
        model_b_img_size,
        transform=test_transforms,
        ttas=None,
        image_list=calib_df["image_id"].tolist(),
    )
    calib_loader = DataLoader(
        calib_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    calib_labels_map = dict(
        zip(calib_df["image_id"].tolist(), calib_df["label"].tolist())
    )

    sims_list = []
    y_list = []
    backbone.eval()
    with torch.no_grad():
        for model_a_inputs, _, filenames in calib_loader:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            feats = _vit_features(backbone, model_a_inputs, apply_head=False)
            feats = torch.nn.functional.normalize(feats, dim=1)
            sims = feats @ centroids.T  # [B,C]
            sims_list.append(sims.detach().float().cpu())
            for fn in filenames:
                y_list.append(int(calib_labels_map[str(fn)]))

    sims_all = torch.cat(sims_list, dim=0)  # CPU float32
    y_all = torch.tensor(y_list, dtype=torch.long)

    logT = torch.tensor(0.0, requires_grad=True)  # T starts at 1.0
    opt = torch.optim.LBFGS([logT], lr=0.25, max_iter=25, line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad(set_to_none=True)
        T = torch.exp(logT).clamp(0.25, 8.0)
        logits = sims_all / T
        loss = torch.nn.functional.cross_entropy(logits, y_all)
        loss.backward()
        return loss

    opt.step(_closure)
    centroid_temp = float(torch.exp(logT).clamp(0.25, 8.0).detach().cpu().item())
    print("Fitted centroid temperature:", centroid_temp)

linear_head_module = None
linear_head_in_dim = None
if not ("use_centroid_fallback" in globals() and use_centroid_fallback):
    try:
        inferred_in_dim = None
        if isinstance(linear_head, dict):
            sd0 = linear_head.get("state_dict", linear_head)
            inferred_in_dim = (
                _infer_linear_in_dim_from_sd(sd0) if isinstance(sd0, dict) else None
            )
        if inferred_in_dim is None:
            inferred_in_dim = 768

        linear_head_module = _ensure_linear_head(linear_head, in_dim=inferred_in_dim)
        if linear_head_module is not None:
            linear_head_module = linear_head_module.to(device).eval()
            linear_head_in_dim = int(inferred_in_dim)
            print("Prepared linear_head_module with in_dim =", linear_head_in_dim)
        else:
            print("No usable linear_head_module; will prefer model forward logits.")
    except Exception as e:
        print("Failed to init linear_head_module:", repr(e))
        linear_head_module = None
        linear_head_in_dim = None

use_external_head_for_a = False
if not ("use_centroid_fallback" in globals() and use_centroid_fallback):
    if (
        "linear_head_module" in globals()
        and linear_head_module is not None
        and linear_head_in_dim is not None
    ):
        try:
            dummy = torch.zeros(1, 3, model_a_img_size, model_a_img_size, device=device)
            feat_probe = _vit_features(model_a, dummy, apply_head=False)
            use_external_head_for_a = int(feat_probe.shape[1]) == int(
                linear_head_in_dim
            )
        except Exception:
            use_external_head_for_a = False
    print("use_external_head_for_a =", use_external_head_for_a)



## === cell 3
all_names = []
all_preds = []

if "use_centroid_fallback" in globals() and use_centroid_fallback:
    backbone.eval()
else:
    model_a.eval()
    model_b.eval()
    if "linear_head_module" in globals() and linear_head_module is not None:
        linear_head_module.eval()
    elif linear_head is not None and hasattr(linear_head, "eval"):
        linear_head.eval()

with torch.no_grad():
    for _, (model_a_inputs, model_b_inputs, filenames) in enumerate(test_loader):
        if tta:
            batch_n = len(filenames)
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(
                device, non_blocking=True
            )
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(
                device, non_blocking=True
            )
            filenames = list(filenames)

            if "use_centroid_fallback" in globals() and use_centroid_fallback:
                feats = _vit_features(backbone, model_a_inputs, apply_head=False)
                feats = torch.nn.functional.normalize(feats, dim=1)
                sims = (feats @ centroids.T) / float(centroid_temp)  # calibrated
                sims_batch = torch.stack(torch.split(sims, batch_n), dim=0)
                sims_mean = torch.mean(sims_batch, dim=0)
                pred_labels = torch.argmax(sims_mean, 1).tolist()
            else:
                if use_external_head_for_a:
                    feats_a = _vit_features(model_a, model_a_inputs, apply_head=False)
                    logits_a = linear_head_module(feats_a)
                    logits_b = model_b(model_b_inputs)
                else:
                    logits_a = model_a(model_a_inputs)
                    logits_b = model_b(model_b_inputs)

                logits_a_batch = torch.stack(torch.split(logits_a, batch_n), dim=0)
                logits_a_mean = torch.mean(logits_a_batch, dim=0)

                logits_b_batch = torch.stack(torch.split(logits_b, batch_n), dim=0)
                logits_b_mean = torch.mean(logits_b_batch, dim=0)

                outputs = 0.9 * logits_a_mean + 0.1 * logits_b_mean
                preds = normalizer(outputs)
                pred_labels = torch.argmax(preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            if "use_centroid_fallback" in globals() and use_centroid_fallback:
                feats = _vit_features(backbone, model_a_inputs, apply_head=False)
                feats = torch.nn.functional.normalize(feats, dim=1)
                sims = (feats @ centroids.T) / float(centroid_temp)  # calibrated
                pred_labels = torch.argmax(sims, 1).tolist()
            else:
                if use_external_head_for_a:
                    feats_a = _vit_features(model_a, model_a_inputs, apply_head=False)
                    model_a_outputs = linear_head_module(feats_a)
                    model_b_outputs = model_b(model_b_inputs)
                else:
                    model_a_outputs = model_a(model_a_inputs)
                    model_b_outputs = model_b(model_b_inputs)

                outputs = 0.9 * model_a_outputs + 0.1 * model_b_outputs
                preds = normalizer(outputs)
                pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

pred_map = dict(zip(all_names, all_preds))



## === cell 4
ordered_preds = sample_sub["image_id"].map(pred_map)
ordered_preds = ordered_preds.fillna(0).astype(int)

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": ordered_preds}
)

assert len(my_submission) == len(sample_sub), (len(my_submission), len(sample_sub))
assert my_submission["image_id"].isna().sum() == 0
assert my_submission["label"].between(0, 4).all()

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
