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

0.8974010275007556

# 6. Current score

0.38266

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38266) has done: 'I fix the dataset root detection so it doesn’t incorrectly append `test_images` twice when you already pass the `.../test_images/` directory, which currently causes the “No image files found” crash and prevents `test_loader` from being created. I keep your inference/ensemble logic unchanged and only adjust the path resolution to robustly handle both “competition root” and “direct images folder” inputs. I also add a small safety check to print the resolved image root and ensure the submission merges correctly against `sample_submission.csv`. This should run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.38266) has done: 'Your current low score is consistent with your models not actually using the competition’s fine-tuned weights: the code tries to load from `/kaggle/input/vit-v1/...` etc., but those datasets typically don’t exist, so it silently falls back to ImageNet weights and a random new 5-class head, producing near-random predictions. I keep your exact ensemble/inference logic, but add a robust search that automatically finds `.pt/.pth` model files inside `/kaggle/input/` and loads them (handling common checkpoint formats) before falling back to torchvision defaults. I also ensure the classifier is replaced after loading (and reinitialized if shapes mismatch) so the forward pass stays valid while actually leveraging your provided weights when present. This should move accuracy substantially upward toward your target without changing architecture or inference semantics.'
- What this solution (achieved 0.38266) has done: 'Your current score suggests the loaded checkpoints (if any) aren’t being applied correctly to the exact model parameter names/shapes, so inference effectively uses mostly random/new heads. I keep your ensemble and inference exactly the same, but make checkpoint loading more robust by (1) filtering out classifier/head weights when their shapes don’t match your 5-class head, and (2) automatically remapping common `timm`-style keys (e.g., `head.weight`/`head.bias`) onto torchvision ViT (`heads.head.*`) so fine-tuned weights actually load. This should substantially increase accuracy toward your target without changing architecture, transforms, averaging, or prediction logic. I also ensure we print how many keys were loaded so you can confirm weights are truly applied.'
- What this solution (achieved 0.38266) has done: 'Your score is far below the target, and the most likely cause (given your current logic) is that at least one ensemble member is effectively untrained on Cassava due to a classifier/head mismatch: `_replace_classifier` doesn’t correctly replace the torchvision ViT head, so your ViT may still output 1000 ImageNet logits, making the ensemble nearly random. I make a minimal, score-relevant fix to `_replace_classifier` to correctly handle torchvision ViT (`model.heads.head`) and also make `_filter_mismatched_shapes` stricter so it drops keys that don’t exist in the current model (avoiding “unexpected” keys from interfering with loading). These changes keep your architecture/ensemble/inference semantics the same, but ensure checkpoints (or at least correct 5-class heads) are actually applied so accuracy moves toward the target. The rest of your pipeline (paths, transforms, averaging, submission merge) remains unchanged.'

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
print("device:", device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_a_img_size = 384
model_b_img_size = 528
model_c_img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _replace_classifier(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Module):
        if isinstance(model.classifier, torch.nn.Sequential):
            seq = list(model.classifier)
            for i in range(len(seq) - 1, -1, -1):
                if isinstance(seq[i], torch.nn.Linear):
                    in_f = seq[i].in_features
                    seq[i] = torch.nn.Linear(in_f, num_classes)
                    break
            model.classifier = torch.nn.Sequential(*seq)
        elif isinstance(model.classifier, torch.nn.Linear):
            in_f = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "heads"):
        heads = getattr(model, "heads")
        if hasattr(heads, "head") and isinstance(
            getattr(heads, "head"), torch.nn.Linear
        ):
            in_f = heads.head.in_features
            heads.head = torch.nn.Linear(in_f, num_classes)
            model.heads = heads
        elif isinstance(heads, torch.nn.Sequential):
            seq = list(heads)
            for i in range(len(seq) - 1, -1, -1):
                if isinstance(seq[i], torch.nn.Linear):
                    in_f = seq[i].in_features
                    seq[i] = torch.nn.Linear(in_f, num_classes)
                    break
            model.heads = torch.nn.Sequential(*seq)
        elif isinstance(heads, torch.nn.Linear):
            in_f = heads.in_features
            model.heads = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        in_f = model.head.in_features
        model.head = torch.nn.Linear(in_f, num_classes)
    return model


def _safe_construct(fallback_ctor, num_classes: int):
    """
    Keep existing behavior: try DEFAULT weights (if available offline), else weights=None.
    """
    try:
        model = fallback_ctor(weights="DEFAULT")
    except Exception as e:
        print(
            f"[WARN] Could not load DEFAULT weights ({type(e).__name__}: {e}). Using weights=None."
        )
        model = fallback_ctor(weights=None)
    model = _replace_classifier(model, num_classes)
    return model


def _infer_required_image_size(model: torch.nn.Module, default: int) -> int:
    """
    Keep existing behavior: ViT models may require fixed image_size.
    """
    sz = getattr(model, "image_size", None)
    if isinstance(sz, int) and sz > 0:
        return sz
    return default


def _discover_checkpoint(preferred_path: str, keywords: list[str]) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = ["/kaggle/input"]
    exts = (".pt", ".pth", ".bin")
    hits = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath)
            if base in {
                "train_images",
                "test_images",
                "train_tfrecords",
                "test_tfrecords",
            }:
                dirnames[:] = []
                continue
            for fn in filenames:
                lfn = fn.lower()
                if not lfn.endswith(exts):
                    continue
                score = 0
                for kw in keywords:
                    if kw.lower() in lfn:
                        score += 1
                if score > 0:
                    hits.append((score, os.path.join(dirpath, fn)))

    if not hits:
        return None
    hits.sort(key=lambda x: (-x[0], x[1]))
    return hits[0][1]


def _load_checkpoint_state(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return obj  # could be a full nn.Module


def _strip_state_dict_prefixes(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_sd[nk] = v
    return new_sd


def _remap_common_head_keys_for_torchvision(model: torch.nn.Module, sd: dict) -> dict:
    """
    Minimal, score-relevant fix: many fine-tuned checkpoints use timm-style keys:
      - head.weight / head.bias
      - classifier.weight / classifier.bias
    Torchvision ViT uses:
      - heads.head.weight / heads.head.bias
    """
    if not isinstance(sd, dict):
        return sd

    model_keys = set(model.state_dict().keys())
    out = dict(sd)

    if "heads.head.weight" in model_keys:
        if "head.weight" in out and "heads.head.weight" not in out:
            out["heads.head.weight"] = out["head.weight"]
        if "head.bias" in out and "heads.head.bias" not in out:
            out["heads.head.bias"] = out["head.bias"]
        if "classifier.weight" in out and "heads.head.weight" not in out:
            out["heads.head.weight"] = out["classifier.weight"]
        if "classifier.bias" in out and "heads.head.bias" not in out:
            out["heads.head.bias"] = out["classifier.bias"]

    return out


def _filter_mismatched_shapes(
    model: torch.nn.Module, sd: dict
) -> tuple[dict, int, int]:
    if not isinstance(sd, dict):
        return sd, 0, 0

    model_sd = model.state_dict()
    kept = {}
    dropped = 0
    for k, v in sd.items():
        if k not in model_sd:
            dropped += 1
            continue
        if hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) == tuple(model_sd[k].shape):
                kept[k] = v
            else:
                dropped += 1
        else:
            kept[k] = v
    return kept, len(kept), dropped


def _try_load_or_fallback(
    pt_path: str, fallback_ctor, num_classes: int, discover_keywords: list[str]
):
    chosen = _discover_checkpoint(pt_path, discover_keywords)

    if chosen is not None:
        print(f"[INFO] Using checkpoint: {chosen}")
        obj = _load_checkpoint_state(chosen)

        if isinstance(obj, torch.nn.Module):
            model = obj
            model = _replace_classifier(model, num_classes)
            return model

        if isinstance(obj, dict):
            model = _safe_construct(fallback_ctor, num_classes)

            sd = _strip_state_dict_prefixes(obj)
            sd = _remap_common_head_keys_for_torchvision(model, sd)

            sd, kept_n, dropped_n = _filter_mismatched_shapes(model, sd)

            try:
                missing, unexpected = model.load_state_dict(sd, strict=False)
                print(
                    f"[INFO] load_state_dict: kept={kept_n} dropped={dropped_n} missing={len(missing)} unexpected={len(unexpected)}"
                )
                if missing:
                    print(f"[WARN] Missing keys (truncated): {missing[:8]} ...")
                if unexpected:
                    print(f"[WARN] Unexpected keys (truncated): {unexpected[:8]} ...")
            except RuntimeError as e:
                print(
                    f"[WARN] load_state_dict RuntimeError: {e}. Falling back to torchvision model."
                )
                return _safe_construct(fallback_ctor, num_classes)

            return model

        print(
            "[WARN] Unrecognized checkpoint format; falling back to torchvision model."
        )
        return _safe_construct(fallback_ctor, num_classes)

    print(
        f"[WARN] Checkpoint not found for keywords={discover_keywords}; using torchvision fallback."
    )
    return _safe_construct(fallback_ctor, num_classes)


from torchvision import models as tvm

path_a = "/kaggle/input/vit-v1/vit_v1.pt"
path_b = "/kaggle/input/efficient-net/efficient_net.pt"
path_c = "/kaggle/input/vit-v6/vit_v6.pt"

model_a = _try_load_or_fallback(
    path_a,
    lambda weights="DEFAULT": tvm.vit_b_16(weights=weights),
    num_classes,
    discover_keywords=["vit", "b16", "vit_v1", "vit-v1"],
)
model_b = _try_load_or_fallback(
    path_b,
    lambda weights="DEFAULT": tvm.efficientnet_b0(weights=weights),
    num_classes,
    discover_keywords=["efficient", "efficientnet", "b0", "efficient_net"],
)
model_c = _try_load_or_fallback(
    path_c,
    lambda weights="DEFAULT": tvm.resnet18(weights=weights),
    num_classes,
    discover_keywords=["resnet18", "resnet_18", "resnet"],
)

model_a_img_size = _infer_required_image_size(model_a, model_a_img_size)
model_b_img_size = _infer_required_image_size(model_b, model_b_img_size)
model_c_img_size = _infer_required_image_size(model_c, model_c_img_size)
print(
    "Using image sizes:",
    {"a": model_a_img_size, "b": model_b_img_size, "c": model_c_img_size},
)

model_a.to(device)
model_b.to(device)
model_c.to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Args:
        data_dir: base directory or direct directory to images.
        transforms: set of transforms to be used.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        model_c_size,
        transform=None,
        ttas=None,
        img_size=384,
    ):
        data_dir = os.path.abspath(data_dir)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def _has_images(d):
            if not os.path.isdir(d):
                return False
            try:
                for n in os.listdir(d):
                    p = os.path.join(d, n)
                    if os.path.isfile(p) and os.path.splitext(n.lower())[1] in exts:
                        return True
            except Exception:
                return False
            return False

        if _has_images(data_dir):
            img_root = data_dir
        else:
            cand_test = os.path.join(data_dir, "test_images")
            cand_train = os.path.join(data_dir, "train_images")
            if _has_images(cand_test):
                img_root = cand_test
            elif _has_images(cand_train):
                img_root = cand_train
            else:
                img_root = data_dir  # will error with a clearer message below

        super().__init__(root=img_root)

        self.transform = transform
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_c = v2.Resize(
            (model_c_size, model_c_size), interpolation=InterpolationMode.BICUBIC
        )

        files = []
        if not os.path.isdir(self.root):
            raise RuntimeError(f"Image root directory does not exist: {self.root}")

        for name in os.listdir(self.root):
            p = os.path.join(self.root, name)
            if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
                files.append(name)
        self.images = sorted(files)

        if len(self.images) == 0:
            raise RuntimeError(
                f"No image files found under: {self.root}. "
                f"Please verify test_dir/train_dir points to a folder containing images."
            )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)
        model_c_img = self.resize_model_c(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
            model_c_img = [self.transform(t(model_c_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)
            model_c_img = self.transform(model_c_img)

        return model_a_img, model_b_img, model_c_img, filename

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
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    model_c_img_size,
    transform=test_transforms,
    ttas=ttas,
)
print("Resolved test image root:", test_dataset.root)
print("Num test images:", len(test_dataset))

use_workers = num_workers if os.name != "nt" else 0
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=use_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
model_c.eval()

with torch.no_grad():
    for batch_idx, (
        model_a_inputs,
        model_b_inputs,
        model_c_inputs,
        filenames,
    ) in enumerate(test_loader):
        bsz = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            model_c_inputs = torch.cat(model_c_inputs, dim=0).to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)
            model_c_outputs = model_c(model_c_inputs)

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bsz), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bsz), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            model_c_batch_logits = torch.stack(torch.split(model_c_outputs, bsz), dim=0)
            model_c_mean_logits = torch.mean(model_c_batch_logits, dim=0)

            outputs = (
                model_a_mean_logits + model_b_mean_logits + model_c_mean_logits
            ) / 3.0
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            model_c_inputs = model_c_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)
            model_c_outputs = model_c(model_c_inputs)

            outputs = (model_a_outputs + model_b_outputs + model_c_outputs) / 3.0
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

sample_sub = pd.read_csv(sample_sub_path)
pred_map = pd.DataFrame({"image_id": all_names, "label": all_preds}).drop_duplicates(
    "image_id"
)

my_submission = sample_sub[["image_id"]].merge(pred_map, on="image_id", how="left")
my_submission["label"] = my_submission["label"].fillna(0).astype(int)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
print("Missing predictions filled with 0:", int(my_submission["label"].isna().sum()))
print("Unique predicted labels:", sorted(my_submission["label"].unique().tolist()))
