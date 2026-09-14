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

0.8937745542459957

# 6. Current score

0.0725

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the missing model checkpoint path by loading the ViT weights from a local Kaggle dataset if present, and otherwise fall back to a torchvision ViT with a compatible 5-class head so inference can run end-to-end. I also fix inference-time issues that break submission validity: test DataLoader must not shuffle (to avoid duplicates/misalignment), and TTA currently uses random transforms (non-deterministic + inconsistent); I replace them with deterministic flip-based TTA while keeping the same “average TTA probabilities then argmax” core logic. Finally, I enforce submission alignment exactly to `sample_submission.csv` order and length, guaranteeing a valid `submission.csv` with correct columns.'
- What this solution (achieved 0.19283) has done: 'The crash comes from a mismatch between your preprocessing size (384) and the torchvision ViT fallback’s fixed `image_size=224`, which asserts on input height/width. I make the image size derive from the loaded model (224 for torchvision ViT; keep 384 only when the custom checkpoint supports it), so the existing transforms and inference loop stay the same but won’t error. I also make the TTA transforms deterministic by using always-on flips (instead of random flip modules) so results are stable run-to-run while preserving the same “average probabilities then argmax” logic. Finally, I keep the submission aligned to `sample_submission.csv` order and ensure `submission.csv` is written.'
- What this solution (achieved 0.19357) has done: 'The crash is due to `os.listdir(test_dir)` picking up a nested `test_images/` directory inside the provided path, so PIL tries to open a directory as an image; I filter the dataset file list to include only actual image files and (as a safety) also fall back to the `sample_submission.csv` image list to guarantee exact coverage/order. I also make the model loading more robust by ensuring we use the correct callable model (some checkpoints store dicts) without changing the core ViT inference logic. These changes are execution-unblocking and should also improve the score versus predicting many missing rows as class 0. Finally, the script still write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.19096) has done: 'Your score is far below target, so the smallest safe way to move upward is to fix likely inference/preprocessing mismatch rather than changing the model. The current `CenterCrop((600,600))` can cut off the leaf for many images and is also inconsistent with common ViT/ImageNet preprocessing; I replace it with `Resize` + `CenterCrop` to the model input size, preserving the same normalization and TTA averaging logic. I also add an automatic unwrapping step for common checkpoint formats (e.g., `state_dict`) to ensure your intended trained weights actually load into the ViT architecture instead of silently falling back or misloading. Finally, I keep submission alignment to `sample_submission.csv` identical and still write `submission.csv`.'
- What this solution (achieved 0.2201) has done: 'Your score is far below target, so the smallest safe improvement is to ensure the inference preprocessing exactly matches the ViT backbone’s expected ImageNet pipeline instead of a custom resize/crop that can shift accuracy. I switch the test transforms to use the official `ViT_B_16_Weights.IMAGENET1K_V1.transforms()` when the fallback torchvision ViT is used (and keep your existing logic otherwise), which preserves the same model/loop/TTA averaging while improving calibration and input scaling. I also make the checkpoint loading stricter when it’s a state_dict (avoid silently missing most keys) so you don’t accidentally run with near-random weights. The submission writing and sample_submission alignment remain identical.'
- What this solution (achieved 0.0725) has done: 'Your score is far below the target, so we should focus on a minimal but high-impact correctness fix rather than changing the model or training: right now the torchvision fallback replaces the 5-class head with a randomly initialized layer, which produce near-random predictions (≈0.20 accuracy) and matches your current score. I keep the exact same ViT backbone and inference/TTA logic, but load a *real* 5-class head by using a torchvision ViT checkpoint fine-tuned on this dataset if it exists; otherwise, I avoid random-head inference by keeping the original 1000-class head and mapping it deterministically to 5 classes (stable, legitimate, and usually better than random). I also make the DataLoader collate explicitly handle the TTA “list of tensors” case to avoid subtle shape issues across PyTorch versions, without changing the prediction semantics. Submission writing/order remain aligned to `sample_submission.csv` exactly.'
- What this solution (achieved 0.0725) has done: 'Your current score is far below target mainly because you’re effectively doing “ImageNet-to-5-class binning” when no real cassava-finetuned checkpoint is found, which behaves close to random for this task. The smallest high-impact improvement (without changing your model/loop/metric semantics) is to make checkpoint discovery reliably pick up an actual cassava 5-class ViT (common filenames don’t include “vit”), and to load it strictly enough that we don’t silently run with mismatched/random weights. Concretely, I (1) expand the checkpoint search to include common cassava fine-tune filenames and only accept checkpoints whose head outputs 5 classes, (2) improve dict-checkpoint unwrapping to handle common keys like `model_state` and Lightning `state_dict`, and (3) only use the ImageNet fallback mapping when we truly have no 5-class head available. This keeps the same ViT inference + deterministic flip TTA + “mean probs then argmax” logic, but should move accuracy upward toward the target.'
- What this solution (achieved 0.0725) has done: 'Your score is far below target, so we should make a minimal but high-impact correctness change: ensure the model used for inference is actually cassava-fine-tuned (5-class) rather than the ImageNet fallback with an arbitrary 1000→5 mapping, which tends to score near random. I tighten checkpoint discovery to preferentially load any state_dict that clearly contains a 5-class classifier head (or a full model whose head outputs 5), and I load it with `strict=True` when it’s a 5-class match to avoid silently running with mostly-missing weights. If no 5-class checkpoint is found, we keep your existing fallback behavior unchanged (so core logic stays the same), and submission alignment/output remain exactly as before.'
- What this solution (achieved 0.0725) has done: 'Your current score is far below target, and the most likely cause is that you’re still not actually loading a real cassava-finetuned 5-class ViT; the ImageNet fallback + 1000→5 grouping is effectively near-random for this dataset. I make checkpoint discovery/load stricter and more effective by (1) searching all `.pt/.pth` files (not only ones containing “vit/best/model”), (2) unwrapping common Lightning/EMA formats and also accepting `{'state_dict': ...}` that uses `model.*` prefixes, and (3) requiring a confident 5-class head match while loading with `strict=True` (and only falling back when no such checkpoint exists). This keeps your exact inference/TTA/argmax logic unchanged, but increases the chance you run the intended trained weights (the biggest lever toward your 0.89 target). Submission writing/order remains aligned to `sample_submission.csv` exactly and still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob

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

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True

vit_model = None
using_torchvision_fallback = False

ckpt_candidates = [
    "/kaggle/input/vit-v1-update/vit_v1_1.pt",
]
for p in ckpt_candidates:
    if os.path.exists(p):
        vit_model = torch.load(p, map_location=device)
        print(f"Loaded checkpoint: {p}")
        break


def _find_ckpt_candidates():
    patterns = [
        "/kaggle/input/**/*.pt",
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.bin",
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    matches = [m for m in matches if os.path.isfile(m)]

    def _rank(p):
        s = p.lower()
        score = 0
        for kw, w in [
            ("cassava", 50),
            ("leaf", 20),
            ("vit", 20),
            ("deit", 15),
            ("swin", 10),
            ("eff", 10),
            ("best", 15),
            ("fold", 10),
            ("epoch", 5),
            ("checkpoint", 5),
            ("model", 5),
        ]:
            if kw in s:
                score += w
        score -= len(s) * 0.001
        return -score

    matches = sorted(set(matches), key=_rank)
    return matches


def _infer_out_features_from_state_dict(sd: dict):
    for head_key in [
        "heads.head.weight",
        "head.weight",
        "classifier.weight",
        "fc.weight",
        "module.heads.head.weight",
        "module.head.weight",
        "module.classifier.weight",
        "module.fc.weight",
        "model.heads.head.weight",
        "model.head.weight",
        "model.classifier.weight",
        "model.fc.weight",
    ]:
        if head_key in sd and hasattr(sd[head_key], "shape"):
            return int(sd[head_key].shape[0])
    return None


def _extract_state_dict_from_obj(obj):
    """Minimal unwrapping for common checkpoint formats."""
    if not isinstance(obj, dict):
        return None
    for k in [
        "state_dict",
        "model_state_dict",
        "model_state",
        "model",
        "net",
        "weights",
        "model_ema",
        "ema_state_dict",
        "teacher",
    ]:
        if k in obj and isinstance(obj[k], dict):
            return obj[k]
    if all(isinstance(v, torch.Tensor) for v in obj.values()) and any(
        kk.endswith(".weight") for kk in obj.keys()
    ):
        return obj
    return None


def _strip_known_prefixes(sd: dict):
    prefixes = ("module.", "model.")
    out = {}
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _load_vit_state_dict_as_torchvision_vit(sd: dict, device, out_features: int):
    """
    Load a state_dict into a torchvision ViT_B_16 with the same classifier out_features.
    """
    from torchvision.models import vit_b_16

    sd = _strip_known_prefixes(sd)

    m = vit_b_16(weights=None)
    if (
        hasattr(m, "heads")
        and hasattr(m.heads, "head")
        and hasattr(m.heads.head, "in_features")
    ):
        m.heads.head = torch.nn.Linear(m.heads.head.in_features, int(out_features))

    m.load_state_dict(sd, strict=True)

    print(
        f"Loaded torchvision ViT_B_16 from state_dict: strict=True, out_features={out_features}"
    )
    return m


def _load_vit_from_checkpoint_obj(obj, device, num_classes=5):
    if hasattr(obj, "forward"):
        try:
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = obj(dummy)
            if out.shape[-1] == num_classes:
                return obj, False
        except Exception:
            return obj, False

    sd = _extract_state_dict_from_obj(obj)
    if sd is None:
        return None, False

    sd = _strip_known_prefixes(sd)
    out_features = _infer_out_features_from_state_dict(sd)

    if out_features != num_classes:
        return None, False

    try:
        m = _load_vit_state_dict_as_torchvision_vit(
            sd, device=device, out_features=out_features
        )
    except Exception as e:
        print(
            f"Found 5-class-looking state_dict but strict load failed: {type(e).__name__}: {e}"
        )
        return None, False

    return m, False


if vit_model is None:
    matches = _find_ckpt_candidates()
    loaded = False

    for mp in matches:
        try:
            obj = torch.load(mp, map_location=device)
        except Exception:
            continue

        m, _ = _load_vit_from_checkpoint_obj(obj, device, num_classes=num_classes)
        if m is not None:
            vit_model = m
            loaded = True
            print(f"Loaded usable 5-class ViT from: {mp}")
            break

    if not loaded:
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        using_torchvision_fallback = True
        print(
            "WARNING: No usable 5-class fine-tuned checkpoint found under /kaggle/input; using torchvision ViT_B_16 "
            "ImageNet weights with its original head (1000 classes) and a deterministic mapping to 5 cassava classes."
        )

imagenet_to_cassava = None

if isinstance(vit_model, dict):
    if "model" in vit_model and hasattr(vit_model["model"], "forward"):
        vit_model = vit_model["model"]
        print("Unwrapped checkpoint dict -> vit_model['model']")
    else:
        sd = _extract_state_dict_from_obj(vit_model)
        if sd is not None:
            sd = _strip_known_prefixes(sd)
            out_features = _infer_out_features_from_state_dict(sd)

            if out_features == num_classes:
                try:
                    _m = _load_vit_state_dict_as_torchvision_vit(
                        sd, device=device, out_features=out_features
                    )
                except Exception:
                    from torchvision.models import vit_b_16, ViT_B_16_Weights

                    _m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                    using_torchvision_fallback = True
                    print(
                        "WARNING: 5-class state_dict found but strict load failed; falling back to torchvision ImageNet ViT_B_16 weights."
                    )
                vit_model = _m
            else:
                from torchvision.models import vit_b_16, ViT_B_16_Weights

                vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                using_torchvision_fallback = True
                print(
                    "WARNING: dict checkpoint does not contain a clear 5-class head; falling back to torchvision ImageNet ViT_B_16 weights."
                )
        else:
            raise TypeError(
                "Loaded checkpoint is a dict but does not contain a usable 'model' or known state_dict key."
            )

if hasattr(vit_model, "image_size"):
    try:
        img_size = int(vit_model.image_size)
        print(f"Using model-enforced image_size={img_size}")
    except Exception:
        pass

vit_model.to(device)

if using_torchvision_fallback:

    def _build_imagenet_to_cassava_map(
        in_classes: int = 1000, out_classes: int = 5
    ) -> torch.Tensor:
        base = in_classes // out_classes
        rem = in_classes % out_classes
        groups = []
        start = 0
        for c in range(out_classes):
            size = base + (1 if c < rem else 0)
            idx = list(range(start, start + size))
            groups.append(idx)
            start += size
        M = torch.zeros((out_classes, in_classes), dtype=torch.float32)
        for c, idx in enumerate(groups):
            M[c, idx] = 1.0 / max(1, len(idx))  # average within group
        return M

    imagenet_to_cassava = _build_imagenet_to_cassava_map(1000, num_classes).to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data (returns image tensor(s), filename)."""

    def __init__(self, data_dir, transform=None, ttas=None, image_ids=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas

        if image_ids is not None:
            self.images = list(image_ids)
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            files = []
            for fn in os.listdir(data_dir):
                full = os.path.join(data_dir, fn)
                if os.path.isfile(full) and os.path.splitext(fn.lower())[1] in exts:
                    files.append(fn)
            self.images = sorted(files)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform is not None:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
if "using_torchvision_fallback" in globals() and using_torchvision_fallback:
    from torchvision.models import ViT_B_16_Weights

    test_transforms = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
else:
    test_transforms = v2.Compose(
        [
            v2.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

if tta:
    ttas = [
        v2.Identity(),
        v2.Lambda(lambda x: v2.functional.horizontal_flip(x)),
        v2.Lambda(lambda x: v2.functional.vertical_flip(x)),
    ]
else:
    ttas = None

sample_sub_for_ids = pd.read_csv(sample_sub_path)
test_ids = sample_sub_for_ids["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir, transform=test_transforms, ttas=ttas, image_ids=test_ids
)


def cassava_collate(batch):
    imgs, fns = zip(*batch)
    if isinstance(imgs[0], list):
        n_tta = len(imgs[0])
        stacked = [
            torch.stack([imgs[b][t] for b in range(len(imgs))], dim=0)
            for t in range(n_tta)
        ]
        return stacked, list(fns)
    else:
        return torch.stack(list(imgs), dim=0), list(fns)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=cassava_collate,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        if tta:
            inputs_cat = torch.cat(inputs, dim=0).to(
                device, non_blocking=True
            )  # [n_tta*B, C, H, W]
            logits = vit_model(inputs_cat)

            if imagenet_to_cassava is not None and logits.shape[-1] == 1000:
                logits = torch.matmul(logits, imagenet_to_cassava.t())  # [n_tta*B, 5]

            preds = normalizer(logits)  # [n_tta*B, num_classes]
            n_tta = len(ttas)
            bsz = len(filenames)

            preds = preds.view(n_tta, bsz, -1)  # [n_tta, B, num_classes]
            mean_preds = preds.mean(dim=0)  # [B, num_classes]
            pred_labels = mean_preds.argmax(dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            logits = vit_model(inputs)
            if imagenet_to_cassava is not None and logits.shape[-1] == 1000:
                logits = torch.matmul(logits, imagenet_to_cassava.t())
            preds = normalizer(logits)
            pred_labels = preds.argmax(dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print(f"Predicted {len(all_preds)} labels for {len(all_names)} images")



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
pred_df["label"] = pred_df["label"].fillna(0).astype(int)

pred_df.to_csv("submission.csv", index=False)
pred_df.head()



## === cell 5
pred_df.shape, pred_df["label"].value_counts().sort_index()
