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

0.8912058023572076

# 6. Current score

0.14761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.1577) has done: 'I fix the Albumentations v2 API mismatch that prevents transforms from constructing (the `RandomResizedCrop` transform now expects `size=(h,w)` instead of `height`/`width`). I also make the transform builder backward/forward compatible so it won’t crash if the environment changes, while keeping the same augmentation intent and inference loop. Finally, I ensure the submission is always written after successful inference and matches the required `image_id,label` format.'
- What this solution (achieved 0.1562) has done: 'Your current score (0.1577) is far below the target (0.8912), and the most likely cause is that the checkpoint is not being found/loaded, so you’re effectively running an ImageNet-pretrained EfficientNet with a random 5-class head (or a mismatched head), which yields near-random accuracy. I make the checkpoint search/load stricter and transparent, and ensure we correctly extract the `state_dict` from common checkpoint formats (e.g., `state_dict`, `model_state_dict`, PyTorch Lightning `state_dict`) so the trained weights actually load. I also enforce `strict=True` once the right state dict is identified (to avoid silently missing most weights), falling back to ImageNet only if loading truly fails. These are minimal changes that preserve your exact inference-time augmentation and TTA loop, but should move the score sharply upward toward the target.'
- What this solution (achieved 0.14761) has done: 'Your score is far below the target, so we should make the smallest changes that plausibly recover the intended trained-model performance rather than “improve” the model. The biggest remaining risk is that the checkpoint still isn’t being found/loaded correctly (or is loading a wrong file), so I (1) fix the checkpoint path to point to an existing location under the provided dataset tree and (2) make the checkpoint search prefer the known cassava dataset root. I also make inference-time augmentations deterministic (use a simple Resize+CenterCrop instead of random/photometric/dropout transforms) so the repeated “epochs” averaging becomes stable and acts like proper test-time averaging rather than injecting label-destroying noise. These changes preserve your core model (EfficientNet-B7 with a 5-class linear head) and your inference loop/averaging logic, but should move accuracy sharply upward toward the target.'
- What this solution (achieved 0.14761) has done: 'Your current score is near-random, which strongly suggests the intended trained checkpoint still isn’t being loaded, so the smallest score-improving change is to reliably locate an actual `.pt/.pth` file and load the correct tensor keys (handling common prefixes like `model.`/`net.` as well as `module.`). I keep your exact model (torchvision EfficientNet-B7 + 5-class linear head), your exact inference loop/averaging semantics, and your deterministic inference transforms, but make checkpoint discovery more robust by searching the whole dataset tree for plausible checkpoints (not just the single hardcoded name). I also print a concise load report (matched/missing/unexpected) so you can verify weights are really applied; if we still can’t find any checkpoint, it fall back to ImageNet as before and still produce a valid `submission.csv`. These changes are directly aimed at moving accuracy sharply upward toward the target by restoring the correct learned weights rather than “optimizing” the model.'
- What this solution (achieved 0.14761) has done: 'Your score is near-random versus the target, so the smallest likely fix is to ensure you are actually using fine-tuned weights rather than an ImageNet model with a randomly initialized 5-class head. I keep your exact model and inference averaging loop, but (1) expand checkpoint discovery to include common weight filenames shipped with many notebooks (e.g., `best.pth`, `model.pth`, etc.) and (2) add a small “sanity check” that rejects checkpoints that don’t contain the EfficientNet classifier head for 5 classes, preventing accidental auto-selection of unrelated `.pth/.pt` files. Finally, I make the cudnn flags consistent with determinism (disable benchmark when deterministic=True) to avoid run-to-run variance while keeping semantics unchanged.'
- What this solution (achieved 0.14761) has done: 'Your score is near-random vs the target, so the most likely issue is still “no real fine-tuned weights are being used”. I keep your exact EfficientNet-B7 model and inference averaging loop, but (1) fix the test image path to a location that actually exists in this environment, and (2) make checkpoint discovery explicitly prefer weight files inside the cassava dataset folder (and avoid scanning all of `/kaggle/input`, which can accidentally select unrelated checkpoints). I also add a tiny, non-invasive runtime check that prints the classifier weight norm after loading so you can verify the head isn’t left randomly initialized. These are minimal changes aimed at restoring intended weights/inputs, which should move accuracy sharply upward toward the target.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import random
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm

import torchvision



## === cell 1
image_size = 380



## === cell 2
DATA_ROOT = "/kaggle/data/cassava-leaf-disease-classification"
ALT_DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.isdir(DATA_ROOT) and os.path.isdir(ALT_DATA_ROOT):
    DATA_ROOT = ALT_DATA_ROOT

config = dict(
    seed=22,
    experiment_name="enb7",
    test_location=os.path.join(DATA_ROOT, "test_images"),
    checkpoint_path=DATA_ROOT,
    checkpoint="enb7best.pt",
    model="efficientnet-b7",
    epochs=10,
    batch_size=16,
    workers=8,
    inference_augmentations=[
        dict(name="Resize", params=dict(height=image_size, width=image_size, p=1.0)),
        dict(
            name="CenterCrop", params=dict(height=image_size, width=image_size, p=1.0)
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
    ],
)



## === cell 3
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample = pd.read_csv(sample_path)
test = sample[["image_id"]].copy()
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 5
seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("DATA_ROOT:", DATA_ROOT)
print(
    "Test images dir exists:",
    os.path.isdir(config["test_location"]),
    "|",
    config["test_location"],
)




## === cell 6
def _build_model_torchvision(use_imagenet_weights: bool = False):
    weights = (
        torchvision.models.EfficientNet_B7_Weights.DEFAULT
        if use_imagenet_weights
        else None
    )
    model = torchvision.models.efficientnet_b7(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 5)
    return model


def _strip_known_prefixes(state_dict: dict):
    """
    Change rationale (score toward target): if the checkpoint was saved under wrappers (DataParallel/Lightning)
    keys often have prefixes like 'module.' or 'model.'; stripping them allows strict loading of the real weights.
    """
    prefixes = ["module.", "model.", "net.", "backbone."]
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _extract_state_dict(ckpt_obj):
    """
    Change rationale (score toward target): robustly extract the actual model weights dict from common
    checkpoint formats so we don't silently run with mostly-random weights.
    """
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if len(ckpt_obj) > 0 and all(
            hasattr(v, "shape") or torch.is_tensor(v) for v in ckpt_obj.values()
        ):
            return ckpt_obj
    return None


def _looks_like_enb7_5class_state_dict(state: dict) -> bool:
    """
    Change rationale (score toward target): prevent auto-selecting unrelated .pth/.pt files.
    Only accept candidates that contain keys compatible with EfficientNet-B7's classifier and a 5-class head.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return False

    w_key = "classifier.1.weight"
    b_key = "classifier.1.bias"
    if w_key in state and b_key in state:
        w = state[w_key]
        b = state[b_key]
        try:
            return (
                torch.is_tensor(w)
                and torch.is_tensor(b)
                and w.ndim == 2
                and w.shape[0] == 5
                and b.ndim == 1
                and b.shape[0] == 5
            )
        except Exception:
            return False
    return False


def _load_state_dict_flex(model, ckpt_obj):
    state = _extract_state_dict(ckpt_obj)
    if state is None:
        raise ValueError(
            "Could not extract a state_dict from the checkpoint object. "
            "Expected keys like 'state_dict'/'model_state_dict'/'model'."
        )

    state = _strip_known_prefixes(state)

    if not _looks_like_enb7_5class_state_dict(state):
        raise ValueError(
            "Checkpoint state_dict does not look like EfficientNet-B7 with 5-class classifier "
            "(missing classifier.1.{weight,bias} with shape [5,*])."
        )

    try:
        model.load_state_dict(state, strict=True)
        print("Loaded checkpoint with strict=True.")
        return model
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            "WARNING: strict=True failed; loaded with strict=False.\n"
            f"RuntimeError: {e}\n"
            f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}"
        )
        if len(missing) > 0:
            print("Example missing keys:", missing[:5])
        if len(unexpected) > 0:
            print("Example unexpected keys:", unexpected[:5])
        return model


def _candidate_checkpoint_files(root: str):
    exts = {".pt", ".pth", ".ckpt", ".bin"}
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                out.append(os.path.join(dirpath, fn))
    return out


def _find_checkpoint_path(checkpoint_name: str, preferred_dir: str):
    """
    Change rationale (score toward target): avoid scanning all of /kaggle/input which can pick unrelated checkpoints.
    We strongly prefer cassava dataset roots only (where the intended fine-tuned weights should live).
    """
    if preferred_dir:
        p = os.path.join(preferred_dir, checkpoint_name)
        if os.path.isfile(p):
            return p

    search_roots = []
    for root in [
        preferred_dir,
        DATA_ROOT,
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
    ]:
        if root and os.path.isdir(root) and root not in search_roots:
            search_roots.append(root)

    for root in search_roots:
        p = os.path.join(root, checkpoint_name)
        if os.path.isfile(p):
            return p

    common_names = [
        "best.pth",
        "best.pt",
        "model.pth",
        "model.pt",
        "checkpoint.pth",
        "checkpoint.pt",
        "enb7.pth",
        "enb7.pt",
        "efficientnet_b7.pth",
        "efficientnet-b7.pth",
        "fold0.pth",
        "fold0.pt",
    ]
    for root in search_roots:
        for nm in common_names:
            p = os.path.join(root, nm)
            if os.path.isfile(p):
                return p

    all_ckpts = []
    for root in search_roots:
        all_ckpts.extend(_candidate_checkpoint_files(root))

    if len(all_ckpts) == 0:
        return None

    def score_path(p: str) -> int:
        name = os.path.basename(p).lower()
        s = 0
        if "enb7" in name or "b7" in name or "efficientnet" in name:
            s += 5
        if "cassava" in p.lower():
            s += 2
        if "best" in name:
            s += 3
        if name.endswith(".pt") or name.endswith(".pth"):
            s += 1
        try:
            sz = os.path.getsize(p)
            if sz > 50_000_000:
                s += 2
            elif sz > 5_000_000:
                s += 1
        except OSError:
            pass
        return s

    all_ckpts_sorted = sorted(all_ckpts, key=lambda p: (score_path(p), p), reverse=True)

    for chosen in all_ckpts_sorted[:50]:
        try:
            ckpt = torch.load(chosen, map_location="cpu")
            st = _extract_state_dict(ckpt)
            if st is None:
                continue
            st = _strip_known_prefixes(st)
            if _looks_like_enb7_5class_state_dict(st):
                print(
                    f"WARNING: Exact checkpoint name '{checkpoint_name}' not found. "
                    f"Auto-selected compatible checkpoint candidate (dataset roots only): {chosen}"
                )
                return chosen
        except Exception:
            continue

    chosen = all_ckpts_sorted[0]
    print(
        f"WARNING: No compatible checkpoint found during validation. "
        f"Falling back to top heuristic candidate (may fail to load): {chosen}"
    )
    return chosen


def load_model():
    ckpt_path = _find_checkpoint_path(
        config["checkpoint"], config.get("checkpoint_path")
    )

    if ckpt_path is None:
        print(
            f"WARNING: No checkpoint found (looked for '{config['checkpoint']}').\n"
            f"Falling back to ImageNet pretrained EfficientNet-B7 weights (architecture unchanged)."
        )
        model = _build_model_torchvision(use_imagenet_weights=True)
        model = model.to(device)
        model.eval()
        return model

    print("Found checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for k in ["epoch", "train_loss", "val_loss", "metrics", "lr"]:
            if k in checkpoint:
                print(f"{k}:", checkpoint[k])

    model = _build_model_torchvision(use_imagenet_weights=False)
    try:
        model = _load_state_dict_flex(model, checkpoint)
    except Exception as e:
        print(
            "WARNING: Failed to load weights from checkpoint; falling back to ImageNet weights.\n"
            f"Reason: {repr(e)}"
        )
        model = _build_model_torchvision(use_imagenet_weights=True)

    with torch.no_grad():
        w = model.classifier[1].weight.detach().float().cpu()
        print("Classifier weight norm (sanity):", float(w.norm().item()))

    model = model.to(device)
    model.eval()
    return model




## === cell 7
def _make_albu_transform(name: str, params: dict):
    cls = getattr(A, name)

    p = dict(params) if params is not None else {}

    if name == "RandomResizedCrop":
        if "size" not in p and ("height" in p and "width" in p):
            h, w = p.pop("height"), p.pop("width")
            p["size"] = (h, w)

        try:
            return cls(**p)
        except Exception:
            p2 = dict(params)
            if "size" in p2 and ("height" not in p2 and "width" not in p2):
                h, w = p2["size"]
                p2.pop("size")
                p2["height"] = h
                p2["width"] = w
            return cls(**p2)

    return cls(**p)


def get_transforms():
    transforms = [
        _make_albu_transform(item["name"], item["params"])
        for item in config["inference_augmentations"]
    ]
    comp = A.Compose(transforms)
    return comp




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        img_path = os.path.join(config["test_location"], self.images[n])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)  # HWC -> CHW
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __len__(self):
        return len(self.images)




## === cell 9
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"])
    data = CassavaDataset(test_data, transforms)

    workers = config["workers"]
    if device.type == "cpu":
        workers = 0

    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=(device.type == "cuda"),
        num_workers=workers,
    )
    return dataloader




## === cell 10
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            batch_hat = model(batch)
            predictions.append(batch_hat.detach().cpu())

    return torch.cat(predictions, dim=0)




## === cell 11
if __name__ == "__main__":
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dataloader = get_dataloader()
    model = load_model()
    predictions = None
    print("Inferring experiment", config["experiment_name"])

    for epoch in range(config["epochs"]):
        print("Epoch:", epoch)
        start_time = time.time()

        if epoch == 0:
            predictions = infer(model, dataloader)
        else:
            predictions += infer(model, dataloader)

        print("Time:", time.time() - start_time)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    predictions /= config["epochs"]
    results = predictions.numpy()
    test["label"] = np.argmax(results, axis=-1).astype(int)

    submission = sample[["image_id"]].merge(
        test[["image_id", "label"]], on="image_id", how="left"
    )
    if submission["label"].isna().any():
        missing = submission[submission["label"].isna()]["image_id"].head().tolist()
        raise RuntimeError(f"Missing predictions for some test images, e.g.: {missing}")

    submission["label"] = submission["label"].astype(int)
    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Wrote submission.csv with rows:", len(submission))
    print("Saved at:", os.path.abspath("submission.csv"))
