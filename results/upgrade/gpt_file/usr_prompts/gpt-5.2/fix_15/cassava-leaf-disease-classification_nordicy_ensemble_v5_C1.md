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

0.8874282260501662

# 6. Current score

0.24589

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'Your current notebook likely fails to yield a Kaggle score because it depends on four external weight files under `/kaggle/input/...` that aren’t guaranteed to exist; if any of those paths are missing, inference stops and no `submission.csv` is produced. I keep your exact ensemble/inference logic, but add a minimal “safe weight loader” that (a) checks each weight file path and (b) if missing, falls back to the model’s random initialization so the notebook always finishes and writes a valid submission. I also fix small path robustness issues (mixed `../input` vs `/kaggle/input` and the accidental `//kaggle/...`) and ensure the submission is aligned to `sample_submission.csv` order (so `image_id,label` rows match exactly). These changes are aimed at producing a valid submission consistently; if your weight files are present, the score should match your intended performance (moving you toward the target).'
- What this solution (achieved 0.61099) has done: 'Your current 0.11024 is consistent with running the ensemble on randomly initialized weights, which likely happened because the checkpoint paths don’t exist in this environment. To move the accuracy up toward your 0.887 target while keeping the same model definitions and ensemble logic, I (1) make the checkpoint loader robust to common checkpoint formats (`state_dict`, `model`, or raw state dict) and strip `module.` prefixes, and (2) add an automatic search for the expected `.pth` filenames under `/kaggle/input` so the intended weights are actually found and loaded when present. I also fix the (currently unused) `resnet_model` by including it in the ensemble only if its weights successfully load, which improves score without changing the overall inference semantics. These are minimal, execution-safe changes that should substantially increase performance if the weights exist anywhere in the dataset inputs.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.61099) is far below the target (0.88743), so we should improve accuracy with minimal, low-risk changes that don’t alter your model definitions or ensemble logic. The biggest likely issue is a preprocessing mismatch: you always apply EfficientNet-style transforms (CLAHE + 384 resize + ImageNet normalize) but then feed the same tensors into ResNet50 when it’s loaded, which can significantly hurt the ResNet contribution and drag the ensemble down. I keep the same models and averaging scheme, but (1) add a ResNet-specific transform (224 resize, no CLAHE, ImageNet normalize) and run ResNet inference on that version only, and (2) ensure the loaded checkpoints are treated correctly by moving the model to device before loading (so GPU tensors in checkpoints won’t cause subtle issues). These changes preserve your overall inference semantics (softmax + equal-weight averaging) while making the ResNet branch consistent and typically improving the ensemble toward your target.'
- What this solution (achieved 0.61099) has done: 'I remove the hard failure when EfficientNet checkpoints can’t be loaded, because it stops execution before a submission is written. To keep the ensemble logic intact while ensuring end-to-end completion, I fall back to using the EfficientNet models with their current (possibly random) weights if no checkpoints are found, and I similarly keep ResNet optional as you intended. I also ensure `image_names`/`ensemble_predictions` are always defined so cell 15 can run, and I keep submission ordering aligned to `sample_submission.csv`. These changes are primarily for correctness/robustness; if checkpoints exist in `/kaggle/input`, the loader still finds and uses them and score should move up toward the target.'
- What this solution (achieved 0.61099) has done: 'Your current gap to the target is large (0.61099 vs 0.88743, higher-is-better), so the smallest meaningful improvement without changing core model/ensemble logic is to eliminate inference-time mismatches and ensure every loaded checkpoint is actually used correctly. I (1) switch EfficientNet preprocessing to the correct input size for `efficientnet_v2_s` (224) to better match how such checkpoints are typically trained, and (2) ensure model weights are loaded onto CPU first and then moved to GPU to avoid any subtle device/serialization incompatibilities. I also enable `torch.inference_mode()` (semantically identical to `no_grad()` for inference) and set deterministic seeds to stabilize results without changing the approach. The ensemble averaging, model definitions, softmax, and submission formatting remain unchanged.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.88743), so we should improve accuracy with the smallest changes that don’t alter your model/ensemble core logic. The most likely drag is that EfficientNet is being fed CLAHE-processed images that may not match how those checkpoints were trained; keeping the same models and averaging, we make the EfficientNet transform “checkpoint-friendly” by removing CLAHE and using standard ImageNet preprocessing only. We keep ResNet preprocessing unchanged and keep the exact same inference/softmax/mean-ensemble semantics. This should move performance upward toward the target while remaining stable and producing the same submission format.'
- What this solution (achieved 0.61099) has done: 'Your gap to the target is large (0.61099 vs 0.88743), so we should raise accuracy with minimal risk while preserving your exact model/ensemble structure. The most likely issue is that `safe_load_state_dict()` currently “filters by shape” and can silently drop many valid weights (especially if key names don’t align perfectly), causing partial/random initialization and a big accuracy hit. I change loading to first try a strict key-based load after cleaning common prefixes, and only fall back to shape-filtering if strict fails; this keeps the same models and inference logic but greatly increases the chance the intended checkpoints fully load. I also ensure models are moved to `device` regardless of load success (so inference never accidentally runs on CPU weights) and keep submission ordering identical to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, so the smallest safe way to move accuracy up (without changing your models/ensemble logic) is to ensure the checkpoints actually load fully and correctly. I adjust the checkpoint key-cleaning to also handle common `backbone.`/`net.` prefixes and also fix a frequent mismatch where EfficientNet checkpoints store the head under `classifier.1.*` but your `eff7/eff8` heads are `classifier.0/1.*` due to `nn.Sequential(Dropout, Linear)`. This preserves the exact architecture and inference averaging, but makes weight loading much more likely to succeed (avoiding partially-random heads that can cap accuracy around ~0.61). The rest of the pipeline (transforms, dataloaders, softmax+mean ensemble, and submission formatting) remains unchanged and it still always write `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current gap to the target is large (0.61099 vs 0.88743, higher-is-better), so the smallest likely win—without touching model architectures or the ensemble logic—is to fix remaining checkpoint-loading mismatches so you stop inadvertently running partially-random heads. I extend the EfficientNet classifier key remapping to handle both common directions (`classifier.1.*` ↔ `classifier.0.*`) and also handle `classifier.1.*` ↔ `classifier.0.1.*` patterns that appear when checkpoints wrap the head in an extra `Sequential`. I also add a tiny diagnostic that prints how many keys were loaded and whether head weights were loaded, so you can confirm the run is using real checkpoints (which should move accuracy up toward your target). The inference loop, transforms, softmax+mean averaging, and `submission.csv` format/order remain unchanged.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, so we should only make changes that increase the chance your intended pretrained checkpoints fully load and are actually used at inference, without changing the model architectures or ensemble averaging logic. The most likely remaining issue is that `safe_load_state_dict()` currently *rejects* otherwise-good partial loads when it can’t detect “head” keys (e.g., when a checkpoint uses `classifier.*` but your model uses `classifier.1.*` or vice versa), causing fallback to random heads and capping accuracy around ~0.61. I (1) improve EfficientNet head-key remapping to cover more real-world patterns, and (2) relax the “head_loaded” gate to accept shape-filtered loads when the backbone mostly loads (still legitimate, and much better than random), while keeping strict-load preference unchanged. This should move accuracy upward toward your target while keeping the same inference semantics and still always writing a valid `submission.csv`.'
- What this solution (achieved 0.24589) has done: 'Your current score is far below the target, so we should only make changes that increase the chance you’re using the intended (fully loaded) checkpoints and that inference matches how those checkpoints were trained, without changing any model architectures or ensemble averaging. The biggest likely accuracy drag here is the very strong `Dropout(p=0.8)` in `efficientnet_model_7`/`efficientnet_model_8`: calling `model.eval()` disables dropout, but many training setups with such heavy dropout implicitly rely on *MC Dropout at inference* (keeping dropout active) to get good results. I add a minimal MC-dropout inference just for the models that have `Dropout(p=0.8)` (eff7/eff8) by running several forward passes and averaging probabilities; this preserves your softmax+mean ensemble semantics and uses your existing `num_tta=5` variable (previously unused) without changing training/architecture. Everything else (checkpoint loading, transforms, dataloaders, submission format/order) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.24589) has done: 'Your current score (0.24589) is far below the target (0.88743), so we should make the smallest changes that plausibly increase accuracy without touching your model architectures or ensemble logic. The biggest low-risk issue is that MC-dropout is currently executed under `torch.inference_mode()`, which can interfere with “training-mode” stochastic layers; switching the forward section to `torch.no_grad()` preserves inference semantics (no gradients) while allowing dropout to behave normally. I also ensure we don’t ever stack an empty `probs_list` (which can happen if all EfficientNet checkpoints failed and some unexpected filtering occurs), and I keep everything else (checkpoint loading, transforms, softmax+mean averaging, submission ordering/format) identical.'
- What this solution (achieved 0.24589) has done: 'Your current score (0.24589) is far below the target (0.88743), so we should make the smallest changes that plausibly *increase real accuracy* without changing your models or ensemble logic. The biggest issue is that MC-dropout is currently only applied to eff7/eff8 while your `eff_models` list *excludes* any EfficientNet whose checkpoint didn’t load, which can easily leave you effectively predicting from a weak subset (or random weights) and tank accuracy. I keep the same architectures and softmax-mean ensembling, but (1) always include all three EfficientNet models in the ensemble while still tracking whether their checkpoints loaded (so a single missing checkpoint doesn’t collapse performance), and (2) run the ResNet inference under `torch.no_grad()` as well (matching the dropout-safe inference context you already use for EfficientNet). This preserves evaluation semantics and output format, but increases the chance you’re using the strongest available signals consistently.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import random
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
num_tta = (
    5  # previously unused; will be used for MC-dropout passes on heavy-dropout models
)



## === cell 3
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = os.path.join(DATA_ROOT, "test_images")



## === cell 4
test_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
test_df.head()



## === cell 5
efficientnet_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

resnet_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 7
test_dataset_eff = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)
test_loader_eff = DataLoader(
    test_dataset_eff, batch_size=32, shuffle=False, num_workers=0
)

test_dataset_res = CassavaTestDataset(
    test_df, test_image_dir, transform=resnet_transforms
)
test_loader_res = DataLoader(
    test_dataset_res, batch_size=32, shuffle=False, num_workers=0
)



## === cell 8
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 9
def find_ckpt_by_filename(filename, search_root="/kaggle/input"):
    for root, _, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        tensorish = any(torch.is_tensor(v) for v in ckpt_obj.values())
        if tensorish:
            return ckpt_obj
    return None


def _clean_state_dict_keys(state):
    cleaned = {}
    prefixes_to_strip = ("module.", "model.", "net.", "backbone.")
    for k, v in state.items():
        for pref in prefixes_to_strip:
            if k.startswith(pref):
                k = k[len(pref) :]
        cleaned[k] = v
    return cleaned


def _remap_efficientnet_classifier_linear_keys(model, state):
    if not hasattr(model, "classifier"):
        return state

    model_sd = model.state_dict()
    out = dict(state)

    def _try_remap(src_prefix, dst_prefix):
        changed = 0
        for k, v in list(out.items()):
            if k.startswith(src_prefix):
                new_k = dst_prefix + k[len(src_prefix) :]
                if (
                    new_k in model_sd
                    and torch.is_tensor(v)
                    and torch.is_tensor(model_sd[new_k])
                    and v.shape == model_sd[new_k].shape
                ):
                    out[new_k] = v
                    changed += 1
        return changed

    remap_pairs = [
        ("classifier.", "classifier.1."),
        ("classifier.weight", "classifier.1.weight"),
        ("classifier.bias", "classifier.1.bias"),
        ("classifier.0.", "classifier.1."),
        ("classifier.1.", "classifier.0."),
        ("classifier.0.1.", "classifier.1."),
        ("classifier.1.", "classifier.0.1."),
        ("head.", "classifier.1."),
        ("fc.", "classifier.1."),
    ]

    total = 0
    for s, d in remap_pairs:
        total += _try_remap(s, d)

    if total > 0:
        print(
            f"[INFO] EfficientNet classifier key remap applied: {total} tensors remapped."
        )
    return out


def _filter_state_dict_by_shape(model, state):
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_sd and torch.is_tensor(v) and v.shape == model_sd[k].shape:
            filtered[k] = v
    return filtered


def safe_load_state_dict(model, ckpt_path, device):
    if ckpt_path is None:
        print("[WARN] No checkpoint path provided; using randomly initialized weights.")
        model.to(device)
        return False

    ckpt_path = os.path.normpath(ckpt_path)
    if not os.path.exists(ckpt_path):
        print(
            f"[WARN] Checkpoint not found: {ckpt_path} -> using randomly initialized weights."
        )
        model.to(device)
        return False

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if state is None:
            raise ValueError(
                "Unrecognized checkpoint format (no state_dict/model found)."
            )

        state = _clean_state_dict_keys(state)
        state = _remap_efficientnet_classifier_linear_keys(model, state)

        try:
            model.load_state_dict(state, strict=True)
            sd_keys = set(model.state_dict().keys())
            head_keys = set()
            if hasattr(model, "fc"):
                head_keys |= {k for k in sd_keys if k.startswith("fc.")}
            if hasattr(model, "classifier"):
                head_keys |= {k for k in sd_keys if k.startswith("classifier.")}
            loaded_keys = set(state.keys())
            head_loaded = len(head_keys & loaded_keys) > 0
            print(
                f"[INFO] Loaded checkpoint (strict): {ckpt_path} | head_loaded={head_loaded}"
            )
            model.to(device)
            return True
        except Exception as e_strict:
            print(
                f"[WARN] Strict load failed ({type(e_strict).__name__}: {e_strict}); trying shape-filtered load."
            )

        filtered = _filter_state_dict_by_shape(model, state)
        if len(filtered) == 0:
            print(
                f"[WARN] No compatible keys found in checkpoint: {ckpt_path} -> using randomly initialized weights."
            )
            model.to(device)
            return False

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(
            f"[INFO] Loaded checkpoint (shape-filtered): {ckpt_path} | loaded_tensors={len(filtered)}"
        )
        if len(missing) > 0:
            print(f"[WARN] Missing keys (showing up to 5): {missing[:5]}")
        if len(unexpected) > 0:
            print(f"[WARN] Unexpected keys (showing up to 5): {unexpected[:5]}")

        sd_keys = set(model.state_dict().keys())
        loaded_keys = set(filtered.keys())

        head_keys = set()
        if hasattr(model, "fc"):
            head_keys |= {k for k in sd_keys if k.startswith("fc.")}
        if hasattr(model, "classifier"):
            head_keys |= {k for k in sd_keys if k.startswith("classifier.")}

        head_loaded = len(head_keys & loaded_keys) > 0
        coverage = len(filtered) / max(1, len(sd_keys))
        print(f"[INFO] head_loaded={head_loaded} | key_coverage={coverage:.3f}")

        if (not head_loaded) and (coverage < 0.85):
            print(
                f"[WARN] Low-coverage partial load and head not loaded: {ckpt_path} -> treating as failed."
            )
            model.to(device)
            return False

        model.to(device)
        return True
    except Exception as e:
        print(
            f"[WARN] Failed to load checkpoint at {ckpt_path} ({type(e).__name__}: {e}) -> using randomly initialized weights."
        )
        model.to(device)
        return False


def resolve_ckpt_path(preferred_path, expected_filename):
    preferred_path = os.path.normpath(preferred_path)
    if os.path.exists(preferred_path):
        return preferred_path
    found = find_ckpt_by_filename(expected_filename, search_root="/kaggle/input")
    if found is not None:
        print(f"[INFO] Resolved {expected_filename} via search: {found}")
        return found
    print(f"[WARN] Could not resolve {expected_filename} under /kaggle/input")
    return preferred_path




## === cell 10
resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_ckpt = resolve_ckpt_path(
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    "cassava_leaf_best_model_fine_aug.pth",
)
resnet_loaded = safe_load_state_dict(resnet_model, resnet_ckpt, device)
resnet_model.eval()



## === cell 11
efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)

eff1_ckpt = resolve_ckpt_path(
    "/kaggle/input/eff-t/pytorch/default/1/Eff.pth",
    "Eff.pth",
)
eff1_loaded = safe_load_state_dict(efficientnet_model_1, eff1_ckpt, device)
efficientnet_model_1.eval()



## === cell 12
efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff7_ckpt = resolve_ckpt_path(
    "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth",
    "Eff_best7.pth",
)
eff7_loaded = safe_load_state_dict(efficientnet_model_7, eff7_ckpt, device)
efficientnet_model_7.eval()



## === cell 13
efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff8_ckpt = resolve_ckpt_path(
    "/kaggle/input/eff-best-8/pytorch/default/1/Eff_best8.pth",
    "Eff_best8.pth",
)
eff8_loaded = safe_load_state_dict(efficientnet_model_8, eff8_ckpt, device)
efficientnet_model_8.eval()



## === cell 14
weight_efficientnet = 0.7  # unchanged (not used in original combination)
weight_resnet = 0.3  # unchanged (not used in original combination)

eff_models = [
    efficientnet_model_1.to(device),
    efficientnet_model_7.to(device),
    efficientnet_model_8.to(device),
]
print(
    f"[INFO] EfficientNet checkpoints loaded: eff1={eff1_loaded}, eff7={eff7_loaded}, eff8={eff8_loaded} "
    f"| Using {len(eff_models)} EfficientNet models in ensemble regardless."
)


def _is_mc_dropout_candidate(m: nn.Module) -> bool:
    if not hasattr(m, "classifier"):
        return False
    cls = getattr(m, "classifier")
    if (
        isinstance(cls, nn.Sequential)
        and len(cls) >= 1
        and isinstance(cls[0], nn.Dropout)
    ):
        return float(getattr(cls[0], "p", 0.0)) >= 0.79  # targets the p=0.8 models only
    return False


mc_dropout_models = [m for m in eff_models if _is_mc_dropout_candidate(m)]
standard_eff_models = [m for m in eff_models if m not in mc_dropout_models]

if len(mc_dropout_models) > 0:
    print(
        f"[INFO] Using MC-dropout passes (n={num_tta}) for {len(mc_dropout_models)} EfficientNet model(s)."
    )

probs_eff_sum = {}
counts_eff = {}

with torch.no_grad():
    for images, img_names in test_loader_eff:
        images = images.to(device)

        probs_list = []

        for m in standard_eff_models:
            m.eval()
            out = m(images)
            probs_list.append(F.softmax(out, dim=1))

        for m in mc_dropout_models:
            m.train()  # activates Dropout
            mc_probs = []
            for _ in range(int(num_tta)):
                out = m(images)
                mc_probs.append(F.softmax(out, dim=1))
            probs_list.append(torch.stack(mc_probs, dim=0).mean(dim=0))
            m.eval()  # restore eval state for safety

        if len(probs_list) == 0:
            raise RuntimeError(
                "No EfficientNet probabilities were produced (empty model list). Check checkpoint loading."
            )

        combined_eff = torch.stack(probs_list, dim=0).mean(dim=0)

        for i, name in enumerate(img_names):
            p = combined_eff[i].detach().cpu()
            if name not in probs_eff_sum:
                probs_eff_sum[name] = p.clone()
                counts_eff[name] = 1
            else:
                probs_eff_sum[name] += p
                counts_eff[name] += 1

if resnet_loaded:
    probs_res = {}
    with torch.no_grad():
        for images, img_names in test_loader_res:
            images = images.to(device)
            outputs_resnet = resnet_model(images)
            probs_batch = F.softmax(outputs_resnet, dim=1).detach().cpu()
            for i, name in enumerate(img_names):
                probs_res[name] = probs_batch[i]
else:
    probs_res = None

ensemble_predictions = []
image_names = list(test_df["image_id"].values)

for name in image_names:
    eff_p = probs_eff_sum[name] / float(counts_eff[name])
    if probs_res is not None and name in probs_res:
        combined = (eff_p + probs_res[name]) / 2.0
    else:
        combined = eff_p
    ensemble_predictions.append(int(torch.argmax(combined).item()))



## === cell 15
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map).astype(int)

if submission_df["label"].isna().any():
    missing = (
        submission_df.loc[submission_df["label"].isna(), "image_id"].head(10).tolist()
    )
    raise RuntimeError(f"Missing predictions for some test images. Examples: {missing}")

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())
