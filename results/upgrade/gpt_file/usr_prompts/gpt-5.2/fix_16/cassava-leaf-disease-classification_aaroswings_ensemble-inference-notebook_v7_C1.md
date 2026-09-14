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

0.8771532184950136

# 6. Current score

0.52803

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24738) has done: 'I fix the immediate runtime issues by updating the Albumentations imports for v2.x and ensuring `torch` is imported before loading models. I also make model loading robust (so it works whether the pickle contains a full model or a state_dict) while keeping the ensemble inference logic the same. Next, I make the test file ordering deterministic and align the written `submission.csv` to `sample_submission.csv` so the output length and image_id set exactly match Kaggle’s expected answers. Finally, I keep the same transforms/prediction approach but ensure the pipeline always completes and writes a valid CSV.'
- What this solution (achieved 0.24514) has done: 'Your current score is consistent with a label–image_id mismatch rather than a weak model: the test dataset is iterated in filename-sorted order, but Kaggle expects predictions aligned exactly to `sample_submission.csv`’s `image_id` order; any missing/duplicated keys in `pred_map` (or filenames not matching due to path/encoding quirks) collapses accuracy toward chance. I make the test dataset index strictly follow `sample_submission.csv` (same list, same order) and read images by joining that id to `test_images/`, so every prediction is for the intended row. I also switch the ensemble to average *softmax probabilities* (instead of summing logits) to stabilize multi-model fusion without changing the underlying models or training. These are minimal inference-only changes that should move the score substantially upward toward your target.'
- What this solution (achieved 0.18087) has done: 'Your score looks like it’s being dragged down by a preprocessing mismatch: you are resizing to 600×600 and using Albumentations’ default `Normalize()` (mean=0, std=1), while the ResNeXt weights you load expect ImageNet normalization and a much smaller input size (typically 224). I keep the exact same ensemble inference logic and model loading, but switch the test transform to use the ImageNet mean/std and a 224 resize (plus center-crop) to match the pretrained backbone’s expected distribution. This is an inference-only change (no architecture/training changes) and is the smallest high-impact adjustment likely to move accuracy substantially toward your target. I also keep the submission alignment checks exactly as you already have.'
- What this solution (achieved 0.1917) has done: 'Your current score suggests the inference-time distribution still doesn’t match what your trained pickled models expect. I keep your exact ensemble logic and model definitions, but change only the test-time preprocessing to use the common “timm-style” evaluation pipeline: resize slightly larger then center-crop to 224 (instead of resizing directly to 224), which often fixes a big accuracy drop without touching architecture or weights. I also make the probability ensembling invariant to the number of models by averaging (divide by N) to keep logits→softmax calibration consistent (this should only improve or stabilize, not alter the core approach). Finally, I keep your strict alignment to `sample_submission.csv` unchanged to avoid any ordering-related score collapse.'
- What this solution (achieved 0.18087) has done: 'Your score (0.1917) is far below the target (0.877), so we should improve accuracy with minimal, inference-only changes. The biggest likely issue is a preprocessing mismatch: these pickled ResNeXt101 WSL-style models are typically trained with a *direct 224 resize* (no center-crop) and sometimes with slightly different normalization; your current “resize 256 then center-crop 224” can cut off disease regions and tank accuracy. I keep your exact ensemble logic and model loading, but switch the test transform to a simple `Resize(224,224)` + ImageNet normalize (still deterministic), and I also ensure the model forward pass runs under `torch.inference_mode()` to avoid any subtle grad-mode overhead without changing outputs. These are the smallest changes most likely to move the score materially upward toward the target.'
- What this solution (achieved 0.16517) has done: 'Your current score (0.18087) is far below the target (0.87715), so we should raise accuracy with minimal, inference-only changes. The most likely remaining issue is that these pickled ResNeXt101 WSL-style models were trained with a specific preprocessing pipeline; a common mismatch is using ImageNet mean/std when the training used the *WSL normalization* (mean/std = 0.5/0.5) or vice-versa, which can collapse accuracy toward chance. I keep your exact model architecture/loading and ensemble logic, but switch the test normalization to the common WSL-style `(0.5,0.5,0.5)` mean/std (still with 224 resize) and add a tiny safety fallback so grayscale/corrupt reads don’t crash the run. This is the smallest change that plausibly moves accuracy substantially upward toward your target without changing training or the ensemble method.'
- What this solution (achieved 0.18087) has done: 'Your score is close to random guessing, which strongly suggests an inference mismatch rather than a weak ensemble. The smallest high-impact fix without changing core model/ensemble logic is to correct the input normalization: `torchvision` ResNeXt101 default weights expect ImageNet mean/std, but your current pipeline uses WSL-style 0.5/0.5 which can collapse accuracy. I switch only the test-time `Normalize` to ImageNet stats (keeping the same 224 resize, loader order aligned to `sample_submission.csv`, and the same softmax-probability averaging). This should move accuracy substantially upward toward your target while preserving the architecture and inference semantics.'
- What this solution (achieved 0.1917) has done: 'Your current score is far below the target, so we should make a minimal inference-only adjustment that’s plausibly correcting a remaining preprocessing mismatch rather than changing the model/ensemble logic. The most common missing piece with ResNeXt-style PyTorch models is ensuring the tensor is in RGB with `float32` and *channel-first* (you already have that), plus matching the *exact* normalization and resize behavior used at training; since we already tried pure ImageNet normalize, the next smallest step is to add a deterministic “resize slightly larger then center-crop to 224” eval pipeline (widely used and often expected). I keep your dataset ordering aligned to `sample_submission.csv` and keep the same softmax-probability averaging ensemble. I also move `torch.inference_mode()` to wrap the whole loader loop (no semantic change) and add a strict assertion that every `image_id` was predicted (to prevent silent key mismatches).'
- What this solution (achieved 0.16517) has done: 'Your score is far below the target, which strongly suggests the models are being fed inputs that don’t match the preprocessing they were trained with. I keep your exact ensemble logic (softmax-prob averaging across the same loaded models) and only adjust the test-time transform to the most likely correct pipeline for these WSL-style ResNeXt pickles: a direct `Resize(224,224)` (no center-crop) plus the 0.5/0.5 normalization commonly used with WSL training. I also add a tiny safety step to ensure the image array is contiguous `uint8` before Albumentations (prevents rare dtype/layout issues) while keeping evaluation semantics unchanged. The rest of the code (ordering aligned to `sample_submission.csv`, CSV writing, assertions) stays the same to avoid reintroducing misalignment.'
- What this solution (achieved 0.18087) has done: 'Your current accuracy is near random (0.165), which most often indicates an inference preprocessing mismatch rather than a weak ensemble or ordering bug (your `sample_submission.csv` alignment is already correct). With minimal, inference-only changes, I switch the test-time normalization to standard ImageNet mean/std (commonly expected by `torchvision` ResNeXt backbones and many fine-tuned checkpoints) while keeping the same resize-to-224 and the same softmax-probability averaging ensemble. I also add a tiny safety cast to `float32` before softmax to avoid any mixed dtype edge cases without changing the model logic. This is the smallest likely high-impact adjustment to move the score upward toward your target.'
- What this solution (achieved 0.18087) has done: 'Your current score is near random, so the smallest likely high-impact fix is to match the *training-time preprocessing* of these ResNeXt101 WSL pickled checkpoints rather than changing any model/ensemble logic. I keep your exact model loading, architecture, DataLoader, and softmax-probability averaging ensemble, but adjust only the test-time normalization to the commonly used WSL-style mean/std (0.5/0.5) while keeping the same deterministic 224 resize. To avoid silently hurting accuracy when checkpoints were trained with a different normalization, I also generate a second submission using ImageNet normalization in the same run so you can submit both and pick the one closer to the target. Both submissions stay perfectly aligned to `sample_submission.csv` and are written as valid `.csv` files.'
- What this solution (achieved 0.1917) has done: 'Your current score is near chance, and your code is already correctly aligned to `sample_submission.csv`, so the most plausible remaining issue is that the models are being loaded incorrectly (e.g., state_dict key prefix mismatches like `model.` / `net.` / `module.`) or are not being found at all, causing you to run essentially untrained ImageNet backbones with random heads. I make model loading more robust by stripping a few common prefixes and by explicitly warning if none of the pickle paths exist (so you don’t silently fall back). I also keep your exact ensemble logic and preprocessing options, but add a third submission using a very common “timm-style” eval transform (Resize to 256 + CenterCrop 224 + ImageNet norm) because it’s a minimal inference-only change that often fixes a big accuracy gap for ResNeXt checkpoints without touching architecture/training. All outputs remain valid CSVs aligned to the sample submission.'
- What this solution (achieved 0.108) has done: 'I remove the hard assertion that stops execution when the external model pickle paths aren’t present, and instead fall back to a built-in torchvision ResNeXt101 model so the notebook always runs end-to-end and writes a valid `submission.csv`. To fix the downstream `NameError: models is not defined`, I ensure `models` is always created (either from available pickles or from the fallback model) and moved to eval/device. I keep your ensemble inference logic, ordering aligned to `sample_submission.csv`, and the three deterministic test-time preprocessing variants unchanged to preserve evaluation semantics while restoring a runnable pipeline. This also improve score versus a crash (and is the only safe, minimal way to “increase toward target” without access to the missing pickle inputs).'
- What this solution (achieved 0.52803) has done: 'Your current score (0.108) is far below the target (0.877), and your code now falls back to an untrained/random 5-class head when the pickled competition models aren’t available—this alone can keep accuracy near chance. The smallest score-moving change is to remove that harmful fallback by switching to a *valid but stronger* inference-only fallback: use a pretrained ImageNet ResNeXt101 and map its 1000-class predictions to 5 cassava classes via lightweight nearest-class prototypes computed from the training set (no extra training loop, same architecture, same preprocessing/inference semantics). This keeps everything deterministic, uses only provided data paths, and should move accuracy substantially upward toward the target without changing the ensemble logic when real pickles are present. I also keep your three submission variants, but ensure the fallback mapping is used only when no external checkpoints exist.'

# 9. Code solution

## === cell 0
import io
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
batch_size = 32

valid_input_size = 224

test_img_path = "../input/cassava-leaf-disease-classification/test_images"
train_img_path = "../input/cassava-leaf-disease-classification/train_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
model_fnames = [
    "../input/cassava-notebook-16-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-19-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-20-models/resnext101wsl_epoch_6.pickle",
]


def _build_base_model(num_classes: int = 5):
    m = torchvision.models.resnext101_32x8d(
        weights=torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT
    )
    in_features = m.fc.in_features
    m.fc = torch.nn.Linear(in_features, num_classes)
    return m


def _strip_known_prefixes(state_dict: dict):
    """
    Minimal robustness: strip wrapper/DDP prefixes.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ("module.", "model.", "net.", "encoder.", "backbone.")
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        new_state[nk] = v
    return new_state


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if len(obj) > 0 and all(hasattr(v, "shape") for v in obj.values()):
            return obj
    return None


def _remap_common_head_keys_to_fc(state: dict):
    """
    Score-critical minimal fix:
    If the checkpoint stores the classifier head under names like 'classifier.*' or 'head.*',
    remap them onto 'fc.*' so we load the trained head rather than leaving a random fc.
    """
    if not isinstance(state, dict):
        return state

    if any(k.startswith("fc.") for k in state.keys()):
        return state

    candidates = []
    for prefix in ("classifier.", "head.", "model.fc.", "net.fc."):
        if any(k.startswith(prefix) for k in state.keys()):
            candidates.append(prefix)

    if len(candidates) == 0:
        for prefix in ("module.classifier.", "module.head."):
            if any(k.startswith(prefix) for k in state.keys()):
                candidates.append(prefix)

    if len(candidates) == 0:
        return state

    src_prefix = candidates[0]
    remapped = dict(state)
    for k in list(state.keys()):
        if k.startswith(src_prefix):
            new_k = "fc." + k[len(src_prefix) :]
            if new_k not in remapped:
                remapped[new_k] = state[k]
    return remapped


def _load_model_any(path: str, num_classes: int = 5):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.nn.Module):
        return obj

    state = _extract_state_dict(obj)
    if isinstance(state, dict):
        m = _build_base_model(num_classes=num_classes)

        state = _strip_known_prefixes(state)
        state = _remap_common_head_keys_to_fc(state)

        missing, unexpected = m.load_state_dict(state, strict=False)

        head_loaded = ("fc.weight" in state) and ("fc.bias" in state)
        print(
            f"[INFO] Loaded {os.path.basename(path)} | head_loaded={head_loaded} | missing={len(missing)} unexpected={len(unexpected)}"
        )
        if len(unexpected) > 0:
            print(f"[WARN] Unexpected keys sample: {unexpected[:8]}")
        if len(missing) > 0:
            print(f"[WARN] Missing keys sample: {missing[:8]}")

        return m

    return _build_base_model(num_classes=num_classes)


available_model_paths = [p for p in model_fnames if os.path.exists(p)]

models = []
using_prototype_fallback = False

if len(available_model_paths) == 0:
    print(
        "[WARN] None of the provided model pickle paths exist in this environment.\n"
        "       Switching fallback to: ImageNet-pretrained ResNeXt101 + 5-class prototype mapping (no training loop).\n"
        "       This should be substantially better than a random 5-class head and move score toward the target."
    )
    using_prototype_fallback = True
    imagenet_model = torchvision.models.resnext101_32x8d(
        weights=torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT
    )
    models = [imagenet_model]
else:
    for p in available_model_paths:
        models.append(_load_model_any(p, num_classes=5))

models = [m.to(device).eval() for m in models]
print(f"Loaded {len(models)} model(s). CUDA: {torch.cuda.is_available()}")



## === cell 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

WSL_MEAN = (0.5, 0.5, 0.5)
WSL_STD = (0.5, 0.5, 0.5)

test_tfms_imagenet = A.Compose(
    [
        A.Resize(valid_input_size, valid_input_size, always_apply=True),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

test_tfms_wsl = A.Compose(
    [
        A.Resize(valid_input_size, valid_input_size, always_apply=True),
        A.Normalize(mean=WSL_MEAN, std=WSL_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

test_tfms_timm = A.Compose(
    [
        A.Resize(256, 256, always_apply=True),
        A.CenterCrop(valid_input_size, valid_input_size, always_apply=True),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, img_dir, image_ids, tfms):
        super(TestDataset, self).__init__()
        self.img_dir = Path(img_dir)
        self.image_ids = list(image_ids)
        self.tfms = tfms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        img_path = self.img_dir / filename

        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            img_pil = Image.new("RGB", (valid_input_size, valid_input_size), (0, 0, 0))

        img = np.ascontiguousarray(np.array(img_pil), dtype=np.uint8)
        img = self.tfms(image=img)["image"]
        return filename, img


sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["image_id"].tolist()




## === cell 4
def _iter_train_images_for_prototypes(
    train_csv_path: str,
    train_img_path: str,
    tfms,
    per_class_limit: int = 200,
    batch_size: int = 32,
):
    df = pd.read_csv(train_csv_path)
    df["label"] = df["label"].astype(int)

    selected = []
    for c in sorted(df["label"].unique().tolist()):
        sub = df[df["label"] == c].head(per_class_limit)
        selected.append(sub)
    sdf = pd.concat(selected, axis=0).reset_index(drop=True)

    class TrainSubsetDataset(Dataset):
        def __init__(self, df, img_dir, tfms):
            self.df = df
            self.img_dir = Path(img_dir)
            self.tfms = tfms

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            filename = row["image_id"]
            y = int(row["label"])
            img_path = self.img_dir / filename
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                img_pil = Image.new(
                    "RGB", (valid_input_size, valid_input_size), (0, 0, 0)
                )

            img = np.ascontiguousarray(np.array(img_pil), dtype=np.uint8)
            img = self.tfms(image=img)["image"]
            return img, y

    ds = TrainSubsetDataset(sdf, train_img_path, tfms)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def _get_backbone_and_head_for_imagenet_resnext101(m: torch.nn.Module):
    backbone = torch.nn.Sequential(*list(m.children())[:-1])  # -> (B, 2048, 1, 1)
    head = m.fc  # -> (B, 1000)
    return backbone, head


def _compute_prototypes_from_train(imagenet_model: torch.nn.Module, tfms):
    train_loader = _iter_train_images_for_prototypes(
        train_csv_path=train_csv_path,
        train_img_path=train_img_path,
        tfms=tfms,
        per_class_limit=200,
        batch_size=batch_size,
    )

    backbone, head = _get_backbone_and_head_for_imagenet_resnext101(imagenet_model)

    sums = None
    counts = np.zeros(5, dtype=np.int64)

    with torch.inference_mode():
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.numpy().astype(np.int64)

            feats = backbone(x).flatten(1)  # (B, 2048)
            logits1000 = head(feats).float()  # (B, 1000)

            logits1000_np = logits1000.detach().cpu().numpy()

            if sums is None:
                sums = np.zeros((5, logits1000_np.shape[1]), dtype=np.float64)

            for i in range(logits1000_np.shape[0]):
                c = int(y[i])
                sums[c] += logits1000_np[i]
                counts[c] += 1

    prototypes = sums / np.maximum(counts[:, None], 1)
    prototypes = prototypes.astype(np.float32)
    print(f"[INFO] Prototype counts per class: {counts.tolist()}")
    return prototypes


class EnsemblePredictor:
    def __init__(
        self,
        models,
        using_prototype_fallback: bool = False,
        prototypes: np.ndarray | None = None,
    ):
        super(EnsemblePredictor, self).__init__()
        self.models = models
        self.using_prototype_fallback = using_prototype_fallback
        self.prototypes = (
            prototypes  # (5, 1000) in ImageNet-logit space if fallback is used
        )

        if self.using_prototype_fallback:
            assert (
                self.prototypes is not None and self.prototypes.shape[0] == 5
            ), "Prototypes required for fallback mode."

    def predict_on_loader(self, loader):
        predictions = []
        filenames = []

        n_models = max(1, len(self.models))

        with torch.inference_mode():
            for file, img in loader:
                img = img.to(device, non_blocking=True)

                if self.using_prototype_fallback:
                    m = self.models[0]
                    backbone, head = _get_backbone_and_head_for_imagenet_resnext101(m)
                    feats = backbone(img).flatten(1)
                    logits1000 = head(feats).float().detach().cpu().numpy()  # (B, 1000)

                    X = logits1000
                    P = self.prototypes
                    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
                    sims = Xn @ Pn.T  # (B, 5)

                    preds = np.argmax(sims, axis=1).astype(int).tolist()
                    predictions += preds
                    filenames += list(file)
                    continue

                prob = None
                for model in self.models:
                    out = model(img).float()  # logits
                    p = torch.softmax(out, dim=1)  # probabilities
                    p = p.detach().cpu().numpy()
                    if prob is None:
                        prob = p
                    else:
                        prob += p

                prob = prob / n_models
                predictions += [int(np.argmax(x)) for x in prob]
                filenames += list(file)

        return predictions, filenames


def _run_inference_and_write(tfms, out_csv_name: str):
    test_ds = TestDataset(test_img_path, test_ids, tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=min(8, os.cpu_count() or 1),
        drop_last=False,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    prototypes = None
    if using_prototype_fallback:
        prototypes = _compute_prototypes_from_train(models[0], tfms=tfms)

    predictor = EnsemblePredictor(
        models, using_prototype_fallback=using_prototype_fallback, prototypes=prototypes
    )
    predictions, filenames = predictor.predict_on_loader(test_loader)

    pred_map = dict(zip(filenames, predictions))
    missing = [
        img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
    ]
    assert (
        len(missing) == 0
    ), f"Missing predictions for {len(missing)} image_ids, e.g. {missing[:5]}"

    aligned_preds = [
        int(pred_map[img_id]) for img_id in sample_sub["image_id"].tolist()
    ]
    submission_df = pd.DataFrame(
        {"image_id": sample_sub["image_id"], "label": aligned_preds}
    )
    submission_df.to_csv(out_csv_name, index=False)

    assert len(submission_df) == len(sample_sub)
    assert (submission_df["image_id"].values == sample_sub["image_id"].values).all()

    print(f"Wrote {out_csv_name} with rows:", len(submission_df))
    print(submission_df.head())
    return submission_df


_ = _run_inference_and_write(test_tfms_wsl, "submission.csv")
_ = _run_inference_and_write(test_tfms_imagenet, "submission_imagenet.csv")
_ = _run_inference_and_write(test_tfms_timm, "submission_timm.csv")
print(
    "Done. Created: submission.csv (WSL norm), submission_imagenet.csv (ImageNet norm), submission_timm.csv (Resize256+Crop224+ImageNet norm)."
)
