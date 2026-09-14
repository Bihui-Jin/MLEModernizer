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

2.7

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-image==0.25.2
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

0.8735267452402539

# 6. Current score

0.23468

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix the Albumentations v2 incompatibility by replacing the custom `ToTensor` object with the official `albumentations.pytorch.ToTensorV2`, which provides the required `available_keys` and returns a proper CHW torch tensor. I also remove the stochastic `HorizontalFlip` from test-time preprocessing because test augmentation makes predictions nondeterministic and typically hurts accuracy for a single-pass submission. Finally, I keep the rest of the model/inference logic intact, but ensure the dataloader and ID list are always created so the submission CSV is written end-to-end.'
- What this solution (achieved 0.26345) has done: 'Your low score is most consistent with the model weights not being loaded (the code silently runs with random EfficientNet weights when the `.pt` file path doesn’t exist), which would produce near-random accuracy. I keep the same EfficientNet-B4 architecture and inference loop, but add a strict check that the checkpoint exists and is actually loaded, and automatically search a few likely `/kaggle/input/**` locations for the `.pt` file to avoid path mismatches. I also print a small diagnostic (checkpoint path + missing/unexpected keys) so you can confirm the model is not running uninitialized. These are minimal changes that should move the score sharply upward toward your target by ensuring you’re using the intended trained weights.'
- What this solution (achieved 0.13677) has done: 'The runtime failure is because the script hard-requires a local `.pt` checkpoint that is not present in this Kaggle environment; I make the checkpoint loading robust by (1) searching for any compatible EfficientNet-B4 checkpoint under `/kaggle/input` and (2) if none is found, falling back to using ImageNet pretrained EfficientNet-B4 weights instead of random initialization (same architecture and inference loop, but much higher accuracy than random). I also add a small, safe input-size fallback so CenterCrop won’t crash on smaller images (resize then crop), without changing the model or training logic. Finally, I keep the submission ordering tied to `sample_submission.csv` and ensure `submission.csv` is always written.'
- What this solution (achieved 0.17377) has done: 'Your current score is far below the target, and the biggest likely cause is that you’re not actually using a properly trained cassava checkpoint (falling back to ImageNet weights won’t reach ~0.87). I keep your exact EfficientNet-B4 + single-pass inference logic, but make checkpoint discovery/load stricter and more compatible by (1) preferring checkpoints that clearly contain 5-class classifier weights and (2) supporting common wrapper formats (plain state_dict, `state_dict`, `model`, `model_state_dict`). I also add a minimal sanity check that the loaded classifier weight has shape `[5, in_features]` to avoid silently loading an incompatible `.pt` and producing near-random predictions. If no compatible checkpoint is found, the code still fall back to ImageNet pretrained weights and produce a valid submission as before.'
- What this solution (achieved 0.43647) has done: 'Your score is far below the target, and with your current “inference-only” setup the only realistic way to move toward ~0.87 is to ensure you’re actually using a real cassava-trained checkpoint (not ImageNet-pretrained or randomly initialized). I keep your EfficientNet-B4 model and single-pass inference identical, but tighten checkpoint discovery to only consider plausible cassava checkpoints (avoid accidentally selecting unrelated `.pt` files) and add support for the very common pattern where checkpoints store `classifier.weight`/`classifier.bias` (not `classifier.1.*`) by remapping those keys—this is a minimal compatibility fix that can immediately recover correct 5-class predictions if such a checkpoint exists. I also make the image transform robust by adding `A.Resize(512,512)` before the crop, preventing silent/edge-case cropping issues that can degrade predictions. These changes should increase accuracy substantially if the dataset contains the intended trained weights, while preserving your core logic and submission format.'
- What this solution (achieved 0.10501) has done: 'Your score gap to the target is large, so the most likely remaining issue is input preprocessing mismatch: EfficientNet-B4 in torchvision expects its own normalization statistics, not the generic ImageNet mean/std you’re using. I keep the same model/inference loop and just switch the test-time transform to the official EfficientNet_B4_Weights preprocessing (resize/crop + correct mean/std), which typically gives a big jump when using ImageNet-pretrained weights and also won’t hurt a properly trained checkpoint. I also make sure the final tensor dtype stays float32 (no semantic change) and keep the submission ordering exactly aligned to `sample_submission.csv`. Everything else (architecture, checkpoint discovery/loading, inference) stays intact.'
- What this solution (achieved 0.23468) has done: 'Your current score is far below the target, and with this inference-only pipeline the most likely reason is that you’re still not actually loading a strong cassava-trained checkpoint (so you’re effectively predicting with an ImageNet model whose 5-class head is random). I keep your EfficientNet-B4 architecture and single-pass inference intact, but I make checkpoint selection stricter and more correct by (1) requiring that the checkpoint contains a *valid 5-class head* and (2) preferring checkpoints whose keys match torchvision EfficientNet-B4 more broadly, instead of accepting a weak “compat score”. This is a minimal change that should substantially increase accuracy if any real cassava checkpoint exists in `/kaggle/input`, while preserving the same transforms, dataloader, and submission writing. I also make the checkpoint search deterministic (sorted) to avoid accidentally picking different files between runs.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import warnings
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage import io
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(SAMPLE_SUB_PATH), "sample_submission.csv not found at {}".format(
    SAMPLE_SUB_PATH
)
assert os.path.isdir(TEST_IMG_DIR), "test_images dir not found at {}".format(
    TEST_IMG_DIR
)




## === cell 1
class TestDataset(Dataset):
    def __init__(self, root_dir, image_ids, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_ids = list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, image_id)
        image = io.imread(img_path)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image_id, image




## === cell 2
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

effnet_weights = EfficientNet_B4_Weights.IMAGENET1K_V1
effnet_mean = tuple(float(x) for x in effnet_weights.transforms().mean)
effnet_std = tuple(float(x) for x in effnet_weights.transforms().std)

transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=512, interpolation=1, p=1.0),
        A.Resize(height=512, width=512, interpolation=1, p=1.0),
        A.CenterCrop(width=512, height=512, p=1.0),
        A.Normalize(
            mean=effnet_mean,
            std=effnet_std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].tolist()

test_ds = TestDataset(
    root_dir=TEST_IMG_DIR, image_ids=test_image_ids, transform=transform
)
testloader = DataLoader(
    test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_cuda
)



## === cell 3
model_full_name = "efficientnet-b4-e10"
folder_name = "effnetmodelb4-2"
expected_path = os.path.join("/kaggle/input", folder_name, model_full_name + ".pt")


def _unwrap_state_dict(obj):
    """
    Minimal robustness: support common checkpoint wrappers without changing model logic.
    """
    if not isinstance(obj, dict):
        return None

    for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    tensor_like = False
    for _, v in obj.items():
        if torch.is_tensor(v):
            tensor_like = True
            break
    if tensor_like:
        return obj

    return None


def _strip_prefixes(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v
    return new_state


def _remap_classifier_keys_to_torchvision_effnet_b4(state):
    """
    Minimal compatibility fix for common head key variants.
    """
    if state is None or not isinstance(state, dict):
        return state

    if ("classifier.weight" in state) and ("classifier.1.weight" not in state):
        state["classifier.1.weight"] = state["classifier.weight"]
    if ("classifier.bias" in state) and ("classifier.1.bias" not in state):
        state["classifier.1.bias"] = state["classifier.bias"]

    if ("classifier.fc.weight" in state) and ("classifier.1.weight" not in state):
        state["classifier.1.weight"] = state["classifier.fc.weight"]
    if ("classifier.fc.bias" in state) and ("classifier.1.bias" not in state):
        state["classifier.1.bias"] = state["classifier.fc.bias"]

    return state


def _has_valid_5class_head(state, in_features):
    """
    Change (score-improving): require an actually compatible 5-class head, otherwise
    we risk selecting a wrong/weak checkpoint and effectively running with a random head.
    """
    if state is None or not isinstance(state, dict):
        return False
    if "classifier.1.weight" not in state or "classifier.1.bias" not in state:
        return False
    w = state["classifier.1.weight"]
    b = state["classifier.1.bias"]
    if not (torch.is_tensor(w) and torch.is_tensor(b)):
        return False
    if not (w.ndim == 2 and b.ndim == 1):
        return False
    return (
        int(w.shape[0]) == 5
        and int(w.shape[1]) == int(in_features)
        and int(b.shape[0]) == 5
    )


def _score_ckpt_compat(state, ref_keys, in_features):
    """
    Change (score-improving): prefer checkpoints that match torchvision EfficientNet-B4 keys
    broadly AND have a valid 5-class head. This helps choose the real cassava-trained ckpt.
    """
    if state is None or not isinstance(state, dict):
        return -(10**9)

    score = 0

    if not _has_valid_5class_head(state, in_features=in_features):
        return -(10**8)

    score += 1000  # strong preference once head is valid

    common = len(set(state.keys()).intersection(ref_keys))
    score += common

    if len(state) < 200:
        score -= 200

    return score


base_weights = EfficientNet_B4_Weights.IMAGENET1K_V1

_ref_model = efficientnet_b4(weights=None)
_ref_model.classifier[1] = nn.Linear(_ref_model.classifier[1].in_features, 5)
ref_keys = set(_ref_model.state_dict().keys())
in_features = _ref_model.classifier[1].in_features
del _ref_model

candidates = []
if os.path.exists(expected_path):
    candidates.append(expected_path)

candidates += glob.glob(
    os.path.join("/kaggle/input", "**", model_full_name + ".pt"), recursive=True
)
candidates += glob.glob(
    os.path.join("/kaggle/input", "**", "*cassava*eff*net*.pt"), recursive=True
)
candidates += glob.glob(
    os.path.join("/kaggle/input", "**", "*cassava*.pt"), recursive=True
)
candidates += glob.glob(
    os.path.join("/kaggle/input", "**", "*eff*net*b4*.pt"), recursive=True
)
candidates += glob.glob(
    os.path.join("/kaggle/input", "cassava-leaf-disease-classification", "**", "*.pt"),
    recursive=True,
)
candidates += glob.glob(
    os.path.join("/kaggle/input", folder_name, "**", "*.pt"), recursive=True
)

candidates = sorted(set([p for p in candidates if os.path.isfile(p)]))

best_path = None
best_score = -(10**9)
best_state = None

for p in candidates:
    try:
        obj = torch.load(p, map_location="cpu")
        state = _unwrap_state_dict(obj)
        if state is None:
            continue
        state = _strip_prefixes(state)
        state = _remap_classifier_keys_to_torchvision_effnet_b4(state)
        sc = _score_ckpt_compat(state, ref_keys=ref_keys, in_features=in_features)
        if sc > best_score:
            best_score = sc
            best_path = p
            best_state = state
    except Exception:
        continue

use_pretrained = True
ckpt_path = None

if best_path is not None and best_score > 0:
    use_pretrained = False
    ckpt_path = best_path

weights = base_weights if use_pretrained else None
if use_pretrained:
    print(
        "No strictly compatible cassava 5-class checkpoint found under /kaggle/input; "
        "falling back to ImageNet pretrained EfficientNet-B4 (random 5-class head)."
    )
else:
    print("Using compatible checkpoint:", ckpt_path, "| compat score:", best_score)

model = efficientnet_b4(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
model = model.to(device)

if (ckpt_path is not None) and os.path.exists(ckpt_path):
    print("Loading checkpoint:", ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu")
    state = _unwrap_state_dict(obj)
    if state is None:
        raise RuntimeError(
            "Checkpoint was selected but could not be unwrapped into a state_dict: {}".format(
                ckpt_path
            )
        )
    state = _strip_prefixes(state)
    state = _remap_classifier_keys_to_torchvision_effnet_b4(state)

    if not _has_valid_5class_head(state, in_features=model.classifier[1].in_features):
        raise RuntimeError(
            "Selected checkpoint does not contain a valid 5-class head after remapping: {}".format(
                ckpt_path
            )
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        "Checkpoint load done. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )
    if len(missing) > 0:
        print("First missing keys:", missing[:5])
    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:5])

model.eval()



## === cell 4
names = []
predicted = []

with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device, non_blocking=True).to(torch.float32)
        output = model(images_batch)
        pred = torch.argmax(output, dim=1).detach().cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(pred.tolist())

pred_map = dict(zip(names, predicted))
ordered_pred = [int(pred_map[iid]) for iid in test_image_ids]



## === cell 5
result = pd.DataFrame({"image_id": test_image_ids, "label": ordered_pred})
result.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert list(result.columns) == ["image_id", "label"]
assert len(result) == len(sample_sub)
print(result.head())
print("Wrote submission.csv with", len(result), "rows")
