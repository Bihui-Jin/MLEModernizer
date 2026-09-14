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

0.8788153520701119

# 6. Current score

0.12407

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the missing `efficientnet_pytorch` dependency by switching to the installed `torchvision` EfficientNet-B4 and loading the provided `.pt` weights in a safe way (falling back cleanly if keys don’t match), so inference can run end-to-end and still follows the same “EfficientNet-B4 → 5-class logits → argmax” core logic. I also fix the Albumentations API error by replacing the removed `A.Flip` with `A.HorizontalFlip` while preserving the same intended augmentation behavior. Finally, I make the test image ordering deterministic (sorted file list) and ensure the submission is written with the correct `.csv` suffix and required columns.'
- What this solution (achieved 0.61099) has done: 'I fix the Albumentations v2 pipeline by replacing the custom `ToTensor` object (which lacks the required Albumentations interface) with `albumentations.pytorch.ToTensorV2`, keeping the same normalize→tensor behavior so batching works. I also fix the missing checkpoint path by searching for an existing `.pt` file under `/kaggle/input/` (including the nested dataset folders) and loading it robustly; if no checkpoint exists, the code still run end-to-end (but accuracy be low). Finally, I ensure the test DataLoader is defined before inference (so `testloader` exists) and that the produced `submission.csv` matches `sample_submission.csv` ordering and required columns.'
- What this solution (achieved 0.56428) has done: 'Your current gap to the target is large (0.61099 → 0.8788), so the most likely minimal, legitimate improvement is to ensure you are using the intended pretrained EfficientNet-B4 weights (ImageNet) before loading the competition fine-tuned checkpoint; without this, partial/mismatched checkpoint loading can leave many layers random and depress accuracy. I keep the same EfficientNet-B4 → 5-class logits → argmax inference core logic, but (1) initialize the backbone with torchvision’s ImageNet weights, (2) make checkpoint loading more robust by explicitly handling common wrapper keys (e.g., `model`, `net`) and reporting missing keys, and (3) use the standard EfficientNet-B4 inference resize+center-crop pipeline (resize to 512 then center-crop 512) to avoid accidental crop failures on smaller images while keeping semantics similar. Submission formatting and ordering remain aligned to `sample_submission.csv` as you already do.'
- What this solution (achieved 0.05531) has done: 'To move your score upward toward the 0.8788 target with minimal logic change, I focus on two high-impact correctness items: (1) match torchvision EfficientNet-B4’s expected ImageNet preprocessing exactly (its internal weights metadata provides the correct resize/crop and normalization), and (2) load the fine-tuned checkpoint more reliably by mapping common key patterns (especially when checkpoints are saved from a 5-class head with different classifier indexing or wrapped modules). These changes keep the same core “EfficientNet-B4 → 5 logits → argmax” inference semantics, but reduce the chance you’re effectively running with partially-random weights or slightly wrong normalization. Submission writing and ordering stay aligned to `sample_submission.csv`.'
- What this solution (achieved 0.07922) has done: 'Your current score (0.05531) is far below the target (0.8788), so the most likely issue is that you are not actually loading the intended fine-tuned weights and/or your preprocessing doesn’t match what the fine-tuned checkpoint expects. I keep the same core “EfficientNet-B4 → 5 logits → argmax” inference logic, but make checkpoint discovery stricter (prefer the exact `/kaggle/input/effnetmodelb4/efficientnet-b4-e10.pt` and avoid accidentally loading the wrong `.pt`) and fix preprocessing to reliably use EfficientNet-B4’s standard 380 center-crop pipeline (the previous code mistakenly used `min_size` and could pick 380 as a resize, which is wrong). These two minimal changes should substantially raise accuracy without changing your model architecture or inference semantics. Submission ordering/format remain aligned to `sample_submission.csv`.'
- What this solution (achieved 0.30904) has done: 'Your score is far below the target, so the most likely remaining issue is that the checkpoint is either not being found/loaded correctly or its weights aren’t actually being applied to the model’s parameters. I make checkpoint discovery more strict (so we don’t accidentally load an unrelated `.pt`) and improve state-dict unwrapping/remapping for common training wrappers while still keeping `strict=False` and the same EfficientNet-B4→5 logits→argmax logic. I also force test ordering to exactly match `sample_submission.csv` (instead of relying on directory sorting) to eliminate any residual ordering/duplication pitfalls. These are minimal changes aimed at a large legitimate accuracy jump toward your target without changing architecture, loss, or training loops.'
- What this solution (achieved 0.12407) has done: 'Your current score is far below the target, so the most likely remaining issue is that the fine-tuned checkpoint isn’t actually being found/loaded (you’re probably running mostly with ImageNet weights only). I make checkpoint discovery robust by searching specifically for the expected filename across `/kaggle/input/**` (not just under a single folder), and I add a small “sanity check” that warns if almost none of the model parameters were loaded (a strong sign of mismatch). I also align preprocessing to EfficientNet_B4_Weights’ exact transforms (resize/crop/mean/std) when available, but keep the same resize→center-crop→normalize→tensor core pipeline. Submission ordering/format remain exactly aligned to `sample_submission.csv`.'

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
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

model_full_name = "efficientnet-b4-e10"
model_name = "efficientnet-b4"
folder_name = "effnetmodelb4"

TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
PATH = "/kaggle/input/{}/{}.pt".format(folder_name, model_full_name)

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"




## === cell 1
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_list=None):
        self.root_dir = root_dir
        self.transform = transform

        if image_list is not None:
            self.images = list(image_list)
        else:
            self.images = sorted(
                [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        if self.transform:
            image = self.transform(image=image)["image"]

        return img_name, image




## === cell 2
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

try:
    backbone_weights = EfficientNet_B4_Weights.IMAGENET1K_V1
except Exception:
    backbone_weights = None

if backbone_weights is not None:
    t = backbone_weights.transforms()
    mean = tuple(float(x) for x in t.mean)
    std = tuple(float(x) for x in t.std)

    crop_size = None
    resize_size = None
    if hasattr(t, "crop_size") and t.crop_size is not None:
        cs = t.crop_size
        crop_size = int(cs[0]) if isinstance(cs, (list, tuple)) else int(cs)
    if hasattr(t, "resize_size") and t.resize_size is not None:
        rs = t.resize_size
        resize_size = int(rs[0]) if isinstance(rs, (list, tuple)) else int(rs)

    if crop_size is None:
        crop_size = 380
    if resize_size is None:
        resize_size = 384
else:
    crop_size = 512
    resize_size = 512
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

transform = A.Compose(
    [
        A.Resize(height=resize_size, width=resize_size, interpolation=1, p=1.0),
        A.CenterCrop(width=crop_size, height=crop_size, p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ]
)

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    test_ids = sample["image_id"].tolist()
else:
    test_ids = None

test_image = TestDataset(root_dir=TEST_DIR, transform=transform, image_list=test_ids)
testloader = DataLoader(
    test_image, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_cuda
)

print("Test images:", len(test_image))
print("Preprocess resize:", resize_size, "crop:", crop_size)
print("Normalize mean/std:", mean, std)




## === cell 3
model = efficientnet_b4(weights=backbone_weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if len(obj) > 0:
            any_v = next(iter(obj.values()))
            if torch.is_tensor(any_v):
                return obj
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "module",
        ]:
            if key in obj and isinstance(obj[key], dict):
                return _unwrap_state_dict(obj[key])
    return obj


def find_checkpoint_path_strict(preferred_path):
    if os.path.exists(preferred_path):
        return preferred_path

    base = os.path.basename(preferred_path)
    patterns = [
        "/kaggle/input/**/{}".format(base),
        "/kaggle/input/**/{}.pt".format(model_full_name),
        "/kaggle/input/{}/{}.pt".format(folder_name, model_full_name),
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))

    candidates = sorted(set(candidates))

    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: (len(p), p))
    return candidates[0]


def _strip_known_prefixes(k):
    for pref in ["module.", "model.", "net.", "network."]:
        if k.startswith(pref):
            return k[len(pref) :]
    return k


def _remap_classifier_keys_if_needed(sd, model_obj):
    """
    Some checkpoints were saved with slightly different classifier key names.
    We only remap when tensor shapes match, preserving core model semantics.
    """
    if not isinstance(sd, dict):
        return sd

    model_sd = model_obj.state_dict()
    out = dict(sd)

    def try_map(src_w, src_b, dst_w, dst_b):
        if src_w in out and dst_w in model_sd:
            if tuple(out[src_w].shape) == tuple(model_sd[dst_w].shape):
                out[dst_w] = out[src_w]
        if src_b in out and dst_b in model_sd:
            if tuple(out[src_b].shape) == tuple(model_sd[dst_b].shape):
                out[dst_b] = out[src_b]

    try_map(
        "classifier.weight",
        "classifier.bias",
        "classifier.1.weight",
        "classifier.1.bias",
    )
    try_map(
        "model.classifier.weight",
        "model.classifier.bias",
        "classifier.1.weight",
        "classifier.1.bias",
    )
    try_map(
        "model.classifier.1.weight",
        "model.classifier.1.bias",
        "classifier.1.weight",
        "classifier.1.bias",
    )
    try_map(
        "net.classifier.1.weight",
        "net.classifier.1.bias",
        "classifier.1.weight",
        "classifier.1.bias",
    )
    return out


def _report_loaded_fraction(model_obj, missing_keys, unexpected_keys):
    model_keys = set(model_obj.state_dict().keys())
    missing_set = (
        set(missing_keys) if isinstance(missing_keys, (list, tuple)) else set()
    )
    loaded = len(model_keys - missing_set)
    total = len(model_keys)
    frac = float(loaded) / float(total) if total > 0 else 0.0
    print(
        "  Approx loaded params:", loaded, "/", total, "({:.1f}%)".format(100.0 * frac)
    )
    if frac < 50.0 / 100.0:
        print(
            "  WARNING: Less than half of model parameters matched the checkpoint. "
            "This usually means wrong checkpoint or incompatible key names/shapes."
        )


ckpt_path = find_checkpoint_path_strict(PATH)
if ckpt_path is None:
    print(
        "WARNING: No fine-tuned checkpoint found; running with ImageNet weights only."
    )
else:
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(ckpt)

    if isinstance(state_dict, dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = _strip_known_prefixes(k)
            new_sd[nk] = v
        state_dict = new_sd

        state_dict = _remap_classifier_keys_if_needed(state_dict, model)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Checkpoint load_state_dict(strict=False) summary:")
        print("  Missing keys:", len(missing))
        print("  Unexpected keys:", len(unexpected))
        if len(missing) > 0:
            print("  Missing keys (first 10):", missing[:10])
        if len(unexpected) > 0:
            print("  Unexpected keys (first 10):", unexpected[:10])
        _report_loaded_fraction(model, missing, unexpected)
    else:
        print(
            "WARNING: Checkpoint format not recognized as state_dict; using ImageNet weights only."
        )

model = model.to(device)
model.eval()




## === cell 4
names = []
predicted = []

with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device).float()
        output = model(images_batch)
        pred = torch.argmax(output, dim=1).cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(list(pred))

print("Predicted:", len(predicted))




## === cell 5
result = pd.DataFrame({"image_id": names, "label": predicted})

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    result = sample[["image_id"]].merge(result, on="image_id", how="left")
    result["label"] = result["label"].fillna(0).astype(int)
else:
    result["label"] = result["label"].astype(int)

result.to_csv("submission.csv", index=False)
print(result.head())
print("Wrote submission.csv with {} rows".format(len(result)))
