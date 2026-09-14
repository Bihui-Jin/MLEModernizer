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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.8070414022363251

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing checkpoint path by adding a safe fallback: if the pretrained weights file isn’t available, the code still run end-to-end by using the (untrained) model and producing a valid `submission.csv`. I also make the image preprocessing compatible with EfficientNet by ensuring the resize produces exactly 224×224 and adding standard ImageNet normalization (this is score-positive if weights are later provided, and score-neutral for runtime correctness otherwise). Finally, I guard against the `names`/`predicted` NameError by ensuring inference always executes and by failing fast only when the test image directory is missing.'
- What this solution (achieved 0.10762) has done: 'Your current score (0.11584) strongly suggests the model is running with random weights (or mismatched weights) at inference, so the smallest score-positive change is to correctly load a real pretrained checkpoint if it exists in the input tree. I keep the same EfficientNet-B3 architecture and the same inference loop, but add a robust “search and load” that finds an `efficientnet-b3*.pt` file under `/kaggle/input/` and loads it with better key handling (including common `model.` prefixes) while still falling back safely if nothing is found. I also make the test transform deterministic by replacing `RandomCrop` with a center crop at inference (same overall preprocessing, just removes randomness that hurts accuracy). These changes are minimal, keep the core logic intact, and should move accuracy upward toward your target.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, which is consistent with either random weights or a checkpoint that doesn’t match the exact EfficientNet-B3 head you’re using. I keep your architecture/inference loop intact, but make checkpoint loading more robust by (1) preferring the “best-looking” Cassava checkpoints (by filename hints like `best`/`final` and fold) and (2) handling the common case where the saved classifier weights use different key names (`classifier.1.*` vs `classifier.0.*`) by remapping them when shapes match. This is a minimal, score-positive change because it increases the chance you actually use trained weights correctly without altering evaluation semantics. Everything else (transforms, argmax predictions, submission merge) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.23804) has done: 'Your current score is far below the target, which strongly indicates you’re still effectively using untrained/random weights (or failing to find/load a useful cassava checkpoint). I keep your EfficientNet-B3 model and your inference loop identical, but add a safe second fallback: if no competition-specific checkpoint is found, load ImageNet pretrained EfficientNet-B3 weights and keep your 5-class head as-is (random), which typically improves accuracy above random guessing because features are meaningful. I also tighten the checkpoint search to prefer fold/best cassava checkpoints while de-prioritizing generic ImageNet-style weights if both exist, and I only mark `loaded_ckpt=True` when a non-trivial number of tensors actually load. These are minimal changes that preserve evaluation semantics and should move accuracy upward toward your target without changing the core approach.'
- What this solution (achieved 0.15845) has done: 'Your score is still far below the target, which is consistent with loading no (or the wrong) Cassava-trained weights and effectively running an ImageNet backbone with a random 5-class head. I keep your EfficientNet-B3 model and the same inference loop, but make checkpoint selection/load stricter: prefer checkpoints that look Cassava-trained (and avoid generic ImageNet weights), require a much higher loaded-key ratio, and ensure the classifier head weights are actually loaded (otherwise we fall back). I also ensure the checkpoint key remapping covers the common `classifier.1.*` vs `classifier.-1.*` / sequential variants more robustly without changing architecture. These minimal changes should increase accuracy toward your target by greatly increasing the chance you use a real Cassava 5-class checkpoint rather than a partially compatible or irrelevant one.'
- What this solution (achieved 0.26756) has done: 'Your score is far below target, and the current pipeline likely never loads a real Cassava-trained 5-class checkpoint (so it’s effectively using an ImageNet backbone with a random head). I keep your EfficientNet-B3 architecture and inference loop intact, but make the checkpoint search/load more Cassava-specific and more permissive: accept “head-missing” Cassava checkpoints by loading the backbone and then loading the classifier if compatible, rather than rejecting the whole checkpoint. I also add a deterministic test-time augmentation (horizontal flip only) and average logits across the two passes, which preserves evaluation semantics and typically gives a meaningful accuracy lift with minimal code changes. Finally, I fix the Python version mismatch issue by removing Python-2-only assumptions while keeping behavior identical in Python 3 (the environment implied by your installed packages).'
- What this solution (achieved 0.09006) has done: 'Your score is far below the target, which is consistent with still not loading an actually Cassava-trained 5-class checkpoint (the current search likely finds nothing useful in `/kaggle/input`, so you fall back to ImageNet-backbone + random head, which tops out around your current score). I keep your EfficientNet-B3 model and inference loop intact, but make the checkpoint finder also search `/kaggle/working` (where Kaggle notebooks commonly place uploaded or generated `.pt/.pth` files) and broaden Cassava-specific filename hints so it’s more likely to pick the right file if it exists. I also slightly relax the “loaded_ckpt” acceptance criteria in a safe way: require that the classifier head actually loads (since this is a 5-class task), otherwise we explicitly treat it as not-loaded and fall back; this avoids the misleading case of “backbone-only loaded” producing low accuracy. These are minimal changes focused on actually using the intended trained weights, which should move accuracy upward toward your target if such a checkpoint is present.'
- What this solution (achieved 0.30942) has done: 'Your current score is far below the target, consistent with the model effectively using a random 5-class head (even when an ImageNet backbone is loaded). I keep your EfficientNet-B3 architecture and inference loop, but make the fallback path produce meaningful 5-class predictions by using EfficientNet-B3 ImageNet weights *end-to-end* and mapping its 1000-class outputs onto the 5 cassava labels via a fixed, legitimate proxy (nearest ImageNet class based on classifier weight cosine similarity). If a real Cassava 5-class checkpoint is found and properly loads, your original 5-class head path is unchanged; otherwise this mapping fallback should substantially increase accuracy versus the current random-head behavior, moving the score toward your target. I also make the checkpoint acceptance slightly more permissive for real cassava checkpoints by not requiring every classifier key to load (some checkpoints store heads differently), while still preferring ones that load the head when possible.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.30942) is far below the target (0.8070), so we should cautiously increase accuracy with minimal changes. The biggest issue is the current “ImageNet-to-cassava weight-similarity mapping” fallback, which is essentially arbitrary because your 5-class head weights are random when no Cassava checkpoint is loaded; I replace that fallback with a deterministic, label-prior baseline derived from `train.csv` class frequencies (still legitimate, no leakage from test labels) which should move accuracy upward substantially. I also add a strict, Cassava-safe checkpoint preference to load any 5-class Cassava EfficientNet-B3 weights if they exist, while keeping the model architecture and inference loop the same. Submission writing stays identical and still produces `submission.csv`.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

warnings.filterwarnings("ignore")

TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_df = None
if os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)




## === cell 1
class Rescale(object):
    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        if type(sample) == dict:
            image, label = sample["image"], sample["label"]
        else:
            image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / float(w), self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / float(h)
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(
            image, (new_h, new_w), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

        if type(sample) == dict:
            return {"image": img, "label": label}
        else:
            return img


class RandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if type(sample) == dict:
            image, label = sample["image"], sample["label"]
        else:
            image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, max(1, h - new_h + 1))
        left = np.random.randint(0, max(1, w - new_w + 1))

        image = image[top : top + new_h, left : left + new_w]
        if type(sample) == dict:
            return {"image": image, "label": label}
        else:
            return image


class CenterCrop(object):
    """Center crop the image in a sample."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if type(sample) == dict:
            image, label = sample["image"], sample["label"]
        else:
            image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = max(0, (h - new_h) // 2)
        left = max(0, (w - new_w) // 2)

        image = image[top : top + new_h, left : left + new_w]
        if type(sample) == dict:
            return {"image": image, "label": label}
        else:
            return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if type(sample) == dict:
            image, label = sample["image"], sample["label"]
        else:
            image = sample

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        image = image.transpose((2, 0, 1))
        if type(sample) == dict:
            return {"image": torch.from_numpy(image), "label": label}
        else:
            return torch.from_numpy(image)




## === cell 2
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True




## === cell 3
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(root_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return img_name, image




## === cell 4
def build_model(num_classes=5):
    m = models.efficientnet_b3(weights=None)
    if isinstance(m.classifier, nn.Sequential):
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, num_classes)
    return m


model = build_model(num_classes=5).to(device)




## === cell 5
class HFlip(object):
    def __call__(self, sample):
        if type(sample) == dict:
            image, label = sample["image"], sample["label"]
        else:
            image = sample
        image = np.ascontiguousarray(image[:, ::-1, ...])
        if type(sample) == dict:
            return {"image": image, "label": label}
        else:
            return image


tsfm = transforms.Compose(
    [
        Rescale(256),
        CenterCrop(224),
        ToTensor(),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

tsfm_flip = transforms.Compose(
    [
        Rescale(256),
        CenterCrop(224),
        HFlip(),
        ToTensor(),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(TEST_DIR):
    raise RuntimeError("Test images directory not found: %s" % TEST_DIR)

test_image = TestDataset(root_dir=TEST_DIR, transform=tsfm)
test_image_flip = TestDataset(root_dir=TEST_DIR, transform=tsfm_flip)

testloader = DataLoader(
    test_image, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_cuda
)
testloader_flip = DataLoader(
    test_image_flip, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_cuda
)


def _find_checkpoint(preferred_path):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = ["/kaggle/input", "/kaggle/working"]

    candidates = []
    for sr in search_roots:
        if not os.path.isdir(sr):
            continue
        for root, dirs, files in os.walk(sr):
            for fn in files:
                lfn = fn.lower()
                if not (lfn.endswith(".pt") or lfn.endswith(".pth")):
                    continue

                cassava_hint = (
                    ("cassava" in lfn)
                    or ("ccldc" in lfn)
                    or ("leaf" in lfn)
                    or ("disease" in lfn)
                    or ("cassavaleaf" in lfn)
                    or ("cldc" in lfn)
                    or ("cassava-leaf-disease-classification" in root.lower())
                )
                arch_hint = (
                    ("efficientnet" in lfn) or ("effnet" in lfn) or ("b3" in lfn)
                )
                bad_hint = ("imagenet" in lfn) or ("swsl" in lfn) or ("swag" in lfn)

                if cassava_hint or arch_hint:
                    candidates.append(
                        (os.path.join(root, fn), cassava_hint, arch_hint, bad_hint)
                    )

    if not candidates:
        return None

    def _score(t):
        p, cassava_hint, arch_hint, bad_hint = t
        b = os.path.basename(p).lower()
        s = 0

        if cassava_hint:
            s -= 5000
        else:
            s += 500

        if "best" in b:
            s -= 400
        if "fold" in b:
            s -= 200
        if "final" in b or "last" in b:
            s -= 100

        if arch_hint:
            s -= 60
        if "b3" in b:
            s -= 30

        if bad_hint:
            s += 2000

        s += len(p) * 0.0001
        return s

    candidates = sorted(candidates, key=_score)
    return candidates[0][0]


def _sanitize_state_dict(state):
    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        return None

    new_state = {}
    for k, v in state.items():
        nk = k
        for pref in ["module.", "model.", "net."]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_state[nk] = v
    return new_state


def _remap_classifier_keys_if_needed(state, model_obj):
    if not isinstance(state, dict):
        return state

    msd = model_obj.state_dict()
    remapped = dict(state)

    possible_src = [
        ("classifier.1.weight", "classifier.0.weight"),
        ("classifier.1.bias", "classifier.0.bias"),
        ("classifier.0.weight", "classifier.1.weight"),
        ("classifier.0.bias", "classifier.1.bias"),
        ("classifier.-1.weight", "classifier.1.weight"),
        ("classifier.-1.bias", "classifier.1.bias"),
        ("classifier.-1.weight", "classifier.0.weight"),
        ("classifier.-1.bias", "classifier.0.bias"),
    ]

    for src, dst in possible_src:
        if src in remapped and dst in msd and dst not in remapped:
            try:
                if tuple(remapped[src].shape) == tuple(msd[dst].shape):
                    remapped[dst] = remapped[src]
            except Exception:
                pass
    return remapped


def _majority_class_from_train(train_df_local):
    if train_df_local is None or "label" not in train_df_local.columns:
        return 0
    vc = train_df_local["label"].value_counts()
    if vc.empty:
        return 0
    return int(vc.idxmax())


PREFERRED_PATH = "/kaggle/input/inferencecassava/efficientnet-b3-e10.pt"
ckpt_path = _find_checkpoint(PREFERRED_PATH)

loaded_ckpt = False
ckpt_msg = ""
if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    state = _sanitize_state_dict(state)
    if state is not None:
        state = _remap_classifier_keys_if_needed(state, model)

        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state.keys())
        backbone_keys = [k for k in model_keys if not k.startswith("classifier.")]
        classifier_keys = [k for k in model_keys if k.startswith("classifier.")]

        backbone_overlap = sum([1 for k in backbone_keys if k in ckpt_keys])
        backbone_ratio = float(backbone_overlap) / float(max(1, len(backbone_keys)))

        missing, unexpected = model.load_state_dict(state, strict=False)

        missing_set = set(missing)
        classifier_overlap = sum(
            [1 for k in classifier_keys if k in ckpt_keys and k not in missing_set]
        )
        classifier_any_loaded = classifier_overlap > 0

        loaded_ckpt = (backbone_ratio >= 0.90) and classifier_any_loaded

        ckpt_msg = (
            "Loaded checkpoint: %s | strict=False; missing=%d unexpected=%d backbone_ratio=%.3f classifier_any_loaded=%s"
            % (
                ckpt_path,
                len(missing),
                len(unexpected),
                backbone_ratio,
                str(classifier_any_loaded),
            )
        )
        print(ckpt_msg)

        if not loaded_ckpt:
            print(
                "WARNING: Checkpoint not sufficiently compatible for 5-class inference. Will use class-prior fallback."
            )
    else:
        print(
            "WARNING: Found checkpoint but could not parse state_dict. Using class-prior fallback. path=%s"
            % ckpt_path
        )
else:
    print(
        "WARNING: No checkpoint found under /kaggle/input or /kaggle/working. Using class-prior fallback."
    )

use_class_prior_fallback = not loaded_ckpt
fallback_label = (
    _majority_class_from_train(train_df) if use_class_prior_fallback else None
)
if use_class_prior_fallback:
    print("Fallback: class-prior majority label =", fallback_label)

model.eval()

names = []
predicted = []

with torch.no_grad():
    for (names_batch, images_batch), (names_batch2, images_batch2) in zip(
        testloader, testloader_flip
    ):
        if list(names_batch) != list(names_batch2):
            raise RuntimeError(
                "Test loader order mismatch between original and flip transforms."
            )

        if not use_class_prior_fallback:
            images_batch = images_batch.to(device)
            images_batch2 = images_batch2.to(device)
            out1 = model(images_batch)
            out2 = model(images_batch2)
            out = (out1 + out2) / 2.0
            out = torch.max(out, 1)[1].cpu().numpy()
            batch_pred = out.tolist()
        else:
            batch_pred = [int(fallback_label)] * len(names_batch)

        names.extend(list(names_batch))
        predicted.extend(batch_pred)

print(
    "Inference done. ckpt_loaded=%s, class_prior_fallback=%s, n_test=%d"
    % (loaded_ckpt, use_class_prior_fallback, len(names))
)



## === cell 6
result = pd.DataFrame({"image_id": names, "label": predicted})

SAMPLE_SUB = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
if os.path.exists(SAMPLE_SUB):
    sub = pd.read_csv(SAMPLE_SUB)
    result = sub[["image_id"]].merge(result, on="image_id", how="left")
    result["label"] = result["label"].fillna(0).astype(int)

result.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", result.shape)
print(result.head())
