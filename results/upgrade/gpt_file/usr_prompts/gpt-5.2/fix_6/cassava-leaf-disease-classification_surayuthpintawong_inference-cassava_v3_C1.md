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

0.23804

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing checkpoint path by adding a safe fallback: if the pretrained weights file isn’t available, the code still run end-to-end by using the (untrained) model and producing a valid `submission.csv`. I also make the image preprocessing compatible with EfficientNet by ensuring the resize produces exactly 224×224 and adding standard ImageNet normalization (this is score-positive if weights are later provided, and score-neutral for runtime correctness otherwise). Finally, I guard against the `names`/`predicted` NameError by ensuring inference always executes and by failing fast only when the test image directory is missing.'
- What this solution (achieved 0.10762) has done: 'Your current score (0.11584) strongly suggests the model is running with random weights (or mismatched weights) at inference, so the smallest score-positive change is to correctly load a real pretrained checkpoint if it exists in the input tree. I keep the same EfficientNet-B3 architecture and the same inference loop, but add a robust “search and load” that finds an `efficientnet-b3*.pt` file under `/kaggle/input/` and loads it with better key handling (including common `model.` prefixes) while still falling back safely if nothing is found. I also make the test transform deterministic by replacing `RandomCrop` with a center crop at inference (same overall preprocessing, just removes randomness that hurts accuracy). These changes are minimal, keep the core logic intact, and should move accuracy upward toward your target.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, which is consistent with either random weights or a checkpoint that doesn’t match the exact EfficientNet-B3 head you’re using. I keep your architecture/inference loop intact, but make checkpoint loading more robust by (1) preferring the “best-looking” Cassava checkpoints (by filename hints like `best`/`final` and fold) and (2) handling the common case where the saved classifier weights use different key names (`classifier.1.*` vs `classifier.0.*`) by remapping them when shapes match. This is a minimal, score-positive change because it increases the chance you actually use trained weights correctly without altering evaluation semantics. Everything else (transforms, argmax predictions, submission merge) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.23804) has done: 'Your current score is far below the target, which strongly indicates you’re still effectively using untrained/random weights (or failing to find/load a useful cassava checkpoint). I keep your EfficientNet-B3 model and your inference loop identical, but add a safe second fallback: if no competition-specific checkpoint is found, load ImageNet pretrained EfficientNet-B3 weights and keep your 5-class head as-is (random), which typically improves accuracy above random guessing because features are meaningful. I also tighten the checkpoint search to prefer fold/best cassava checkpoints while de-prioritizing generic ImageNet-style weights if both exist, and I only mark `loaded_ckpt=True` when a non-trivial number of tensors actually load. These are minimal changes that preserve evaluation semantics and should move accuracy upward toward your target without changing the core approach.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

warnings.filterwarnings("ignore")

TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
if os.path.exists(TRAIN_CSV):
    _df = pd.read_csv(TRAIN_CSV)




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
tsfm = transforms.Compose(
    [
        Rescale(256),
        CenterCrop(224),
        ToTensor(),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(TEST_DIR):
    raise RuntimeError("Test images directory not found: %s" % TEST_DIR)

test_image = TestDataset(root_dir=TEST_DIR, transform=tsfm)
testloader = DataLoader(
    test_image, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_cuda
)


def _find_checkpoint(preferred_path):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    candidates = []
    for root, dirs, files in os.walk("/kaggle/input"):
        for fn in files:
            lfn = fn.lower()
            if not (lfn.endswith(".pt") or lfn.endswith(".pth")):
                continue
            if (
                ("cassava" in lfn)
                or ("efficientnet" in lfn)
                or ("b3" in lfn)
                or ("effnet" in lfn)
            ):
                candidates.append(os.path.join(root, fn))

    if not candidates:
        return None

    def _score(p):
        b = os.path.basename(p).lower()
        s = 0
        if "cassava" in b:
            s -= 200
        if "fold" in b:
            s -= 50
        if "best" in b:
            s -= 80
        if "final" in b or "last" in b:
            s -= 30
        if "efficientnet" in b or "effnet" in b:
            s -= 10
        if "b3" in b:
            s -= 10
        if "imagenet" in b or "swsl" in b or "swag" in b:
            s += 80
        s += len(p) * 0.0001
        return s

    candidates = sorted(candidates, key=_score)
    return candidates[0]


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

    pairs = [
        ("classifier.1.weight", "classifier.0.weight"),
        ("classifier.1.bias", "classifier.0.bias"),
        ("classifier.0.weight", "classifier.1.weight"),
        ("classifier.0.bias", "classifier.1.bias"),
    ]

    for src, dst in pairs:
        if src in remapped and dst in msd and dst not in remapped:
            try:
                if tuple(remapped[src].shape) == tuple(msd[dst].shape):
                    remapped[dst] = remapped[src]
            except Exception:
                pass
    return remapped


def _load_imagenet_backbone_into(model_obj):
    try:
        imagenet = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
    except Exception:
        try:
            imagenet = models.efficientnet_b3(pretrained=True)
        except Exception:
            return False, "failed_to_construct_imagenet_model"

    sd = imagenet.state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
    missing, unexpected = model_obj.load_state_dict(sd, strict=False)
    return True, "loaded_imagenet_backbone missing=%d unexpected=%d" % (
        len(missing),
        len(unexpected),
    )


PREFERRED_PATH = "/kaggle/input/inferencecassava/efficientnet-b3-e10.pt"
ckpt_path = _find_checkpoint(PREFERRED_PATH)

loaded_ckpt = False
ckpt_msg = ""
if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    state = _sanitize_state_dict(state)
    if state is not None:
        state = _remap_classifier_keys_if_needed(state, model)
        missing, unexpected = model.load_state_dict(state, strict=False)

        total_keys = len(model.state_dict())
        loaded_keys = total_keys - len(missing)
        loaded_ratio = float(loaded_keys) / float(max(1, total_keys))
        loaded_ckpt = loaded_ratio >= 0.5

        ckpt_msg = (
            "Loaded checkpoint: %s | strict=False; missing=%d unexpected=%d loaded_ratio=%.3f"
            % (ckpt_path, len(missing), len(unexpected), loaded_ratio)
        )
        print(ckpt_msg)

        if not loaded_ckpt:
            print(
                "WARNING: Checkpoint appears largely incompatible/empty (loaded_ratio=%.3f). Will fall back to ImageNet backbone."
                % loaded_ratio
            )
    else:
        print(
            "WARNING: Found checkpoint but could not parse state_dict. Using fallback weights. path=%s"
            % ckpt_path
        )
else:
    print(
        "WARNING: No checkpoint found under /kaggle/input. Using fallback weights (ImageNet backbone)."
    )

if not loaded_ckpt:
    ok, msg = _load_imagenet_backbone_into(model)
    print("Fallback:", msg)

model.eval()

names = []
predicted = []

with torch.no_grad():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device)
        output = model(images_batch)
        output = torch.max(output, 1)[1].cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(output.tolist())

print("Inference done. ckpt_loaded=%s, n_test=%d" % (loaded_ckpt, len(names)))



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
