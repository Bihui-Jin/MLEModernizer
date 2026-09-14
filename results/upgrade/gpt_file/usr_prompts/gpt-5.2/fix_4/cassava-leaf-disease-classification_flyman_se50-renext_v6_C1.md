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
timm==1.0.19
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

0.8958899969779389

# 6. Current score

0.61061

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17152) has done: 'I fix the missing model directory error by falling back to a lightweight, deterministic timm pretrained model when the external weights folder isn’t present, so the notebook runs end-to-end in this environment. I also correct submission alignment bugs: the current code mixes `image_ids` from only the last loop with a dict-based aggregation, which can scramble predictions; I instead preserve a stable filename order and build the submission from that order. I remove notebook-only `!pip`/`!ls` usages to avoid syntax/runtime issues in a pure `.py` execution context, while keeping the same inference core (TTA + softmax averaging + argmax). Finally, I ensure `submission.csv` is always written with the required `image_id,label` columns.'
- What this solution (achieved 0.14985) has done: 'Your current score is far below target, and the main reason is that the fallback model (ResNet50 pretrained on ImageNet) is being used without any cassava-specific finetuned weights, so predictions are close to random for 5 classes. To move the score up toward the target with minimal core-logic changes, I keep your same inference approach (TTA + softmax averaging + argmax), but switch the fallback to a timm model that is actually pretrained for this competition via `hf-hub` (if available in this environment) and use the correct input normalization for timm pretrained models. I also make the weight loading robust to common checkpoint key formats (`state_dict`, `model`, `module.` prefixes) so that if your external directory does exist, the intended weights load correctly. These changes keep your architecture/inference semantics intact but materially improve accuracy toward the target.'
- What this solution (achieved 0.61061) has done: 'Your score is far below the target because the current pipeline usually runs with generic (ImageNet) pretrained weights, so the 5-way cassava labels are effectively guessed. With minimal changes and without altering your model architecture or inference semantics, I (1) ensure the timm backbone is created with `num_classes=0` so `forward_features()` is consistent across backbones, (2) correctly load external checkpoints into `model.model` and head weights into `_fc` when present (your current loader tries to load everything into the wrapper `Net`, so even correct cassava checkpoints won’t map), and (3) use the backbone’s own pretrained `mean/std` (keeps the same normalization logic but matches pretrained expectations when using hf-hub/timm weights). These are small, execution-safe changes that should materially increase accuracy toward your target if any cassava-finetuned weights are available or hf-hub works in this environment. The submission writing and TTA+softmax averaging+argmax remain unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import timm




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaggle_root = "/kaggle/input"




## === cell 2
def gem(x, p=3, eps=1e-5):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-5):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )




## === cell 3
class Net(nn.Module):
    def __init__(
        self, num_classes=5, backbone_name="seresnext50_32x4d", pretrained=False
    ):
        super().__init__()

        self.model = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)

        if hasattr(self.model, "num_features"):
            in_features = self.model.num_features
        else:
            in_features = 2048
        self._fc = nn.Linear(in_features, num_classes, bias=True)

        cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer(
            "_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, inputs):
        x = inputs / 255.0
        x = (x - self._mean.to(device=x.device, dtype=x.dtype)) / self._std.to(
            device=x.device, dtype=x.dtype
        )

        bs = x.size(0)

        feats = self.model.forward_features(x)
        fm = self._avg_pooling(feats).view(bs, -1)
        fm = self.dropout(fm)
        out = self._fc(fm)
        return out




## === cell 4
class DatasetTest:
    def __init__(self, test_data_dir):
        self.root_dir = test_data_dir
        self.ds = self.get_list(test_data_dir)

    def get_list(self, dir_path):
        return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(".jpg")])

    def __len__(self):
        return len(self.ds)

    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))

        image_90 = np.rot90(image, 1)
        image_180 = np.rot90(image, 2)
        image_270 = np.rot90(image, 3)

        image_fliplr = np.fliplr(image)
        image_fliplr_90 = np.rot90(image_fliplr, 1)
        image_fliplr_180 = np.rot90(image_fliplr, 2)
        image_fliplr_270 = np.rot90(image_fliplr, 3)

        image_batch = np.stack(
            [
                image,
                image_90,
                image_180,
                image_270,
                image_fliplr,
                image_fliplr_90,
                image_fliplr_180,
                image_fliplr_270,
            ]
        )
        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        return image, image_batch

    def __getitem__(self, item):
        fname = self.ds[item]
        image_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(image_path, -1)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image, float_image = self.preprocess_func(image)
        return fname, image, float_image




## === cell 5
test_datadir = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification", "test_images"
)
if not os.path.isdir(test_datadir):
    alt = os.path.join(kaggle_root, "test_images")
    if os.path.isdir(alt):
        test_datadir = alt
    else:
        raise FileNotFoundError(
            f"Test images directory not found at {test_datadir} or {alt}"
        )

dataiter = DatasetTest(test_datadir)



## === cell 6
candidate_model_dirs = [
    os.path.join(kaggle_root, "cassva-models-se50-640"),
    os.path.join(kaggle_root, "cassava-models-se50-640"),
]

weights = []
model_dir = None
for d in candidate_model_dirs:
    if os.path.isdir(d):
        model_dir = d
        weights = [
            os.path.join(d, f)
            for f in sorted(os.listdir(d))
            if f.endswith((".pth", ".pt", ".bin"))
        ]
        if len(weights) > 0:
            break

use_external_weights = len(weights) > 0

if use_external_weights:
    model = Net(num_classes=5, backbone_name="seresnext50_32x4d", pretrained=False).to(
        device
    )
else:
    try:
        backbone_name = "hf-hub:nateraw/cassava_leaf_disease_efficientnet_b4"
        model = Net(num_classes=5, backbone_name=backbone_name, pretrained=True).to(
            device
        )
    except Exception as e:
        print(
            f"hf-hub fallback not available ({e}); using ImageNet pretrained resnet50."
        )
        model = Net(num_classes=5, backbone_name="resnet50", pretrained=True).to(device)




## === cell 7
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _split_wrapper_keys_for_loading(state_dict):
    """
    Change (score-relevant, minimal): many cassava checkpoints save keys for the wrapper Net, e.g.
    'model.*' for backbone and '_fc.*' for the head. Loading all keys into Net with strict=False
    often fails silently due to mismatched naming across timm backbones. We explicitly route keys.
    """
    if not isinstance(state_dict, dict):
        return None, None, state_dict

    sd = state_dict

    sd = _strip_prefix_if_present(sd, "module.")

    backbone_sd = {}
    head_sd = {}
    other_sd = {}

    for k, v in sd.items():
        if k.startswith("model."):
            backbone_sd[k[len("model.") :]] = v
        elif k.startswith("_fc."):
            head_sd[k[len("_fc.") :]] = v
        else:
            other_sd[k] = v

    return backbone_sd, head_sd, other_sd


def predict_with_model(model, weights, dataiter, device):
    model = model.to(device)
    model.eval()

    filenames = list(dataiter.ds)
    merge_res = {fn: None for fn in filenames}

    if len(weights) == 0:
        weights = [None]  # run single-pass with current model weights

    for j, weight in enumerate(weights):
        if weight is not None:
            state = torch.load(weight, map_location=device)
            state = _extract_state_dict(state)

            backbone_sd, head_sd, other_sd = _split_wrapper_keys_for_loading(state)

            loaded_any = False
            msg_parts = []

            if isinstance(backbone_sd, dict) and len(backbone_sd) > 0:
                missing_b, unexpected_b = model.model.load_state_dict(
                    backbone_sd, strict=False
                )
                loaded_any = True
                msg_parts.append(
                    f"backbone missing={len(missing_b)} unexpected={len(unexpected_b)}"
                )

            if isinstance(head_sd, dict) and len(head_sd) > 0:
                missing_h, unexpected_h = model._fc.load_state_dict(
                    head_sd, strict=False
                )
                loaded_any = True
                msg_parts.append(
                    f"head missing={len(missing_h)} unexpected={len(unexpected_h)}"
                )

            if not loaded_any:
                missing, unexpected = model.load_state_dict(other_sd, strict=False)
                msg_parts.append(
                    f"wrapper missing={len(missing)} unexpected={len(unexpected)}"
                )

            if j == 0:
                print(f"Loaded {os.path.basename(weight)}: " + " | ".join(msg_parts))

            model.eval()

        for i, fname in enumerate(filenames):
            if (i + 1) % 250 == 0 or (i + 1) == len(filenames):
                print(f"weight {j+1}/{len(weights)}: data {i+1}/{len(filenames)}")

            _, _, float_image = dataiter.__getitem__(i)
            inp = torch.from_numpy(float_image).to(device).float()

            with torch.no_grad():
                out = model(inp)
                out = torch.nn.functional.softmax(out, dim=-1).cpu().numpy()
                out = np.mean(out, axis=0)  # TTA mean over 8 views

            if merge_res[fname] is None:
                merge_res[fname] = out
            else:
                merge_res[fname] += out

    pred_labels = [int(np.argmax(merge_res[fname])) for fname in filenames]
    sub = pd.DataFrame({"image_id": filenames, "label": pred_labels})
    sub.to_csv("submission.csv", index=False)
    return sub


sub = predict_with_model(model, weights, dataiter, device)
print(sub.head())
print(f"Wrote submission.csv with {len(sub)} rows")
