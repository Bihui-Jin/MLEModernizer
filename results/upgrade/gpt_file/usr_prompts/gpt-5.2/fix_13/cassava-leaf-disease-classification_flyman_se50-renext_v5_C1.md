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

0.8925657298277425

# 6. Current score

0.08483

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.321) has done: 'I fix the runtime failure by making the code robust to the missing external weights directory (`/kaggle/input/cassva-models-se50-640`) and to the unavailable pip install source, so the notebook runs in this environment. Since the pretrained weights aren’t available, I fall back to using timm’s built-in pretrained ImageNet weights (same model architecture) to produce reasonable predictions and a valid `submission.csv`. I also fix the TTA bug where the flipped/rotated variants were incorrectly generated from the original image, and ensure deterministic, sorted test filenames so the output is stable. Finally, I ensure the script always writes a properly formatted submission CSV with `image_id,label` columns.'
- What this solution (achieved 0.06764) has done: 'Your current 0.321 score is consistent with a model whose classifier head is random (ImageNet pretrained backbone + brand-new 5-class head), so the smallest meaningful fix is to load cassava-trained weights if they exist anywhere in `/kaggle/input` (your code only checks one hardcoded, likely-misspelled directory). To keep core logic identical, I only (1) add a robust search for `.pth/.pt/.bin` weights across the competition dataset folders and (2) make weight loading handle common checkpoint formats (`state_dict`, `model`, `module.` prefixes) so the same architecture can actually use the weights. If no weights are found, it still fall back to the current behavior (ImageNet backbone) and write a valid `submission.csv`. This should move accuracy sharply upward toward your ~0.892 target if any reasonable cassava checkpoint is present.'
- What this solution (achieved 0.1009) has done: 'Your current score (0.06764) is far below the target (0.8926), which is consistent with running a cassava architecture but effectively predicting with an untrained/random 5-class head (or not actually loading a compatible cassava checkpoint). The smallest change that should move accuracy sharply upward (without changing the model/training logic) is to (1) pick the *most likely compatible* checkpoint file rather than iterating over every `.pth/.pt/.bin`, and (2) make checkpoint loading more robust by filtering to matching tensor shapes (so a cassava-trained head loads while unrelated weights are ignored). If no compatible checkpoint exists in `/kaggle/input`, the behavior stays the same (ImageNet backbone + random head), but if one exists anywhere under `/kaggle/input`, this should substantially increase score toward your target. I also ensure the test output is strictly aligned to `sample_submission.csv` ordering to avoid any accidental row-order mismatches that can destroy accuracy.'
- What this solution (achieved 0.23468) has done: 'Your score (0.1009) is far below the target (0.8926), which strongly suggests the cassava-trained checkpoint is still not being found/loaded and you’re effectively using an untrained 5-class head. The smallest high-impact fix (without changing model or inference core logic) is to (1) explicitly search the *entire* `/kaggle/input` for likely cassava checkpoints but rank them by architecture/competition keywords, then (2) pick the best checkpoint by compatibility score as you already do, and (3) ensure the model is moved to `eval()` after loading and uses a consistent image normalization expected by timm pretrained backbones (ImageNet mean/std) which typically improves accuracy without changing the architecture/training approach. These changes are directly targeted at getting you closer to the target by making it much more likely you actually load a relevant cassava checkpoint and by matching the backbone’s expected input scaling. The submission writing and alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved 0.21562) has done: 'Your current score is far below the target, which is consistent with the cassava head weights not being loaded even when a checkpoint exists (due to different key names and/or different head naming). I keep the same model and inference/TTA logic, but make checkpoint loading more architecture-aware by remapping common classifier head keys (e.g., `fc.*`, `classifier.*`, `head.fc.*`) to your `_fc.*` when shapes match. I also ensure the loaded checkpoint is applied after moving the model to device, and I bias the compatibility scoring toward checkpoints that contain a 5-class head so the “best” discovered file is more likely to be a real cassava-trained model. These are minimal, targeted changes intended to move accuracy substantially upward toward the target without changing the core approach.'
- What this solution (achieved 0.46749) has done: 'Your current score is far below the target, which strongly suggests the code is still not loading any cassava-trained weights and is effectively predicting with a random 5-class head. I keep the exact same model and TTA inference, but (1) ensure we search `/kaggle/input` more effectively for likely cassava checkpoints by also scanning common “working” mirrors and ranking higher for 5-class heads, and (2) make the checkpoint loader robust to additional common key prefixes (`model.model.`, `backbone.`) and common head names used by timm (`model.fc.*`, `model.classifier.*`) so the head actually loads when present. Finally, I try the top few ranked checkpoints (not all) and pick the one that loads the most tensors (especially the head), which is a minimal change that should move accuracy substantially toward your target if any suitable checkpoint exists in the environment.'
- What this solution (achieved 0.23393) has done: 'We need to move accuracy up from 0.467 toward 0.892, so the highest-impact minimal change is to ensure we load the *right* cassava-trained checkpoint if it exists and that we load it in a way that matches your `Net` key structure. I keep the same model and the same 8-way TTA inference, but make the checkpoint loader understand the common case where the checkpoint was saved for a wrapper module (so keys look like `model._fc.*` and `model.model.*`) and where the classifier head is named `model._fc.*` rather than `_fc.*`. I also broaden the head remapping slightly to cover `model._fc.*` and apply prefix stripping iteratively so nested prefixes get removed reliably. These changes are targeted to increase the chance that your 5-class head weights actually load, which should move the score much closer to the target without changing core evaluation semantics.'
- What this solution (achieved 0.13789) has done: 'Your score is far below the target, which is consistent with still not loading any *cassava-trained* checkpoint (so the 5-class head stays random and predictions are near chance). To move accuracy upward with minimal changes and identical model/TTA logic, I (1) search more effectively for checkpoints by also scanning `/kaggle/input` and `/kaggle/working` for common weight filenames and (2) fix the checkpoint compatibility scoring/selection so it prefers files that actually contain `seresnext50` backbone keys **and** a matching 5-class head, rather than just “loads some tensors”. Finally, I make weight loading tolerant of more common key patterns (`state_dict` tensors wrapped under `ema`, `model_ema`, etc.) and ensure BatchNorm running stats are included when shapes match—this should substantially increase score if any relevant checkpoint exists, without changing the architecture or inference behavior.'
- What this solution (achieved 0.14798) has done: 'Your current score is far below the target, which most often happens here when inference normalization doesn’t match what the checkpoint/backbone expects and/or when BatchNorm running stats aren’t being used correctly at eval time. I keep your exact architecture and TTA loop, but (1) switch the resize to the standard timm pipeline behavior (INTER_LINEAR) and ensure the numpy array is contiguous to avoid subtle layout issues, and (2) load checkpoints in a way that also prefers/keeps BatchNorm running stats (when shapes match) and sets the model to `eval()` immediately after loading. These are minimal changes aimed at making predictions consistent with the intended pretrained backbone/checkpoint behavior and should move accuracy upward toward your target without altering core logic or training.'
- What this solution (achieved 0.39686) has done: 'Your current score (0.14798) is far below the target (0.8926), which is consistent with predictions coming from an effectively untrained cassava head (checkpoint not found/loaded) and/or a normalization mismatch to the checkpoint’s training pipeline. I keep your exact model, TTA, and inference loop, but (1) ensure the input normalization matches the original common Cassava baseline (no ImageNet mean/std; just scale to [0,1]) by gating it to only apply when truly using timm ImageNet-pretrained weights, and (2) improve checkpoint selection to try a few top-ranked candidates and pick the one that yields the highest internal “cassava-ness” (loads the 5-class head and many backbone tensors). This is minimal and directly targeted at making it much more likely you actually use a cassava-trained checkpoint when present, which should move accuracy sharply upward toward your target while preserving the core logic and producing the same submission format.'
- What this solution (achieved 0.21039) has done: 'Your score is far below the target, so the smallest likely high-impact fix is to ensure we actually load a cassava-trained checkpoint *from the competition dataset itself* (these are often stored as `.pth` inside “dataset” folders like `/kaggle/input/**`, but your scan currently skips large parts and doesn’t consider common `.ckpt` files). I minimally extend the checkpoint search to include `.ckpt`, and I stop excluding whole directories by name (only exclude the huge image/tfrecord folders) so we don’t miss a model file located under the dataset tree. I also make the checkpoint loader extract weights from a few additional common wrapper keys (`'model_state'`, `'state'`, `'params'`) used in Lightning/other training scripts, without changing the model or inference logic. Everything else (architecture, TTA, softmax-mean, submission alignment) stays identical, and it still always write `submission.csv`.'
- What this solution (achieved 0.08483) has done: 'Your score (0.210) is far below the target (0.893), which is most consistent with not actually loading a cassava-trained checkpoint (so the 5-class head stays random-ish). I keep your model/TTA/inference core logic identical, but make checkpoint discovery and selection more likely to find a real cassava model by (1) explicitly including common weight file extensions like `.pth.tar` and (2) boosting the ranking for filenames that look like real trained checkpoints (and downranking tiny “optimizer-only” or “last” artifacts). I also fix a subtle selection bug where compatibility was computed on a different model instance than the one used to load, by scoring candidates against the same architecture config you later load (still `Net(..., use_pretrained_backbone=True)`). These are minimal, targeted changes intended to move accuracy up toward your target without changing architecture, TTA, or post-processing.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd



## === cell 1
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import cv2




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
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"




## === cell 3
class Net(nn.Module):
    def __init__(self, num_classes=5, use_pretrained_backbone=True):
        super().__init__()

        self.model = timm.create_model(
            "seresnext50_32x4d",
            pretrained=use_pretrained_backbone,
            num_classes=0,  # use as feature extractor
            global_pool="",  # we'll pool ourselves
        )

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, num_classes, bias=True)

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.use_imagenet_norm = bool(use_pretrained_backbone)

    def forward(self, inputs):
        x = inputs / 255.0

        if self.use_imagenet_norm:
            x = (x - self.img_mean) / self.img_std

        bs = x.size(0)
        x = self.model.forward_features(x)
        fm = self._avg_pooling(x).view(bs, -1)
        fm = self.dropout(fm)
        x = self._fc(fm)
        return x




## === cell 4
class DatasetTest:
    def __init__(self, test_data_dir):
        self.root_dir = test_data_dir
        self.ds = self.get_list(test_data_dir)

    def get_list(self, dir_path):
        pic_list = [f for f in os.listdir(dir_path) if f.lower().endswith(".jpg")]
        pic_list.sort()
        return pic_list

    def __len__(self):
        return len(self.ds)

    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

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
            ],
            axis=0,
        )

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        image_batch = np.ascontiguousarray(image_batch)

        return image, image_batch

    def __getitem__(self, item):
        fname = self.ds[item]
        image_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image, float_image = self.preprocess_func(image)
        return fname, image, float_image




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaggle_root = "/kaggle/input"

test_datadir = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification/test_images"
)
if not os.path.isdir(test_datadir):
    test_datadir = os.path.join(kaggle_root, "test_images")
if not os.path.isdir(test_datadir):
    raise FileNotFoundError(f"Could not find test_images directory under {kaggle_root}")

dataiter = DatasetTest(test_datadir)

sample_path = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.isfile(sample_path):
    sample_path = os.path.join(kaggle_root, "sample_submission.csv")
if not os.path.isfile(sample_path):
    raise FileNotFoundError("Could not find sample_submission.csv under /kaggle/input")

sample_sub = pd.read_csv(sample_path)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)


def find_weights_under(root_dir):
    base_exts = (".pth", ".pt", ".bin", ".ckpt", ".pth.tar", ".pt.tar")
    found = []
    if not os.path.isdir(root_dir):
        return found
    for base, dirs, files in os.walk(root_dir):
        bn = os.path.basename(base)
        if bn in ("train_images", "test_images", "train_tfrecords", "test_tfrecords"):
            dirs[:] = []
            continue
        for fn in files:
            lf = fn.lower()
            if lf.endswith(base_exts):
                found.append(os.path.join(base, fn))
    found.sort()
    return found


def rank_weight_paths(paths):
    def score(p):
        lp = p.lower()
        s = 0
        if "cassava" in lp:
            s += 160
        if "leaf" in lp:
            s += 20
        if "disease" in lp:
            s += 20
        if "cbsd" in lp or "cmd" in lp:
            s += 10

        if "seresnext" in lp or "se_resnext" in lp or "seresnext50" in lp:
            s += 120
        if "32x4d" in lp:
            s += 12

        if "best" in lp:
            s += 40
        if "final" in lp:
            s += 18
        if "fold" in lp:
            s += 8
        if "epoch" in lp:
            s += 4

        if "last" in lp:
            s -= 8
        if "optimizer" in lp or "sched" in lp or "scheduler" in lp or "adam" in lp:
            s -= 80

        try:
            size = os.path.getsize(p)
            if 20_000_000 <= size <= 400_000_000:
                s += 18
            elif 5_000_000 <= size < 20_000_000:
                s += 3
            elif size < 2_000_000:
                s -= 40
        except Exception:
            pass
        return s

    scored = [(score(p), p) for p in paths]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for _, p in scored]


weights = []
model_dir = os.path.join(kaggle_root, "cassva-models-se50-640")
if os.path.isdir(model_dir):
    weights = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
    weights = [
        w
        for w in weights
        if os.path.isfile(w)
        and (
            w.lower().endswith(".pth")
            or w.lower().endswith(".pt")
            or w.lower().endswith(".bin")
            or w.lower().endswith(".ckpt")
            or w.lower().endswith(".pth.tar")
            or w.lower().endswith(".pt.tar")
        )
    ]
    weights.sort()

if len(weights) == 0:
    weights = find_weights_under(kaggle_root)

working_root = "/kaggle/working"
if os.path.isdir(working_root):
    weights_working = find_weights_under(working_root)
    weights = sorted(set(weights + weights_working))

weights = rank_weight_paths(weights)

print(f"Found {len(weights)} weight file(s).")
if len(weights) > 0:
    print("Top-ranked weights:", weights[:10])




## === cell 6
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "model_state",
            "model",
            "net",
            "weights",
            "params",
            "state",
            "ema",
            "model_ema",
            "ema_state_dict",
        ):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj  # may already be a state_dict


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def _iteratively_strip_prefixes(state_dict, prefixes):
    if not isinstance(state_dict, dict):
        return state_dict
    changed = True
    sd = state_dict
    while changed:
        changed = False
        for pref in prefixes:
            if any(k.startswith(pref) for k in sd.keys()):
                sd = _strip_prefix_if_present(sd, pref)
                changed = True
    return sd


def _remap_head_keys_to_custom_fc(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    candidates = (
        ("fc.weight", "_fc.weight"),
        ("fc.bias", "_fc.bias"),
        ("classifier.weight", "_fc.weight"),
        ("classifier.bias", "_fc.bias"),
        ("head.fc.weight", "_fc.weight"),
        ("head.fc.bias", "_fc.bias"),
        ("head.weight", "_fc.weight"),
        ("head.bias", "_fc.bias"),
        ("model.fc.weight", "_fc.weight"),
        ("model.fc.bias", "_fc.bias"),
        ("model.classifier.weight", "_fc.weight"),
        ("model.classifier.bias", "_fc.bias"),
        ("model.head.fc.weight", "_fc.weight"),
        ("model.head.fc.bias", "_fc.bias"),
        ("model.head.weight", "_fc.weight"),
        ("model.head.bias", "_fc.bias"),
        ("model._fc.weight", "_fc.weight"),
        ("model._fc.bias", "_fc.bias"),
    )

    sd = dict(state_dict)
    for src, dst in candidates:
        if src in sd and dst not in sd:
            sd[dst] = sd[src]
    return sd


def _filter_state_dict_by_shape(model, state_dict):
    if not isinstance(state_dict, dict):
        return state_dict, []
    model_sd = model.state_dict()
    kept = {}
    dropped = []
    for k, v in state_dict.items():
        if (
            k in model_sd
            and hasattr(v, "shape")
            and hasattr(model_sd[k], "shape")
            and tuple(v.shape) == tuple(model_sd[k].shape)
        ):
            kept[k] = v
        else:
            dropped.append(k)
    return kept, dropped


def load_checkpoint_flexible(model, weight_path, device):
    obj = torch.load(weight_path, map_location=device)
    sd = _extract_state_dict(obj)
    if isinstance(sd, dict):
        sd = _iteratively_strip_prefixes(
            sd,
            prefixes=(
                "module.",
                "model.model.",
                "model.",
                "net.",
                "backbone.",
                "encoder.",
                "student.",
            ),
        )
        sd = _remap_head_keys_to_custom_fc(sd)
        sd, dropped = _filter_state_dict_by_shape(model, sd)
    else:
        dropped = []

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected, dropped


def score_weight_compatibility(model, weight_path, device):
    try:
        obj = torch.load(weight_path, map_location="cpu")
        sd = _extract_state_dict(obj)
        if isinstance(sd, dict):
            sd = _iteratively_strip_prefixes(
                sd,
                prefixes=(
                    "module.",
                    "model.model.",
                    "model.",
                    "net.",
                    "backbone.",
                    "encoder.",
                    "student.",
                ),
            )
            sd = _remap_head_keys_to_custom_fc(sd)

            filt, _ = _filter_state_dict_by_shape(model, sd)

            n_total = len(filt)
            n_backbone = sum(1 for k in filt.keys() if k.startswith("model."))
            n_bn_stats = sum(
                1 for k in filt.keys() if ("running_mean" in k or "running_var" in k)
            )

            bonus = 0
            if "_fc.weight" in filt:
                bonus += 5000
            if "_fc.bias" in filt:
                bonus += 1000

            bonus += min(1200, n_bn_stats * 15)

            if n_backbone < 50:
                bonus -= 2000

            return n_total + bonus
    except Exception:
        return -1
    return -1


def predict_with_model(model, weights, dataset, device, sample_sub):
    chosen_weight = None

    if len(weights) > 0:
        candidates = weights[:60]

        scoring_model = Net(num_classes=5, use_pretrained_backbone=True).to("cpu")
        scored = [
            (score_weight_compatibility(scoring_model, w, device), w)
            for w in candidates
        ]
        scored.sort(reverse=True, key=lambda x: x[0])

        tried = 0
        best_loaded_score = -1
        best_loaded_weight = None

        for comp_score, w in scored[:8]:
            if comp_score <= 0:
                continue
            tried += 1

            tmp_model = Net(num_classes=5, use_pretrained_backbone=True).to(device)
            missing, unexpected, dropped = load_checkpoint_flexible(
                tmp_model, w, device
            )

            loaded_keys = len(tmp_model.state_dict()) - len(missing)
            head_loaded = int(
                ("_fc.weight" not in missing) and ("_fc.bias" not in missing)
            )
            loaded_score = loaded_keys + head_loaded * 10000

            if loaded_score > best_loaded_score:
                best_loaded_score = loaded_score
                best_loaded_weight = w

        if best_loaded_weight is not None:
            chosen_weight = best_loaded_weight
            print(
                f"Chose checkpoint after trying {tried} candidate(s): {chosen_weight}"
            )
        else:
            print(
                "No compatible checkpoint found among discovered files; will run with ImageNet backbone + random head."
            )
    else:
        print("No weights found; will run with ImageNet backbone + random head.")

    if chosen_weight is not None:
        missing, unexpected, dropped = load_checkpoint_flexible(
            model, chosen_weight, device
        )

        model.use_imagenet_norm = False
        model.eval()

        print(f"Loaded weights: {chosen_weight}")
        if len(dropped) > 0:
            print(f"  Dropped incompatible keys (showing up to 10): {dropped[:10]}")
        if len(missing) > 0:
            print(f"  Missing keys (showing up to 10): {missing[:10]}")
        if len(unexpected) > 0:
            print(f"  Unexpected keys (showing up to 10): {unexpected[:10]}")
    else:
        model.use_imagenet_norm = True
        model.eval()

    image_ids = []
    predictions = []

    for i in range(len(dataset)):
        fname, _, float_image = dataset[i]
        inp = torch.from_numpy(float_image).to(device).float()
        with torch.no_grad():
            out = model(inp)
            out = torch.softmax(out, dim=-1).detach().cpu().numpy()
            out = np.mean(out, axis=0)

        image_ids.append(fname)
        predictions.append(int(np.argmax(out)))

    cur_result = pd.DataFrame({"image_id": image_ids, "label": predictions})

    cur_result["image_id"] = cur_result["image_id"].astype(str)
    cur_result = sample_sub[["image_id"]].merge(cur_result, on="image_id", how="left")

    if cur_result["label"].isna().any():
        mode_label = (
            int(pd.Series(predictions).mode().iloc[0]) if len(predictions) else 0
        )
        cur_result["label"] = cur_result["label"].fillna(mode_label).astype(int)
    else:
        cur_result["label"] = cur_result["label"].astype(int)

    if chosen_weight is not None:
        cur_result.to_csv(f"{os.path.basename(chosen_weight)}.csv", index=False)

    cur_result.to_csv("submission.csv", index=False)
    return cur_result




## === cell 7
model = Net(num_classes=5, use_pretrained_backbone=True).to(device)

_ = predict_with_model(model, weights, dataiter, device, sample_sub)

sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample_sub)
assert sub["label"].between(0, 4).all()
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
