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
- What this solution (achieved 0.61061) has done: 'Your current gap to the target is large (0.61061 → 0.89589), so we should increase accuracy without changing the core inference logic (same backbone wrapper, same 640 resize, same 8-view TTA, same softmax-mean + argmax). The biggest low-risk gain here is fixing the dropout-at-inference bug: `model.eval()` does not disable dropout because your `Net.forward()` uses `F.dropout` semantics via an `nn.Dropout` module but currently it still be active if not correctly tied to eval—however in PyTorch modules it should disable; the real issue is that your `inp` is `float()` but the model uses dropout regardless? Actually `nn.Dropout` respects `eval()`, so the more likely score loss is from not using the pretrained model’s expected input size/crop (your 640 resize is mismatched for many timm backbones) and from the hf-hub model name often failing, falling back to ImageNet resnet50. To move score up minimally, I keep your architecture and TTA exactly the same, but (1) make the hf-hub fallback more robust by trying a small list of known cassava finetuned timm/hf-hub identifiers (no new packages), and (2) ensure the external-weight loading also handles checkpoints saved as a full `Net` but with keys not prefixed by `model.` / `_fc.` by attempting a second pass that routes keys by shape when needed. These changes are targeted to get you onto cassava-specific weights when available, which is the main lever to approach your target.'
- What this solution (achieved 0.61061) has done: 'To move your score up toward the target while keeping your core inference logic unchanged (640-resize, 8-view TTA, softmax-mean + argmax, same Net wrapper), the safest lever is to ensure you actually use cassava-finetuned weights rather than generic ImageNet weights. I add a minimal offline-first fallback that searches `/kaggle/input` for any `.pth/.pt/.bin` checkpoints (common in Kaggle datasets) and tries to load them with your existing robust key-routing, before falling back to hf-hub and then ImageNet. I also fix a subtle ensembling bug: when combining multiple checkpoints you were summing per-weight probabilities but not dividing by number of weights, which can slightly skew results due to floating point accumulation (argmax can change); I normalize by the number of checkpoints to match intended “mean over models”. These changes preserve evaluation semantics and should increase accuracy substantially if any cassava-trained checkpoints exist in the environment, otherwise behavior remains essentially the same.'
- What this solution (achieved 0.61061) has done: 'Your current score (0.61061) is far below the target (0.89589), so we should improve accuracy by ensuring you actually load cassava-finetuned weights when they exist, without changing your model/inference core (same Net wrapper, 640 resize, 8-view TTA, softmax-mean + argmax). The most likely blocker is that many Kaggle checkpoints are saved as full training dicts (`{'model_state_dict': ...}` / `{'state_dict': ...}`) and your loader may miss them, silently leaving the model on generic ImageNet weights. I minimally expand state-dict extraction to cover common key names and make checkpoint scanning prefer “best/fold/epoch/acc” filenames so the first loaded weights are higher quality. This keeps evaluation semantics identical but increases the chance you are using the intended finetuned weights, moving the score toward the target.'
- What this solution (achieved 0.61061) has done: 'Your current score (0.61061) is well below the target (0.89589), so we should increase accuracy with the smallest changes that keep your inference core (same Net wrapper, 640 resize, 8-view TTA, softmax-mean + argmax) intact. The most likely reason you’re not reaching ~0.89 is that you’re still not reliably loading cassava-finetuned weights; scanning only for filenames containing “cassava/leaf” can miss the actual checkpoint, and your loader can still miss common nesting keys. I (1) broaden offline checkpoint discovery under `/kaggle/input` with better prioritization while still excluding huge image folders, (2) expand checkpoint state-dict extraction to include more common keys (including nested `checkpoint` and `model_ema` patterns), and (3) load each checkpoint into a fresh `Net` instance to avoid partial/incompatible loads accumulating across weights. These are execution-safe, minimal changes aimed specifically at getting the model onto the correct finetuned weights so the score moves toward the target.'
- What this solution (achieved 0.61061) has done: 'We keep your model/inference core exactly the same (same Net wrapper, 640 resize, 8-view TTA, softmax-mean + argmax) and focus only on ensuring you actually use cassava-finetuned weights when they exist, since 0.61061 strongly suggests you’re still often running ImageNet weights. The most likely miss is that many Kaggle checkpoints store the best weights under `model_ema` (or `state_dict_ema`) and your loader currently doesn’t reliably prefer EMA over non-EMA. I minimally adjust `_extract_state_dict` to prioritize EMA keys when present, and also ensure the “backbone_name” used for re-instantiated `model_j` is always the same as the original (so we don’t accidentally create a mismatched architecture that prevents proper loading). These changes are small, execution-safe, and directly aimed at increasing accuracy toward your target by improving correct weight loading.'
- What this solution (achieved 0.61061) has done: 'Your current score (0.61061) is far below the target (0.89589), so the most likely path toward the target—without changing your model/TTA/inference semantics—is to make sure you actually load high-quality cassava-finetuned weights when they exist. I keep your exact Net wrapper, 640 resize, 8-view TTA, softmax-mean + argmax, and only make the checkpoint discovery/loading more reliable: (1) scan slightly deeper but still safely under `/kaggle/input` while explicitly skipping huge folders, and (2) prioritize and correctly extract EMA/best state_dicts (including `model_ema`, `ema`, and `model_ema.module` patterns) so you don’t silently fall back to ImageNet weights. I also add a tiny “sanity print” of how many tensors matched when loading the first checkpoint to quickly confirm you’re not doing a near-empty load. These are minimal, score-relevant changes aimed at moving accuracy up toward your target.'
- What this solution (achieved 0.61061) has done: 'We keep your exact inference core (same Net wrapper, 640-resize, 8-view TTA, softmax-mean + argmax, model-mean-over-checkpoints) and only make the weight-loading more reliable, because 0.61061 strongly suggests you’re still frequently running ImageNet weights. The smallest score-relevant fix is to (1) expand checkpoint discovery to also consider `.ckpt` files and raise the cap slightly, and (2) improve state-dict extraction to prefer `model_ema`/`ema` when present and handle common Lightning `state_dict` key formats. Finally, we ensure that if a checkpoint contains a full wrapper (`model.*` and `_fc.*`) we load both parts, but if it contains a plain backbone we still route by exact key+shape match as you already do—this preserves semantics while increasing the chance of actually using cassava-finetuned weights. These changes should move accuracy upward toward the target without changing architecture, TTA, or post-processing.'
- What this solution (achieved 0.61061) has done: 'The current score (0.61061) is far below the target (0.89589), so we should increase accuracy while keeping your inference core unchanged (same Net wrapper, 640 resize, 8-view TTA, softmax-mean + argmax, and checkpoint ensembling). The biggest likely issue is that when external checkpoints are found, you always re-instantiate the model with `backbone_name_used="seresnext50_32x4d"` even if the checkpoint was trained with a different backbone, causing partial/failed loads and effectively ImageNet-level predictions. I minimally add backbone auto-detection from checkpoint keys (EfficientNet/ResNet/SE-ResNeXt patterns) and only use it to choose the correct timm backbone when creating `model_j` for that checkpoint. This keeps the model architecture/loop the same, but makes weight loading actually work, which should move the score substantially toward the target. I also avoid forcing `model.` prefix stripping too early (it can destroy the wrapper’s `model.*` namespace) so wrapper/backbone splitting works as intended.'
- What this solution (achieved 0.61061) has done: 'Your current score (0.61061) is far below the target (0.89589), so we should increase accuracy with the smallest changes that keep your inference/TTA/argmax semantics intact. The biggest likely miss is that most “good” cassava checkpoints are EfficientNet-based (often b4) and your auto-backbone inference is too brittle (it rarely triggers), leading to wrong-backbone instantiation and near-empty loads. I minimally improve backbone inference using robust key-patterns (including classifier/fc and EfficientNet block naming) and also allow loading “head” weights when checkpoints use common names like `classifier.*`/`head.fc.*` instead of `_fc.*`, so more finetuned weights actually take effect. Everything else (640 resize, 8-view TTA, softmax averaging, checkpoint ensembling, submission writing) stays the same.'

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
def _find_checkpoints_under(
    root, exts=(".pth", ".pt", ".bin", ".ckpt"), max_files=200, max_depth=7
):
    """
    Keep: consider .ckpt (Lightning) and allow a few more files, increasing the chance we find
    cassava-finetuned weights.
    """
    found = []
    skip_basenames = {
        "train_images",
        "test_images",
        "train_tfrecords",
        "test_tfrecords",
        ".git",
        "__pycache__",
    }

    root = os.path.abspath(root)
    root_depth = root.rstrip(os.sep).count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if base in skip_basenames:
            dirnames[:] = []
            continue

        depth = os.path.abspath(dirpath).rstrip(os.sep).count(os.sep) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if d not in skip_basenames]

        for fn in filenames:
            if fn.lower().endswith(exts):
                found.append(os.path.join(dirpath, fn))
                if len(found) >= max_files:
                    break
        if len(found) >= max_files:
            break

    def _prio(p):
        name = os.path.basename(p).lower()
        score = 0

        if "best" in name:
            score -= 120
        if "final" in name or "last" in name:
            score -= 20
        if "fold" in name:
            score -= 25
        if "ema" in name:
            score -= 35

        if "optimizer" in name or "sched" in name or "scaler" in name:
            score += 150

        score += min(len(name) / 1000.0, 0.1)
        return (score, name)

    return sorted(found, key=_prio)


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
            if f.lower().endswith((".pth", ".pt", ".bin", ".ckpt"))
        ]
        if len(weights) > 0:
            break

if len(weights) == 0:
    weights = _find_checkpoints_under(kaggle_root, max_files=120, max_depth=7)

use_external_weights = len(weights) > 0

backbone_name_used = "seresnext50_32x4d"

if use_external_weights:
    model = Net(num_classes=5, backbone_name=backbone_name_used, pretrained=False).to(
        device
    )
    print(f"Using external weights ({len(weights)}), showing up to 15:")
    for p in weights[:15]:
        print("  ", p)
    if len(weights) > 15:
        print("  ...")
else:
    cassava_backbone_candidates = [
        "hf-hub:nateraw/cassava_leaf_disease_efficientnet_b4",
        "hf-hub:nateraw/cassava_leaf_disease_efficientnet_b3",
        "hf-hub:nateraw/cassava_leaf_disease_efficientnet_b0",
    ]
    last_err = None
    model = None
    for backbone_name in cassava_backbone_candidates:
        try:
            model = Net(num_classes=5, backbone_name=backbone_name, pretrained=True).to(
                device
            )
            backbone_name_used = backbone_name
            print(f"Using cassava pretrained backbone: {backbone_name}")
            break
        except Exception as e:
            last_err = e
            continue

    if model is None:
        print(
            f"hf-hub cassava backbones not available ({last_err}); using ImageNet pretrained resnet50."
        )
        backbone_name_used = "resnet50"
        model = Net(
            num_classes=5, backbone_name=backbone_name_used, pretrained=True
        ).to(device)




## === cell 7
def _extract_state_dict(obj):
    """
    Keep: handle more checkpoint layouts and prefer EMA weights when available.
    """
    if not isinstance(obj, dict):
        return obj

    for k in ("state_dict", "model_state_dict"):
        if k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            break

    for k in ("checkpoint", "ckpt", "trainer_state", "train_state"):
        if k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            break

    for k in (
        "model_ema_state_dict",
        "state_dict_ema",
        "ema_state_dict",
        "model_ema",
        "ema",
        "ema_model",
    ):
        if k in obj:
            v = obj[k]
            if isinstance(v, dict):
                for kk in ("state_dict", "model_state_dict", "model"):
                    if kk in v and isinstance(v[kk], dict):
                        return v[kk]
                return v

    for k in (
        "model",
        "net",
        "weights",
        "model_state_dict",
        "state_dict",
    ):
        if k in obj:
            v = obj[k]
            if isinstance(v, dict):
                for kk in ("state_dict", "model_state_dict", "model"):
                    if kk in v and isinstance(v[kk], dict):
                        return v[kk]
                return v

    return obj


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _split_wrapper_keys_for_loading(state_dict):
    """
    Change (score-relevant, minimal): support common head key namespaces used in many timm training
    scripts (classifier/head/fc), not just '_fc'. This increases the chance head weights load.
    """
    if not isinstance(state_dict, dict):
        return None, None, state_dict

    sd = _strip_prefix_if_present(state_dict, "module.")
    sd = _strip_prefix_if_present(sd, "model_ema.")
    sd = _strip_prefix_if_present(sd, "ema.")

    backbone_sd = {}
    head_sd = {}
    other_sd = {}

    for k, v in sd.items():
        if k.startswith("model."):
            backbone_sd[k[len("model.") :]] = v
        elif k.startswith("_fc."):
            head_sd[k[len("_fc.") :]] = v
        elif k.startswith("classifier."):
            head_sd[k[len("classifier.") :]] = v
        elif k.startswith("head.fc."):
            head_sd[k[len("head.fc.") :]] = v
        elif k.startswith("fc."):
            head_sd[k[len("fc.") :]] = v
        else:
            other_sd[k] = v

    return backbone_sd, head_sd, other_sd


def _route_by_shape_if_needed(model, sd):
    """
    Keep: route by exact key+shape match when checkpoint doesn't use wrapper prefixes.
    """
    if not isinstance(sd, dict):
        return None, None

    model_sd = model.model.state_dict()
    head_sd_ref = model._fc.state_dict()

    backbone_sd = {}
    head_sd = {}

    for k, v in sd.items():
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
            backbone_sd[k] = v
        elif (
            k in head_sd_ref and hasattr(v, "shape") and v.shape == head_sd_ref[k].shape
        ):
            head_sd[k] = v

    if len(backbone_sd) == 0 and len(head_sd) == 0:
        return None, None
    return backbone_sd, head_sd


def _infer_backbone_name_from_state_dict(sd, default_name):
    """
    Change (score-relevant, minimal): make backbone inference robust so we don't instantiate
    the wrong architecture (which would prevent loading finetuned weights and keep score low).
    """
    if not isinstance(sd, dict) or len(sd) == 0:
        return default_name

    keys = list(sd.keys())
    joined = " ".join(keys[:500]).lower()

    is_effnet = (
        ("conv_stem" in joined and "blocks." in joined)
        or ("conv_stem" in joined and "conv_head" in joined)
        or ("blocks." in joined and "bn1." in joined and "conv_stem" in joined)
    )
    if is_effnet:
        if (
            "efficientnet_b4" in joined
            or "tf_efficientnet_b4" in joined
            or "b4" in joined
        ):
            return "efficientnet_b4"
        if (
            "efficientnet_b3" in joined
            or "tf_efficientnet_b3" in joined
            or "b3" in joined
        ):
            return "efficientnet_b3"
        if (
            "efficientnet_b2" in joined
            or "tf_efficientnet_b2" in joined
            or "b2" in joined
        ):
            return "efficientnet_b2"
        if (
            "efficientnet_b1" in joined
            or "tf_efficientnet_b1" in joined
            or "b1" in joined
        ):
            return "efficientnet_b1"
        return "efficientnet_b0"

    if (
        "layer1." in joined
        and "layer4." in joined
        and ("conv1." in joined or "conv1.weight" in joined)
    ):
        return "resnet50"

    if (
        "layer1." in joined
        and "layer4." in joined
        and ("se_module" in joined or "semodule" in joined)
    ):
        return "seresnext50_32x4d"

    return default_name


def predict_with_model(model, weights, dataiter, device, backbone_name_used):
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

            inferred_backbone = _infer_backbone_name_from_state_dict(
                state, default_name=backbone_name_used
            )

            model_j = Net(
                num_classes=5,
                backbone_name=inferred_backbone,
                pretrained=False,
            ).to(device)
            model_j.eval()

            backbone_sd, head_sd, other_sd = _split_wrapper_keys_for_loading(state)

            loaded_any = False
            msg_parts = [f"backbone={inferred_backbone}"]

            if isinstance(backbone_sd, dict) and len(backbone_sd) > 0:
                missing_b, unexpected_b = model_j.model.load_state_dict(
                    backbone_sd, strict=False
                )
                loaded_any = True
                msg_parts.append(
                    f"backbone loaded={len(backbone_sd)} missing={len(missing_b)} unexpected={len(unexpected_b)}"
                )

            if isinstance(head_sd, dict) and len(head_sd) > 0:
                missing_h, unexpected_h = model_j._fc.load_state_dict(
                    head_sd, strict=False
                )
                loaded_any = True
                msg_parts.append(
                    f"head loaded={len(head_sd)} missing={len(missing_h)} unexpected={len(unexpected_h)}"
                )

            if not loaded_any:
                missing, unexpected = model_j.load_state_dict(other_sd, strict=False)
                msg_parts.append(
                    f"wrapper missing={len(missing)} unexpected={len(unexpected)}"
                )

                routed_backbone_sd, routed_head_sd = _route_by_shape_if_needed(
                    model_j, other_sd
                )
                if routed_backbone_sd is not None:
                    missing_b, unexpected_b = model_j.model.load_state_dict(
                        routed_backbone_sd, strict=False
                    )
                    msg_parts.append(
                        f"shape-route backbone loaded={len(routed_backbone_sd)} missing={len(missing_b)} unexpected={len(unexpected_b)}"
                    )
                if routed_head_sd is not None:
                    missing_h, unexpected_h = model_j._fc.load_state_dict(
                        routed_head_sd, strict=False
                    )
                    msg_parts.append(
                        f"shape-route head loaded={len(routed_head_sd)} missing={len(missing_h)} unexpected={len(unexpected_h)}"
                    )

            if j == 0:
                print(f"Loaded {os.path.basename(weight)}: " + " | ".join(msg_parts))

            model_j.eval()
        else:
            model_j = model
            model_j.eval()

        for i, fname in enumerate(filenames):
            if (i + 1) % 250 == 0 or (i + 1) == len(filenames):
                print(f"weight {j+1}/{len(weights)}: data {i+1}/{len(filenames)}")

            _, _, float_image = dataiter.__getitem__(i)
            inp = torch.from_numpy(float_image).to(device).float()

            with torch.no_grad():
                out = model_j(inp)
                out = torch.nn.functional.softmax(out, dim=-1).cpu().numpy()
                out = np.mean(out, axis=0)  # TTA mean over 8 views

            if merge_res[fname] is None:
                merge_res[fname] = out
            else:
                merge_res[fname] += out

    denom = float(len(weights)) if len(weights) > 0 else 1.0
    pred_labels = [int(np.argmax(merge_res[fname] / denom)) for fname in filenames]

    sub = pd.DataFrame({"image_id": filenames, "label": pred_labels})
    sub.to_csv("submission.csv", index=False)
    return sub


sub = predict_with_model(model, weights, dataiter, device, backbone_name_used)
print(sub.head())
print(f"Wrote submission.csv with {len(sub)} rows")
