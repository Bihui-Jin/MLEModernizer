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

0.8747355696585071

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.40247) has done: 'Your notebook didn’t yield a score because it fail to run in this environment due to missing external weight files (`/kaggle/input/casava-aug/...` and `/kaggle/input/eff-t/...`), so the first minimal fix is to make weight loading robust and still always produce a valid `submission.csv`. To keep the same core ensemble logic, we (1) try multiple likely weight locations and fall back to torchvision ImageNet weights if competition weights aren’t available, and (2) load weights with `map_location=device` for both models to avoid device mismatches. Finally, we guarantee submission row order matches `sample_submission.csv` (so Kaggle reads it correctly) and add a small safety check for unreadable images so the dataloader doesn’t crash.'
- What this solution (achieved 0.16667) has done: 'Your current score is low because the fallback path uses ImageNet backbones but replaces the classifiers with random 5-class heads, so predictions are close to random. To move the score toward the 0.8747 target while preserving your inference/ensemble logic, the minimal fix is to actually load valid 5-class competition weights from the dataset you already have: the provided TFRecords contain a well-known public pretrained fold checkpoint (`tf_efficientnet_b4_ns_0.916_best.pth`) that can be used directly for inference. I keep your ResNet branch intact (still tries to load your ResNet weights, otherwise falls back), and I keep the same “softmax then weighted average then argmax” ensemble semantics, only swapping the EfficientNet-V2-S branch to a compatible EfficientNet-B4 model if that checkpoint exists. This should substantially increase accuracy and move your score much closer to the target without changing the overall approach.'
- What this solution (achieved 0.14312) has done: 'Your low score is consistent with the EfficientNet-B4 “cassava checkpoint” path never being found (that `.pth` file is not part of the official dataset), so the code falls back to ImageNet backbones with a randomly initialized 5-class head (near-random predictions). To move accuracy toward the 0.8747 target while keeping the same two-model softmax-weighted ensemble logic, the minimal fix is to (1) ensure we only use ImageNet weights for the backbone when we cannot find real 5-class cassava weights, and (2) avoid leaving any model with a random classifier head by defaulting to a single strong model (ResNet) when the EfficientNet head is untrained. I also fix the submission dtype creation (`float` -> `int`) to avoid any unintended casting issues and keep row order exactly as `sample_submission.csv`. These changes preserve your architecture/inference semantics but eliminate the “random head” failure mode that drives the 0.16667 score.'
- What this solution (achieved 0.61099) has done: 'Your current low score is mainly driven by both branches often falling back to ImageNet backbones with a randomly initialized 5‑class head, which makes predictions near-random. To move accuracy toward the 0.8747 target while preserving your exact two-model “softmax → weighted average → argmax” ensemble logic, I (1) ensure that if a branch has no real 5-class cassava checkpoint we *do not use it at all* (weight=0) rather than ensembling in random outputs, and (2) add a safe “make backbone features usable” fallback by using a deterministic class-prior head computed from `train.csv` when no cassava checkpoints exist (still outputs valid logits; no training loop changes). I also fix the misleading EfficientNet-B4 checkpoint path (it’s not in the official dataset) by searching a small set of realistic locations under `/kaggle/input` and only enabling that branch if actually found. These are minimal changes that should substantially raise accuracy compared to near-random predictions, without changing your architecture or inference semantics.'
- What this solution (achieved 0.61099) has done: 'Your score (0.61099) is far below the target (0.8747), and the biggest gap-driver in your current logic is that the ensemble often degenerates into a class-prior fallback because no real cassava-trained checkpoints are found/loaded. The minimal, core-logic-preserving improvement is to make checkpoint discovery actually work in this environment by (1) recursively searching under `/kaggle/input` for likely `.pth` filenames you already reference, and (2) fixing the common “key prefix mismatch” (`model.`, `module.`, etc.) so a found checkpoint properly loads into the existing architectures. This keeps your exact inference semantics (softmax → weighted average → argmax) and only increases the chance that one/both branches use trained 5-class heads, which should move accuracy substantially toward the target without changing the modeling approach. Finally, I keep submission ordering aligned to `sample_submission.csv` as you already do and still write `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your score gap is large (0.61099 vs target 0.87474), and with your current constraints the most likely cause is that neither branch is actually loading real cassava-trained 5-class weights, so you’re either using only one branch or falling back to a class-prior. The smallest high-impact change is to broaden checkpoint discovery to include common EfficientNet/ResNet cassava checkpoint filenames and to make loading stricter in a safe way: we only mark a model as “trained head” if the classifier/fc weights actually load with matching shapes. This preserves your exact inference semantics (softmax → weighted average → argmax) and only increases the chance that at least one branch uses true cassava weights, pushing accuracy toward the target. Finally, we keep submission ordering identical to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your score gap is large (0.61099 vs 0.87474), so the most likely minimal win is to make sure at least one branch actually loads a real cassava-trained 5-class head rather than falling back to an untrained head or priors. I keep your exact inference semantics (softmax → weighted average → argmax) and architectures, but improve checkpoint discovery and loading robustness by (1) also searching for `.pt/.bin` files and common Kaggle Cassava checkpoint names, and (2) fixing the common “head key mismatch” (e.g., `fc.*` vs `classifier.*`) by remapping keys only when shapes match your existing heads. Finally, if both trained heads are found, I keep your ensemble but add a tiny “renormalize weights” safety so combined probabilities are correctly scaled even if weights change, without altering predictions when weights already sum to 1.'
- What this solution (achieved 0.61099) has done: 'Your current gap to the target is large (0.61099 vs 0.8747), and the most likely reason (given your existing safeguards) is that you still aren’t actually loading any real cassava-trained checkpoints, so predictions collapse to a label-prior or a single weak branch. I keep your exact inference semantics (softmax → weighted average → argmax) and the same two-branch pipeline, but make the smallest high-impact fix: add a robust, shape-checked loader that can pull cassava-trained weights from common checkpoint structures and key names (including `state_dict` nesting and `fc/classifier` mismatches) without accidentally “accepting” partial/random heads. I also fix the EfficientNet transform size to match the actual model chosen (B4=380, V2-S=384) so you don’t silently lose accuracy via an input-size mismatch. Finally, submission creation stays identical and still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your score is far below the target, so we should improve accuracy with the smallest change that preserves your ensemble inference semantics. The biggest likely issue is that your checkpoint discovery still often misses real cassava-trained weights, so both branches can end up disabled and you fall back to a class-prior (~0.61). I keep your exact models and “softmax → weighted average → argmax” logic, but (1) broaden checkpoint search to include any `.pth/.pt/.bin` under `/kaggle/input` and score them by “has matching 5-class head + lots of tensors”, and (2) fix a key mismatch for EfficientNet-B4 heads (`classifier.weight/bias` vs `classifier.1.*`) so valid checkpoints actually load as “trained head.” This should more reliably activate at least one trained branch and move accuracy toward the target while still producing the same submission format.'
- What this solution (achieved 0.61099) has done: 'Your gap to the target is large (0.61099 vs 0.87474), and the most likely reason is that you are still not actually finding/loading any real cassava-trained 5-class checkpoints, so predictions collapse to the train-label prior. I keep your exact two-branch “softmax → weighted average → argmax” ensemble semantics and model definitions, but make checkpoint discovery actually work by scanning `/kaggle/input` for *any* plausible cassava checkpoints and selecting the best one using a stricter “matching 5-class head + many tensors” heuristic (instead of stopping at the first filename hit). I also make the state-dict loader handle a few additional common wrappers (`ema_state_dict`, `model_ema`) and only mark a branch “trained” if its head weights truly match, so we don’t accidentally ensemble random heads. These are minimal, execution-safe changes that should activate at least one real trained branch and push accuracy upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current 0.61099 is consistent with both models still not loading any real cassava 5-class checkpoints (so the ensemble collapses to the train-label prior). The smallest high-impact change that preserves your exact “softmax → weighted average → argmax” semantics is to make checkpoint selection actually find *the best available* cassava checkpoint by scoring candidates more intelligently (prefer matching 5-class head plus strong “cassava-like” signatures), instead of relying on filename guesses. I also expand state-dict extraction to handle a few additional common wrappers (`model_state`, `model_ema_state_dict`, etc.) so found checkpoints load correctly without changing your model definitions. Finally, submission ordering/format stays identical and still writes `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your score (0.61099) suggests you’re still often falling back to the train-label prior because no true cassava-trained 5-class checkpoint is being found/loaded. The smallest high-impact improvement (without changing your models or ensemble semantics) is to make checkpoint discovery both broader and smarter: scan `/kaggle/input` for plausible `.pth/.pt/.bin` files, then select the best one by (a) requiring a matching 5-class head and (b) preferring checkpoints whose backbone keys match the target architecture (ResNet vs EfficientNet) rather than just filename tokens. I also make the state-dict extraction handle a few additional common wrappers and ensure we don’t “accept” partial/incorrect checkpoints by verifying a minimum overlap of keys with the model. This should activate at least one genuinely trained branch more reliably and move accuracy upward toward your target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'The current 0.61099 score strongly indicates you’re still falling back to the train-label prior (or a single weak branch) because no real cassava-trained 5-class checkpoint is actually being loaded. To move accuracy toward the 0.8747 target with minimal semantic change, I keep your exact two-branch softmax→weighted-average→argmax ensemble, but broaden checkpoint *discovery* to include common safe locations already present in this dataset (especially `/kaggle/input/cassava-leaf-disease-classification/**` and the duplicated `/kaggle/input/**` trees) and then choose the best checkpoint by your existing “matching 5-class head + overlap + backbone signature” scoring. I also add one small, execution-safe improvement: use `num_workers=2` (still deterministic and no approximations) to speed I/O so the broader search + inference stays under the time limit. Submission formatting/order is kept identical and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = os.path.join(DATA_ROOT, "test_images")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 3
resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def make_efficientnet_transforms(img_size: int):
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )




## === cell 4
class CassavaTestDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, transform_resnet=None, transform_efficientnet=None
    ):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform_resnet = transform_resnet
        self.transform_efficientnet = transform_efficientnet

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform_resnet:
            image_resnet = self.transform_resnet(image=image)["image"]
        else:
            image_resnet = None

        if self.transform_efficientnet:
            image_efficientnet = self.transform_efficientnet(image=image)["image"]
        else:
            image_efficientnet = None

        return image_resnet, image_efficientnet, img_name




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 6
def _normalize_state_dict_keys(sd: dict) -> dict:
    """
    Keep existing robustness: strip common wrappers (DDP/Lightning).
    """
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        if k.startswith("model."):
            k = k.replace("model.", "", 1)
        if k.startswith("net."):
            k = k.replace("net.", "", 1)
        out[k] = v
    return out


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model_state",
            "model",
            "net",
            "weights",
            "ema_state_dict",
            "model_ema",
            "model_ema_state_dict",
            "model_ema_weights",
            "ema",
            "ema_model",
            "model_dict",
            "model_state",
            "model_weights",
        ):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        for key in ("state_dict", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                inner = ckpt_obj[key]
                for key2 in ("state_dict", "model", "net"):
                    if key2 in inner and isinstance(inner[key2], dict):
                        return inner[key2]
    return ckpt_obj


def _state_dict_has_param(sd: dict, key: str, expected_shape=None) -> bool:
    if not isinstance(sd, dict):
        return False
    if key not in sd:
        return False
    v = sd[key]
    if not torch.is_tensor(v):
        return False
    if expected_shape is not None and tuple(v.shape) != tuple(expected_shape):
        return False
    return True


def _remap_head_keys_if_compatible(sd: dict, model) -> dict:
    """
    Many cassava checkpoints use different head key names.
    We only remap when tensor shapes match the current model head exactly.
    """
    if not isinstance(sd, dict):
        return sd

    new_sd = dict(sd)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        fc_w_key, fc_b_key = "fc.weight", "fc.bias"
        if (fc_w_key not in new_sd) and ("classifier.weight" in new_sd):
            if tuple(new_sd["classifier.weight"].shape) == tuple(model.fc.weight.shape):
                new_sd[fc_w_key] = new_sd["classifier.weight"]
        if (fc_b_key not in new_sd) and ("classifier.bias" in new_sd):
            if tuple(new_sd["classifier.bias"].shape) == tuple(model.fc.bias.shape):
                new_sd[fc_b_key] = new_sd["classifier.bias"]

        if (fc_w_key not in new_sd) and ("head.weight" in new_sd):
            if tuple(new_sd["head.weight"].shape) == tuple(model.fc.weight.shape):
                new_sd[fc_w_key] = new_sd["head.weight"]
        if (fc_b_key not in new_sd) and ("head.bias" in new_sd):
            if tuple(new_sd["head.bias"].shape) == tuple(model.fc.bias.shape):
                new_sd[fc_b_key] = new_sd["head.bias"]

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        if len(model.classifier) >= 2 and isinstance(model.classifier[1], nn.Linear):
            w_key, b_key = "classifier.1.weight", "classifier.1.bias"

            if (w_key not in new_sd) and ("fc.weight" in new_sd):
                if tuple(new_sd["fc.weight"].shape) == tuple(
                    model.classifier[1].weight.shape
                ):
                    new_sd[w_key] = new_sd["fc.weight"]
            if (b_key not in new_sd) and ("fc.bias" in new_sd):
                if tuple(new_sd["fc.bias"].shape) == tuple(
                    model.classifier[1].bias.shape
                ):
                    new_sd[b_key] = new_sd["fc.bias"]

            if (w_key not in new_sd) and ("classifier.weight" in new_sd):
                if tuple(new_sd["classifier.weight"].shape) == tuple(
                    model.classifier[1].weight.shape
                ):
                    new_sd[w_key] = new_sd["classifier.weight"]
            if (b_key not in new_sd) and ("classifier.bias" in new_sd):
                if tuple(new_sd["classifier.bias"].shape) == tuple(
                    model.classifier[1].bias.shape
                ):
                    new_sd[b_key] = new_sd["classifier.bias"]

            if (w_key not in new_sd) and ("head.weight" in new_sd):
                if tuple(new_sd["head.weight"].shape) == tuple(
                    model.classifier[1].weight.shape
                ):
                    new_sd[w_key] = new_sd["head.weight"]
            if (b_key not in new_sd) and ("head.bias" in new_sd):
                if tuple(new_sd["head.bias"].shape) == tuple(
                    model.classifier[1].bias.shape
                ):
                    new_sd[b_key] = new_sd["head.bias"]

    return new_sd


def _try_load_state_dict(model, candidate_paths, device):
    """
    Keep behavior but add head-key remapping when shape-compatible.
    """
    for p in candidate_paths:
        if p is None:
            continue
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=device)
            sd = _extract_state_dict(ckpt)
            if isinstance(sd, dict):
                sd = _normalize_state_dict_keys(sd)
                sd = _remap_head_keys_if_compatible(sd, model)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            return True, p, missing, unexpected, sd
    return False, None, None, None, None


def _fast_search_kaggle_input_for_filenames(
    filenames, root="/kaggle/input", max_hits_per_name=8
):
    filenames = set(filenames)
    hits = {fn: [] for fn in filenames}
    for dirpath, _, files in os.walk(root):
        common = filenames.intersection(files)
        if common:
            for fn in common:
                if len(hits[fn]) < max_hits_per_name:
                    hits[fn].append(os.path.join(dirpath, fn))
        if all(len(v) >= max_hits_per_name for v in hits.values()):
            break

    flattened = []
    for fn in filenames:
        flattened.extend(hits[fn])
    seen = set()
    out = []
    for p in flattened:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _collect_ckpt_files_prioritized(
    root="/kaggle/input",
    exts=(".pth", ".pt", ".bin"),
    max_files=900,
    prefer_tokens=(
        "cassava",
        "leaf",
        "disease",
        "eff",
        "efficient",
        "resnet",
        "b4",
        "v2",
        "fold",
        "best",
        "seresnext",
        "se_resnext",
        "tf_efficientnet",
    ),
):
    preferred = []
    others = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            lo = fn.lower()
            if not lo.endswith(exts):
                continue
            full = os.path.join(dirpath, fn)
            if any(tok in lo for tok in prefer_tokens):
                preferred.append(full)
            else:
                others.append(full)
            if len(preferred) >= max_files:
                return preferred[:max_files]
            if len(preferred) + len(others) >= max_files:
                break
        if len(preferred) + len(others) >= max_files:
            break
    return preferred + others[: max(0, max_files - len(preferred))]


def _ckpt_has_matching_5class_head(sd: dict, model) -> bool:
    if not isinstance(sd, dict):
        return False

    sd = _normalize_state_dict_keys(sd)
    sd = _remap_head_keys_if_compatible(sd, model)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return _state_dict_has_param(
            sd, "fc.weight", expected_shape=tuple(model.fc.weight.shape)
        ) and _state_dict_has_param(
            sd, "fc.bias", expected_shape=tuple(model.fc.bias.shape)
        )

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        if len(model.classifier) >= 2 and isinstance(model.classifier[1], nn.Linear):
            w_key, b_key = "classifier.1.weight", "classifier.1.bias"
            return _state_dict_has_param(
                sd, w_key, expected_shape=tuple(model.classifier[1].weight.shape)
            ) and _state_dict_has_param(
                sd, b_key, expected_shape=tuple(model.classifier[1].bias.shape)
            )

    return False


def _backbone_signature_score(sd: dict, model) -> int:
    """
    Prefer checkpoints whose backbone keys match the intended architecture.
    """
    if not isinstance(sd, dict):
        return -1
    sd = _normalize_state_dict_keys(sd)

    keys = sd.keys()
    score = 0

    if hasattr(model, "layer1") and hasattr(model, "conv1"):
        sig = (
            "conv1.weight",
            "bn1.weight",
            "layer1.0.conv1.weight",
            "layer2.0.conv1.weight",
            "layer3.0.conv1.weight",
            "layer4.0.conv1.weight",
        )
        score += sum(1 for k in sig if k in keys) * 30

    if hasattr(model, "features"):
        sig = (
            "features.0.0.weight",
            "features.1.0.block.0.0.weight",
            "features.1.0.block.0.1.weight",
        )
        score += sum(1 for k in sig if k in keys) * 30

    return score


def _state_dict_overlap_score(sd: dict, model) -> float:
    """
    Require reasonable overlap so we don't pick a checkpoint that only contains a head.
    """
    if not isinstance(sd, dict):
        return 0.0
    sd = _normalize_state_dict_keys(sd)
    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    inter = len(model_keys.intersection(sd_keys))
    return inter / max(1, len(model_keys))


def _score_candidate_ckpt_for_model(path: str, model, device) -> int:
    """
    Prefer checkpoints that:
    (a) have a matching 5-class head
    (b) overlap model backbone
    (c) look like real full fine-tunes
    """
    try:
        ckpt = torch.load(path, map_location=device)
        sd = _extract_state_dict(ckpt)
        if not isinstance(sd, dict):
            return -1

        sd = _normalize_state_dict_keys(sd)
        sd = _remap_head_keys_if_compatible(sd, model)

        if not _ckpt_has_matching_5class_head(sd, model):
            return -1

        overlap = _state_dict_overlap_score(sd, model)
        if overlap < 0.35:
            return -1

        n = len(sd)
        lo = os.path.basename(path).lower()

        bonus = 0
        for tok in ("cassava", "leaf", "disease"):
            if tok in lo:
                bonus += 200
        for tok in ("fold", "best", "final"):
            if tok in lo:
                bonus += 75
        for tok in ("resnet50", "efficientnet", "tf_efficientnet", "eff", "b4", "v2"):
            if tok in lo:
                bonus += 30

        bonus += _backbone_signature_score(sd, model)

        if n < 50:
            return -1
        if n < 150:
            bonus -= 50

        bonus += int(400 * overlap)

        return n + bonus
    except Exception:
        return -1


def _select_best_ckpt(candidate_paths, model, device):
    existing = [p for p in candidate_paths if p and os.path.exists(p)]
    if not existing:
        return None
    scored = []
    for p in existing:
        s = _score_candidate_ckpt_for_model(p, model, device)
        if s >= 0:
            scored.append((s, p))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def _collect_ckpt_files_multi_roots(
    roots,
    exts=(".pth", ".pt", ".bin"),
    max_files_total=1400,
):
    out = []
    seen = set()
    per_root = max(200, max_files_total // max(1, len(roots)))
    for r in roots:
        if not r or (not os.path.exists(r)):
            continue
        files = _collect_ckpt_files_prioritized(root=r, exts=exts, max_files=per_root)
        for p in files:
            if p not in seen:
                seen.add(p)
                out.append(p)
            if len(out) >= max_files_total:
                return out
    return out




## === cell 7
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_ckpt_filenames = [
    "cassava_leaf_best_model_fine_aug.pth",
    "cassava_leaf_best_model_fine_aug.pt",
    "resnet50.pth",
    "resnet50.pt",
    "resnet50_fold0.pth",
    "resnet50_fold1.pth",
    "resnet50_best.pth",
    "resnet_best.pth",
    "best.pth",
    "model.pth",
    "checkpoint.pth",
]
resnet_weight_candidates = [
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    "/kaggle/input/casava-aug/cassava_leaf_best_model_fine_aug.pth",
    "/kaggle/input/cassava-leaf-disease-classification/cassava_leaf_best_model_fine_aug.pth",
]

resnet_weight_candidates += _fast_search_kaggle_input_for_filenames(
    resnet_ckpt_filenames, root="/kaggle/input"
)
resnet_weight_candidates += _fast_search_kaggle_input_for_filenames(
    resnet_ckpt_filenames, root=DATA_ROOT
)

resnet_weight_candidates += _collect_ckpt_files_multi_roots(
    roots=[
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input",
    ],
    max_files_total=1400,
)

best_resnet_ckpt = _select_best_ckpt(resnet_weight_candidates, resnet_model, device)
resnet_load_list = [best_resnet_ckpt] if best_resnet_ckpt else resnet_weight_candidates

loaded_resnet, resnet_path, resnet_missing, resnet_unexpected, resnet_sd = (
    _try_load_state_dict(resnet_model, resnet_load_list, device)
)

resnet_has_trained_head = False
if loaded_resnet and isinstance(resnet_sd, dict):
    try:
        resnet_sd_norm = _normalize_state_dict_keys(
            _extract_state_dict(torch.load(resnet_path, map_location=device))
            if resnet_path
            else resnet_sd
        )
    except Exception:
        resnet_sd_norm = _normalize_state_dict_keys(resnet_sd)
    resnet_sd_norm = _remap_head_keys_if_compatible(resnet_sd_norm, resnet_model)
    resnet_has_trained_head = _ckpt_has_matching_5class_head(
        resnet_sd_norm, resnet_model
    )

if not loaded_resnet:
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 5)
    resnet_has_trained_head = False

resnet_model = resnet_model.to(device)
resnet_model.eval()

print(
    "ResNet weights loaded from:",
    (
        resnet_path
        if loaded_resnet
        else "torchvision ImageNet (fallback backbone + UNTRAINED 5-class head)"
    ),
)
print("ResNet trained-head flag:", resnet_has_trained_head)



## === cell 8
eff_b4_candidates = [
    "/kaggle/input/tf-efficientnet-pytorch/tf_efficientnet_b4_ns_0.916_best.pth",
    "/kaggle/input/efficientnet-b4-cassava/tf_efficientnet_b4_ns_0.916_best.pth",
    "/kaggle/input/cassava-checkpoints/tf_efficientnet_b4_ns_0.916_best.pth",
    os.path.join(
        "/kaggle/input/cassava-leaf-disease-classification",
        "train_tfrecords",
        "tf_efficientnet_b4_ns_0.916_best.pth",
    ),
]
eff_ckpt_filenames = [
    "tf_efficientnet_b4_ns_0.916_best.pth",
    "tf_efficientnet_b4_ns.pth",
    "efficientnet_b4.pth",
    "efficientnet-b4.pth",
    "eff_b4.pth",
    "effnet_b4.pth",
    "efficientnet_b4_best.pth",
    "effnet_b4_best.pth",
    "best.pth",
    "model.pth",
    "checkpoint.pth",
    "Eff.pth",
    "Eff.pt",
]

eff_b4_candidates += _fast_search_kaggle_input_for_filenames(
    eff_ckpt_filenames, root="/kaggle/input"
)
eff_b4_candidates += _fast_search_kaggle_input_for_filenames(
    eff_ckpt_filenames, root=DATA_ROOT
)

eff_b4_candidates += _collect_ckpt_files_multi_roots(
    roots=[
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input",
    ],
    max_files_total=1400,
)

efficientnet_img_size = 384

efficientnet_model = models.efficientnet_b4(weights=None)
num_features_efficientnet = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

best_eff_ckpt = _select_best_ckpt(eff_b4_candidates, efficientnet_model, device)

loaded_eff = False
eff_path = None
eff_sd = None
efficientnet_has_trained_head = False

if best_eff_ckpt is not None:
    loaded_eff, eff_path, eff_missing, eff_unexpected, eff_sd = _try_load_state_dict(
        efficientnet_model, [best_eff_ckpt], device
    )

    if loaded_eff and isinstance(eff_sd, dict):
        try:
            eff_sd_norm = _normalize_state_dict_keys(
                _extract_state_dict(torch.load(eff_path, map_location=device))
                if eff_path
                else eff_sd
            )
        except Exception:
            eff_sd_norm = _normalize_state_dict_keys(eff_sd)
        eff_sd_norm = _remap_head_keys_if_compatible(eff_sd_norm, efficientnet_model)
        efficientnet_has_trained_head = _ckpt_has_matching_5class_head(
            eff_sd_norm, efficientnet_model
        )

if not efficientnet_has_trained_head:
    efficientnet_model = models.efficientnet_v2_s(weights=None)
    num_features_efficientnet = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

    efficientnet_weight_candidates = [
        "/kaggle/input/eff-t/pytorch/default/1/Eff.pth",
        "/kaggle/input/eff-t/Eff.pth",
        "/kaggle/input/cassava-leaf-disease-classification/Eff.pth",
    ]
    efficientnet_weight_candidates += _fast_search_kaggle_input_for_filenames(
        [
            "Eff.pth",
            "Eff.pt",
            "eff.pth",
            "efficientnet_v2_s.pth",
            "efficientnetv2s.pth",
        ],
        root="/kaggle/input",
    )
    efficientnet_weight_candidates += _fast_search_kaggle_input_for_filenames(
        [
            "Eff.pth",
            "Eff.pt",
            "eff.pth",
            "efficientnet_v2_s.pth",
            "efficientnetv2s.pth",
        ],
        root=DATA_ROOT,
    )
    efficientnet_weight_candidates += _collect_ckpt_files_multi_roots(
        roots=[
            "/kaggle/input/cassava-leaf-disease-classification",
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
            "/kaggle/input",
        ],
        max_files_total=1400,
    )

    best_eff_ckpt2 = _select_best_ckpt(
        efficientnet_weight_candidates, efficientnet_model, device
    )
    eff_load_list = (
        [best_eff_ckpt2] if best_eff_ckpt2 else efficientnet_weight_candidates
    )

    loaded_eff, eff_path, eff_missing, eff_unexpected, eff_sd = _try_load_state_dict(
        efficientnet_model, eff_load_list, device
    )

    efficientnet_has_trained_head = False
    if loaded_eff and isinstance(eff_sd, dict):
        try:
            eff_sd_norm = _normalize_state_dict_keys(
                _extract_state_dict(torch.load(eff_path, map_location=device))
                if eff_path
                else eff_sd
            )
        except Exception:
            eff_sd_norm = _normalize_state_dict_keys(eff_sd)
        eff_sd_norm = _remap_head_keys_if_compatible(eff_sd_norm, efficientnet_model)
        efficientnet_has_trained_head = _ckpt_has_matching_5class_head(
            eff_sd_norm, efficientnet_model
        )

    if not loaded_eff:
        efficientnet_model = models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        num_features_efficientnet = efficientnet_model.classifier[1].in_features
        efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)
        efficientnet_has_trained_head = False

    efficientnet_img_size = 384

    print(
        "EfficientNet (v2-s) weights loaded from:",
        (
            eff_path
            if loaded_eff
            else "torchvision ImageNet (fallback backbone + UNTRAINED 5-class head)"
        ),
    )
    print("EfficientNet (v2-s) trained-head flag:", efficientnet_has_trained_head)
else:
    efficientnet_img_size = 380  # match EfficientNet-B4 common training size
    print("EfficientNet (b4) weights loaded from (selected best):", eff_path)
    print("EfficientNet (b4) trained-head flag:", efficientnet_has_trained_head)

efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()



## === cell 9
efficientnet_transforms = make_efficientnet_transforms(efficientnet_img_size)

test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

if resnet_has_trained_head and efficientnet_has_trained_head:
    weight_efficientnet = 0.7
    weight_resnet = 0.3
elif resnet_has_trained_head and (not efficientnet_has_trained_head):
    weight_efficientnet = 0.0
    weight_resnet = 1.0
elif (not resnet_has_trained_head) and efficientnet_has_trained_head:
    weight_efficientnet = 1.0
    weight_resnet = 0.0
else:
    weight_efficientnet = 0.0
    weight_resnet = 0.0

w_sum = weight_resnet + weight_efficientnet
if w_sum > 0:
    weight_resnet = weight_resnet / w_sum
    weight_efficientnet = weight_efficientnet / w_sum

print(
    "Ensemble weights:", {"resnet": weight_resnet, "efficientnet": weight_efficientnet}
)

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    label_prior = (
        train_df["label"]
        .value_counts(normalize=True)
        .reindex(range(5), fill_value=0.0)
        .values
    )
else:
    label_prior = np.ones(5, dtype=np.float64) / 5.0
label_prior = label_prior / (label_prior.sum() + 1e-12)
label_prior_tensor = torch.tensor(label_prior, dtype=torch.float32, device=device)

ensemble_predictions = []
image_names = []

with torch.no_grad():
    for images_resnet, images_efficientnet, img_names in test_loader:
        images_resnet = images_resnet.to(device, non_blocking=True)
        images_efficientnet = images_efficientnet.to(device, non_blocking=True)

        if weight_resnet > 0:
            outputs_resnet = resnet_model(images_resnet)
            probs_resnet = F.softmax(outputs_resnet, dim=1)
        else:
            probs_resnet = None

        if weight_efficientnet > 0:
            outputs_efficientnet = efficientnet_model(images_efficientnet)
            probs_efficientnet = F.softmax(outputs_efficientnet, dim=1)
        else:
            probs_efficientnet = None

        if (weight_resnet + weight_efficientnet) > 0:
            combined_probs = 0.0
            if probs_resnet is not None:
                combined_probs = combined_probs + (weight_resnet * probs_resnet)
            if probs_efficientnet is not None:
                combined_probs = combined_probs + (
                    weight_efficientnet * probs_efficientnet
                )
        else:
            combined_probs = label_prior_tensor.unsqueeze(0).repeat(
                images_resnet.size(0), 1
            )

        preds = combined_probs.argmax(dim=1).cpu().numpy()
        ensemble_predictions.extend(preds.tolist())
        image_names.extend(list(img_names))

len(image_names), len(ensemble_predictions)



## === cell 10
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df.copy()

submission_df["label"] = submission_df["image_id"].map(pred_map)

if submission_df["label"].isna().any():
    mode_label = (
        int(pd.Series(ensemble_predictions).mode().iloc[0])
        if len(ensemble_predictions)
        else 0
    )
    submission_df["label"] = submission_df["label"].fillna(mode_label)

submission_df["label"] = submission_df["label"].astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv' with shape:", submission_df.shape)
print(submission_df.head())
