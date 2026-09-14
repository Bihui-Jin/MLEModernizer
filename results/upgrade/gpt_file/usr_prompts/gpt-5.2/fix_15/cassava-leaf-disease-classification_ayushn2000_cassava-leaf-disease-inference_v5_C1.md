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

0.819431852523421

# 6. Current score

0.10762

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10912) has done: 'I fix the fallback model selection bug so `timm.create_model(..., pretrained=True)` never tries to load nonexistent pretrained weights for `vit_base_resnet50d_224`. I also make checkpoint discovery robust and ensure that, if no checkpoints are found, we reliably fall back to a known-pretrained ViT architecture and still produce `submission.csv`. These changes are execution-unblocking and preserve the existing inference-only logic (no training changes). Finally, I add a small guard so the script always writes a correctly formatted submission with the sample_submission order.'
- What this solution (achieved 0.09828) has done: 'Your current score (0.10912) is far below the target (0.8194), and the biggest issue is that you’re not actually using any trained cassava checkpoints—so you’re effectively submitting random-ish predictions from an ImageNet-pretrained model with a fresh 5-class head. I keep the same inference-only pipeline and model code, but change the fallback behavior to (1) use timm’s real pretrained weights correctly (don’t replace the classifier during creation), and (2) apply the model’s default preprocessing (mean/std + resize/crop behavior) via `timm.data.resolve_data_config/create_transform`, which is a minimal, metric-aligned fix. This should dramatically increase accuracy vs. the current fallback, moving the score much closer toward the target without changing training/loops/loss (still no training). I also keep submission order identical to `sample_submission.csv`.'
- What this solution (achieved 0.49066) has done: 'Your current score is far below the target because the fallback path is still effectively producing near-random 5-class outputs: it runs an ImageNet model with 1000 classes and then “folds” those probabilities into 5 bins, which has no reason to align with cassava labels. I keep the same inference-only approach and preserve the existing model class/training setup, but add a minimal, legitimate improvement: use the provided `train.csv` to compute per-class ImageNet-prototype logits (mean 1000-d logits per cassava class) and classify test images by nearest prototype (argmax over class-prototype dot products). This keeps the model architecture unchanged, uses no training loop/loss, and should move accuracy substantially upward toward the target band while still fitting within Kaggle constraints/time. I also keep timm’s default preprocessing and preserve the sample_submission row order to avoid alignment issues.'
- What this solution (achieved 0.11584) has done: 'Your current gap to the target is large (0.49066 → 0.81943), and it’s coming from the fallback path’s weak “ImageNet-logit prototype” classifier. I keep the same inference-only approach and the same backbone model creation, but make a minimal, legitimate change to the fallback classifier: instead of averaging raw 1000-d logits per class, compute a **regularized LDA-style linear classifier** in that 1000-d feature/logit space using a small, class-balanced subset from `train.csv`. This is still not a training loop (just closed-form statistics), preserves evaluation semantics, and should move accuracy significantly upward while staying within the 600s constraint. I also keep timm’s own preprocessing and preserve the `sample_submission.csv` row order to avoid alignment-related score loss.'
- What this solution (achieved 0.10762) has done: 'Your current score (0.11584) is far below the target (0.81943), so we should improve accuracy substantially while keeping your inference-only core logic intact. The biggest leverage with minimal change is to stop using the 1000-class ImageNet logits as “features” and instead extract the model’s penultimate embedding features (timm `forward_features`) and fit the same closed-form regularized LDA classifier in that feature space. This preserves your overall approach (pretrained backbone + closed-form statistics, no SGD/training loops), but makes the representation far more transferable to cassava classes. I also add a safe, tiny fallback in case a model doesn’t expose `forward_features`, reverting to logits so it still runs end-to-end and writes `submission.csv` correctly.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so we should improve accuracy substantially with minimal changes while keeping your inference-only + closed-form LDA approach intact. The biggest issue is that your fallback model is still producing weak “features” because `forward_features` often returns a token map and we’re mean-pooling it; switching to timm’s built-in `forward_head(..., pre_logits=True)` (when available) gives a much better penultimate embedding without changing architecture or adding training. I also L2-normalize embeddings before fitting/applying LDA (a standard, tiny calibration step that tends to stabilize closed-form classifiers), and I compute the covariance in float64 more robustly. Everything still runs end-to-end, uses the same data/paths, and writes a valid `submission.csv` in sample order.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.81943), so we need a real accuracy lift with minimal disruption to your existing inference-only + closed-form LDA approach. The biggest issue is that `_forward_features_any` is being called on the **wrapper** (`CassvaImgClassifier`) instead of the underlying timm model, so it usually falls back to `m(x)` logits (or mismatched embeddings), which makes the LDA classifier very weak. I minimally fix feature extraction to always target the underlying timm backbone (`model.model` when wrapped), and I also ensure we set the pretrained model to `eval()` and disable gradients during feature extraction/inference to keep behavior consistent. This keeps your architecture, no-SGD approach, transforms, and submission semantics intact, but should move the score substantially upward toward the target band.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.81943), so we need a real accuracy lift without changing the overall “pretrained backbone + closed-form classifier (no SGD)” core. The biggest issue is that you’re using a generic ImageNet-pretrained model with a weak linear LDA fit on a small sample; a minimal but high-leverage fix is to use **ImageNet21k-pretrained ViT weights** (better transfer to plant datasets) while keeping the same feature extraction + LDA pipeline. I also fix a subtle determinism/performance bug in `seed_everything` (you currently set `deterministic=True` and `benchmark=True` simultaneously) and ensure all feature extraction runs under `torch.inference_mode()` to avoid any accidental grad/mode side effects. Finally, I keep submission ordering exactly aligned to `sample_submission.csv` as before.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so we need a real lift without changing the overall “pretrained timm backbone + closed-form classifier (no SGD)” approach. The main issue is that in the fallback path you build the model with `num_classes=1000`, so `forward_head(..., pre_logits=True)` often returns 1000-d classifier features (or logits) instead of the true penultimate embedding, which makes LDA ineffective; switching to `num_classes=0` forces timm to expose the backbone representation cleanly while keeping the architecture the same. I also make the feature extractor explicitly prefer timm’s `forward_features` + `global_pool` path when available (more reliable across backbones) and ensure the transform wrapper handles non-uint8 arrays safely. These are minimal, inference-only changes that should move accuracy substantially upward toward the target band while still writing a valid `submission.csv` in sample order.'
- What this solution (achieved 0.10762) has done: 'Your current score (0.05531) is far below the target (0.8194), and the biggest issue is that the fallback path is using a generic ImageNet-pretrained backbone without any cassava-specific supervision; the closed-form LDA on a small sample is not strong enough. Keeping your core “inference-only + closed-form classifier (no SGD training loop)” logic, I minimally switch the fallback to use a **self-supervised ViT backbone (DINOv2)** from timm, which typically yields much more transferable embeddings for plant/disease imagery. I also make two small, directly score-relevant fixes: use a larger, still-bounded per-class cap (to stabilize LDA statistics) and set `persistent_workers` to keep the pipeline fast enough under the 600s constraint without changing semantics. Everything still writes a valid `submission.csv` in the exact `sample_submission.csv` order.'

# 9. Code solution

## === cell 0
package_path = "../input/pytorch-image-dataset"
import sys

sys.path.append(package_path)



## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch as Apy

from glob import glob
import os
import time
import random
import cv2
import warnings
import timm

from tqdm import tqdm
from datetime import datetime
from skimage import io
from sklearn.model_selection import GroupKFold, StratifiedKFold

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler

import sklearn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")



## === cell 2
config = {
    "fold_num": 5,
    "seed": 719,
    "model_arch": "vit_base_resnet50d_224",
    "img_size": 224,
    "resize_to": 224,
    "epochs": 3,
    "train_bs": 32,
    "valid_bs": 32,
    "lr": 1e-4,
    "num_workers": 4,
    "accum_iter": 1,
    "verbose_step": 1,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "tta": 3,
    "used_epochs": [0, 1, 2],
    "weights": [1, 1, 1, 1],
}



## === cell 3
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission.head()




## === cell 4
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_img(path):
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return im_bgr[:, :, ::-1]


seed_everything(config["seed"])




## === cell 5
class CassavaDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        self.resize_image = torchvision.transforms.Resize(
            size=(config["resize_to"], config["resize_to"])
        )
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        if self.output_label:
            target = self.df.iloc[index]["label"]

        path = "{}/{}".format(self.data_root, self.df.iloc[index]["image_id"])

        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        if self.output_label == True:
            return img, target
        else:
            return img




## === cell 6
def get_inference_transforms():
    return A.Compose(
        [
            A.Resize(height=config["img_size"], width=config["img_size"], p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            Apy.ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 7
class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch,
            pretrained=pretrained,
            num_classes=n_class,
        )

    def forward(self, x):
        return self.model(x)




## === cell 8
def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.inference_mode():
        for step, imgs in pbar:
            imgs = imgs.to(device).float()
            image_preds = model(imgs)
            image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def load_checkpoint_safely(model, ckpt_path, device):
    if not os.path.exists(ckpt_path) and os.path.exists(ckpt_path + ".pth"):
        ckpt_path = ckpt_path + ".pth"
    if not os.path.exists(ckpt_path) and os.path.exists(ckpt_path + ".pt"):
        ckpt_path = ckpt_path + ".pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}(.pth/.pt)")

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state

    model.load_state_dict(state, strict=False)
    return ckpt_path


def find_ckpt_dir(preferred_dir, model_arch):
    """
    Bugfix: make checkpoint discovery robust.
    Search ../input for any file starting with f"{model_arch}_fold_" (with optional .pth/.pt).
    """
    if preferred_dir is not None and os.path.isdir(preferred_dir):
        if len(glob(os.path.join(preferred_dir, f"{model_arch}_fold_*"))) > 0:
            return preferred_dir

    base = "../input"
    candidates = []
    for d in sorted(glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        hit = False
        for pat in (
            os.path.join(d, f"{model_arch}_fold_*"),
            os.path.join(d, f"{model_arch}_fold_*.pth"),
            os.path.join(d, f"{model_arch}_fold_*.pt"),
        ):
            if len(glob(pat)) > 0:
                hit = True
                break
        if hit:
            candidates.append(d)
    return candidates[0] if len(candidates) > 0 else None


def resolve_fallback_arch(model_arch: str) -> str:
    """
    Score-improvement: deterministically fall back from the hybrid ViT to a pretrained ViT.
    """
    if model_arch == "vit_base_resnet50d_224":
        return "vit_base_patch16_224"
    return model_arch


class TimmTransformWrapper:
    """
    Use the model's default timm preprocessing config to match pretrained weights.
    Returns a CHW float tensor, same contract as ToTensorV2.
    """

    def __init__(self, timm_transform):
        self.timm_transform = timm_transform

    def __call__(self, image):
        if isinstance(image, np.ndarray) and image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil = torchvision.transforms.functional.to_pil_image(image)
        x = self.timm_transform(pil)
        return {"image": x}


def _l2_normalize_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)


def fit_regularized_lda_classifier(X, y, n_classes=5, shrink=0.20):
    """
    Closed-form, regularized LDA-style linear classifier (no SGD/training loop).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    D = X.shape[1]
    C = n_classes

    means = np.zeros((C, D), dtype=np.float64)
    counts = np.zeros((C,), dtype=np.float64)
    for c in range(C):
        Xc = X[y == c]
        if Xc.shape[0] == 0:
            continue
        means[c] = Xc.mean(axis=0)
        counts[c] = Xc.shape[0]

    cov = np.zeros((D, D), dtype=np.float64)
    denom = max(int(X.shape[0] - C), 1)
    for c in range(C):
        Xc = X[y == c]
        if Xc.shape[0] <= 1:
            continue
        X0 = Xc - means[c]
        cov += X0.T @ X0
    cov /= float(denom)

    tr = float(np.trace(cov))
    avg_var = tr / float(max(D, 1))
    cov_shrunk = (1.0 - shrink) * cov + shrink * (avg_var * np.eye(D, dtype=np.float64))

    inv_cov = np.linalg.inv(cov_shrunk)

    priors = counts / np.maximum(counts.sum(), 1.0)
    priors = np.clip(priors, 1e-12, 1.0)
    log_priors = np.log(priors)

    W = (inv_cov @ means.T).T  # (C, D)
    quad = np.einsum("cd,dd,cd->c", means, inv_cov, means)  # (C,)
    b = -0.5 * quad + log_priors
    return W.astype(np.float32), b.astype(np.float32)


def _unwrap_timm_backbone(model: nn.Module) -> nn.Module:
    """
    Minimal: when we use CassvaImgClassifier, the real timm model is model.model.
    """
    return model.model if hasattr(model, "model") else model


def _forward_features_any(model, x):
    """
    Score-improvement (minimal): reliably extract the penultimate embedding from timm backbones.
    Prefer forward_features + global_pool (when present) to avoid returning token maps / logits.
    """
    m = _unwrap_timm_backbone(model)

    if hasattr(m, "forward_features") and callable(getattr(m, "forward_features")):
        feats = m.forward_features(x)

        if hasattr(m, "global_pool") and callable(getattr(m, "global_pool")):
            try:
                pooled = m.global_pool(feats)
                if torch.is_tensor(pooled) and pooled.dim() == 2:
                    return pooled
            except Exception:
                pass

        if hasattr(m, "forward_head") and callable(getattr(m, "forward_head")):
            try:
                pl = m.forward_head(feats, pre_logits=True)
                if isinstance(pl, (list, tuple)):
                    pl = pl[0]
                if torch.is_tensor(pl) and pl.dim() == 2:
                    return pl
            except TypeError:
                pass
            except Exception:
                pass

        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if torch.is_tensor(feats):
            if feats.dim() == 4:
                feats = feats.mean(dim=(2, 3))
            elif feats.dim() == 3:
                feats = feats[:, 0] if feats.shape[1] > 0 else feats.mean(dim=1)
        return feats

    return m(x)


def features_one_epoch(model, data_loader, device):
    model.eval()
    all_feats = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.inference_mode():
        for step, batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, _ = batch
            else:
                imgs = batch
            imgs = imgs.to(device).float()
            feats = _forward_features_any(model, imgs)
            all_feats.append(feats.detach().cpu())
    return torch.cat(all_feats, dim=0).numpy()




## === cell 9
test_root = "../input/cassava-leaf-disease-classification/test_images/"
train_root = "../input/cassava-leaf-disease-classification/train_images/"

test = submission[["image_id"]].copy()
test["image_id"] = test["image_id"].astype(str)

tst_loader = None
device = torch.device(config["device"])

preferred_ckpt_dir = "../input/cassava-leave-disease"
ckpt_dir = find_ckpt_dir(preferred_ckpt_dir, config["model_arch"])
print("Resolved ckpt_dir:", ckpt_dir)

tst_preds = None

if ckpt_dir is not None:
    test_ds = CassavaDataset(
        test, test_root, transforms=get_inference_transforms(), output_label=False
    )
    tst_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config["valid_bs"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(config["num_workers"] > 0),
    )

    total_weight = float(sum(config["weights"])) * float(config["tta"])

    for fold in range(config["fold_num"]):
        for i, epoch in enumerate(config["used_epochs"]):
            model = CassvaImgClassifier(config["model_arch"], 5, pretrained=False).to(
                device
            )
            ckpt_path = f'{ckpt_dir}/{config["model_arch"]}_fold_{fold}_{epoch}'
            loaded_path = load_checkpoint_safely(model, ckpt_path, device)

            with torch.inference_mode():
                for _ in range(config["tta"]):
                    pred = inference_one_epoch(model, tst_loader, device)
                    w = float(config["weights"][i]) / total_weight
                    if tst_preds is None:
                        tst_preds = w * pred
                    else:
                        tst_preds += w * pred

            del model
            torch.cuda.empty_cache()
else:
    fallback_arch = resolve_fallback_arch(config["model_arch"])
    print(
        f"No compatible checkpoints found under ../input. Falling back to timm pretrained model: {fallback_arch}"
    )

    fallback_candidates = [
        "vit_base_patch14_dinov2.lvd142m",  # DINOv2, strong transferable embeddings
        "vit_small_patch14_dinov2.lvd142m",  # smaller if base not available in this timm build
        "vit_base_patch16_224.augreg_in21k",
        "vit_base_patch16_224_in21k",
        fallback_arch,
        "vit_base_patch16_224",
        "resnet50",
    ]

    model = None
    chosen_arch = None
    for arch in fallback_candidates:
        try:
            m = timm.create_model(arch, pretrained=True, num_classes=0)
            model = m.to(device)
            chosen_arch = arch
            break
        except Exception as e:
            print(f"Pretrained load failed for arch={arch} with error: {repr(e)}")

    if model is None:
        raise RuntimeError("Could not create any pretrained fallback model.")

    model.eval()

    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    data_cfg = resolve_data_config({}, model=model)
    timm_tf = create_transform(**data_cfg, is_training=False)
    infer_tf = TimmTransformWrapper(timm_tf)

    train_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
    train_df["image_id"] = train_df["image_id"].astype(str)
    train_df["label"] = train_df["label"].astype(int)

    rng = np.random.RandomState(config["seed"])

    per_class_cap = 1200

    parts = []
    for c in range(5):
        sub = train_df[train_df["label"] == c]
        if len(sub) > per_class_cap:
            idx = rng.choice(sub.index.values, size=per_class_cap, replace=False)
            sub = sub.loc[idx]
        parts.append(sub)
    proto_df = pd.concat(parts, axis=0).reset_index(drop=True)
    print(
        "Fallback LDA sample size:",
        len(proto_df),
        "class counts:",
        proto_df["label"].value_counts().to_dict(),
    )

    proto_ds = CassavaDataset(
        proto_df, train_root, transforms=infer_tf, output_label=True
    )
    proto_loader = torch.utils.data.DataLoader(
        proto_ds,
        batch_size=config["valid_bs"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(config["num_workers"] > 0),
    )

    test_ds = CassavaDataset(test, test_root, transforms=infer_tf, output_label=False)
    tst_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config["valid_bs"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        persistent_workers=(config["num_workers"] > 0),
    )

    with torch.inference_mode():
        proto_feats = []
        proto_labels = []
        pbar = tqdm(enumerate(proto_loader), total=len(proto_loader))
        for step, (imgs, y) in pbar:
            imgs = imgs.to(device).float()
            feats = _forward_features_any(model, imgs).detach().cpu()
            proto_feats.append(feats)
            proto_labels.append(y.detach().cpu())
        proto_feats = torch.cat(proto_feats, dim=0).float().numpy()
        proto_labels = torch.cat(proto_labels, dim=0).long().numpy()

        proto_feats = _l2_normalize_rows(proto_feats)

        W, b = fit_regularized_lda_classifier(
            proto_feats, proto_labels, n_classes=5, shrink=0.10
        )

        test_feats = []
        pbar = tqdm(enumerate(tst_loader), total=len(tst_loader))
        for step, imgs in pbar:
            imgs = imgs.to(device).float()
            feats = _forward_features_any(model, imgs).detach().cpu().float().numpy()
            test_feats.append(feats)
        test_feats = np.concatenate(test_feats, axis=0).astype(np.float32)

        test_feats = _l2_normalize_rows(test_feats)

        scores = test_feats @ W.T + b[None, :]  # (N,5)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        tst_preds = probs

    print("Fallback chosen_arch:", chosen_arch)
    del model
    torch.cuda.empty_cache()



## === cell 10
if tst_preds is None:
    raise RuntimeError("Inference produced no predictions (tst_preds is None).")

test["label"] = np.argmax(tst_preds, axis=1).astype(int)
out = test[["image_id", "label"]].copy()

out = out.merge(submission[["image_id"]], on="image_id", how="right")
out["label"] = out["label"].fillna(0).astype(int)

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
