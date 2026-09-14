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

No external packages required in the script and installed.

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

0.899365367180417

# 6. Current score

0.70142

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17302) has done: 'I remove the failing external pip installs and missing third‑party imports (VisionTransformer/albumentations/skimage/cv2/efficientnet), and replace them with lightweight, built-in equivalents that keep the same overall inference flow (multi-model ensemble → argmax label → submission.csv). I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it isn’t needed for producing a valid submission here) and keep everything in PyTorch+timm, which is available in Kaggle. I also fix multiple NameErrors caused by out-of-order cells by consolidating required imports/definitions and ensuring every referenced symbol exists before use. Finally, I ensure the submission has exactly the required columns (`image_id,label`), correct row count (matching `sample_submission.csv`), and is written with a `.csv` suffix.'
- What this solution (achieved 0.77317) has done: 'Your current score is low because the code is doing pure ImageNet-pretrained inference with random classification heads (timm replaces the final layer when `num_classes=5`, so outputs are essentially untrained), so predictions are near-random. To move accuracy toward the target with minimal change in “core logic” (still timm models + ensemble + argmax + submission), I (1) load the provided Cassava train labels, (2) train only the classification head for a single short epoch on resized images, and (3) reuse the same inference/ensemble pipeline afterward. This keeps the architecture and inference flow intact while making the predictions dataset-relevant instead of random. The rest of the submission formatting and paths remain unchanged.'
- What this solution (achieved 0.77466) has done: 'You’re currently under the target (0.77317 vs 0.89936), so we should improve accuracy with the smallest possible changes while keeping the same timm models → train head → ensemble → argmax pipeline. The biggest issue is that your “head-only” training is still running the full model in train mode with dropout/BN updates and without a stable split, which makes the quick head-fit less effective; we freeze the backbone and run it in eval mode during training so features stay consistent, and we train the head for a couple of epochs (still the same approach, just slightly longer) with a tiny validation split to avoid obvious overfitting. We also switch to timm’s built-in `resolve_data_config` + `create_transform` so preprocessing matches each pretrained model (same semantics: normalize+resize), which usually yields a meaningful accuracy bump without changing architecture or loss. Everything else (models, loss, ensemble, submission format/paths) stays the same and it still produce `submission.csv`.'
- What this solution (achieved 0.77466) has done: 'Your current score (0.77466) is below the target (0.89936), so we should improve accuracy with small, low-risk changes that keep the same timm models → train head-only → ensemble → argmax pipeline. The main issue is that “head-only training” still runs the full model forward, so BatchNorm/dropout behavior and autograd through the backbone can make this quick fit unstable/inefficient; we keep the backbone in eval mode and use `torch.no_grad()` to extract fixed features, then train only the classifier head on those features (same head-only approach, just done correctly). We also compute the train transform once per model (as you do) but use the model’s own `default_cfg` via timm transforms as-is, and add a tiny validation accuracy printout (no early stopping) to confirm the head is actually learning. These changes are directly aimed at improving the learned head quality without changing architectures, losses, or the overall training/inference semantics.'
- What this solution (achieved 0.77466) has done: 'Your current score (0.77466) is well below the target (0.89937), so we should improve accuracy with minimal, low-risk changes that keep the same overall pipeline: timm pretrained models → head-only training → ensemble → argmax → submission. The main bottleneck is that the head is being trained on features whose shape may not match the classifier input (especially for ResNet/EfficientNet), making the head training ineffective; we instead use timm’s own `forward_head(pre_logits=True)` when available and otherwise apply the model’s global pooling before the head so the head always receives the right feature vector. We also ensure the extracted features are flattened to `(B, D)` to match `nn.Linear` heads, and keep everything else (models, loss, epochs, batch size, ensemble weights, submission formatting/paths) unchanged. This should move the score upward toward the target without changing the core approach.'
- What this solution (achieved 0.74141) has done: 'Your score (0.77466) is well below the target (0.89937), so we need a modest accuracy lift while keeping your exact “timm pretrained models → head-only training → ensemble → argmax” core pipeline. The biggest low-risk gain is to make the head-only training more dataset-relevant by (1) using a stratified split (your current random split can underrepresent some classes) and (2) applying class-weighted cross-entropy while training the head (Cassava is imbalanced), without changing architecture, loss type, or inference semantics. I’m also making the head-training path robust by always using the model’s own classifier forward (so it matches internal dropout/BN expectations) while still freezing the backbone; this preserves “head-only” training but avoids subtle shape/feature mismatches. These changes are directly aimed at moving accuracy upward toward your target, and the script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.72197) has done: 'Your current score (0.74141) is far below the target (0.89937), so we should improve accuracy with minimal, low-risk changes that keep your exact pipeline (timm models → head-only training → ensemble → argmax → submission). The main issue is that head-only training is learning from heavily augmented “training” transforms (random crop/flip/autoaugment) while inference uses clean center-crop transforms, creating a train/test mismatch that hurts generalization; we train the head on the same deterministic (eval) transform used at inference. We also make the head-training LR a bit smaller and run a few more epochs to better fit the head without changing architecture, loss, or the overall approach. Finally, we add a lightweight validation accuracy print (no early stopping) to confirm the head is learning and reduce the risk of silent regressions.'
- What this solution (achieved 0.69581) has done: 'Your current score (0.72197) is far below the target (0.89937), so we need a modest accuracy gain while keeping your exact “timm pretrained models → head-only training → ensemble → argmax → submission” pipeline unchanged. The largest low-risk issue is that `create_transform(is_training=False)` is using evaluation/center-crop style preprocessing for training too, which makes the head underfit; we switch the head-training transform to `is_training=True` (still timm-native, still the same resize/normalize semantics) while keeping inference on `is_training=False`. To stabilize improvement without changing the training approach, we also add a short LR warmup (via a built-in PyTorch scheduler) and compute class weights on the *actual* training labels (unchanged loss: still CrossEntropyLoss, just weighted as you already do). Everything else—models, frozen backbone, head-only optimization, ensemble, and submission format/paths—stays the same.'
- What this solution (achieved 0.65658) has done: 'Your gap to target is large (0.69581 vs 0.89937), so we need a real accuracy lift while keeping your exact pipeline (timm pretrained backbones → head-only training → ensemble → argmax → submission). The biggest minimal win is to stop training the head on heavy random augmentations and instead train on the same deterministic eval preprocessing used at inference, which reduces train/test mismatch for “fixed-feature head training”. We also adjust the head LR slightly down for stability with weighted CE (still the same loss/optimizer/scheduler approach) and increase batch size a bit to reduce gradient noise without changing training semantics. Finally, we keep the rest identical and still write a valid `submission.csv`.'
- What this solution (achieved 0.66218) has done: 'Your current score (0.65658) is far below the target (0.89937), so we should make a small, low-risk change that improves generalization without changing your core pipeline (timm pretrained backbones → head-only training on fixed features → ensemble → argmax → submission). The biggest issue is that you’re training the head with the eval (center-crop) transform, which reduces effective data diversity and tends to underfit the head; we switch only the head-training transform to `is_training=True` while keeping inference/validation on `is_training=False` exactly as-is. To keep the change safe and consistent with your fixed-feature training, we also set the backbone to `eval()` once before training and explicitly keep `head_module.train()` (already done) so only the head learns. Everything else (models, loss, optimizer, scheduler, epochs, submission format/paths) remains unchanged.'
- What this solution (achieved 0.65658) has done: 'Your current score (0.66218) is far below the target (0.89937), so we need a modest accuracy lift while keeping your same pipeline (timm pretrained backbones → fixed-feature head-only training → ensemble → argmax → submission). The most direct low-risk issue is that your head training is using `is_training=True` transforms, which include heavy random augmentations; this injects label-noise for fixed-feature training and tends to hurt generalization on the clean eval transform used for inference. I switch the head-training transform to the same deterministic eval transform used at inference/validation, and I keep everything else (models, loss, optimizer, epochs, ensemble, submission formatting) unchanged. This should move accuracy upward toward the target without changing the core logic.'
- What this solution (achieved 0.66181) has done: 'Your score is far below the target (0.6566 vs 0.8994), so the smallest reliable way to move toward the target while preserving your exact pipeline (timm pretrained backbones → fixed-feature head-only training → ensemble → argmax) is to (1) fix the train/inference preprocessing mismatch by using timm’s *training* transform for head training and *eval* transform for validation/inference, and (2) make head training actually learn by using a simple feature cache so each epoch sees consistent backbone features (still head-only; backbone remains frozen/eval). These are minimal changes that directly improve generalization without changing architecture, loss, or inference semantics. Everything else (models, weights, epochs, submission format/paths) is kept the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.70142) has done: 'Your score (0.66181) is far below the target (0.89937), so we should increase accuracy with the smallest changes that keep your exact pipeline (timm pretrained backbones → head-only training on fixed features → ensemble → argmax). The most limiting factor is that training on raw pixel images is slow/noisy for head-only learning; we can (1) cache fixed backbone features once using the *eval* transform (matching inference) for stable, generalizable head training, and (2) train a bit longer on those cached features (still only the head, same loss/optimizer/scheduler). I keep inference transforms and submission formatting identical, and only adjust the head-training data/epochs in a way that remains within Kaggle runtime constraints. This should move the score upward toward your target without changing model architectures or the fundamental training/inference semantics.'

# 9. Code solution

## === cell 0
import os, sys, glob, random, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import timm
from timm.data import resolve_data_config, create_transform



## === cell 1
IMAGE_SIZE = 512
CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 2
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_CSV_PATH = f"{DATA_ROOT}/train.csv"

assert os.path.exists(TEST_IMG_DIR), f"Missing test_images at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images at {TRAIN_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub = sample_sub[["image_id", "label"]].copy()

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = train_df[["image_id", "label"]].copy()
train_df["label"] = train_df["label"].astype(int)




## === cell 3
def build_timm_transform(
    model_name: str, image_size_fallback: int = IMAGE_SIZE, is_train: bool = False
):
    tmp_model = timm.create_model(model_name, pretrained=True, num_classes=CLASSES)
    cfg = resolve_data_config({}, model=tmp_model)
    if (
        "input_size" in cfg
        and isinstance(cfg["input_size"], (tuple, list))
        and len(cfg["input_size"]) == 3
    ):
        pass
    else:
        cfg["input_size"] = (3, image_size_fallback, image_size_fallback)
    tfm = create_transform(**cfg, is_training=is_train)
    del tmp_model
    return tfm


class PytorchCassavaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transforms=None,
        with_label: bool = False,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        image_id = self.df.iloc[idx]["image_id"]
        path = os.path.join(self.data_root, image_id)
        img = Image.open(path).convert("RGB")

        if self.transforms is None:
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1)
        else:
            x = self.transforms(img)

        if self.with_label:
            y = int(self.df.iloc[idx]["label"])
            return x, y
        return x




## === cell 4
class TimmModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=CLASSES
        )

    def forward(self, x):
        return self.model(x)


def inference_simple(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device, non_blocking=True).float()
            logits = model(images)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _get_head_module(model: nn.Module):
    head_module = None
    if hasattr(model.model, "classifier") and isinstance(
        model.model.classifier, nn.Module
    ):
        head_module = model.model.classifier
    elif hasattr(model.model, "fc") and isinstance(model.model.fc, nn.Module):
        head_module = model.model.fc
    elif hasattr(model.model, "head") and isinstance(model.model.head, nn.Module):
        head_module = model.model.head
    return head_module


def _forward_features_pre_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Keep head-only training, but ensure we extract the pooled feature vector the classifier expects.
    """
    m = model.model

    if hasattr(m, "forward_features"):
        feats = m.forward_features(x)

        if hasattr(m, "forward_head"):
            try:
                pre = m.forward_head(feats, pre_logits=True)
                if pre.ndim > 2:
                    pre = torch.flatten(pre, 1)
                return pre
            except TypeError:
                pass

        if hasattr(m, "global_pool") and callable(getattr(m, "global_pool")):
            pooled = m.global_pool(feats)
            if pooled.ndim > 2:
                pooled = torch.flatten(pooled, 1)
            return pooled

        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return feats

    out = model(x)
    if out.ndim > 2:
        out = torch.flatten(out, 1)
    return out


def _compute_class_weights_from_df(
    df: pd.DataFrame, num_classes: int = CLASSES
) -> torch.Tensor:
    counts = (
        df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values.astype(np.float32)
    )
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def eval_head_on_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    head_module = _get_head_module(model)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True).float()
            targets = torch.as_tensor(targets, device=device, dtype=torch.long)
            if head_module is None:
                logits = model(images)
            else:
                feats = _forward_features_pre_logits(model, images)
                logits = head_module(feats)
            correct += (torch.argmax(logits, dim=1) == targets).sum().item()
            total += targets.numel()
    return float(correct) / max(1, total)


class _FeatureCacheDataset(Dataset):
    """
    Cache backbone features once (backbone frozen/eval), then train head on fixed features.
    """

    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], int(self.targets[idx])


def _cache_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True).float()
            feats = _forward_features_pre_logits(model, images).detach().cpu()
            xs.append(feats)
            ys.append(torch.as_tensor(targets, dtype=torch.long).cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_head_n_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 4,
    class_weights: torch.Tensor = None,
):
    for p in model.parameters():
        p.requires_grad = False

    head_module = _get_head_module(model)
    if head_module is None:
        for p in model.parameters():
            p.requires_grad = True
        head_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(head_params, lr=lr)
        crit = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        model.train()
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
        for _ in range(epochs):
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True).float()
                targets = torch.as_tensor(targets, device=device, dtype=torch.long)
                opt.zero_grad(set_to_none=True)
                logits = model(images)
                loss = crit(logits, targets)
                loss.backward()
                opt.step()
                sched.step()
        return

    for p in head_module.parameters():
        p.requires_grad = True

    feats_cpu, targets_cpu = _cache_features(model, train_loader, device=device)

    cache_ds = _FeatureCacheDataset(feats_cpu, targets_cpu)
    cache_loader = DataLoader(
        cache_ds,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    head_module.train()
    opt = torch.optim.AdamW(head_module.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(cache_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    for _ in range(epochs):
        for feats, targets in cache_loader:
            feats = feats.to(device, non_blocking=True).float()
            targets = torch.as_tensor(targets, device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits = head_module(feats)
            loss = crit(logits, targets)
            loss.backward()
            opt.step()
            sched.step()




## === cell 5
seed_everything(42)


def stratified_split_indices(
    labels: np.ndarray, val_frac: float = 0.05, seed: int = 42
):
    rng = np.random.RandomState(seed)
    trn_idx, val_idx = [], []
    for c in range(CLASSES):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(len(idx_c) * val_frac))
        val_idx.append(idx_c[:n_val])
        trn_idx.append(idx_c[n_val:])
    trn_idx = np.concatenate(trn_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(trn_idx)
    rng.shuffle(val_idx)
    return trn_idx, val_idx


labels_np = train_df["label"].values.astype(int)
trn_idx, val_idx = stratified_split_indices(labels_np, val_frac=0.05, seed=42)

train_df_trn = train_df.iloc[trn_idx].reset_index(drop=True)
train_df_val = train_df.iloc[val_idx].reset_index(drop=True)

class_w = _compute_class_weights_from_df(train_df_trn, num_classes=CLASSES)

test_df = sample_sub[["image_id"]].copy()



## === cell 6
models_to_ensemble = [
    ("resnet50", 0.5),
    ("tf_efficientnet_b0_ns", 0.5),
]

all_preds = []
for mname, w in models_to_ensemble:
    seed_everything(42)

    train_tfm = build_timm_transform(mname, is_train=True)
    test_tfm = build_timm_transform(mname, is_train=False)

    test_dataset = PytorchCassavaDataset(
        df=test_df, data_root=TEST_IMG_DIR, transforms=test_tfm, with_label=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )

    train_dataset = PytorchCassavaDataset(
        df=train_df_trn, data_root=TRAIN_IMG_DIR, transforms=test_tfm, with_label=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )

    val_dataset = PytorchCassavaDataset(
        df=train_df_val, data_root=TRAIN_IMG_DIR, transforms=test_tfm, with_label=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )

    model = TimmModel(mname, pretrained=True).to(device)

    train_head_n_epochs(
        model,
        train_loader,
        device=device,
        lr=7.5e-4,
        epochs=10,
        class_weights=class_w,
    )

    val_acc = eval_head_on_loader(model, val_loader, device=device)
    print(f"[{mname}] val_acc={val_acc:.4f}")

    preds = inference_simple(model, test_loader, device)
    all_preds.append(w * preds)

    del (
        model,
        test_dataset,
        test_loader,
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

ensemble_preds = np.sum(all_preds, axis=0)  # (N, CLASSES)



## === cell 7
pred_labels = np.argmax(ensemble_preds, axis=1).astype(int)

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

submission_df = submission_df.merge(
    sample_sub[["image_id"]], on="image_id", how="right"
)
submission_df["label"] = submission_df["label"].fillna(0).astype(int)
submission_df = submission_df[["image_id", "label"]]

assert submission_df.shape[0] == sample_sub.shape[0], "Submission row count mismatch"
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns mismatch"



## === cell 8
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
