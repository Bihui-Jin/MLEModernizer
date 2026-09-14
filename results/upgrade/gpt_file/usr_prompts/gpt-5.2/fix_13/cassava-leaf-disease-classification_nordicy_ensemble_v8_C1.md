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

0.8785131459655485

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Your pipeline doesn’t yield a Kaggle score primarily because it can fail before writing `submission.csv` (missing external weight files / wrong paths) and because the saved predictions may not be aligned to the sample submission order. I make minimal, execution-unblocking changes: robustly resolve input paths, load checkpoints safely (with a clear fallback that still produces a valid CSV), and reorder the final submission to exactly match `sample_submission.csv`’s `image_id` order. I also enable deterministic settings to avoid run-to-run drift without changing the modeling logic. These changes preserve your ensemble/inference logic and only touch reliability + submission correctness.'
- What this solution (achieved 0.39499) has done: 'Your current score (0.61099) is far below the target (0.8785), and the main reason is that the models are being instantiated with `weights=None` and your external checkpoints are not available in the provided file tree—so inference is effectively using random weights. The smallest legitimate change that preserves your ensemble/inference core logic is to load ImageNet pretrained weights when custom checkpoints are missing, so the models produce meaningful features. I keep the exact same architecture heads, transforms, and ensemble averaging, only adding a safe fallback to torchvision pretrained weights and ensuring the submission order matches `sample_submission.csv`. This should move accuracy up substantially toward the target without changing your overall approach.'
- What this solution (achieved 0.09791) has done: 'Your score is far below target because, when custom checkpoints are missing, you currently keep randomly initialized 5-class classifier heads on top of ImageNet backbones, which yields near-random predictions. To move accuracy toward the target while preserving your ensemble/inference core logic, I (1) add a tiny fallback “head calibration” step that initializes the 5-class heads from the corresponding ImageNet 1000-class heads (so the model produces meaningful logits even without cassava fine-tuning), and (2) use the correct ImageNet preprocessing for ResNet50 (your current pipeline uses EfficientNet normalization for all models). These are minimal, inference-only changes that keep your architecture/loops intact and should raise the score substantially toward the target. I also keep the submission alignment logic but make it stricter by forcing the output order to exactly match `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score is near-random because the EfficientNet models are being fed 384×384 images, but EfficientNetV2-S is pretrained on 384×384 only for the *L* variant; V2-S pretrained expects 224×224 and its ImageNet preprocessing (including resize/crop behavior) is different from the simple resize you’re using. To move the score sharply upward toward the target while keeping the same ensemble/models/inference loop, I only change the EfficientNet test transform to match `EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()` (correct size + normalization), and keep ResNet as-is. I also make the DataLoader use a small number of workers for faster, more stable throughput under Kaggle, without changing predictions. Submission alignment logic is kept identical and still forces the exact `sample_submission.csv` order.'
- What this solution (achieved 0.09903) has done: 'I fix the transform construction bug causing the early crash by extracting EfficientNetV2-S input size/mean/std in a version-robust way (the current `crop_size` unpack fails under this torchvision). That restore definition of `efficientnet_transforms`, which unblocks dataset/dataloader creation and therefore the inference loop. I also make the DataLoader’s `persistent_workers` conditional on `num_workers>0` to avoid runtime errors on some Kaggle setups, and keep the rest of your ensemble/inference logic unchanged so the score should move up from near-random purely by enabling the intended preprocessing and end-to-end execution.'
- What this solution (achieved 0.09268) has done: 'Your current score is near-random because the test DataLoader feeds EfficientNet-preprocessed images into the ResNet model as well, so the ResNet branch is effectively broken and drags down the ensemble. To move the score sharply upward toward the target while keeping the same models and inference loop, I make the smallest change: run two DataLoaders (one with EfficientNet transforms, one with ResNet transforms) and combine probabilities per image_id. I also keep strict submission alignment to `sample_submission.csv` order and add a sanity check that both loaders see the exact same image_id sequence (so averaging is correct). Everything else (architectures, checkpoint loading fallback, averaging scheme, softmax+argmax, CSV format) stays the same.'
- What this solution (achieved 0.0938) has done: 'Your score is far below the target, so we should make a small change that legitimately improves accuracy without changing your ensemble/models/inference structure. Right now both branches apply CLAHE, which can distort color/texture distributions relative to ImageNet-pretrained features and hurt performance when you’re relying on pretrained backbones and only a weak/fallback head. I remove CLAHE from both the EfficientNet and ResNet test-time transforms (keeping resize + ImageNet normalization intact), leaving everything else (models, checkpoints/fallbacks, loaders, averaging, softmax+argmax, submission alignment) unchanged. This should move accuracy materially upward toward the target while staying within your “minimal change / same core logic” constraint.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.0938) is far below the target (0.8785), so we need a small but high-impact fix that preserves your ensemble/inference structure. The biggest issue is that your “fallback head init from ImageNet” uses arbitrary 0–200/…/800–1000 class groups, which produces essentially meaningless 5-class logits and near-random predictions when your cassava checkpoints are missing. I keep the exact same models, transforms, loaders, softmax+average+argmax ensemble, but change the fallback to a safer “bias-only” head init (zero weights, bias set to log of the train label prior), which typically yields a much better-than-random baseline on this imbalanced dataset and should move accuracy materially toward the target. I also add a strict check that train.csv is found and used only for computing class priors (no training, no leakage).'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8785), so we should make a small, high-impact correction that preserves your ensemble/inference core logic but fixes a likely checkpoint-loading failure. The biggest issue is that `safe_load_state_dict()` loads checkpoints onto `device` even though the model is still on CPU at load time; this often results in most weights not being applied (silently, due to `strict=False`), leaving you effectively with pretrained backbones + prior-bias heads, which caps accuracy around your current range. I change `safe_load_state_dict()` to always load checkpoints on CPU (map_location="cpu") and then load into the model (still on CPU), which is the most robust approach and should allow your custom cassava fine-tuned weights (if present) to actually take effect. Everything else (models, transforms, averaging, argmax, submission alignment) remains the same.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is well below the target (0.87851), so we need a small change that increases accuracy without altering your ensemble/models/inference structure. The most likely limiter is that even when checkpoints exist, they may be loaded but not actually used because the keys are prefixed (e.g., `module.` from DataParallel) and because `strict=False` can silently skip critical layers; this can leave you with mostly-pretrained backbones + prior-bias heads. I make `safe_load_state_dict()` more robust by stripping common prefixes (`model.`, `module.`, `net.`) and, when available, preferentially loading an EMA state dict (`state_dict_ema`) which often performs better at inference. These changes keep the same architectures, transforms, averaging, and argmax logic, but should move your score upward toward the target by correctly applying your fine-tuned weights.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8785), so we should make a small, high-impact fix that preserves your exact ensemble/inference structure but increases the likelihood that your fine-tuned checkpoints actually load. The most probable issue is that many Kaggle-exported checkpoints store weights under nested keys (e.g., `model_state_dict`, `state_dict`, `ema_state_dict`) and/or contain non-tensor entries; with your current loader, those cases silently fall back to pretrained backbones + prior-bias heads, capping accuracy around where you are now. I make `safe_load_state_dict()` more robust by (1) searching a broader set of common keys, (2) filtering to tensor-only items, and (3) stripping a few more common prefixes, while keeping `strict=False` and everything else (models, transforms, averaging, argmax, submission alignment) unchanged. This should move the score upward toward your target without changing core logic.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is well below the target (0.8785), so we should make a small change that legitimately improves accuracy without changing your ensemble/models/inference structure. The most likely remaining issue is that your EfficientNet models with `classifier = nn.Sequential(Dropout, Linear)` are left in `.eval()` mode, which disables dropout; if the fine-tuned checkpoints were trained expecting that dropout layer to be active, this mismatch can noticeably hurt accuracy. I keep the same architectures, checkpoints, transforms, and averaging logic, but enable MC-dropout only for those two EfficientNet models during inference (dropout active, batchnorm stays in eval), and average multiple stochastic forward passes; this is still the same inference approach (no training/optimization) and should move the score upward toward the target. Everything else (CSV format, sample_submission order alignment, robust checkpoint loading) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_V2_S_Weights
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 3
num_tta = 5  # will be used for MC-dropout passes on models that contain dropout



## === cell 4
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
test_df = pd.read_csv(sample_path)
test_df.head()



## === cell 6
_eff_weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
_eff_preset = _eff_weights.transforms()

_eff_size = None
for attr in ("crop_size", "resize_size"):
    if hasattr(_eff_preset, attr):
        v = getattr(_eff_preset, attr)
        if isinstance(v, (tuple, list)):
            if len(v) == 2:
                _eff_size = (int(v[0]), int(v[1]))
                break
            if len(v) == 1:
                _eff_size = (int(v[0]), int(v[0]))
                break
        else:
            _eff_size = (int(v), int(v))
            break
if _eff_size is None:
    _eff_size = (224, 224)

if hasattr(_eff_preset, "mean"):
    eff_mean = tuple(float(x) for x in _eff_preset.mean)
else:
    eff_mean = (0.485, 0.456, 0.406)
if hasattr(_eff_preset, "std"):
    eff_std = tuple(float(x) for x in _eff_preset.std)
else:
    eff_std = (0.229, 0.224, 0.225)

eff_h, eff_wid = _eff_size

efficientnet_transforms = A.Compose(
    [
        A.Resize(eff_h, eff_wid),
        A.Normalize(mean=eff_mean, std=eff_std),
        ToTensorV2(),
    ]
)

resnet_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 7
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 8
test_dataset_eff = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)
test_dataset_res = CassavaTestDataset(
    test_df, test_image_dir, transform=resnet_transforms
)

_num_workers = 2
_common_loader_kwargs = dict(
    batch_size=32,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
)

test_loader_eff = DataLoader(test_dataset_eff, **_common_loader_kwargs)
test_loader_res = DataLoader(test_dataset_res, **_common_loader_kwargs)



## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 10
def safe_load_state_dict(model, ckpt_path, device):
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        print(f"[WARN] Checkpoint not found, skipping load: {ckpt_path}")
        return False

    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict):
        candidate_keys = [
            "state_dict_ema",
            "ema_state_dict",
            "model_ema",
            "model_state_dict_ema",
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]
        for k in candidate_keys:
            if k in state and isinstance(state[k], dict):
                if k in (
                    "state_dict_ema",
                    "ema_state_dict",
                    "model_ema",
                    "model_state_dict_ema",
                ):
                    print(f"[INFO] Using EMA weights from checkpoint key: {k}")
                state = state[k]
                break

    if not isinstance(state, dict):
        print("[WARN] Unrecognized checkpoint format; skipping load.")
        return False

    def _strip_prefix(k: str) -> str:
        for pref in ("model.", "module.", "net.", "encoder.", "backbone."):
            if k.startswith(pref):
                return k[len(pref) :]
        return k

    new_state = {}
    for k, v in state.items():
        if not isinstance(k, str):
            continue
        if not torch.is_tensor(v):
            continue
        nk = _strip_prefix(k)
        new_state[nk] = v
    state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"[INFO] Loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
    return True


def compute_train_label_prior(train_csv_path, num_classes=5, eps=1e-6):
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    if "label" not in df.columns:
        raise ValueError("train.csv missing required column: label")
    counts = (
        df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values.astype(np.float64)
    )
    prior = counts / max(counts.sum(), 1.0)
    prior = np.clip(prior, eps, 1.0)
    prior = prior / prior.sum()
    return prior


def init_5class_head_bias_from_prior_linear(linear_layer: nn.Linear, prior: np.ndarray):
    if not isinstance(linear_layer, nn.Linear) or linear_layer.out_features != len(
        prior
    ):
        raise ValueError("Expected nn.Linear with out_features == num_classes")
    with torch.no_grad():
        linear_layer.weight.zero_()
        linear_layer.bias.copy_(
            torch.tensor(
                np.log(prior),
                dtype=linear_layer.bias.dtype,
                device=linear_layer.bias.device,
            )
        )


def init_effv2s_head_bias_from_prior(model, prior: np.ndarray):
    head = None
    if isinstance(model.classifier, nn.Sequential):
        for m in reversed(model.classifier):
            if isinstance(m, nn.Linear):
                head = m
                break
    else:
        try:
            if isinstance(model.classifier[1], nn.Linear):
                head = model.classifier[1]
        except Exception:
            head = None
    if head is None:
        print(
            "[WARN] Could not locate EfficientNetV2-S 5-class head for prior-bias init."
        )
        return
    init_5class_head_bias_from_prior_linear(head, prior)
    print(
        "[INFO] Initialized EfficientNetV2-S 5-class head with prior-bias fallback (zero weights)."
    )


def init_resnet50_head_bias_from_prior(model, prior: np.ndarray):
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        print("[WARN] Could not locate ResNet50 5-class fc for prior-bias init.")
        return
    init_5class_head_bias_from_prior_linear(model.fc, prior)
    print(
        "[INFO] Initialized ResNet50 5-class fc with prior-bias fallback (zero weights)."
    )




## === cell 11
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
label_prior = compute_train_label_prior(train_csv_path, num_classes=5)
print("[INFO] Train label prior:", label_prior.round(6).tolist())



## === cell 12
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_ckpt = (
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth"
)
loaded_resnet = safe_load_state_dict(resnet_model, resnet_ckpt, device)
if not loaded_resnet:
    print(
        "[INFO] Using torchvision pretrained ResNet50 backbone weights (no custom ckpt loaded)."
    )
    init_resnet50_head_bias_from_prior(resnet_model, label_prior)

resnet_model = resnet_model.to(device)
resnet_model.eval()



## === cell 13
efficientnet_model_1 = models.efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)

eff1_ckpt = "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth"
loaded_eff1 = safe_load_state_dict(efficientnet_model_1, eff1_ckpt, device)
if not loaded_eff1:
    print(
        "[INFO] Using torchvision pretrained EfficientNetV2-S backbone weights for model_1 (no custom ckpt loaded)."
    )
    init_effv2s_head_bias_from_prior(efficientnet_model_1, label_prior)

efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()



## === cell 14
efficientnet_model_7 = models.efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff7_ckpt = "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth"
loaded_eff7 = safe_load_state_dict(efficientnet_model_7, eff7_ckpt, device)
if not loaded_eff7:
    print(
        "[INFO] Using torchvision pretrained EfficientNetV2-S backbone weights for model_7 (no custom ckpt loaded)."
    )
    init_effv2s_head_bias_from_prior(efficientnet_model_7, label_prior)

efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()



## === cell 15
efficientnet_model_8 = models.efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff8_ckpt = "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth"
loaded_eff8 = safe_load_state_dict(efficientnet_model_8, eff8_ckpt, device)
if not loaded_eff8:
    print(
        "[INFO] Using torchvision pretrained EfficientNetV2-S backbone weights for model_8 (no custom ckpt loaded)."
    )
    init_effv2s_head_bias_from_prior(efficientnet_model_8, label_prior)

efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()




## === cell 16
def enable_mc_dropout(model: nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model


def mc_softmax_probs(model: nn.Module, x: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 1:
        return F.softmax(model(x), dim=1)
    probs = None
    for _ in range(n):
        p = F.softmax(model(x), dim=1)
        probs = p if probs is None else (probs + p)
    return probs / float(n)


_uses_mc_dropout = True
if _uses_mc_dropout:
    enable_mc_dropout(efficientnet_model_7)
    enable_mc_dropout(efficientnet_model_8)
    mc_passes = int(num_tta) if isinstance(num_tta, int) and num_tta > 1 else 5
else:
    mc_passes = 1

ensemble_predictions = []
image_names = []

with torch.no_grad():
    for (images_eff, img_names_eff), (images_res, img_names_res) in zip(
        test_loader_eff, test_loader_res
    ):
        if list(img_names_eff) != list(img_names_res):
            raise RuntimeError(
                "Mismatch between EfficientNet and ResNet loader image order; cannot safely ensemble."
            )

        images_eff = images_eff.to(device, non_blocking=True)
        images_res = images_res.to(device, non_blocking=True)

        outputs_efficientnet1 = efficientnet_model_1(images_eff)
        probs_efficientnet1 = F.softmax(outputs_efficientnet1, dim=1)

        probs_efficientnet7 = mc_softmax_probs(
            efficientnet_model_7, images_eff, mc_passes
        )
        probs_efficientnet8 = mc_softmax_probs(
            efficientnet_model_8, images_eff, mc_passes
        )

        outputs_resnet = resnet_model(images_res)
        probs_resnet = F.softmax(outputs_resnet, dim=1)

        combined_probs = (
            probs_efficientnet1
            + probs_efficientnet7
            + probs_efficientnet8
            + probs_resnet
        ) / 4.0
        preds = combined_probs.argmax(dim=1).cpu().numpy()

        ensemble_predictions.extend(preds.tolist())
        image_names.extend(list(img_names_eff))



## === cell 17
pred_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})

submission_df = test_df[["image_id"]].merge(
    pred_df, on="image_id", how="left", validate="one_to_one"
)

if submission_df["label"].isna().any():
    n_missing = int(submission_df["label"].isna().sum())
    print(
        f"[WARN] {n_missing} test images missing predictions; filling with 0 to keep submission valid."
    )
    submission_df["label"] = submission_df["label"].fillna(0)

submission_df["label"] = submission_df["label"].astype(int)

assert len(submission_df) == len(
    test_df
), "Submission row count mismatch vs sample_submission."

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
