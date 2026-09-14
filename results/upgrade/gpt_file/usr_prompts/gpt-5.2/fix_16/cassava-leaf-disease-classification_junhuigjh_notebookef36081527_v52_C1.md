# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8578120278029616

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.68386) has done: 'I remove the hard dependency on missing external model files under `/kaggle/input/...` by falling back to a built-in torchvision ResNet50 feature extractor (same overall approach: extract per-image model features then train a DecisionTreeClassifier). I also fix the cell-order/state bug that caused `use_model1` to be undefined by ensuring it is always defined before `get_probs_for_image()` is used. To keep runtime under control while preserving the same “train on extracted features → predict test” semantics, I batch image loading and feature extraction and avoid TensorFlow entirely (since it’s not guaranteed in this environment). Finally, I ensure a valid `submission.csv` with the exact required columns is always written.'
- What this solution (achieved 0.6633) has done: 'Your current gap to the target is large (0.68386 → 0.8578, about +25%), so we should improve accuracy but keep the same “extract CNN features → train DecisionTree → predict” core pipeline. The biggest minimal win is fixing image preprocessing: ResNet50 pretrained weights expect a 224×224 crop-style input; your 512×512 resize distorts scale and wastes compute, hurting both accuracy and runtime. We switch to the standard ResNet50 validation transform (resize shorter side to 256 then center-crop 224) while keeping the same model, same classifier, and same training loop. We also add safe PIL file-handle closing (context managers) to avoid intermittent I/O issues without changing semantics.'
- What this solution (achieved 0.51158) has done: 'You’re currently far below the target (0.6633 vs 0.8578), so we need a modest, low-risk accuracy lift while keeping the same “ResNet50 feature extractor → DecisionTreeClassifier” pipeline. The biggest issue is that you train a single shallow tree on highly imbalanced classes; a minimal change that usually boosts accuracy substantially is to use `class_weight="balanced"` and allow a bit more tree capacity (`min_samples_leaf` + slightly deeper `max_depth`) without changing the modeling family. I also make feature extraction more stable by ensuring deterministic ordering and by using `torch.no_grad()`/`inference_mode()` already present while avoiding any accidental randomness; core feature extractor and training loop remain the same. The submission format and paths stay unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.55344) has done: 'Your current score (0.51158) is far below the target (0.85781), so we should improve accuracy while keeping the same “ResNet50 feature extractor → DecisionTreeClassifier” pipeline. The most impactful minimal fix is to use the ResNet50 feature extractor exactly as intended by the pretrained weights: apply the weights’ official preprocessing (already done) and, crucially, use the weights’ built-in inference-time normalization consistently and avoid any accidental CPU/GPU nondeterminism. Then we make a small, low-risk DecisionTree adjustment that usually improves generalization on this task without changing the model family: tune `max_depth`/`min_samples_leaf` slightly toward a better bias/variance point and add `min_samples_split`. Finally, we keep submission generation identical but ensure strict alignment and stable ordering.'
- What this solution (achieved 0.54185) has done: 'Your current score (0.55344) is far below the target (0.85781), so we should improve accuracy while keeping the same “ResNet50 feature extractor → DecisionTreeClassifier” pipeline. The smallest high-impact fix is to extract stronger, more generalizable features by adding a second, standard pretrained backbone (ResNet18) and concatenating its embeddings with the existing ResNet50 embeddings—this preserves the core logic (CNN feature extraction then a DecisionTree) while usually giving a sizable lift. To keep runtime within limits, we increase feature-extraction batch size when possible and use `torch.inference_mode()` consistently. The submission generation and required column alignment remain unchanged.'
- What this solution (achieved 0.5725) has done: 'Your current score (0.54185) is far below the target (0.85781), so we need a real accuracy lift while keeping the same core pipeline: pretrained CNN feature extraction → DecisionTreeClassifier → predict test. The biggest low-risk improvement is to use a stronger single pretrained backbone for embeddings (swap the auxiliary ResNet18 to EfficientNet-B0) while keeping the exact same extraction+tree training semantics and submission format. This is a minimal architecture change within the same “torchvision pretrained feature extractor” concept and typically boosts cassava accuracy noticeably. I also keep transforms tied to the chosen weights to ensure correct preprocessing, but won’t change your training loop or loss/metric behavior.'
- What this solution (achieved 0.58109) has done: 'Your current score (0.5725) is far below the target (0.8578), so we should improve accuracy while keeping the same core pipeline (torchvision pretrained CNN feature extraction → DecisionTreeClassifier → predict test). The smallest high-impact issue here is that you are applying ResNet50’s preprocessing transform to EfficientNet-B0 inputs; that mismatch can significantly degrade the EfficientNet embeddings, hurting the tree. I keep the exact same models and DecisionTree, but use each model’s own official weights `.transforms()` (separate transforms per backbone) and extract both embeddings in the same batched loop. This preserves evaluation semantics and training approach while providing a meaningful accuracy lift with minimal code change and similar runtime.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # keep as in original (fast for fixed 224x224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TRAIN_DIR = f"{DATA_ROOT}/train_images"
TEST_DIR = f"{DATA_ROOT}/test_images"
SAMPLE_SUB = f"{DATA_ROOT}/sample_submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

use_model1 = False
model1 = None

common_tfms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 1
@torch.inference_mode()
def pil_to_tensor_common(img_pil: Image.Image) -> torch.Tensor:
    return common_tfms(img_pil)


def build_feature_extractor_resnet50() -> nn.Module:
    try:
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
    except Exception as e:
        print(
            f"WARNING: Could not load pretrained ResNet50 weights; using random init. Root error: {repr(e)}"
        )
        backbone = models.resnet50(weights=None)

    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


def build_feature_extractor_efficientnet_b0() -> nn.Module:
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
    except Exception as e:
        print(
            f"WARNING: Could not load pretrained EfficientNet-B0 weights; using random init. Root error: {repr(e)}"
        )
        backbone = models.efficientnet_b0(weights=None)

    backbone.classifier = nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


model2 = build_feature_extractor_resnet50()
model3 = build_feature_extractor_efficientnet_b0()

if torch.cuda.is_available():
    model2 = model2.to(memory_format=torch.channels_last)
    model3 = model3.to(memory_format=torch.channels_last)

try:
    model2 = torch.compile(model2, mode="reduce-overhead")
    model3 = torch.compile(model3, mode="reduce-overhead")
except Exception:
    pass


def get_features_for_image(img_path: str) -> np.ndarray:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        x = pil_to_tensor_common(img).unsqueeze(0).to(device)

    if torch.cuda.is_available():
        x = x.contiguous(memory_format=torch.channels_last)

    emb50 = model2(x).detach().float().cpu().numpy()[0].astype(np.float32)  # (2048,)
    embef = model3(x).detach().float().cpu().numpy()[0].astype(np.float32)  # (1280,)

    if use_model1:
        p1 = np.array(model1.predict(None), dtype=np.float32)  # unreachable
    else:
        p1 = np.zeros((5,), dtype=np.float32)

    return np.concatenate([p1, emb50, embef], axis=0)  # (5 + 2048 + 1280,)




## === cell 2
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.io import decode_jpeg, read_file
    import torchvision.transforms.functional as TF

    _HAS_TV_DECODE = True
except Exception:
    _HAS_TV_DECODE = False
    TF = None

_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_STD_CPU = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _center_crop_tensor_bchw(img_bchw: torch.Tensor, out_hw: int = 224) -> torch.Tensor:
    h = int(img_bchw.shape[2])
    w = int(img_bchw.shape[3])
    top = (h - out_hw) // 2
    left = (w - out_hw) // 2
    return img_bchw[:, :, top : top + out_hw, left : left + out_hw]


class _CassavaPathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        fp = self.paths[idx]
        if _HAS_TV_DECODE:
            data = read_file(fp)  # uint8 1D tensor
            img = decode_jpeg(data, device="cpu")  # uint8, 3xHxW (RGB)

            h, w = int(img.shape[1]), int(img.shape[2])
            if h < w:
                new_h = 256
                new_w = int(round(w * (256.0 / h)))
            else:
                new_w = 256
                new_h = int(round(h * (256.0 / w)))

            img = img.to(dtype=torch.float32).div_(255.0)  # ToTensor equivalent
            img = TF.resize(
                img,
                [new_h, new_w],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            top = (int(img.shape[1]) - 224) // 2
            left = (int(img.shape[2]) - 224) // 2
            img = img[:, top : top + 224, left : left + 224]
            return img
        else:
            with Image.open(fp) as img:
                img = img.convert("RGB")
                x = pil_to_tensor_common(img)
            return x


@torch.inference_mode()
def get_features_for_paths(paths, batch_size: int = 256) -> np.ndarray:
    paths = list(paths)
    n = len(paths)
    out = np.empty((n, 5 + 2048 + 1280), dtype=np.float32)

    cpu_cnt = os.cpu_count() or 2
    num_workers = min(8, max(2, cpu_cnt // 2))
    pin = torch.cuda.is_available()

    loader = DataLoader(
        _CassavaPathDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(
            4 if num_workers > 0 else None
        ),  # Speedup: deeper prefetch reduces GPU idle.
    )

    mean = _MEAN_CPU.to(device) if torch.cuda.is_available() else _MEAN_CPU
    std = _STD_CPU.to(device) if torch.cuda.is_available() else _STD_CPU

    write_pos = 0
    for x in loader:
        b = x.shape[0]

        if _HAS_TV_DECODE:
            x = x.to(device, non_blocking=pin)
            x = x.view(b, 3, 224, 224)
            x = (x - mean) / std
        else:
            x = x.to(device, non_blocking=pin)

        if torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)

        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                emb50_t = model2(x)
                embef_t = model3(x)
            emb50_t = emb50_t.float().cpu()
            embef_t = embef_t.float().cpu()
        else:
            emb50_t = model2(x).float().cpu()
            embef_t = model3(x).float().cpu()

        sl = slice(write_pos, write_pos + b)
        out[sl, 0:5] = 0.0
        out[sl, 5 : 5 + 2048] = np.asarray(emb50_t, dtype=np.float32)
        out[sl, 5 + 2048 :] = np.asarray(embef_t, dtype=np.float32)

        write_pos += b

    return out


train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)

train_df["filepath"] = (
    train_df["image_id"].astype(str).map(lambda x: f"{TRAIN_DIR}/{x}")
)
train_df = train_df.sort_values("image_id").reset_index(drop=True)

train_labels = train_df["label"].astype(int).to_numpy()
train_paths = train_df["filepath"].tolist()

train_features = get_features_for_paths(train_paths, batch_size=256)

X_tr, X_va, y_tr, y_va = train_test_split(
    train_features,
    train_labels,
    test_size=0.20,
    random_state=SEED,
    stratify=train_labels,
)

base_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=SEED,
)
path = base_tree.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas = np.unique(path.ccp_alphas)

if ccp_alphas.shape[0] > 25:
    idx = np.linspace(0, ccp_alphas.shape[0] - 1, 25).round().astype(int)
    ccp_grid = ccp_alphas[idx]
else:
    ccp_grid = ccp_alphas

best_alpha = float(ccp_grid[0])
best_acc = -1.0
for a in ccp_grid:
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        ccp_alpha=float(a),
    )
    clf.fit(X_tr, y_tr)
    acc = float((clf.predict(X_va) == y_va).mean())
    if acc > best_acc:
        best_acc = acc
        best_alpha = float(a)

print(f"Selected ccp_alpha={best_alpha:.8g} (holdout acc={best_acc:.5f})")

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=SEED,
    ccp_alpha=best_alpha,
)
decision_tree.fit(train_features, train_labels)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/2455723559.py in <cell line: 0>()
    142 train_paths = train_df["filepath"].tolist()
    143 
--> 144 train_features = get_features_for_paths(train_paths, batch_size=256)
    145 
    146 X_tr, X_va, y_tr, y_va = train_test_split(

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_56/2455723559.py in get_features_for_paths(paths, batch_size)
    114                 emb50_t = model2(x)
    115                 embef_t = model3(x)
--> 116             emb50_t = emb50_t.float().cpu()
    117             embef_t = embef_t.float().cpu()
    118         else:

RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py", line 285, in forward
    return self._forward_impl(x)
  File "/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py", line 279, in _forward_impl
    x = torch.flatten(x, 1). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
assert "image_id" in sample_sub.columns

test_image_ids = sample_sub["image_id"].tolist()
test_filepaths = [os.path.join(TEST_DIR, iid) for iid in test_image_ids]

missing = [fp for fp in test_filepaths if not os.path.exists(fp)]
assert len(missing) == 0, f"Missing {len(missing)} test images; example: {missing[0]}"

test_features = get_features_for_paths(test_filepaths, batch_size=256)

prediction = decision_tree.predict(test_features).astype(int)

submission = pd.DataFrame({"image_id": test_image_ids, "label": prediction})
assert submission.shape[0] == len(sample_sub), "Submission row count mismatch"
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch"

submission.to_csv("submission.csv", index=False)
submission.head(10)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/2174273008.py in <cell line: 0>()
      8 assert len(missing) == 0, f"Missing {len(missing)} test images; example: {missing[0]}"
      9 
---> 10 test_features = get_features_for_paths(test_filepaths, batch_size=256)
     11 
     12 prediction = decision_tree.predict(test_features).astype(int)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_56/2455723559.py in get_features_for_paths(paths, batch_size)
    114                 emb50_t = model2(x)
    115                 embef_t = model3(x)
--> 116             emb50_t = emb50_t.float().cpu()
    117             embef_t = embef_t.float().cpu()
    118         else:

RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py", line 285, in forward
    return self._forward_impl(x)
  File "/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py", line 279, in _forward_impl
    x = torch.flatten(x, 1). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
