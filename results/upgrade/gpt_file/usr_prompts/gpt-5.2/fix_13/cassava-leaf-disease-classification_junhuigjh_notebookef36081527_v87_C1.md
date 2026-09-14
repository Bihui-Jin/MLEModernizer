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

0.8869749168933212

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The runtime error comes from using `torch.compile()` with CUDA graphs where one model’s output buffer gets reused/overwritten before you concatenate it. The minimal fix is to avoid returning a cudagraph-managed tensor by cloning the outputs (or disabling cudagraphs); cloning is the smallest change and keeps core logic identical. I also make the `pin_memory_device` argument conditional (it can error on some Kaggle/PyTorch builds) and add a safe fallback that disables `torch.compile` if the error persists. These changes are score-neutral but unblock end-to-end execution and ensure `submission.csv` is written correctly.'
- What this solution (achieved 0.61061) has done: 'Your current score (0.61099) is far below the target (0.88697), so we need a real accuracy lift while keeping the same overall pipeline (two pretrained backbones → concatenate logits → RandomForest meta-model). The biggest issue is that you never train the new classification heads (they’re random), so the “features” are essentially noise; the minimal core-logic-preserving fix is to extract meaningful pretrained features by removing/replacing the random heads with identity so the concatenated vectors are real backbone embeddings. Then we keep the same RandomForest training/inference, but tune only its capacity slightly (more trees, a bit deeper) to move accuracy upward toward the target without changing the approach. All I/O paths and the submission format remain unchanged, and the script still runs end-to-end within Kaggle constraints.'

# 9. Code solution

## === cell 0
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import vit_h_14, efficientnet_v2_l
from torchvision.models.vision_transformer import ViT_H_14_Weights
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

Image.MAX_IMAGE_PIXELS = None
try:
    Image.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

torch.cuda.empty_cache()




## === cell 1
def invert_square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size

    dx, dy = w // 2, h // 2
    if dx or dy:
        img2 = Image.new(img.mode, (w, h))
        img2.paste(img.crop((dx, dy, w, h)), (0, 0))
        img2.paste(img.crop((0, dy, dx, h)), (w - dx, 0))
        img2.paste(img.crop((dx, 0, w, dy)), (0, h - dy))
        img2.paste(img.crop((0, 0, dx, dy)), (w - dx, h - dy))
    else:
        img2 = img

    max_side = max(w, h)
    pad_left = (max_side - w) // 2
    pad_top = (max_side - h) // 2
    pad_right = (max_side - w) - pad_left
    pad_bottom = (max_side - h) - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    return transforms.functional.pad(img2, padding, padding_mode="reflect")


VIT_IMAGE_SIZE = 224

vit_weights_default = ViT_H_14_Weights.DEFAULT
eff_weights_default = EfficientNet_V2_L_Weights.DEFAULT

torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        vit_weights_default.transforms(),
    ]
)

torch_transforms_EfficientNet = eff_weights_default.transforms()



## === cell 2
NUM_CLASSES = 5


def _try_local_torchhub_weights(weights_enum):
    try:
        url = weights_enum.url
        filename = os.path.basename(url)
        cached = os.path.expanduser(
            os.path.join("~", ".cache", "torch", "hub", "checkpoints", filename)
        )
        if os.path.exists(cached) and os.path.getsize(cached) > 0:
            return weights_enum
    except Exception:
        pass
    return None


vit_weights = _try_local_torchhub_weights(ViT_H_14_Weights.DEFAULT)
eff_weights = _try_local_torchhub_weights(EfficientNet_V2_L_Weights.DEFAULT)

model_vit = vit_h_14(weights=vit_weights)
model_eff = efficientnet_v2_l(weights=eff_weights)

model_vit.heads = nn.Identity()
model_eff.classifier = nn.Identity()

model_vit.to(device).eval()
model_eff.to(device).eval()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_USE_COMPILE = False
print(
    "Backbones ready:",
    type(model_vit).__name__,
    type(model_eff).__name__,
    "| vit_weights:",
    vit_weights is not None,
    "| eff_weights:",
    eff_weights is not None,
    "| compiled:",
    _USE_COMPILE,
)




## === cell 3
def _load_rgb_pil(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class _PILCache:
    __slots__ = ("_cache",)

    def __init__(self):
        self._cache = {}

    def get(self, path: str) -> Image.Image:
        im = self._cache.get(path)
        if im is None:
            im = _load_rgb_pil(path)
            self._cache[path] = im
        return im


class CassavaTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, tfm_vit, tfm_eff):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm_vit = tfm_vit
        self.tfm_eff = tfm_eff

        self.image_ids = self.df["image_id"].to_list()
        self.labels = self.df["label"].to_numpy(dtype=np.int64)

        self._pil_cache = _PILCache()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        label = int(self.labels[i])
        path = os.path.join(self.img_dir, image_id)

        img = self._pil_cache.get(path)

        x_vit = self.tfm_vit(img)
        x_eff = self.tfm_eff(img)
        return x_vit, x_eff, label

    def __getitems__(self, indices):
        out = []
        for i in indices:
            out.append(self.__getitem__(i))
        return out


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, img_dir: str, tfm_vit, tfm_eff):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.tfm_vit = tfm_vit
        self.tfm_eff = tfm_eff

        self._pil_cache = _PILCache()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        path = os.path.join(self.img_dir, image_id)

        img = self._pil_cache.get(path)

        x_vit = self.tfm_vit(img)
        x_eff = self.tfm_eff(img)
        return x_vit, x_eff, image_id

    def __getitems__(self, indices):
        out = []
        for i in indices:
            out.append(self.__getitem__(i))
        return out


@torch.inference_mode()
def extract_feats(
    dataloader: DataLoader, model_vit: nn.Module, model_eff: nn.Module, pin: bool
):
    n = len(dataloader.dataset)

    x_vit0, x_eff0, _ = next(iter(dataloader))
    x_vit0 = x_vit0.to(device, non_blocking=True)
    x_eff0 = x_eff0.to(device, non_blocking=True)
    f_v0 = model_vit(x_vit0)
    f_e0 = model_eff(x_eff0)
    d_v = int(f_v0.shape[1])
    d_e = int(f_e0.shape[1])

    feats = np.empty((n, d_e + d_v), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)

    offset = 0
    for x_vit, x_eff, y in dataloader:
        bs = x_vit.shape[0]

        x_vit = x_vit.to(device, non_blocking=True)
        x_eff = x_eff.to(device, non_blocking=True)

        out_v = model_vit(x_vit).clone()
        out_e = model_eff(x_eff).clone()

        comb = torch.cat([out_e, out_v], dim=1)
        comb_np = comb.detach().to("cpu", non_blocking=pin).float().numpy()
        np.copyto(feats[offset : offset + bs], comb_np, casting="no")
        labels[offset : offset + bs] = np.asarray(y, dtype=np.int64)
        offset += bs

    return feats, labels


@torch.inference_mode()
def extract_feats_test(
    dataloader: DataLoader, model_vit: nn.Module, model_eff: nn.Module, pin: bool
):
    n = len(dataloader.dataset)

    x_vit0, x_eff0, _ = next(iter(dataloader))
    x_vit0 = x_vit0.to(device, non_blocking=True)
    x_eff0 = x_eff0.to(device, non_blocking=True)
    f_v0 = model_vit(x_vit0)
    f_e0 = model_eff(x_eff0)
    d_v = int(f_v0.shape[1])
    d_e = int(f_e0.shape[1])

    feats = np.empty((n, d_e + d_v), dtype=np.float32)
    ids = [None] * n

    offset = 0
    for x_vit, x_eff, image_id in dataloader:
        bs = x_vit.shape[0]

        x_vit = x_vit.to(device, non_blocking=True)
        x_eff = x_eff.to(device, non_blocking=True)

        out_v = model_vit(x_vit).clone()
        out_e = model_eff(x_eff).clone()

        comb = torch.cat([out_e, out_v], dim=1)
        comb_np = comb.detach().to("cpu", non_blocking=pin).float().numpy()
        np.copyto(feats[offset : offset + bs], comb_np, casting="no")
        ids[offset : offset + bs] = list(image_id)
        offset += bs

    return feats, ids




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
assert set(train_df.columns) == {"image_id", "label"}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_labels = train_df["label"].to_numpy(dtype=np.int64)

BATCH = 24 if torch.cuda.is_available() else 8
NUM_WORKERS = min(6, max(2, (os.cpu_count() or 2) - 2))
PIN = torch.cuda.is_available()

full_train_ds = CassavaTrainDataset(
    train_df, TRAIN_IMG_DIR, torch_transforms_VIT, torch_transforms_EfficientNet
)

loader_kwargs = dict(
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
)
if NUM_WORKERS > 0:
    loader_kwargs["prefetch_factor"] = 4
    try:
        loader_kwargs["multiprocessing_context"] = "fork"
    except Exception:
        pass

if PIN:
    try:
        _ = DataLoader(full_train_ds, **{**loader_kwargs, "pin_memory_device": "cuda"})
        loader_kwargs["pin_memory_device"] = "cuda"
    except TypeError:
        pass

full_train_loader = DataLoader(full_train_ds, **loader_kwargs)

full_train_feats, full_train_y = extract_feats(
    full_train_loader, model_vit, model_eff, pin=PIN
)
assert np.array_equal(full_train_y, oof_labels), "Train feature/label order mismatch."

oof_feats = np.zeros_like(full_train_feats, dtype=np.float32)
for fold, (tr_idx, va_idx) in enumerate(
    skf.split(train_df["image_id"], train_df["label"]), start=1
):
    oof_feats[va_idx] = full_train_feats[va_idx]
    print(f"Fold {fold}: assigned {len(va_idx)} val rows from precomputed features")

decision_tree = RandomForestClassifier(
    n_estimators=400,
    criterion="gini",
    max_depth=22,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
decision_tree.fit(oof_feats, oof_labels)
print("Meta-model trained on OOF features:", oof_feats.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/3062463794.py in <cell line: 0>()
     37 full_train_loader = DataLoader(full_train_ds, **loader_kwargs)
     38 
---> 39 full_train_feats, full_train_y = extract_feats(
     40     full_train_loader, model_vit, model_eff, pin=PIN
     41 )

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/1565597585.py in extract_feats(dataloader, model_vit, model_eff, pin)
     89     x_vit0 = x_vit0.to(device, non_blocking=True)
     90     x_eff0 = x_eff0.to(device, non_blocking=True)
---> 91     f_v0 = model_vit(x_vit0)
     92     f_e0 = model_eff(x_eff0)
     93     d_v = int(f_v0.shape[1])

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in forward(self, x)
    289     def forward(self, x: torch.Tensor):
    290         # Reshape and permute the input tensor
--> 291         x = self._process_input(x)
    292         n = x.shape[0]
    293 

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _process_input(self, x)
    269         n, c, h, w = x.shape
    270         p = self.patch_size
--> 271         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    272         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    273         n_h = h // p

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _assert(condition, message)
   2130             _assert, (condition,), condition, message
   2131         )
-> 2132     assert condition, message
   2133 
   2134 

AssertionError: Wrong image height! Expected 224 but got 518!

## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_sub["image_id"].tolist()

test_ds = CassavaTestDataset(
    test_image_ids, TEST_IMG_DIR, torch_transforms_VIT, torch_transforms_EfficientNet
)
test_loader = DataLoader(test_ds, **loader_kwargs)

test_feats, out_ids = extract_feats_test(test_loader, model_vit, model_eff, pin=PIN)

assert (
    out_ids == test_image_ids
), "Test ID order mismatch; submission would be misaligned."

prediction = decision_tree.predict(test_feats).astype(int)

submission = pd.DataFrame({"image_id": out_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "rows:", len(submission))
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0

_check = pd.read_csv("submission.csv")
assert list(_check.columns) == ["image_id", "label"]
assert len(_check) == len(sample_sub)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/3639915663.py in <cell line: 0>()
      7 test_loader = DataLoader(test_ds, **loader_kwargs)
      8 
----> 9 test_feats, out_ids = extract_feats_test(test_loader, model_vit, model_eff, pin=PIN)
     10 
     11 assert (

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/1565597585.py in extract_feats_test(dataloader, model_vit, model_eff, pin)
    125     x_vit0 = x_vit0.to(device, non_blocking=True)
    126     x_eff0 = x_eff0.to(device, non_blocking=True)
--> 127     f_v0 = model_vit(x_vit0)
    128     f_e0 = model_eff(x_eff0)
    129     d_v = int(f_v0.shape[1])

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in forward(self, x)
    289     def forward(self, x: torch.Tensor):
    290         # Reshape and permute the input tensor
--> 291         x = self._process_input(x)
    292         n = x.shape[0]
    293 

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _process_input(self, x)
    269         n, c, h, w = x.shape
    270         p = self.patch_size
--> 271         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    272         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    273         n_h = h // p

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _assert(condition, message)
   2130             _assert, (condition,), condition, message
   2131         )
-> 2132     assert condition, message
   2133 
   2134 

AssertionError: Wrong image height! Expected 224 but got 518!
