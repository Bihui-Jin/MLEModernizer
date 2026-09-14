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

# 8. Previous improvement plan

N/A

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


torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((518, 518)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

torch_transforms_EfficientNet = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((480, 480)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 2
NUM_CLASSES = 5


class Head(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


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
vit_in = model_vit.heads.head.in_features
model_vit.heads.head = Head(vit_in, NUM_CLASSES)
model_vit.to(device).eval()

model_eff = efficientnet_v2_l(weights=eff_weights)
eff_in = model_eff.classifier[-1].in_features
model_eff.classifier[-1] = nn.Linear(eff_in, NUM_CLASSES)
model_eff.to(device).eval()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    if hasattr(torch, "compile"):
        model_vit = torch.compile(model_vit, mode="reduce-overhead")
        model_eff = torch.compile(model_eff, mode="reduce-overhead")
except Exception:
    pass

print(
    "Backbones ready:",
    type(model_vit).__name__,
    type(model_eff).__name__,
    "| vit_weights:",
    vit_weights is not None,
    "| eff_weights:",
    eff_weights is not None,
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
def extract_logits(
    dataloader: DataLoader, model_vit: nn.Module, model_eff: nn.Module, pin: bool
):
    n = len(dataloader.dataset)
    feats = np.empty((n, 2 * NUM_CLASSES), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)

    offset = 0
    for x_vit, x_eff, y in dataloader:
        bs = x_vit.shape[0]

        x_vit = x_vit.to(device, non_blocking=True)
        x_eff = x_eff.to(device, non_blocking=True)

        if hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        out_v = model_vit(x_vit)

        if hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        out_e = model_eff(x_eff)

        comb = torch.cat([out_e, out_v], dim=1)  # (B, 10)
        comb_np = comb.detach().to("cpu", non_blocking=pin).float().numpy()
        np.copyto(feats[offset : offset + bs], comb_np, casting="no")
        labels[offset : offset + bs] = np.asarray(y, dtype=np.int64)
        offset += bs

    return feats, labels


@torch.inference_mode()
def extract_logits_test(
    dataloader: DataLoader, model_vit: nn.Module, model_eff: nn.Module, pin: bool
):
    n = len(dataloader.dataset)
    feats = np.empty((n, 2 * NUM_CLASSES), dtype=np.float32)
    ids = [None] * n

    offset = 0
    for x_vit, x_eff, image_id in dataloader:
        bs = x_vit.shape[0]

        x_vit = x_vit.to(device, non_blocking=True)
        x_eff = x_eff.to(device, non_blocking=True)

        if hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        out_v = model_vit(x_vit)

        if hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        out_e = model_eff(x_eff)

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

oof_feats = np.zeros((len(train_df), 2 * NUM_CLASSES), dtype=np.float32)
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
    loader_kwargs["pin_memory_device"] = "cuda"

full_train_loader = DataLoader(full_train_ds, **loader_kwargs)

full_train_feats, full_train_y = extract_logits(
    full_train_loader, model_vit, model_eff, pin=PIN
)
assert np.array_equal(full_train_y, oof_labels), "Train feature/label order mismatch."

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(train_df["image_id"], train_df["label"]), start=1
):
    oof_feats[va_idx] = full_train_feats[va_idx]
    print(f"Fold {fold}: assigned {len(va_idx)} val rows from precomputed features")

decision_tree = RandomForestClassifier(
    n_estimators=30, criterion="gini", max_depth=8, random_state=SEED, n_jobs=-1
)
decision_tree.fit(oof_feats, oof_labels)
print("Meta-model trained on OOF features:", oof_feats.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/1092138788.py in <cell line: 0>()
     33 full_train_loader = DataLoader(full_train_ds, **loader_kwargs)
     34 
---> 35 full_train_feats, full_train_y = extract_logits(
     36     full_train_loader, model_vit, model_eff, pin=PIN
     37 )

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/1865273223.py in extract_logits(dataloader, model_vit, model_eff, pin)
    100         ):
    101             torch.compiler.cudagraph_mark_step_begin()
--> 102         out_v = model_vit(x_vit)
    103 
    104         if hasattr(torch, "compiler") and hasattr(

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

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

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

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    376             fn = getattr(self.obj.value, self.fn.__name__)
    377             return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
--> 378         return super().call_function(tx, args, kwargs)
    379 
    380     def inspect_parameter_names(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    315                 with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
    316                     return super().call_function(tx, args, kwargs)
--> 317         return super().call_function(tx, args, kwargs)
    318 
    319 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    116         kwargs: "Dict[str, VariableTracker]",
    117     ) -> "VariableTracker":
--> 118         return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
    119 
    120     def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_user_function_return(self, fn, args, kwargs)
    901         A call to some user defined function by inlining it.
    902         """
--> 903         return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
    904 
    905     def get_line_of_code_header(self, lineno=None):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call(cls, parent, func, args, kwargs)
   3070     def inline_call(cls, parent, func, args, kwargs):
   3071         with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3072             return cls.inline_call_(parent, func, args, kwargs)
   3073 
   3074     @staticmethod

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call_(parent, func, args, kwargs)
   3196         try:
   3197             with strict_ctx:
-> 3198                 tracer.run()
   3199         except exc.ObservedException as e:
   3200             msg = f"Observed exception DURING INLING {code} : {e}"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/torch.py in call_function(self, tx, args, kwargs)
    895             # constant fold
    896             return ConstantVariable.create(
--> 897                 self.as_python_constant()(
    898                     *[x.as_python_constant() for x in args],
    899                     **{k: v.as_python_constant() for k, v in kwargs.items()},

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _assert(condition, message)
   2130             _assert, (condition,), condition, message
   2131         )
-> 2132     assert condition, message
   2133 
   2134 

AssertionError: Wrong image height! Expected 224 but got 518!

from user code:
   File "/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py", line 291, in forward
    x = self._process_input(x)
  File "/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py", line 271, in _process_input
    torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_sub["image_id"].tolist()

test_ds = CassavaTestDataset(
    test_image_ids, TEST_IMG_DIR, torch_transforms_VIT, torch_transforms_EfficientNet
)
test_loader = DataLoader(test_ds, **loader_kwargs)

test_feats, out_ids = extract_logits_test(test_loader, model_vit, model_eff, pin=PIN)
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
/tmp/ipykernel_55/2639220678.py in <cell line: 0>()
      7 test_loader = DataLoader(test_ds, **loader_kwargs)
      8 
----> 9 test_feats, out_ids = extract_logits_test(test_loader, model_vit, model_eff, pin=PIN)
     10 assert (
     11     out_ids == test_image_ids

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/1865273223.py in extract_logits_test(dataloader, model_vit, model_eff, pin)
    136         ):
    137             torch.compiler.cudagraph_mark_step_begin()
--> 138         out_v = model_vit(x_vit)
    139 
    140         if hasattr(torch, "compiler") and hasattr(

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

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

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

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    376             fn = getattr(self.obj.value, self.fn.__name__)
    377             return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
--> 378         return super().call_function(tx, args, kwargs)
    379 
    380     def inspect_parameter_names(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    315                 with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
    316                     return super().call_function(tx, args, kwargs)
--> 317         return super().call_function(tx, args, kwargs)
    318 
    319 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    116         kwargs: "Dict[str, VariableTracker]",
    117     ) -> "VariableTracker":
--> 118         return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
    119 
    120     def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_user_function_return(self, fn, args, kwargs)
    901         A call to some user defined function by inlining it.
    902         """
--> 903         return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
    904 
    905     def get_line_of_code_header(self, lineno=None):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call(cls, parent, func, args, kwargs)
   3070     def inline_call(cls, parent, func, args, kwargs):
   3071         with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3072             return cls.inline_call_(parent, func, args, kwargs)
   3073 
   3074     @staticmethod

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call_(parent, func, args, kwargs)
   3196         try:
   3197             with strict_ctx:
-> 3198                 tracer.run()
   3199         except exc.ObservedException as e:
   3200             msg = f"Observed exception DURING INLING {code} : {e}"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/torch.py in call_function(self, tx, args, kwargs)
    895             # constant fold
    896             return ConstantVariable.create(
--> 897                 self.as_python_constant()(
    898                     *[x.as_python_constant() for x in args],
    899                     **{k: v.as_python_constant() for k, v in kwargs.items()},

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _assert(condition, message)
   2130             _assert, (condition,), condition, message
   2131         )
-> 2132     assert condition, message
   2133 
   2134 

AssertionError: Wrong image height! Expected 224 but got 518!

from user code:
   File "/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py", line 291, in forward
    x = self._process_input(x)
  File "/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py", line 271, in _process_input
    torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
