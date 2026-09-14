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

0.8859171955273496

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.63341) has done: 'I fix the CUDA OOM and the later AssertionError by disabling `torch.compile` (it is creating large CUDA graph/private-pool allocations and triggering an inductor/cudagraph assertion in this environment) and by running inference in `torch.autocast` to reduce activation memory while keeping the exact same model/backbone logic. I also make the batch size adaptive (reduce on OOM) so feature extraction completes reliably on both train-meta and test without manual tuning. These changes are runtime/stability focused and should also improve accuracy versus the current “no submission” state by allowing the full pipeline to finish and produce `submission.csv`. Finally, I keep all paths and the meta-model logic intact and ensure a valid CSV is always written.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageChops, ImageOps, ImageFile
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import (
    vit_h_14,
    ViT_H_14_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
)

from torchvision.models.feature_extraction import create_feature_extractor

try:
    from torchvision.io import read_image, ImageReadMode

    _HAS_TVIO = True
except Exception:
    _HAS_TVIO = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


ImageFile.LOAD_TRUNCATED_IMAGES = True

cpu = os.cpu_count() or 1
torch.set_num_threads(max(1, min(8, cpu // 2)))
torch.set_num_interop_threads(1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.isdir(
        os.path.join(p, "train_images")
    ):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset root. Expected train.csv + train_images under one of: "
        + ", ".join(DATA_ROOT_CANDIDATES)
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("Device:", device)

USE_TORCH_COMPILE = False
print("USE_TORCH_COMPILE:", USE_TORCH_COMPILE)

AMP_DTYPE = torch.float16 if device.type == "cuda" else None




## === cell 1
def invert_square_pad(img: Image.Image) -> Image.Image:
    width, height = img.size
    img2 = ImageChops.offset(img, width // 2, height // 2)

    max_side = max(width, height)
    pad_left = (max_side - width) // 2
    pad_top = (max_side - height) // 2
    pad_right = (max_side - width) - pad_left
    pad_bottom = (max_side - height) - pad_top

    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return img2

    base = img2
    mirror_x = ImageOps.mirror(base)
    mirror_y = ImageOps.flip(base)
    mirror_xy = ImageOps.mirror(mirror_y)

    w, h = base.size
    canvas = Image.new("RGB", (w * 3, h * 3))
    tiles = [
        [mirror_xy, mirror_y, mirror_xy],
        [mirror_x, base, mirror_x],
        [mirror_xy, mirror_y, mirror_xy],
    ]
    for r in range(3):
        for c in range(3):
            canvas.paste(tiles[r][c], (c * w, r * h))

    left = w + (w - width) // 2 - pad_left
    top = h + (h - height) // 2 - pad_top
    right = left + width + pad_left + pad_right
    bottom = top + height + pad_top + pad_bottom

    padded = canvas.crop((left, top, right, bottom))
    return padded


torch_transforms_ResNet = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

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

torch_transforms_DenseNet = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## === cell 2


def build_backbone_and_extractor(name: str):
    if name == "vit_h_14":
        m = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1).eval()
        return_nodes = {"heads.pre_logits": "feat"}  # shape [B, D]
        ex = create_feature_extractor(m, return_nodes=return_nodes).eval()
        feat_dim = m.heads.head.in_features
        return m, ex, feat_dim

    if name == "efficientnet_v2_l":
        m = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1).eval()
        return_nodes = {"classifier.0": "feat"}  # after dropout; shape [B, D]
        ex = create_feature_extractor(m, return_nodes=return_nodes).eval()
        feat_dim = m.classifier[-1].in_features
        return m, ex, feat_dim

    if name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        return_nodes = {"avgpool": "feat"}  # shape [B, 2048, 1, 1]
        ex = create_feature_extractor(m, return_nodes=return_nodes).eval()
        feat_dim = m.fc.in_features
        return m, ex, feat_dim

    if name == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
        return_nodes = {"features": "feat"}  # shape [B, C, H, W]
        ex = create_feature_extractor(m, return_nodes=return_nodes).eval()
        feat_dim = m.classifier.in_features
        return m, ex, feat_dim

    raise ValueError(f"Unknown backbone: {name}")


model1, ex1, dim1 = build_backbone_and_extractor("densenet121")
model2, ex2, dim2 = build_backbone_and_extractor("resnet50")
model3, ex3, dim3 = build_backbone_and_extractor("vit_h_14")
model4, ex4, dim4 = build_backbone_and_extractor("efficientnet_v2_l")

model1 = model1.to(device).eval()
model2 = model2.to(device).eval()
model3 = model3.to(device).eval()
model4 = model4.to(device).eval()
ex1 = ex1.to(device).eval()
ex2 = ex2.to(device).eval()
ex3 = ex3.to(device).eval()
ex4 = ex4.to(device).eval()

if device.type == "cuda":
    model1 = model1.to(memory_format=torch.channels_last)
    model2 = model2.to(memory_format=torch.channels_last)
    model4 = model4.to(memory_format=torch.channels_last)
    ex1 = ex1.to(memory_format=torch.channels_last)
    ex2 = ex2.to(memory_format=torch.channels_last)
    ex4 = ex4.to(memory_format=torch.channels_last)

if USE_TORCH_COMPILE and hasattr(torch, "compile"):
    try:
        model1 = torch.compile(model1, mode="reduce-overhead", fullgraph=False)
        model2 = torch.compile(model2, mode="reduce-overhead", fullgraph=False)
        model3 = torch.compile(model3, mode="reduce-overhead", fullgraph=False)
        model4 = torch.compile(model4, mode="reduce-overhead", fullgraph=False)
        ex1 = torch.compile(ex1, mode="reduce-overhead", fullgraph=False)
        ex2 = torch.compile(ex2, mode="reduce-overhead", fullgraph=False)
        ex3 = torch.compile(ex3, mode="reduce-overhead", fullgraph=False)
        ex4 = torch.compile(ex4, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled for models.")
    except Exception as e:
        print("torch.compile not used (fallback to eager):", repr(e))

FEAT_DIMS = (dim1, dim2, dim3, dim4)
TOTAL_FEATS = sum(FEAT_DIMS)
print(
    "Models initialized (pretrained backbones) with feature dims:",
    FEAT_DIMS,
    "total:",
    TOTAL_FEATS,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/3534559931.py in <cell line: 0>()
     46 model1, ex1, dim1 = build_backbone_and_extractor("densenet121")
     47 model2, ex2, dim2 = build_backbone_and_extractor("resnet50")
---> 48 model3, ex3, dim3 = build_backbone_and_extractor("vit_h_14")
     49 model4, ex4, dim4 = build_backbone_and_extractor("efficientnet_v2_l")
     50 

/tmp/ipykernel_54/3534559931.py in build_backbone_and_extractor(name)
     12         # Extract representation right before classification head.
     13         return_nodes = {"heads.pre_logits": "feat"}  # shape [B, D]
---> 14         ex = create_feature_extractor(m, return_nodes=return_nodes).eval()
     15         feat_dim = m.heads.head.in_features
     16         return m, ex, feat_dim

/usr/local/lib/python3.11/dist-packages/torchvision/models/feature_extraction.py in create_feature_extractor(model, return_nodes, train_return_nodes, eval_return_nodes, tracer_kwargs, suppress_diff_warning, concrete_args)
    543             # one of the available names starts with it up to a .
    544             if not any([re.match(rf"^{query}(\.|$)", n) is not None for n in available_nodes]):
--> 545                 raise ValueError(
    546                     f"node: '{query}' is not present in model. Hint: use "
    547                     "`get_graph_node_names` to make sure the "

ValueError: node: 'heads.pre_logits' is not present in model. Hint: use `get_graph_node_names` to make sure the `return_nodes` you specified are present. It may even be that you need to specify `train_return_nodes` and `eval_return_nodes` separately.

## === cell 3
class CassavaImageDataset(Dataset):
    """
    Speed optimization (correctness-preserving):
    - Use torchvision.io.read_image for faster JPEG decode into uint8 tensor (CPU).
    - Fall back to PIL if torchvision.io is not available.
    """

    def __init__(self, df, image_dir):
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].tolist()
        self.has_label = "label" in df.columns
        self.labels = df["label"].astype(np.int64).tolist() if self.has_label else None
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        y = int(self.labels[idx]) if self.has_label else -1
        fp = os.path.join(self.image_dir, image_id)

        if _HAS_TVIO:
            t = read_image(fp, mode=ImageReadMode.RGB)
            return image_id, t, y
        else:
            im = Image.open(fp).convert("RGB")
            return image_id, im, y


_tf_dn = torch_transforms_DenseNet
_tf_rn = torch_transforms_ResNet
_tf_en = torch_transforms_EfficientNet


def _invert_square_pad_tensor_uint8_chw(t: torch.Tensor) -> torch.Tensor:
    _, h, w = t.shape
    t2 = torch.roll(t, shifts=(h // 2, w // 2), dims=(1, 2))
    max_side = h if h >= w else w
    pad_left = (max_side - w) // 2
    pad_top = (max_side - h) // 2
    pad_right = (max_side - w) - pad_left
    pad_bottom = (max_side - h) - pad_top
    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return t2
    return F.pad(t2, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")


def quad_collate_fn(batch):
    bs = len(batch)
    image_ids = [None] * bs
    ys = torch.empty((bs,), dtype=torch.long)

    x1_out = x2_out = x3_out = x4_out = None

    for i, (image_id, im_or_t, y) in enumerate(batch):
        image_ids[i] = image_id
        ys[i] = y

        if isinstance(im_or_t, torch.Tensor):
            t = im_or_t  # uint8 CHW RGB
            t_dn = F.interpolate(
                t.unsqueeze(0).float().div_(255.0),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            t_rn = t_dn
            t_vit = _invert_square_pad_tensor_uint8_chw(t)
            t_vit = F.interpolate(
                t_vit.unsqueeze(0).float().div_(255.0),
                size=(518, 518),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            t_en = F.interpolate(
                t.unsqueeze(0).float().div_(255.0),
                size=(480, 480),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            t_dn = t_dn.mul_(2.0).sub_(1.0)
            t_rn = t_rn.mul_(2.0).sub_(1.0)
            t_vit = t_vit.mul_(2.0).sub_(1.0)
            t_en = t_en.mul_(2.0).sub_(1.0)

            t1, t2, t3, t4 = t_dn, t_rn, t_vit, t_en
        else:
            im = im_or_t
            t1 = _tf_dn(im)
            t2 = _tf_rn(im)
            im_vit = invert_square_pad(im)
            t3 = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Resize((518, 518)),
                    v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )(im_vit)
            t4 = _tf_en(im)

        if x1_out is None:
            x1_out = torch.empty((bs,) + tuple(t1.shape), dtype=t1.dtype)
            x2_out = torch.empty((bs,) + tuple(t2.shape), dtype=t2.dtype)
            x3_out = torch.empty((bs,) + tuple(t3.shape), dtype=t3.dtype)
            x4_out = torch.empty((bs,) + tuple(t4.shape), dtype=t4.dtype)

        x1_out[i].copy_(t1)
        x2_out[i].copy_(t2)
        x3_out[i].copy_(t3)
        x4_out[i].copy_(t4)

    return image_ids, (x1_out, x2_out, x3_out, x4_out), ys


@torch.inference_mode()
def _predict_features_from_4tensors(x1, x2, x3, x4) -> np.ndarray:
    if device.type == "cuda":
        x1 = x1.contiguous(memory_format=torch.channels_last).to(
            device, non_blocking=True
        )
        x2 = x2.contiguous(memory_format=torch.channels_last).to(
            device, non_blocking=True
        )
        x3 = x3.to(device, non_blocking=True)
        x4 = x4.contiguous(memory_format=torch.channels_last).to(
            device, non_blocking=True
        )
    else:
        x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            f1 = ex1(x1)["feat"]
            f2 = ex2(x2)["feat"]
            f3 = ex3(x3)["feat"]
            f4 = ex4(x4)["feat"]
    else:
        f1 = ex1(x1)["feat"]
        f2 = ex2(x2)["feat"]
        f3 = ex3(x3)["feat"]
        f4 = ex4(x4)["feat"]

    if f1.ndim == 4:
        f1 = F.adaptive_avg_pool2d(F.relu(f1, inplace=False), (1, 1)).flatten(1)
    if f2.ndim == 4:
        f2 = f2.flatten(1)
    f3 = f3.flatten(1)
    f4 = f4.flatten(1)

    out_t = torch.cat((f1, f2, f3, f4), dim=1).to("cpu")
    return out_t.numpy().astype(np.float32, copy=False)


def extract_features_dataframe(
    df: pd.DataFrame, image_dir: str, *, loader_bs: int
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    ds = CassavaImageDataset(df, image_dir)

    cpu = os.cpu_count() or 1
    if device.type == "cuda":
        num_workers = min(8, max(4, cpu // 2))
    else:
        num_workers = min(4, max(0, cpu // 2))

    g = torch.Generator()
    g.manual_seed(SEED)

    dl_kwargs = dict(
        batch_size=loader_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=quad_collate_fn,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 8 if device.type == "cuda" else 4
    dl = DataLoader(ds, **dl_kwargs)

    X = np.empty((len(ds), TOTAL_FEATS), dtype=np.float32)
    has_label = "label" in df.columns
    y = np.empty((len(ds),), dtype=np.int64) if has_label else None
    image_ids_all = [None] * len(ds)

    ofs = 0
    for image_ids, (x1, x2, x3, x4), ys in dl:
        feats = _predict_features_from_4tensors(x1, x2, x3, x4)
        bs = feats.shape[0]
        X[ofs : ofs + bs] = feats
        image_ids_all[ofs : ofs + bs] = image_ids
        if has_label:
            y[ofs : ofs + bs] = ys.numpy()
        ofs += bs
        if (ofs % 800 == 0) or (ofs == len(ds)):
            print(f"Features: {ofs}/{len(ds)}", end="\r")
    print()
    return X, y, image_ids_all


def extract_features_dataframe_adaptive(
    df: pd.DataFrame, image_dir: str, *, loader_bs: int
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    bs = int(loader_bs)
    while True:
        try:
            return extract_features_dataframe(df, image_dir, loader_bs=bs)
        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            new_bs = max(1, bs // 2)
            print(
                f"CUDA OOM during feature extraction at batch_size={bs}; retrying with batch_size={new_bs}"
            )
            bs = new_bs




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain columns: image_id, label")

MAX_META_TRAIN = 6000
if len(train_df) > MAX_META_TRAIN:
    meta_train_df, _ = train_test_split(
        train_df,
        train_size=MAX_META_TRAIN,
        stratify=train_df["label"],
        random_state=SEED,
    )
else:
    meta_train_df = train_df

meta_train_df = meta_train_df.reset_index(drop=True)
print("Meta-train size:", len(meta_train_df))

loader_bs = 96 if device.type == "cuda" else 16
X_meta, y_meta, _ = extract_features_dataframe_adaptive(
    meta_train_df, TRAIN_DIR, loader_bs=loader_bs
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2690816304.py in <cell line: 0>()
     18 
     19 loader_bs = 96 if device.type == "cuda" else 16
---> 20 X_meta, y_meta, _ = extract_features_dataframe_adaptive(
     21     meta_train_df, TRAIN_DIR, loader_bs=loader_bs
     22 )

/tmp/ipykernel_54/1731393045.py in extract_features_dataframe_adaptive(df, image_dir, loader_bs)
    213     while True:
    214         try:
--> 215             return extract_features_dataframe(df, image_dir, loader_bs=bs)
    216         except torch.OutOfMemoryError:
    217             if device.type != "cuda":

/tmp/ipykernel_54/1731393045.py in extract_features_dataframe(df, image_dir, loader_bs)
    187     dl = DataLoader(ds, **dl_kwargs)
    188 
--> 189     X = np.empty((len(ds), TOTAL_FEATS), dtype=np.float32)
    190     has_label = "label" in df.columns
    191     y = np.empty((len(ds),), dtype=np.int64) if has_label else None

NameError: name 'TOTAL_FEATS' is not defined

## === cell 5
decision_tree = RandomForestClassifier(
    n_estimators=90,
    criterion="gini",
    max_depth=6,
    random_state=42,
    n_jobs=-1,
)
decision_tree.fit(X_meta, y_meta)
print("Meta-model fitted.")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/288753876.py in <cell line: 0>()
      6     n_jobs=-1,
      7 )
----> 8 decision_tree.fit(X_meta, y_meta)
      9 print("Meta-model fitted.")
     10 

NameError: name 'X_meta' is not defined

## === cell 6
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Missing test_images directory at: {TEST_DIR}")

with os.scandir(TEST_DIR) as it:
    image_ids = sorted(
        [e.name for e in it if e.is_file() and e.name.lower().endswith(".jpg")]
    )

print("Test images:", len(image_ids))

test_df = pd.DataFrame({"image_id": image_ids})

loader_bs = 128 if device.type == "cuda" else 16
combined_output, _, image_ids_out = extract_features_dataframe_adaptive(
    test_df, TEST_DIR, loader_bs=loader_bs
)

if image_ids_out != image_ids:
    raise RuntimeError(
        "Image order mismatch during feature extraction; submission would misalign."
    )

prediction = decision_tree.predict(combined_output).astype(int)
print("Predictions:", prediction.shape, "unique:", np.unique(prediction))

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    if "image_id" in sample.columns and len(sample) == len(submission):
        submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
        if submission["label"].isna().any():
            raise RuntimeError(
                "Submission merge produced NaNs; check image_id alignment."
            )
        submission["label"] = submission["label"].astype(int)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/825267484.py in <cell line: 0>()
     12 
     13 loader_bs = 128 if device.type == "cuda" else 16
---> 14 combined_output, _, image_ids_out = extract_features_dataframe_adaptive(
     15     test_df, TEST_DIR, loader_bs=loader_bs
     16 )

/tmp/ipykernel_54/1731393045.py in extract_features_dataframe_adaptive(df, image_dir, loader_bs)
    213     while True:
    214         try:
--> 215             return extract_features_dataframe(df, image_dir, loader_bs=bs)
    216         except torch.OutOfMemoryError:
    217             if device.type != "cuda":

/tmp/ipykernel_54/1731393045.py in extract_features_dataframe(df, image_dir, loader_bs)
    187     dl = DataLoader(ds, **dl_kwargs)
    188 
--> 189     X = np.empty((len(ds), TOTAL_FEATS), dtype=np.float32)
    190     has_label = "label" in df.columns
    191     y = np.empty((len(ds),), dtype=np.int64) if has_label else None

NameError: name 'TOTAL_FEATS' is not defined
