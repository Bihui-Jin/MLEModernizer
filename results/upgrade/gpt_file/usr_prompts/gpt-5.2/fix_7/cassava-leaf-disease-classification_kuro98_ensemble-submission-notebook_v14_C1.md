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

3.12

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.9008763977032336

# 6. Current score

0.17227

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the ViT input-size assertion by automatically aligning `vit_img_size`/`eff_img_size` to the loaded models’ expected image sizes (224 for the torchvision fallback ViT), without changing the ensemble/training-free inference logic. I also make the TTA deterministic and collate-safe: `Random*` transforms be applied with fixed seeds per item/tta so the DataLoader can stack tensors reliably and results are reproducible. Finally, I ensure `pred_map` is always created (even if inference fails early) and that the submission strictly matches `sample_submission.csv` ordering/length and is written as `submission.csv`. These changes are correctness/stability fixes and should also improve score versus the broken run by enabling the intended inference/ensemble.'
- What this solution (achieved 0.22235) has done: 'I fix the DataLoader crash by ensuring the dataset only indexes actual image files (filtering out nested `test_images/` directories that exist inside the provided path). This is a correctness fix that allows inference to run end-to-end and produce a valid `submission.csv`. I also add a small safety fallback to build the image list from `sample_submission.csv` if needed, ensuring the prediction map aligns with the required submission rows and avoids missing-label issues. No model/ensemble logic is changed, so score should improve substantially versus the current broken/mostly-empty prediction run.'
- What this solution (achieved 0.22235) has done: 'Your current score (0.22235) is far below the target (0.90088), so we should improve accuracy while keeping the same inference-only ensemble core. The biggest issue is that your torchvision fallback models are ImageNet-pretrained but *not* Cassava-trained, and `linear_head` is never actually applied—so predictions are essentially random for the competition labels. The minimal, core-logic-preserving fix is to (1) correctly load the provided `.pt` checkpoints as `state_dict` when needed, (2) ensure the loaded objects are put into eval mode on the right device, and (3) actually apply `linear_head` (when present) to the combined logits before softmax. These changes keep your architecture/loop/TTA/ensemble intact but make the pipeline use the intended trained weights and head, which should move the score sharply toward the target.'
- What this solution (achieved 0.21226) has done: 'Your current score (0.22235) is far below the target (0.90088), so we should make minimal changes that improve correctness of inference rather than tuning for marginal gains. The biggest likely issue is a mismatch between your pretrained normalization/resize/crop pipeline and what your Cassava-trained checkpoints expect, plus the `linear_head` being applied to already-class logits (which can severely distort predictions if the head was trained on features). I (1) infer whether the loaded checkpoints already output 5-class logits and only apply `linear_head` when its input dimension matches the logits/features, and (2) make the input preprocessing closer to standard Cassava pipelines by switching from a fixed CenterCrop(600) to a simpler Resize to each model’s native size (keeping the same transforms/TTA logic), which often fixes large accuracy drops caused by aggressive cropping. These are small, inference-only adjustments that preserve your ensemble/TTA approach and should move the score sharply upward toward the target.'
- What this solution (achieved 0.17227) has done: 'Your score is far below the target, so we should focus on a small, correctness-focused fix that can materially improve accuracy without changing the ensemble/inference core. The biggest likely remaining issue is a preprocessing mismatch: using ImageNet mean/std for Cassava checkpoints often collapses accuracy, while Cassava solutions commonly use simple `[0.5,0.5,0.5]` normalization (or none) depending on training. I keep your resize/TTA/ensemble and model-loading logic intact, but switch to a safer Cassava-style normalization and also add a minimal safeguard to ensure both models really output 5 logits (otherwise fallback to argmax over whatever they output is effectively random). This should move the score substantially upward toward the target while remaining within your constraints and still producing `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

random.seed(3407)
np.random.seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

eff_img_size = 528
vit_img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

VIT_PATH = "/kaggle/input/vit-v1-update/vit_v1_1.pt"
EFF_PATH = "/kaggle/input/efficient-net/vit_cont_3.pt"
HEAD_PATH = "/kaggle/input/linear-head/linear_cls.pt"


def _try_load_torch_obj(path: str):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
    return None


def _is_state_dict(x):
    return isinstance(x, dict) and any(
        isinstance(k, str) and (k.endswith("weight") or k.endswith("bias"))
        for k in x.keys()
    )


vit_obj = _try_load_torch_obj(VIT_PATH)
eff_obj = _try_load_torch_obj(EFF_PATH)
head_obj = _try_load_torch_obj(HEAD_PATH)

vit_model = None
eff_model = None
linear_head = None

if isinstance(vit_obj, torch.nn.Module):
    vit_model = vit_obj
if isinstance(eff_obj, torch.nn.Module):
    eff_model = eff_obj
if isinstance(head_obj, torch.nn.Module):
    linear_head = head_obj

need_fallback_arch = (vit_model is None) or (eff_model is None)

if need_fallback_arch:
    import torchvision

    print(
        "Loading torchvision backbones to host state_dict checkpoints (or as a final fallback)."
    )

    if vit_model is None:
        vit_model = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        )
        vit_model.heads = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)

    if eff_model is None:
        eff_model = torchvision.models.efficientnet_b4(
            weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        eff_model.classifier[1] = torch.nn.Linear(
            eff_model.classifier[1].in_features, num_classes
        )


def _load_state_dict_forgiving(model, obj, name: str):
    if obj is None:
        return False
    sd = None
    if _is_state_dict(obj):
        sd = obj
    elif isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in obj and _is_state_dict(obj[key]):
                sd = obj[key]
                break
    if sd is None:
        return False

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(
        f"{name}: loaded state_dict with strict=False; missing={len(missing)}, unexpected={len(unexpected)}"
    )
    return True


_ = _load_state_dict_forgiving(vit_model, vit_obj, "vit_model")
_ = _load_state_dict_forgiving(eff_model, eff_obj, "eff_model")

if linear_head is None:
    if head_obj is not None and (
        _is_state_dict(head_obj) or isinstance(head_obj, dict)
    ):
        sd = head_obj
        if isinstance(head_obj, dict) and not _is_state_dict(head_obj):
            for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
                if key in head_obj and _is_state_dict(head_obj[key]):
                    sd = head_obj[key]
                    break
        w = None
        for k, v in sd.items():
            if (
                isinstance(k, str)
                and k.endswith("weight")
                and isinstance(v, torch.Tensor)
            ):
                w = v
                break
        if w is not None and w.ndim == 2 and w.shape[0] == num_classes:
            linear_head = torch.nn.Linear(int(w.shape[1]), num_classes)
            _load_state_dict_forgiving(linear_head, sd, "linear_head")
        else:
            linear_head = torch.nn.Identity()
    else:
        linear_head = torch.nn.Identity()


def _infer_vit_image_size(model):
    if hasattr(model, "image_size"):
        try:
            return int(model.image_size)
        except Exception:
            pass
    return 224


def _infer_eff_image_size(model, default):
    return int(default)


vit_img_size = _infer_vit_image_size(vit_model)
eff_img_size = _infer_eff_image_size(eff_model, eff_img_size)
print(f"Using vit_img_size={vit_img_size}, eff_img_size={eff_img_size}")

vit_model = vit_model.to(device)
eff_model = eff_model.to(device)
linear_head = linear_head.to(device)


def _get_linear_in_features(m):
    if isinstance(m, torch.nn.Linear):
        return int(m.in_features)
    return None


linear_in = _get_linear_in_features(linear_head)
print(f"linear_head type={type(linear_head).__name__}, in_features={linear_in}")


def apply_head_if_compatible(x: torch.Tensor) -> torch.Tensor:
    if isinstance(linear_head, torch.nn.Identity):
        return x
    if linear_in is not None and x.shape[-1] == linear_in:
        return linear_head(x)
    return x


def ensure_5_logits(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 2:
        x = x.view(x.shape[0], -1)
    if x.shape[1] == num_classes:
        return x
    proj = torch.nn.Linear(int(x.shape[1]), num_classes, bias=True).to(x.device)
    proj.eval()
    with torch.no_grad():
        return proj(x)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test images.

    Bugfix:
    - Filter to only actual image files to avoid IsADirectoryError in PIL.Image.open().
    - Use sample_submission.csv order as the image list to guarantee alignment.

    Change (score-related correctness):
    - Avoid hard CenterCrop(600,600) which can crop away leaf context and hurt accuracy.
      Instead, do a mild resize to each model's required size directly (common Cassava inference practice).
      This preserves the core inference/TTA/ensemble logic while fixing a frequent preprocessing mismatch.
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        transform=None,
        tta_transforms=None,
        tta_seed=3407,
        image_ids=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        if image_ids is not None:
            self.images = list(image_ids)
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            items = []
            for name in os.listdir(data_dir):
                p = os.path.join(data_dir, name)
                if os.path.isfile(p) and Path(name).suffix.lower() in exts:
                    items.append(name)
            self.images = sorted(items)

        self.tta_transforms = tta_transforms or []
        self.tta_seed = int(tta_seed)

        self.resize_vit = v2.Resize(
            (vit_size, vit_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _apply_with_seed(self, t, img, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return t(img)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        vit_base = self.resize_vit(img)
        eff_base = self.resize_efficient(img)

        if self.transform is None:
            raise RuntimeError("transform must be provided to produce tensors.")

        if len(self.tta_transforms) > 0:
            vit_views = []
            eff_views = []
            for j, t in enumerate(self.tta_transforms):
                seed = self.tta_seed + idx * 1000 + j
                vit_aug = self._apply_with_seed(t, vit_base, seed)
                eff_aug = self._apply_with_seed(t, eff_base, seed)
                vit_views.append(self.transform(vit_aug))
                eff_views.append(self.transform(eff_aug))
            return (
                torch.stack(vit_views, dim=0),
                torch.stack(eff_views, dim=0),
                filename,
            )

        vit_t = self.transform(vit_base)
        eff_t = self.transform(eff_base)
        return vit_t, eff_t, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

if tta:
    tta_transforms = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    tta_transforms = []

sample_sub_for_ids = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub_for_ids["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    transform=test_transforms,
    tta_transforms=tta_transforms,
    tta_seed=3407,
    image_ids=test_image_ids,
)


def seed_worker(worker_id):
    worker_seed = (3407 + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(3407)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 3
all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

pred_map = {}

with torch.no_grad():
    for vit_inputs, eff_inputs, filenames in test_loader:
        filenames = list(filenames)
        base_bs = len(filenames)

        if tta:
            if vit_inputs.ndim != 5 or eff_inputs.ndim != 5:
                raise RuntimeError(
                    f"Expected TTA tensors with shape [B,T,C,H,W], got vit={tuple(vit_inputs.shape)}, eff={tuple(eff_inputs.shape)}"
                )
            B, T = vit_inputs.shape[0], vit_inputs.shape[1]
            vit_inputs = vit_inputs.permute(1, 0, 2, 3, 4).reshape(
                T * B, *vit_inputs.shape[2:]
            )
            eff_inputs = eff_inputs.permute(1, 0, 2, 3, 4).reshape(
                T * B, *eff_inputs.shape[2:]
            )

            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            if isinstance(vit_outputs, (tuple, list)):
                vit_outputs = vit_outputs[0]
            if isinstance(eff_outputs, (tuple, list)):
                eff_outputs = eff_outputs[0]

            vit_batch_logits = vit_outputs.view(T, base_bs, -1)
            vit_mean_logits = vit_batch_logits.mean(dim=0)

            eff_batch_logits = eff_outputs.view(T, base_bs, -1)
            eff_mean_logits = eff_batch_logits.mean(dim=0)

            vit_mean_logits = ensure_5_logits(vit_mean_logits, "vit")
            eff_mean_logits = ensure_5_logits(eff_mean_logits, "eff")

            outputs = 0.6 * vit_mean_logits + 0.4 * eff_mean_logits

            outputs = apply_head_if_compatible(outputs)

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            if isinstance(vit_outputs, (tuple, list)):
                vit_outputs = vit_outputs[0]
            if isinstance(eff_outputs, (tuple, list)):
                eff_outputs = eff_outputs[0]

            vit_outputs = ensure_5_logits(vit_outputs, "vit")
            eff_outputs = ensure_5_logits(eff_outputs, "eff")

            outputs = (vit_outputs + eff_outputs) / 2.0

            outputs = apply_head_if_compatible(outputs)

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

pred_map = dict(zip(all_names, all_preds))
print(
    f"Predicted {len(pred_map)} unique test images out of {len(all_names)} total entries."
)




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
sample_sub["label"] = sample_sub["image_id"].map(pred_map)

if sample_sub["label"].isna().any():
    mode_label = (
        int(pd.Series(list(pred_map.values())).mode().iloc[0]) if len(pred_map) else 0
    )
    sample_sub["label"] = sample_sub["label"].fillna(mode_label).astype(int)
else:
    sample_sub["label"] = sample_sub["label"].astype(int)

submission_path = "submission.csv"
sample_sub.to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape={sample_sub.shape}")
print(sample_sub.head())
