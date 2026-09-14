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

0.8931701420368692

# 6. Current score

0.30493

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I first fix the hard runtime blocker: the code tries to `torch.load` three external model files that are not present in this environment, which causes `model_a`/`model_b` to be undefined and cascades into later errors. To keep the core “two-model logits ensemble + softmax + argmax” inference logic intact while making the notebook runnable end-to-end, I fall back to standard torchvision ViT backbones (same type of model) when those `.pt` files aren’t available. I also fix submission validity by ensuring `test_loader` is not shuffled (so ordering matches `sample_submission.csv` exactly) and by building predictions in the exact sample order (with an explicit reindex to sample order as a final guard). Finally, I remove the unused `linear_head` usage (it was never applied) but keep it loaded if present to preserve compatibility.'
- What this solution (achieved 0.61099) has done: 'I fix the runtime error by ensuring the fallback torchvision ViT models expect 384×384 inputs (your pipeline resizes to 384, but `vit_b_16` defaults to 224, causing the assertion). To keep core inference logic intact, I switch the fallback backbone to a torchvision ViT variant that natively uses 384 (and still outputs 5 classes via the same head replacement). I also make the dataloader worker/pin_memory settings robust across CPU/GPU so the notebook runs end-to-end reliably, and keep the submission ordering guard via `sample_submission.csv` unchanged.'
- What this solution (achieved 0.19656) has done: 'I fix the runtime assertion by ensuring the fallback torchvision ViT models are instantiated with `image_size=384` so they accept your existing 384×384 pipeline without changing the overall “two-model logits ensemble + softmax + argmax” inference logic. I also make the fallback creation robust across torchvision versions by trying the newer constructor signature first and falling back safely if needed. No training/evaluation semantics are changed beyond making the model accept the correct input resolution. The script still write `submission.csv` with the required `image_id,label` columns and enforce the exact `sample_submission.csv` order.'
- What this solution (achieved 0.19544) has done: 'I fix the DataLoader crash by filtering out directories/non-image entries from `test_images/` (your folder contains a nested `test_images` directory that gets picked up by `os.listdir`). I also add a small safety check to ensure we only include typical image extensions and keep a deterministic, sorted file list, preserving your existing inference/ensemble logic. Finally, I keep the submission ordering guard against `sample_submission.csv` exactly as you already intended, so the produced `submission.csv` is valid and aligned.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.19544) is far below the target (0.89317), and the main cause is that you’re running randomly initialized fallback ViT models when the external `.pt` weights aren’t available—this yields near-random predictions. To move the score toward the target while preserving your core “two-model logits ensemble + softmax + argmax” inference logic, I keep your pipeline intact but switch the fallback to use strong ImageNet-pretrained ViT weights and adapt the classifier head to 5 classes (head is new, but the backbone is meaningful). I also fix a subtle but important scaling issue in your ensembling: you currently divide by `2.0` even though weights already sum to 1.0, which unnecessarily shrinks logits and can hurt argmax stability; removing that keeps semantics the same but improves calibration. Submission ordering/format safeguards remain unchanged and it still writes `submission.csv`.'
- What this solution (achieved 0.19544) has done: 'I fix the root runtime blocker in the fallback ViT creation: torchvision’s SWAG weights force a specific `image_size` (512), which conflicts with your 384×384 pipeline and raises the `ValueError`. The minimal fix is to pick pretrained ViT weights that allow `image_size=384` (or fall back safely), keeping your two-model logits ensemble + softmax + argmax inference unchanged. This also ensure `model_a`/`model_b` are always constructed (never `None`), eliminating the downstream `.eval()` crash and producing a valid `submission.csv`. These changes should substantially improve score versus near-random heads while staying within your original approach.'
- What this solution (achieved 0.61099) has done: 'The timeout is dominated by (1) running ViT-L forward passes over the entire 18.7k-image training set just to fit the linear head, and (2) slow/duplicated Python work in the training loop (per-batch dict/list lookups, CPU-side label building, non-persistent workers). We keep the exact same models, feature mixing, loss, optimizer, and 1-epoch training loop, but remove avoidable overhead by moving label resolution into the dataset, enabling persistent DataLoader workers + prefetching, and compiling the fixed feature-extractor models to speed up inference-style forward passes. These changes are provably equivalent in semantics (same inputs/labels, same math), but reduce CPU overhead and improve GPU utilization to fit within 600s.'
- What this solution (achieved 0.61435) has done: 'The timeout is dominated by running two ViT-L/16 forward passes over the entire *training* set just to fit the linear head, plus slower CPU-side image decoding/transform overhead. I keep the exact same model logic (same feature mixing weights, same linear head training loop, same inference semantics), but eliminate wasted work by (1) extracting train features with `shuffle=False` and larger, more efficient batches, (2) enabling inference-mode + autocast for the frozen ViT feature extractors (safe for correctness within negligible FP drift), and (3) using faster Pillow settings and DataLoader knobs to reduce input pipeline stalls. I also avoid unnecessary Python overhead in the TTA path and make feature caching more memory/transfer efficient without changing outputs.'
- What this solution (achieved 0.61099) has done: 'You’re far below the target accuracy, so the smallest meaningful improvement is to stop training the linear head on features from a randomly re-initialized (5-class) ViT head and instead extract features from the ImageNet-pretrained backbone *before* replacing the classifier. This keeps your core logic intact (two ViT feature extractors mixed with fixed weights → a single linear head trained for 1 epoch → softmax/argmax submission), but makes the cached features informative. I also fix a subtle bug where `_get_vit_feature_extractor` mutates `model_a/model_b` in-place, potentially breaking later use and causing inconsistent state; we now create separate backbone instances for feature extraction while preserving your ensemble weights and pipeline. Submission ordering/format remains unchanged and `submission.csv` is still written.'
- What this solution (achieved 0.61323) has done: 'I fix the runtime blocker by replacing the incorrect `torch.deepcopy` call with Python’s `copy.deepcopy`, which allow feature-extractor construction to run and unblock the rest of the pipeline. I also add a small compatibility import and make sure the code never crashes if a model load returns a non-Module object by validating and falling back safely. These changes are execution-only (score-neutral by themselves), preserving your existing two-ViT feature mixing + 1-epoch linear head training + softmax/argmax submission semantics. The script then run end-to-end and write a valid `submission.csv` in the required `image_id,label` format and sample order.'
- What this solution (achieved 0.29447) has done: 'Your current score (0.613) is far below the target (0.893), and the main driver is that the “linear head trained for 1 epoch” is being trained while your ViT backbones are still in `eval()` mode, so BatchNorm/Dropout (if present) and, more importantly, the optimizer sees features without any train-time stochasticity; the bigger issue is that the head training is effectively underpowered and under-regularized for the domain shift. To move score upward while keeping your exact core pipeline (two frozen ViT feature extractors mixed with fixed weights → train a linear head for 1 epoch → softmax/argmax submission), I make two minimal, metric-aligned changes: (1) use standard class-weighted CrossEntropy (computed from `train.csv`) to reduce majority-class bias, and (2) add a tiny bit of label smoothing to improve calibration/generalization without changing the model architecture or loops. These are small, safe adjustments that typically yield a noticeable accuracy gain for Cassava while preserving your semantics and producing the same valid `submission.csv` format/order.'
- What this solution (achieved 0.30493) has done: 'Your score is far below the target, so we should improve generalization without changing the core pipeline (two frozen ViT feature extractors mixed with fixed weights → train a linear head for 1 epoch → softmax/argmax submission). The biggest low-risk gain here is fixing a train/test mismatch: you currently train the head on center-cropped images but evaluate on a (different) center-crop plus resize without any augmentation, which tends to overfit and underperform on Cassava. I keep the same architecture and 1-epoch training loop, but add a standard light training augmentation pipeline (random resized crop + horizontal flip) only for head training, while keeping test transforms unchanged. I also make the feature-mix weights consistent (0.91/0.09 everywhere) to avoid a small systematic mismatch between training/inference.'

# 9. Code solution

## === cell 0
import os
import copy

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = False

W_A = 0.91
W_B = 0.09

try:
    Image.MAX_IMAGE_PIXELS = None
    ImageFile = __import__("PIL.ImageFile", fromlist=["ImageFile"]).ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass


def _safe_torch_load(path, map_location):
    try:
        if path and os.path.exists(path):
            obj = torch.load(path, map_location=map_location)
            return obj
    except Exception:
        return None
    return None


model_a = _safe_torch_load("/kaggle/input/vit-v1/vit_v1.pt", map_location=device)
model_b = _safe_torch_load(
    "/kaggle/input/vit-boosted/vit_boosted.pt", map_location=device
)
linear_head = _safe_torch_load(
    "/kaggle/input/linear-head/linear_cls.pt", map_location=device
)

if model_a is not None and not isinstance(model_a, torch.nn.Module):
    model_a = None
if model_b is not None and not isinstance(model_b, torch.nn.Module):
    model_b = None
if linear_head is not None and not isinstance(linear_head, torch.nn.Module):
    linear_head = None

if model_a is None or model_b is None:
    import torchvision

    def _choose_vit_weights_for_image_size(image_size: int):
        W = getattr(torchvision.models, "ViT_L_16_Weights", None)
        if W is None:
            return None

        candidates = []
        for attr in ["DEFAULT", "IMAGENET1K_V1"]:
            if hasattr(W, attr):
                candidates.append(getattr(W, attr))

        for w in candidates:
            try:
                min_size = w.meta.get("min_size", None)
                if min_size is None:
                    return w
                if isinstance(min_size, (list, tuple)) and len(min_size) >= 1:
                    if int(min_size[0]) <= int(image_size):
                        return w
            except Exception:
                continue
        return candidates[0] if candidates else None

    def _make_vit_backbone(image_size: int = 384):
        weights = _choose_vit_weights_for_image_size(image_size)

        try:
            try:
                m = torchvision.models.vit_l_16(weights=weights, image_size=image_size)
            except TypeError:
                m = torchvision.models.vit_l_16(weights=weights)
                if hasattr(m, "image_size"):
                    m.image_size = image_size
        except ValueError:
            try:
                m = torchvision.models.vit_l_16(weights=None, image_size=image_size)
            except TypeError:
                m = torchvision.models.vit_l_16(weights=None)
                if hasattr(m, "image_size"):
                    m.image_size = image_size
        return m

    if model_a is None:
        model_a = _make_vit_backbone(image_size=model_a_img_size).to(device)
    if model_b is None:
        model_b = _make_vit_backbone(image_size=model_b_img_size).to(device)

model_a = model_a.to(device)
model_b = model_b.to(device)
if linear_head is not None and hasattr(linear_head, "to"):
    linear_head = linear_head.to(device)

USE_AUTOCAST = torch.cuda.is_available()
AUTOCast_DTYPE = (
    torch.float16
)  # feature extractors only; head math remains fp32 by default.




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Args:
        data_dir: base directory to the images.
        transforms: set of transforms to be used.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        image_ids=None,  # optional explicit ordering list (used for test to match sample_submission exactly)
        labels=None,  # optional labels aligned with image_ids
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if image_ids is not None:
            self.images = list(image_ids)
        else:
            files = []
            for name in os.listdir(data_dir):
                path = os.path.join(data_dir, name)
                if os.path.isfile(path) and os.path.splitext(name.lower())[1] in exts:
                    files.append(name)
            self.images = sorted(files)

        if labels is not None:
            if len(labels) != len(self.images):
                raise ValueError("labels length must match image_ids/images length")
            self.labels = list(labels)
        else:
            self.labels = None

        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )

        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        if self.labels is None:
            return model_a_img, model_b_img, filename
        else:
            return model_a_img, model_b_img, int(self.labels[idx]), filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(model_a_img_size, model_a_img_size),
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_ids=sample_sub["image_id"].tolist(),
)

pin_memory = torch.cuda.is_available()

cpu_cnt = os.cpu_count() or 1
nw = min(max(num_workers, 4), cpu_cnt)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=nw,
    pin_memory=pin_memory,
    persistent_workers=(nw > 0),
    prefetch_factor=4 if nw > 0 else None,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
train_df = pd.read_csv(train_csv_path)
train_image_ids = train_df["image_id"].tolist()
train_labels = train_df["label"].astype(int).tolist()

train_dataset = CassavaDataset(
    train_dir,
    model_a_img_size,
    model_b_img_size,
    transform=train_transforms,
    ttas=None,
    image_ids=train_image_ids,
    labels=train_labels,
)

feat_extract_batch = max(batch_size, 32) if torch.cuda.is_available() else batch_size

train_loader_feat = DataLoader(
    train_dataset,
    batch_size=feat_extract_batch,
    shuffle=False,
    num_workers=nw,
    pin_memory=pin_memory,
    persistent_workers=(nw > 0),
    prefetch_factor=4 if nw > 0 else None,
)

train_loader_head = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=nw,
    pin_memory=pin_memory,
    persistent_workers=(nw > 0),
    prefetch_factor=4 if nw > 0 else None,
)


def _get_vit_feature_extractor(vit_model: torch.nn.Module):
    m = vit_model
    feat_dim = None
    if (
        hasattr(m, "heads")
        and hasattr(m.heads, "head")
        and isinstance(m.heads.head, torch.nn.Linear)
    ):
        feat_dim = m.heads.head.in_features
        m.heads.head = torch.nn.Identity()
        return m, feat_dim
    if hasattr(m, "classifier") and isinstance(m.classifier, torch.nn.Linear):
        feat_dim = m.classifier.in_features
        m.classifier = torch.nn.Identity()
        return m, feat_dim
    raise RuntimeError("Cannot locate ViT classifier head to extract features.")


model_a_feat, feat_dim_a = _get_vit_feature_extractor(copy.deepcopy(model_a).eval())
model_b_feat, feat_dim_b = _get_vit_feature_extractor(copy.deepcopy(model_b).eval())

model_a_feat = model_a_feat.to(device)
model_b_feat = model_b_feat.to(device)

feat_dim = feat_dim_a  # vit_l_16 should match
if feat_dim_a != feat_dim_b:
    raise RuntimeError(f"Feature dims differ: {feat_dim_a} vs {feat_dim_b}")

if linear_head is None or not isinstance(linear_head, torch.nn.Module):
    linear_head = torch.nn.Linear(feat_dim, num_classes).to(device)
else:
    linear_head = linear_head.to(device)

model_a_feat.eval()
model_b_feat.eval()
linear_head.train()

USE_COMPILE = False

if USE_COMPILE and hasattr(torch, "compile"):
    try:
        model_a_feat = torch.compile(model_a_feat, mode="reduce-overhead")
        model_b_feat = torch.compile(model_b_feat, mode="reduce-overhead")
        linear_head = torch.compile(linear_head, mode="reduce-overhead")
    except Exception:
        pass

label_counts = train_df["label"].value_counts().sort_index()
counts = torch.tensor(
    [label_counts.get(i, 0) for i in range(num_classes)], dtype=torch.float32
)
counts = torch.clamp(counts, min=1.0)
class_weights = counts.sum() / counts
class_weights = (class_weights / class_weights.mean()).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

optimizer = torch.optim.AdamW(linear_head.parameters(), lr=3e-4, weight_decay=1e-2)

epochs = 1


@torch.inference_mode()
def _extract_train_features_to_cpu(loader):
    n = len(loader.dataset)
    feats_cpu = torch.empty((n, feat_dim), dtype=torch.float32, device="cpu")
    ys_cpu = torch.empty((n,), dtype=torch.int64, device="cpu")
    offset = 0

    for model_a_inputs, model_b_inputs, y, _fn in loader:
        b = y.shape[0]
        model_a_inputs = model_a_inputs.to(device, non_blocking=True)
        model_b_inputs = model_b_inputs.to(device, non_blocking=True)

        if USE_AUTOCAST:
            with torch.autocast(device_type="cuda", dtype=AUTOCast_DTYPE):
                fa = model_a_feat(model_a_inputs)
                fb = model_b_feat(model_b_inputs)
                f = W_A * fa + W_B * fb
        else:
            fa = model_a_feat(model_a_inputs)
            fb = model_b_feat(model_b_inputs)
            f = W_A * fa + W_B * fb

        feats_cpu[offset : offset + b].copy_(
            f.float().detach().cpu(), non_blocking=False
        )
        ys_cpu[offset : offset + b].copy_(
            y.to(torch.int64).detach().cpu(), non_blocking=False
        )
        offset += b

    return feats_cpu, ys_cpu


cached_feat, cached_y = _extract_train_features_to_cpu(train_loader_feat)

cached_ds = TensorDataset(cached_feat, cached_y)
cached_loader = DataLoader(
    cached_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # tensors already in-memory; workers add overhead
    pin_memory=pin_memory,
)

for epoch in range(epochs):
    total_loss = 0.0
    seen = 0
    for f_cpu, y_cpu in cached_loader:
        f = f_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)

        logits = linear_head(f)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(y.shape[0])
        seen += int(y.shape[0])

    print(f"linear_head epoch={epoch+1}/{epochs} loss={total_loss/max(seen,1):.5f}")

linear_head.eval()



## === cell 4
all_names = []
all_preds = []

model_a_feat.eval()
model_b_feat.eval()
linear_head.eval()

with torch.inference_mode():
    for batch_idx, batch in enumerate(test_loader):
        model_a_inputs, model_b_inputs, filenames = batch
        bsz = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(
                device, non_blocking=True
            )
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(
                device, non_blocking=True
            )
            filenames = list(filenames)

            if USE_AUTOCAST:
                with torch.autocast(device_type="cuda", dtype=AUTOCast_DTYPE):
                    fa = model_a_feat(model_a_inputs)
                    fb = model_b_feat(model_b_inputs)
            else:
                fa = model_a_feat(model_a_inputs)
                fb = model_b_feat(model_b_inputs)

            fa_batch = fa.view(-1, bsz, fa.shape[1]).mean(dim=0)
            fb_batch = fb.view(-1, bsz, fb.shape[1]).mean(dim=0)

            feats = W_A * fa_batch + W_B * fb_batch
            outputs = linear_head(feats)
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            if USE_AUTOCAST:
                with torch.autocast(device_type="cuda", dtype=AUTOCast_DTYPE):
                    fa = model_a_feat(model_a_inputs)
                    fb = model_b_feat(model_b_inputs)
                    feats = W_A * fa + W_B * fb
                    outputs = linear_head(feats)
            else:
                fa = model_a_feat(model_a_inputs)
                fb = model_b_feat(model_b_inputs)
                feats = W_A * fa + W_B * fb
                outputs = linear_head(feats)

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 5
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = pred_df.set_index("image_id").reindex(sample_sub["image_id"]).reset_index()

pred_df["label"] = pred_df["label"].fillna(3).astype(int)

pred_df.to_csv("submission.csv", index=False)
print(pred_df.shape)
print(pred_df.head())
