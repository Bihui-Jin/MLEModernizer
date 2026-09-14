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

0.862042913266848

# 6. Current score

0.08146

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the pipeline so it runs end-to-end and always writes a valid `submission.csv` with exactly the same `image_id` ordering and length as `sample_submission.csv`. The main runtime failure is missing external model files (`/kaggle/input/vit-v1/...` etc.), so I keep the same inference/ensemble logic but add a safe fallback to a torchvision ViT model if those files aren’t available. I also fix dataset ordering (remove `shuffle=True`, sort filenames) and align predictions to `sample_submission.csv` to eliminate the “same length as the answers” submission error. Finally, I ensure images are converted to RGB to avoid shape issues with some JPEGs.'
- What this solution (achieved 0.23019) has done: 'I fix the runtime error by ensuring the fallback torchvision ViT models receive inputs at the image size they expect (224), while keeping your existing dual-model averaging inference logic unchanged. I do this by dynamically reading each model’s expected `image_size` and resizing per-model accordingly in the dataset, instead of hard-coding 384. I also make `_safe_torch_load` robust to both full-model and state_dict checkpoints (a common cause of silent “fallback-only” behavior), which should increase accuracy toward your target without changing the overall approach. Finally, I keep the submission alignment logic intact so `submission.csv` is always valid and ordered like `sample_submission.csv`.'
- What this solution (achieved 0.08146) has done: 'Your current score strongly suggests you’re effectively submitting predictions from randomly initialized fallback ViTs (the custom checkpoints aren’t being used), so the smallest meaningful improvement is to make the fallback models actually perform inference with ImageNet-pretrained features. I keep the same dual-model averaging, transforms, and submission alignment logic, but change the fallback to keep the original pretrained ViT backbone and only adapt the classifier head in a way that preserves pretrained weights (instead of replacing the head with a randomly initialized 5-class layer). Concretely: we keep the 1000-class ImageNet head and map it to 5 classes via a fixed, deterministic “class-to-class aggregation” layer (no training), which is a minimal inference-only tweak that usually lifts accuracy well above random while preserving the overall approach. This should move accuracy substantially toward your target without changing the ensemble or adding training.'
- What this solution (achieved 0.08146) has done: 'Your current score is far below the target, and the most likely reason is that you’re still effectively using an untrained/random mapping from ImageNet logits to 5 cassava classes in the fallback path, so predictions are near-random. I keep your exact two-model averaging inference pipeline, but make the fallback head a deterministic, label-aware mapper that uses ImageNet semantics (via `label_num_to_disease_map.json`) to aggregate relevant ImageNet classes into each cassava class, which should substantially increase accuracy without adding training or changing the model architecture/loop. I also ensure TTA (if enabled) is deterministic by seeding per-worker, and keep the submission alignment logic unchanged so the CSV is always valid and ordered like `sample_submission.csv`. These are minimal inference-only changes aimed at moving the score upward toward your target band.'

# 9. Code solution

## === cell 0
import os
import json
import re

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
label_map_path = (
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _load_label_num_to_disease_map(path: str):
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return {int(k): str(v) for k, v in d.items()}
    except Exception as e:
        print(f"WARNING: Could not read label map at {path}: {e}")
        return None


def _tokenize(text: str):
    return re.findall(r"[a-z]+", text.lower())


def _build_semantic_projector_from_imagenet(
    weights_meta, cassava_label_map, num_classes: int
):
    """
    Build a fixed (non-trained) Linear(1000->5) that aggregates ImageNet classes
    whose labels match tokens in the cassava disease names.

    This preserves your inference-only approach and avoids a random 5-class head,
    aiming to improve score toward target while keeping core logic unchanged.
    """
    categories = (
        list(getattr(weights_meta, "categories", []))
        if weights_meta is not None
        else []
    )
    if len(categories) != 1000 or not cassava_label_map:
        return None

    cassava_tokens = {}
    for k in range(num_classes):
        name = cassava_label_map.get(k, "")
        toks = set(_tokenize(name))
        if k == 4:  # healthy
            toks |= {"healthy", "leaf", "leaves"}
        cassava_tokens[k] = toks

    matches = {k: [] for k in range(num_classes)}
    for idx, cat in enumerate(categories):
        cat_toks = set(_tokenize(cat))
        if not cat_toks:
            continue
        for k in range(num_classes):
            if len(cat_toks & cassava_tokens[k]) > 0:
                matches[k].append(idx)

    proj = torch.nn.Linear(1000, num_classes, bias=False)
    with torch.no_grad():
        proj.weight.zero_()
        for k in range(num_classes):
            idxs = matches[k]
            if len(idxs) > 0:
                proj.weight[k, idxs] = 1.0 / float(len(idxs))

        proj.weight += 1e-3 / 1000.0

    print(
        "Fallback semantic projector match counts:",
        {k: len(v) for k, v in matches.items()},
    )
    return proj


def _build_fallback_vit(num_classes: int, label_map_path: str):
    """
    Change (score-critical, minimal): keep ImageNet-pretrained ViT backbone, and
    use a deterministic semantic projector (based on ImageNet class labels and cassava disease names)
    instead of a random or arbitrary partitioning of 1000 classes into 5.
    """
    import torchvision

    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    base = torchvision.models.vit_b_16(weights=weights)
    base.eval()

    cassava_map = _load_label_num_to_disease_map(label_map_path)
    proj = _build_semantic_projector_from_imagenet(
        weights.meta, cassava_map, num_classes
    )
    if proj is None:
        proj = torch.nn.Linear(1000, num_classes, bias=False)
        with torch.no_grad():
            proj.weight.zero_()
            edges = [0, 200, 400, 600, 800, 1000]
            for k in range(num_classes):
                a, b = edges[k], edges[k + 1]
                proj.weight[k, a:b] = 1.0 / float(b - a)

    class FallbackWrapped(torch.nn.Module):
        def __init__(self, backbone, projector):
            super().__init__()
            self.backbone = backbone
            self.projector = projector
            self.image_size = getattr(backbone, "image_size", 224)

        def forward(self, x):
            logits1000 = self.backbone(x)  # [B, 1000]
            return self.projector(logits1000)  # [B, 5]

    return FallbackWrapped(base, proj)


def _safe_torch_load(path: str, num_classes: int):
    if not os.path.exists(path):
        return None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        if isinstance(sd, dict):
            m = _build_fallback_vit(num_classes, label_map_path)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(
                    f"WARNING: Loaded state_dict from {path} with missing={len(missing)} unexpected={len(unexpected)}"
                )
            return m
    print(f"WARNING: Unrecognized checkpoint format at {path}; ignoring.")
    return None


model_a = _safe_torch_load("/kaggle/input/vit-v1/vit_v1.pt", num_classes)
model_b = _safe_torch_load("/kaggle/input/vit-boosted/vit_boosted.pt", num_classes)
linear_head = _safe_torch_load(
    "/kaggle/input/linear-head/linear_cls.pt", num_classes
)  # not used in original inference; keep for compatibility

if model_a is None or model_b is None:
    print(
        "WARNING: Pretrained competition model files not found/loaded; using torchvision ViT fallback models."
    )
    model_a = _build_fallback_vit(num_classes, label_map_path)
    model_b = _build_fallback_vit(num_classes, label_map_path)


def _get_model_image_size(m, default_size: int) -> int:
    s = getattr(m, "image_size", None)
    if isinstance(s, int):
        return s
    return int(default_size)


model_a_img_size = _get_model_image_size(model_a, model_a_img_size)
model_b_img_size = _get_model_image_size(model_b, model_b_img_size)
print(
    "Using model_a_img_size:", model_a_img_size, "model_b_img_size:", model_b_img_size
)

model_a = model_a.to(device)
model_b = model_b.to(device)
if linear_head is not None:
    linear_head = linear_head.to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (int(model_a_size), int(model_a_size)),
            interpolation=InterpolationMode.BICUBIC,
        )
        self.resize_model_b = v2.Resize(
            (int(model_b_size), int(model_b_size)),
            interpolation=InterpolationMode.BICUBIC,
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

        return model_a_img, model_b_img, filename

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

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, model_a_img_size, model_b_img_size, transform=test_transforms, ttas=ttas
)


def _seed_worker(worker_id):
    worker_seed = (3407 + worker_id) % 2**32
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(3407)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=g,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
if linear_head is not None:
    linear_head.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bs), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bs), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = (model_a_mean_logits + model_b_mean_logits) / 2.0
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            outputs = (model_a_outputs + model_b_outputs) / 2.0
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print("Predictions:", len(all_preds), "Files:", len(all_names))



## === cell 4
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if missing:
    for img_id in missing:
        pred_map[img_id] = 4

sample_sub["label"] = sample_sub["image_id"].map(pred_map).astype(int)
sample_sub.to_csv("submission.csv", index=False)

print(sample_sub.head())
print("Wrote submission.csv with rows:", len(sample_sub))
