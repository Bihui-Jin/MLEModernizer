# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

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


def _safe_torch_load(path, map_location):
    try:
        if path and os.path.exists(path):
            return torch.load(path, map_location=map_location)
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

if model_a is None or model_b is None:
    import torchvision

    def _choose_vit_weights_for_image_size(image_size: int):
        W = getattr(torchvision.models, "ViT_L_16_Weights", None)
        if W is None:
            return None

        candidates = []
        for attr in ["IMAGENET1K_V1", "IMAGENET1K_SWAG_E2E_V1", "DEFAULT"]:
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
        return None

    def _make_vit(num_classes: int, image_size: int = 384):
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

        if hasattr(m, "heads") and hasattr(m.heads, "head"):
            in_features = m.heads.head.in_features
            m.heads.head = torch.nn.Linear(in_features, num_classes)
        elif hasattr(m, "classifier") and isinstance(m.classifier, torch.nn.Linear):
            in_features = m.classifier.in_features
            m.classifier = torch.nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError(
                "Unexpected ViT head structure; cannot set num_classes=5."
            )
        return m

    if model_a is None:
        model_a = _make_vit(num_classes, image_size=model_a_img_size).to(device)
    if model_b is None:
        model_b = _make_vit(num_classes, image_size=model_b_img_size).to(device)

model_a = model_a.to(device)
model_b = model_b.to(device)
if linear_head is not None and hasattr(linear_head, "to"):
    linear_head = linear_head.to(device)




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
        labels=None,  # Optimization: optional labels aligned with image_ids to avoid per-batch Python dict lookups
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
    transform=test_transforms,  # same normalization pipeline as inference
    ttas=None,
    image_ids=train_image_ids,
    labels=train_labels,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # fine for head training
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


model_a_feat, feat_dim_a = _get_vit_feature_extractor(model_a)
model_b_feat, feat_dim_b = _get_vit_feature_extractor(model_b)

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

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(linear_head.parameters(), lr=3e-4, weight_decay=1e-2)

epochs = 1


@torch.no_grad()
def _extract_train_features_to_cpu(loader):
    feats = []
    ys = []
    for model_a_inputs, model_b_inputs, y, _fn in loader:
        model_a_inputs = model_a_inputs.to(device, non_blocking=True)
        model_b_inputs = model_b_inputs.to(device, non_blocking=True)

        fa = model_a_feat(model_a_inputs)
        fb = model_b_feat(model_b_inputs)
        f = 0.91 * fa + 0.09 * fb

        feats.append(f.detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(ys, dim=0)


cached_feat, cached_y = _extract_train_features_to_cpu(train_loader)

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

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        model_a_inputs, model_b_inputs, filenames = batch
        bsz = len(filenames)

        if tta:
            model_a_inputs, model_b_inputs, filenames = (
                torch.cat(model_a_inputs, dim=0).to(device, non_blocking=True),
                torch.cat(model_b_inputs, dim=0).to(device, non_blocking=True),
                list(filenames),
            )

            fa = model_a_feat(model_a_inputs)
            fb = model_b_feat(model_b_inputs)

            fa_batch = torch.stack(torch.split(fa, bsz), dim=0)
            fa_mean = torch.mean(fa_batch, dim=0)

            fb_batch = torch.stack(torch.split(fb, bsz), dim=0)
            fb_mean = torch.mean(fb_batch, dim=0)

            feats = 0.9 * fa_mean + 0.1 * fb_mean
            outputs = linear_head(feats)
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs, model_b_inputs, filenames = (
                model_a_inputs.to(device, non_blocking=True),
                model_b_inputs.to(device, non_blocking=True),
                list(filenames),
            )

            fa = model_a_feat(model_a_inputs)
            fb = model_b_feat(model_b_inputs)

            feats = 0.91 * fa + 0.09 * fb
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
