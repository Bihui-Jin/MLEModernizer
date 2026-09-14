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
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
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

finetune_fallback = True
finetune_epochs = 2
finetune_lr = 3e-5
finetune_batch_size = 32  # separate from inference batch size
finetune_num_workers = num_workers


def _safe_torch_load(path: str, map_location):
    obj = torch.load(path, map_location=map_location)
    return obj


def _resize_vit_pos_embedding_(
    model: torch.nn.Module, image_size: int, patch_size: int = 16
):
    """
    Bugfix: torchvision ViT asserts input H/W match model.image_size.
    When adapting to 384px inputs, update both:
      - positional embeddings (grid resized)
      - model.image_size attribute (so the forward assertion matches)
    """
    if hasattr(model, "image_size"):
        try:
            model.image_size = image_size
        except Exception:
            pass

    if not hasattr(model, "encoder") or not hasattr(model.encoder, "pos_embedding"):
        return model

    pos = model.encoder.pos_embedding  # [1, 1+N, D]
    if not isinstance(pos, torch.nn.Parameter):
        return model

    with torch.no_grad():
        pos_data = pos.data
        if pos_data.ndim != 3 or pos_data.shape[0] != 1:
            return model

        n_tokens, dim = pos_data.shape[1], pos_data.shape[2]
        if n_tokens < 2:
            return model

        cls_pos = pos_data[:, :1, :]  # [1,1,D]
        grid_pos = pos_data[:, 1:, :]  # [1,N,D]

        old_n = grid_pos.shape[1]
        old_g = int(old_n**0.5)
        if old_g * old_g != old_n:
            return model

        new_g = image_size // patch_size
        if new_g * new_g == old_n:
            return model

        grid_pos = grid_pos.reshape(1, old_g, old_g, dim).permute(0, 3, 1, 2)
        grid_pos = torch.nn.functional.interpolate(
            grid_pos, size=(new_g, new_g), mode="bicubic", align_corners=False
        )
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, new_g * new_g, dim)

        new_pos = torch.cat([cls_pos, grid_pos], dim=1)
        model.encoder.pos_embedding = torch.nn.Parameter(new_pos)

    return model


def _build_fallback_vit(num_classes: int, image_size: int):
    """
    Bugfix: torchvision ViT with pretrained weights enforces weights.meta["min_size"]=224
    and raises if image_size is passed as 384. We therefore:
      1) instantiate pretrained vit_b_16 at its default (224),
      2) resize positional embeddings + set model.image_size to desired size (384),
      3) replace classification head for 5 cassava classes.
    This preserves the ViT-B/16 core architecture and provides a strong baseline.
    """
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1

    m = vit_b_16(weights=weights)
    m = _resize_vit_pos_embedding_(m, image_size=image_size, patch_size=16)

    if hasattr(m, "heads") and hasattr(m.heads, "head"):
        in_features = m.heads.head.in_features
        m.heads.head = torch.nn.Linear(in_features, num_classes)
    else:
        in_features = m.classifier.in_features
        m.classifier = torch.nn.Linear(in_features, num_classes)
    return m


def _unwrap_loaded_model(obj):
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], torch.nn.Module):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj  # caller can decide; we don't have the architecture here
    return None


def _maybe_load_models(device):
    path_a = "/kaggle/input/vit-v1/vit_v1.pt"
    path_b = "/kaggle/input/vit-boosted/vit_boosted.pt"
    path_head = "/kaggle/input/linear-head/linear_cls.pt"

    model_a = None
    model_b = None
    linear_head = None

    try:
        if os.path.exists(path_a):
            model_a = _unwrap_loaded_model(_safe_torch_load(path_a, map_location="cpu"))
            if isinstance(model_a, dict):
                model_a = None  # cannot reconstruct safely; fall back
        if os.path.exists(path_b):
            model_b = _unwrap_loaded_model(_safe_torch_load(path_b, map_location="cpu"))
            if isinstance(model_b, dict):
                model_b = None
        if os.path.exists(path_head):
            lh_obj = _safe_torch_load(path_head, map_location="cpu")
            linear_head = lh_obj if isinstance(lh_obj, torch.nn.Module) else None
    except Exception as e:
        print(
            f"Warning: failed to load provided .pt models, will use fallback. Error: {e}"
        )
        model_a = None
        model_b = None
        linear_head = None

    using_fallback_a = model_a is None
    using_fallback_b = model_b is None

    if model_a is None:
        model_a = _build_fallback_vit(num_classes, image_size=model_a_img_size)
    if model_b is None:
        model_b = _build_fallback_vit(num_classes, image_size=model_b_img_size)

    if linear_head is None:
        linear_head = torch.nn.Identity()

    model_a = model_a.to(device)
    model_b = model_b.to(device)
    linear_head = linear_head.to(device)
    return model_a, model_b, linear_head, using_fallback_a, using_fallback_b


model_a, model_b, linear_head, using_fallback_a, using_fallback_b = _maybe_load_models(
    device
)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        img_size=384,
        image_ids=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        if image_ids is not None:
            images = list(image_ids)
        else:
            images = sorted(os.listdir(data_dir))

        self.images = [
            fn for fn in images if os.path.isfile(os.path.join(data_dir, fn))
        ]

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

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    def __init__(self, data_dir, df, image_size, transform):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform
        self.cc = v2.CenterCrop((600, 600))
        self.resize = v2.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BICUBIC
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = row["image_id"]
        y = int(row["label"])
        img = Image.open(os.path.join(self.root, fn)).convert("RGB")
        img = self.cc(img)
        img = self.resize(img)
        img = self.transform(img) if self.transform is not None else img
        return img, y


def _set_trainable(m: torch.nn.Module, trainable: bool):
    for p in m.parameters():
        p.requires_grad = trainable


def _finetune_model(model: torch.nn.Module, image_size: int):
    if not finetune_fallback:
        return model

    train_df = pd.read_csv(train_csv_path)

    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = CassavaTrainDataset(
        train_dir, train_df, image_size=image_size, transform=train_transforms
    )
    loader = DataLoader(
        ds,
        batch_size=finetune_batch_size,
        shuffle=True,
        num_workers=finetune_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.train()

    _set_trainable(model, False)
    head_params = []
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        _set_trainable(model.heads.head, True)
        head_params = list(model.heads.head.parameters())
    elif hasattr(model, "classifier"):
        _set_trainable(model.classifier, True)
        head_params = list(model.classifier.parameters())
    else:
        _set_trainable(model, True)
        head_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(head_params, lr=finetune_lr)
    criterion = torch.nn.CrossEntropyLoss()

    for ep in range(finetune_epochs):
        running_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if not isinstance(out, torch.Tensor):
                if (
                    isinstance(out, (tuple, list))
                    and len(out) > 0
                    and isinstance(out[0], torch.Tensor)
                ):
                    out = out[0]
                elif isinstance(out, dict) and "logits" in out:
                    out = out["logits"]
                else:
                    raise TypeError(
                        f"Unsupported model output type during finetune: {type(out)}"
                    )
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(
            f"finetune epoch {ep+1}/{finetune_epochs} loss={running_loss/max(n,1):.4f}"
        )

    model.eval()
    return model


if using_fallback_a:
    model_a = _finetune_model(model_a, image_size=model_a_img_size)
if using_fallback_b:
    model_b = _finetune_model(model_b, image_size=model_b_img_size)



## === cell 3
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

sample = pd.read_csv(sample_sub_path)
test_image_ids = sample["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_ids=test_image_ids,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)


def _as_logits(model_out):
    if isinstance(model_out, torch.Tensor):
        return model_out
    if (
        isinstance(model_out, (tuple, list))
        and len(model_out) > 0
        and isinstance(model_out[0], torch.Tensor)
    ):
        return model_out[0]
    if isinstance(model_out, dict):
        for k in ("logits", "out", "output"):
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
    raise TypeError(f"Unsupported model output type: {type(model_out)}")


all_names = []
all_preds = []

model_a.eval()
model_b.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)

        if tta:
            model_a_inputs, model_b_inputs = (
                torch.cat(model_a_inputs, dim=0).to(device),
                torch.cat(model_b_inputs, dim=0).to(device),
            )
            filenames = list(filenames)

            model_a_outputs = _as_logits(model_a(model_a_inputs))
            model_b_outputs = _as_logits(model_b(model_b_inputs))

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bs), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bs), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = (0.95 * model_a_mean_logits + 0.05 * model_b_mean_logits) / 2
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = _as_logits(model_a(model_a_inputs))
            model_b_outputs = _as_logits(model_b(model_b_inputs))

            outputs = (0.97 * model_a_outputs + 0.03 * model_b_outputs) / 2
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print("Predicted:", len(all_preds), "files:", len(all_names))

pred_map = dict(zip(all_names, all_preds))
labels = [int(pred_map.get(img_id, 0)) for img_id in sample["image_id"].tolist()]
my_submission = pd.DataFrame({"image_id": sample["image_id"], "label": labels})

out_path = "submission.csv"
my_submission.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", my_submission.shape)
print(my_submission.head())
