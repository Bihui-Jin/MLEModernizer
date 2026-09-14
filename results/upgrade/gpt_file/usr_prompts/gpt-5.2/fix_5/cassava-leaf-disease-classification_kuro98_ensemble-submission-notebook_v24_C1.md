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

0.8919613176186159

# 6. Current score

0.15022

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10426) has done: 'You’re hitting a ViT input-size assertion because your fallback `vit_b_16` expects 224×224 images but the pipeline resizes to 384×384; we align the resize sizes to the actual loaded models’ expected `image_size` to fix this robustly for both external and fallback models. The “missing predictions” error is a downstream consequence of the crash in the prediction loop; once inference completes, the merge be fully populated. I also make the ensemble math consistent (currently you halve the logits after already using 0.8/0.2 weights), which is a minimal logic fix that should slightly improve accuracy without changing the core approach. Finally, I ensure `linear_head` (if present) is moved to the right device to avoid a future device-mismatch crash.'
- What this solution (achieved 0.23206) has done: 'Your current score (0.10426) is far below the target (0.89196), and the most likely cause is that you’re running untrained fallback ViTs (weights=None) because the external model files are not being found/loaded in this environment. I make the model loading robust by (1) searching common Kaggle input locations for the provided `.pt` files, and (2) if still missing, switching the fallback to a pretrained torchvision ViT (same architecture) and only replacing the classifier head to 5 classes—this preserves the core ViT approach while avoiding near-random predictions. I also ensure the inference output handling works whether the loaded `.pt` models return logits directly or return a dict/tuple, which prevents silently broken ensembling. These are minimal, execution-safe changes aimed directly at improving accuracy toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.15022) has done: 'Your score gap is large (0.23206 vs target 0.89196), and with this inference-only script the most likely remaining cause is that the “fallback” models are still effectively random because the classifier head is newly initialized. I keep the same ViT architecture and ensemble logic, but change the fallback construction to use an ImageNet-pretrained ViT and keep its pretrained head (1000 classes), then add a small linear adapter to 5 classes using the already-declared `linear_head` concept (so we don’t change the inference loop style). This makes predictions non-random without changing the overall approach (two ViTs + weighted logits + softmax + argmax) and keeps execution under the time limit. I also ensure the adapter is always on the right device and that logits extraction works consistently.'

# 9. Code solution

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

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_ROOT}/test_images/"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"

model_b_img_size = 384
model_a_img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _build_fallback_vit_with_adapter(num_classes: int):
    """
    Change (score-directed, minimal): avoid a randomly initialized 5-class head.
    Keep the same ViT backbone/architecture, but:
      - use pretrained ViT-B/16 (ImageNet),
      - keep its pretrained 1000-class head,
      - add a small linear adapter (1000 -> 5) for cassava labels.
    This preserves the core ViT inference approach while making fallback predictions non-random.
    """
    import torchvision

    backbone = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.DEFAULT
    )
    adapter = torch.nn.Linear(backbone.heads.head.out_features, num_classes)
    return backbone, adapter


def _infer_vit_image_size(model) -> int | None:
    if hasattr(model, "image_size"):
        try:
            return int(model.image_size)
        except Exception:
            pass
    for attr in ("img_size", "input_size", "image_resolution"):
        if hasattr(model, attr):
            try:
                v = getattr(model, attr)
                if isinstance(v, (tuple, list)):
                    return int(v[0])
                return int(v)
            except Exception:
                pass
    return None


def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _candidate_paths(rel_name: str):
    roots = [
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
    ]
    cands = []
    if rel_name.startswith("/"):
        cands.append(rel_name)
    for r in roots:
        cands.append(os.path.join(r, rel_name.lstrip("/")))
    dataset_names = ["vit-v1", "vit-boosted", "linear-head"]
    for dn in dataset_names:
        cands.append(f"/kaggle/input/{dn}/{os.path.basename(rel_name)}")
    return cands


vit_a_path = _find_first_existing(_candidate_paths("/kaggle/input/vit-v1/vit_v1.pt"))
vit_b_path = _find_first_existing(
    _candidate_paths("/kaggle/input/vit-boosted/vit_boosted.pt")
)
linear_head_path = _find_first_existing(
    _candidate_paths("/kaggle/input/linear-head/linear_cls.pt")
)

have_external_models = False
model_a = None
model_b = None
linear_head = None

try:
    if vit_a_path is not None and vit_b_path is not None:
        model_a = torch.load(vit_a_path, map_location=device)
        model_b = torch.load(vit_b_path, map_location=device)
        have_external_models = True
    if linear_head_path is not None:
        linear_head = torch.load(linear_head_path, map_location=device)
except Exception as e:
    print(
        "Warning: failed to load external checkpoints, falling back to pretrained torchvision ViT. Error:",
        repr(e),
    )
    have_external_models = False
    model_a = None
    model_b = None
    linear_head = None

if not have_external_models:
    model_a, linear_head_a = _build_fallback_vit_with_adapter(num_classes)
    model_b, linear_head_b = _build_fallback_vit_with_adapter(num_classes)

    linear_head = torch.nn.ModuleList([linear_head_a, linear_head_b])

    model_a = model_a.to(device)
    model_b = model_b.to(device)
    linear_head = linear_head.to(device)

model_a.to(device)
model_b.to(device)
if linear_head is not None:
    linear_head.to(device)

a_size = _infer_vit_image_size(model_a)
b_size = _infer_vit_image_size(model_b)
if a_size is not None:
    model_a_img_size = a_size
if b_size is not None:
    model_b_img_size = b_size

print(f"Using image sizes -> model_a: {model_a_img_size}, model_b: {model_b_img_size}")
print(f"External models loaded: {have_external_models}")
if have_external_models:
    print("Loaded:", vit_a_path, vit_b_path, linear_head_path)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test images.

    If `image_list` is provided, images are returned in that exact order to match sample_submission.csv.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        image_list=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.ttas = ttas

        if image_list is None:
            self.images = sorted(os.listdir(data_dir))
        else:
            self.images = list(image_list)

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

sample_sub = pd.read_csv(sample_sub_path)
test_image_list = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_list=test_image_list,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 4
def _to_logits(model_out):
    if isinstance(model_out, torch.Tensor):
        return model_out
    if isinstance(model_out, dict):
        for k in ("logits", "out", "output"):
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
        for v in model_out.values():
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        if isinstance(model_out[0], torch.Tensor):
            return model_out[0]
    raise TypeError(
        f"Unsupported model output type for logits extraction: {type(model_out)}"
    )


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

            model_a_outputs = _to_logits(model_a(model_a_inputs))
            model_b_outputs = _to_logits(model_b(model_b_inputs))

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bs), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bs), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = 0.8 * model_a_mean_logits + 0.2 * model_b_mean_logits

            if linear_head is not None:
                if (
                    isinstance(linear_head, torch.nn.ModuleList)
                    and len(linear_head) == 2
                ):
                    cassava_logits = 0.8 * linear_head[0](outputs) + 0.2 * linear_head[
                        1
                    ](outputs)
                else:
                    cassava_logits = linear_head(outputs)
                mean_preds = normalizer(cassava_logits)
            else:
                mean_preds = normalizer(outputs)

            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = _to_logits(model_a(model_a_inputs))
            model_b_outputs = _to_logits(model_b(model_b_inputs))

            outputs = 0.8 * model_a_outputs + 0.2 * model_b_outputs

            if linear_head is not None:
                if (
                    isinstance(linear_head, torch.nn.ModuleList)
                    and len(linear_head) == 2
                ):
                    cassava_logits = 0.8 * linear_head[0](outputs) + 0.2 * linear_head[
                        1
                    ](outputs)
                else:
                    cassava_logits = linear_head(outputs)
                preds = normalizer(cassava_logits)
            else:
                preds = normalizer(outputs)

            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

if len(all_names) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(all_names)} predictions, expected {len(sample_sub)}."
    )



## === cell 5
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})

my_submission = sample_sub[["image_id"]].merge(my_submission, on="image_id", how="left")
if my_submission["label"].isna().any():
    missing = my_submission[my_submission["label"].isna()]["image_id"].head(10).tolist()
    raise RuntimeError(f"Missing predictions for some images, e.g.: {missing}")

my_submission["label"] = my_submission["label"].astype(int)
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
my_submission.head()
