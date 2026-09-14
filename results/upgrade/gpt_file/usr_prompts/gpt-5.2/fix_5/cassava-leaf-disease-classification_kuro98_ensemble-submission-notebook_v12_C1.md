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

0.900725294650952

# 6. Current score

0.76756

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external model files and replacing it with a lightweight, deterministic fallback that still produces valid class predictions. I also fix the submission-length issue by ensuring we iterate test images in the exact order of `sample_submission.csv` (no shuffling) and by enforcing that every `image_id` in the sample gets exactly one prediction. Finally, I make TTA deterministic and compatible with this fallback so the pipeline runs end-to-end and writes a correct `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.12294) has done: 'I fix the ViT input-size mismatch by aligning `vit_img_size` with the model’s expected `image_size` (224 for `vit_b_16`), which is the immediate runtime blocker. I also make the TTA batching robust by stacking TTA views per sample instead of relying on DataLoader’s default list collation, preventing shape/ordering issues and ensuring one prediction per test image. Finally, I keep the ensemble and submission ordering exactly as-is (using `sample_submission.csv` order) so the script runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.76756) has done: 'Your score is far below the target, and the main reason is that your current pipeline loads no trained cassava weights and falls back to ImageNet-pretrained models with randomly initialized 5-class heads, which yields near-random predictions. The smallest score-improving change that preserves your core inference logic is to add a lightweight training step on `train.csv` using the exact same two backbones you already use, then reuse the same ensemble + (optional) linear head at test time. To stay minimal and stable, I freeze both backbones and train only their classification heads (and keep the same transforms/TTA/inference code paths), which should move accuracy substantially toward the target without changing the architecture or loss. I also keep submission ordering locked to `sample_submission.csv` and write `submission.csv` exactly as required.'

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
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = f"{DATA_DIR}/train.csv"
train_dir = f"{DATA_DIR}/train_images/"
test_dir = f"{DATA_DIR}/test_images/"
sample_path = f"{DATA_DIR}/sample_submission.csv"

eff_img_size = 528
vit_img_size = 224

batch_size = 16
num_workers = 2
num_classes = 5
tta = True

train_epochs = 2
train_lr = 3e-3


def _safe_torch_load(path: str, map_location):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location=map_location)
    except Exception:
        pass
    return None


vit_model = _safe_torch_load(
    "/kaggle/input/vit-v1-update/vit_v1_1.pt", map_location=device
)
eff_model = _safe_torch_load(
    "/kaggle/input/efficient-net/vit_cont_3.pt", map_location=device
)
linear_head = _safe_torch_load(
    "/kaggle/input/linear-head/linear_cls.pt", map_location=device
)

if vit_model is None or eff_model is None:
    import torchvision

    vit_backbone = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    )
    in_features_vit = vit_backbone.heads.head.in_features
    vit_backbone.heads.head = torch.nn.Linear(in_features_vit, num_classes)

    eff_backbone = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features_eff = eff_backbone.classifier[-1].in_features
    eff_backbone.classifier[-1] = torch.nn.Linear(in_features_eff, num_classes)

    vit_model = vit_backbone
    eff_model = eff_backbone

if linear_head is None:
    linear_head = torch.nn.Identity()

vit_model = vit_model.to(device)
eff_model = eff_model.to(device)
linear_head = linear_head.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava data (supports train with labels and test without)."""

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        image_ids,
        labels=None,
        transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = list(image_ids)
        self.labels = None if labels is None else list(labels)
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")
        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.ttas is not None and self.transform is not None:
            vit_stack = torch.stack(
                [self.transform(t(vit_img)) for t in self.ttas], dim=0
            )
            eff_stack = torch.stack(
                [self.transform(t(eff_img)) for t in self.ttas], dim=0
            )
            if self.labels is None:
                return vit_stack, eff_stack, filename
            return vit_stack, eff_stack, int(self.labels[idx])
        elif self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        if self.labels is None:
            return vit_img, eff_img, filename
        return vit_img, eff_img, int(self.labels[idx])

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
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=(90, 90)),
        v2.RandomRotation(degrees=(270, 270)),
    ]
else:
    ttas = None

sample_df = pd.read_csv(sample_path)
test_image_ids = sample_df["image_id"].tolist()

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
need_train = (
    "vit-v1-update" not in str(vit_model.__class__).lower()
    and "efficient-net" not in str(eff_model.__class__).lower()
)

if (
    _safe_torch_load("/kaggle/input/vit-v1-update/vit_v1_1.pt", map_location="cpu")
    is None
    or _safe_torch_load("/kaggle/input/efficient-net/vit_cont_3.pt", map_location="cpu")
    is None
):
    need_train = True
else:
    need_train = False

if need_train:
    train_df = pd.read_csv(train_csv_path)

    train_dataset = CassavaDataset(
        train_dir,
        vit_img_size,
        eff_img_size,
        image_ids=train_df["image_id"].tolist(),
        labels=train_df["label"].tolist(),
        transform=test_transforms,  # keep same normalization pipeline
        ttas=None,  # no TTA in training to keep minimal and fast
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    for p in vit_model.parameters():
        p.requires_grad = False
    for p in eff_model.parameters():
        p.requires_grad = False

    vit_head_params = []
    eff_head_params = []

    if hasattr(vit_model, "heads") and hasattr(vit_model.heads, "head"):
        for p in vit_model.heads.head.parameters():
            p.requires_grad = True
        vit_head_params = list(vit_model.heads.head.parameters())

    if hasattr(eff_model, "classifier") and isinstance(
        eff_model.classifier, torch.nn.Sequential
    ):
        for p in eff_model.classifier[-1].parameters():
            p.requires_grad = True
        eff_head_params = list(eff_model.classifier[-1].parameters())

    params = vit_head_params + eff_head_params
    if len(params) == 0:
        raise RuntimeError("Could not find classification head parameters to train.")

    optimizer = torch.optim.Adam(params, lr=train_lr)
    criterion = torch.nn.CrossEntropyLoss()

    vit_model.train()
    eff_model.train()

    for epoch in range(train_epochs):
        running_loss = 0.0
        seen = 0
        for vit_inputs, eff_inputs, y in train_loader:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            vit_logits = vit_model(vit_inputs)
            eff_logits = eff_model(eff_inputs)
            logits = (vit_logits + eff_logits) / 2.0  # keep same ensemble core logic

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            seen += bs

        print(f"epoch {epoch+1}/{train_epochs} loss={running_loss/max(seen,1):.4f}")

    vit_model.eval()
    eff_model.eval()
    if not isinstance(linear_head, torch.nn.Identity):
        linear_head.eval()



## === cell 5
test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    image_ids=test_image_ids,
    labels=None,
    transform=test_transforms,
    ttas=ttas,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        cur_bs = len(filenames)

        if tta:
            vit_inputs = vit_inputs.flatten(0, 1).to(device, non_blocking=True)
            eff_inputs = eff_inputs.flatten(0, 1).to(device, non_blocking=True)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            n_tta = vit_outputs.shape[0] // cur_bs
            vit_mean_logits = vit_outputs.view(cur_bs, n_tta, -1).mean(dim=1)
            eff_mean_logits = eff_outputs.view(cur_bs, n_tta, -1).mean(dim=1)

            outputs = (vit_mean_logits + eff_mean_logits) / 2.0
            outputs = (
                linear_head(outputs)
                if not isinstance(linear_head, torch.nn.Identity)
                else outputs
            )

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            outputs = (vit_outputs + eff_outputs) / 2.0
            outputs = (
                linear_head(outputs)
                if not isinstance(linear_head, torch.nn.Identity)
                else outputs
            )

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

pred_map = dict(zip(all_names, all_preds))
final_preds = [int(pred_map[iid]) for iid in test_image_ids]

assert len(final_preds) == len(
    sample_df
), "Prediction length mismatch with sample_submission."

my_submission = pd.DataFrame({"image_id": test_image_ids, "label": final_preds})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with rows:", len(my_submission))
