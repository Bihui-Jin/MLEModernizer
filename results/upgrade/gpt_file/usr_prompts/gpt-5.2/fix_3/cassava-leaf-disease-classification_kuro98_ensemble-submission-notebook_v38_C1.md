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

0.8987609549712904

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
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)
random.seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_DIR}/test_images/"
sample_path = f"{DATA_DIR}/sample_submission.csv"

model_a_img_size = 224  # was 384
model_b_img_size = 528
model_c_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True


def _load_model_or_fallback(
    model_path: str, arch: str, num_classes: int, device: torch.device
):
    """
    Bugfixes:
    - Handle checkpoints saved as full nn.Module OR state_dict (common on Kaggle).
    - Ensure loaded model is moved to device and set to eval().
    """
    p = Path(model_path)

    def _build_arch():
        if arch == "vit_b_16":
            m = torchvision.models.vit_b_16(
                weights=torchvision.models.ViT_B_16_Weights.DEFAULT
            )
            m.heads.head = torch.nn.Linear(m.heads.head.in_features, num_classes)
        elif arch == "efficientnet_b4":
            m = torchvision.models.efficientnet_b4(
                weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT
            )
            m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
        else:
            m = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            )
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m

    if p.exists():
        ckpt = torch.load(str(p), map_location="cpu")
        if isinstance(ckpt, torch.nn.Module):
            m = ckpt
        elif isinstance(ckpt, dict):
            state = None
            for k in ("state_dict", "model_state_dict", "model"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]
                    break
            if state is None:
                state = ckpt
            m = _build_arch()
            cleaned = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                cleaned[nk] = v
            m.load_state_dict(cleaned, strict=False)
        else:
            m = _build_arch()

        m.to(device).eval()
        return m

    m = _build_arch()
    m.to(device).eval()
    return m


model_a = _load_model_or_fallback(
    "/kaggle/input/vit-v1/vit_v1.pt", "vit_b_16", num_classes, device
)
model_b = _load_model_or_fallback(
    "/kaggle/input/efficient-net/efficient_net.pt",
    "efficientnet_b4",
    num_classes,
    device,
)
model_c = _load_model_or_fallback(
    "/kaggle/input/vit-v6/vit_v6.pt", "resnet50", num_classes, device
)

sample_df = pd.read_csv(sample_path)
sample_image_ids = sample_df["image_id"].tolist()
sample_set = set(sample_image_ids)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test images with optional deterministic TTA.

    Bugfixes:
    - Use sample_submission ordering to avoid mismatched submission length.
    - Ensure images are always RGB.
    - Deterministic TTA per-image via seeding in __getitem__.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        model_c_size,
        transform=None,
        ttas=None,
        image_ids=None,
        base_seed=3407,
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.base_seed = int(base_seed)

        if image_ids is None:
            self.images = sorted([p.name for p in Path(data_dir).glob("*.jpg")])
        else:
            self.images = list(image_ids)

        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_c = v2.Resize(
            (model_c_size, model_c_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)
        model_c_img = self.resize_model_c(img)

        if self.ttas is not None and self.transform is not None:
            model_a_list, model_b_list, model_c_list = [], [], []
            for j, t in enumerate(self.ttas):
                torch.manual_seed(self.base_seed + idx * 1000 + j)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(self.base_seed + idx * 1000 + j)

                a = self.transform(t(model_a_img))
                b = self.transform(t(model_b_img))
                c = self.transform(t(model_c_img))
                model_a_list.append(a)
                model_b_list.append(b)
                model_c_list.append(c)
            model_a_img, model_b_img, model_c_img = (
                model_a_list,
                model_b_list,
                model_c_list,
            )

        elif self.transform is not None:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)
            model_c_img = self.transform(model_c_img)

        return model_a_img, model_b_img, model_c_img, filename

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
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    model_c_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_ids=sample_image_ids,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
model_c.eval()

with torch.no_grad():
    for model_a_inputs, model_b_inputs, model_c_inputs, filenames in test_loader:
        bs = len(filenames)

        if tta:
            T = len(model_a_inputs[0])

            a = torch.stack([torch.stack(x, dim=0) for x in model_a_inputs], dim=1)
            b = torch.stack([torch.stack(x, dim=0) for x in model_b_inputs], dim=1)
            c = torch.stack([torch.stack(x, dim=0) for x in model_c_inputs], dim=1)

            a = a.flatten(0, 1).to(device, non_blocking=True)  # [T*bs, C, H, W]
            b = b.flatten(0, 1).to(device, non_blocking=True)
            c = c.flatten(0, 1).to(device, non_blocking=True)

            out_a = model_a(a)
            out_b = model_b(b)
            out_c = model_c(c)

            out_a = out_a.view(T, bs, -1).mean(dim=0)
            out_b = out_b.view(T, bs, -1).mean(dim=0)
            out_c = out_c.view(T, bs, -1).mean(dim=0)

            outputs = (out_a + out_b + out_c) / 3.0
        else:
            a = model_a_inputs.to(device, non_blocking=True)
            b = model_b_inputs.to(device, non_blocking=True)
            c = model_c_inputs.to(device, non_blocking=True)

            out_a = model_a(a)
            out_b = model_b(b)
            out_c = model_c(c)
            outputs = (out_a + out_b + out_c) / 3.0

        probs = normalizer(outputs)
        pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

pred_map = dict(zip(all_names, all_preds))
ordered_preds = [int(pred_map[iid]) for iid in sample_image_ids]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1851867557.py in <cell line: 0>()
     16             T = len(model_a_inputs[0])
     17 
---> 18             a = torch.stack([torch.stack(x, dim=0) for x in model_a_inputs], dim=1)
     19             b = torch.stack([torch.stack(x, dim=0) for x in model_b_inputs], dim=1)
     20             c = torch.stack([torch.stack(x, dim=0) for x in model_c_inputs], dim=1)

/tmp/ipykernel_56/1851867557.py in <listcomp>(.0)
     16             T = len(model_a_inputs[0])
     17 
---> 18             a = torch.stack([torch.stack(x, dim=0) for x in model_a_inputs], dim=1)
     19             b = torch.stack([torch.stack(x, dim=0) for x in model_b_inputs], dim=1)
     20             c = torch.stack([torch.stack(x, dim=0) for x in model_c_inputs], dim=1)

TypeError: stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Tensor

## === cell 4
my_submission = pd.DataFrame({"image_id": sample_image_ids, "label": ordered_preds})
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
my_submission.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4085623106.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({"image_id": sample_image_ids, "label": ordered_preds})
      2 my_submission.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", my_submission.shape)
      4 my_submission.head()

NameError: name 'ordered_preds' is not defined
