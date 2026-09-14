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

0.9008763977032336

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'I fix the ViT input-size assertion by automatically aligning `vit_img_size`/`eff_img_size` to the loaded models’ expected image sizes (224 for the torchvision fallback ViT), without changing the ensemble/training-free inference logic. I also make the TTA deterministic and collate-safe: `Random*` transforms be applied with fixed seeds per item/tta so the DataLoader can stack tensors reliably and results are reproducible. Finally, I ensure `pred_map` is always created (even if inference fails early) and that the submission strictly matches `sample_submission.csv` ordering/length and is written as `submission.csv`. These changes are correctness/stability fixes and should also improve score versus the broken run by enabling the intended inference/ensemble.'

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



## === cell 1
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


def _try_load_torch_model(path: str):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location=device)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
    return None


vit_model = _try_load_torch_model(VIT_PATH)
eff_model = _try_load_torch_model(EFF_PATH)
linear_head = _try_load_torch_model(HEAD_PATH)

if vit_model is None or eff_model is None:
    import torchvision

    print(
        "Pretrained .pt models not found; using torchvision pretrained backbones as a fallback."
    )

    vit_model = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    )
    vit_model.heads = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)

    eff_model = torchvision.models.efficientnet_b4(
        weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )
    eff_model.classifier[1] = torch.nn.Linear(
        eff_model.classifier[1].in_features, num_classes
    )

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




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test images.

    Bugfixes:
    - Make TTA deterministic and collate-safe by producing a fixed number of tensor views per sample.
    - Apply random transforms with controlled seeds so DataLoader can stack into [T, B, C, H, W].
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        transform=None,
        tta_transforms=None,
        tta_seed=3407,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(os.listdir(data_dir))

        self.tta_transforms = tta_transforms or []
        self.tta_seed = int(tta_seed)

        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def _apply_with_seed(self, t, img, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return t(img)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)

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




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    transform=test_transforms,
    tta_transforms=tta_transforms,
    tta_seed=3407,
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



## === cell 4
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

            outputs = 0.6 * vit_mean_logits + 0.4 * eff_mean_logits
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

            outputs = (vit_outputs + eff_outputs) / 2.0
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

pred_map = dict(zip(all_names, all_preds))
print(
    f"Predicted {len(pred_map)} unique test images out of {len(all_names)} total entries."
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/2172990362.py in <cell line: 0>()
     10 
     11 with torch.no_grad():
---> 12     for vit_inputs, eff_inputs, filenames in test_loader:
     13         filenames = list(filenames)
     14         base_bs = len(filenames)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1453                 data = self._task_info.pop(self._rcvd_idx)[1]
   1454                 self._rcvd_idx += 1
-> 1455                 return self._process_data(data)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

IsADirectoryError: Caught IsADirectoryError in DataLoader worker process 3.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/1094644441.py", line 43, in __getitem__
    img = Image.open(os.path.join(self.root, filename)).convert("RGB")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/cassava-leaf-disease-classification/test_images/test_images'


## === cell 5
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
