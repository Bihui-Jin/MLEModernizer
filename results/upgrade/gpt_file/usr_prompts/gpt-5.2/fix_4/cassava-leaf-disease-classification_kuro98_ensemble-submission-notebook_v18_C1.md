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

0.8694469628286491

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it runs end-to-end and always writes a valid `submission.csv` with the exact rows/order expected by `sample_submission.csv`. The immediate blocker is missing model files under `/kaggle/input/vit-v1/...`, so I add a safe fallback that loads a standard torchvision ViT if those assets aren’t present, keeping the same inference flow and TTA logic. I also fix DataLoader shuffling (must be `False`) and align predictions to `sample_submission.csv` to prevent length/order mismatches. Finally, I make the dataset/TTA path deterministic and robust (RGB conversion, sorted filenames, correct TTA stacking) so it doesn’t crash or produce invalid outputs.'
- What this solution (achieved 0.58744) has done: 'I fix the ViT input-size assertion by making the fallback ViT models use an `image_size` that matches your configured 384px preprocessing, so inference runs without crashing. This is a bug fix only (no architecture/training loop changes) and preserves your current ensemble + TTA flow. I also make the TTA transforms deterministic at inference by switching to functional flips/rotations/perspective with fixed parameters; this avoids randomness that can destabilize accuracy and typically improves it for a fixed model. Finally, I keep the submission alignment merge against `sample_submission.csv` exactly as you already do, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so the safest way to move it upward (without changing your ensemble/TTA core logic) is to ensure you are not accidentally running with random, untrained ViT weights. I keep your exact inference flow, but change the fallback model creation to use ImageNet-pretrained ViT weights (same architecture, same input size), and ensure the linear head is actually applied (it’s currently loaded but unused). These two fixes typically yield a large accuracy jump while preserving your model+TTA approach and submission semantics. I also keep the sample_submission alignment exactly as you already do so the output CSV remains valid.'

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

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True


def _safe_load_or_build_vit(checkpoint_path: str, img_size: int, num_classes: int):
    state_dict = None
    if os.path.exists(checkpoint_path):
        obj = torch.load(checkpoint_path, map_location=device)
        if isinstance(obj, torch.nn.Module):
            return obj
        state_dict = obj

    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if state_dict is None else None
    model = vit_b_16(weights=weights, image_size=img_size)
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, num_classes)

    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        _ = (missing, unexpected)
    return model


model_a_path = "/kaggle/input/vit-v1/vit_v1.pt"
model_b_path = "/kaggle/input/vit-boosted/vit_boosted.pt"
linear_head_path = "/kaggle/input/linear-head/linear_cls.pt"

model_a = _safe_load_or_build_vit(model_a_path, model_a_img_size, num_classes).to(
    device
)
model_b = _safe_load_or_build_vit(model_b_path, model_b_img_size, num_classes).to(
    device
)

if os.path.exists(linear_head_path):
    linear_head = torch.load(linear_head_path, map_location=device)
    if isinstance(linear_head, torch.nn.Module):
        linear_head = linear_head.to(device)
    else:
        linear_head = torch.nn.Identity().to(device)
else:
    linear_head = torch.nn.Identity().to(device)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3070858402.py in <cell line: 0>()
     57 linear_head_path = "/kaggle/input/linear-head/linear_cls.pt"
     58 
---> 59 model_a = _safe_load_or_build_vit(model_a_path, model_a_img_size, num_classes).to(
     60     device
     61 )

/tmp/ipykernel_55/3070858402.py in _safe_load_or_build_vit(checkpoint_path, img_size, num_classes)
     43 
     44     weights = ViT_B_16_Weights.IMAGENET1K_V1 if state_dict is None else None
---> 45     model = vit_b_16(weights=weights, image_size=img_size)
     46     in_features = model.heads.head.in_features
     47     model.heads.head = torch.nn.Linear(in_features, num_classes)

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in wrapper(*args, **kwargs)
    140             kwargs.update(keyword_only_kwargs)
    141 
--> 142         return fn(*args, **kwargs)
    143 
    144     return wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in inner_wrapper(*args, **kwargs)
    226                 kwargs[weights_param] = default_weights_arg
    227 
--> 228             return builder(*args, **kwargs)
    229 
    230         return inner_wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in vit_b_16(weights, progress, **kwargs)
    639     weights = ViT_B_16_Weights.verify(weights)
    640 
--> 641     return _vision_transformer(
    642         patch_size=16,
    643         num_layers=12,

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _vision_transformer(patch_size, num_layers, num_heads, hidden_dim, mlp_dim, weights, progress, **kwargs)
    319         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    320         assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
--> 321         _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    322     image_size = kwargs.pop("image_size", 224)
    323 

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in _ovewrite_named_param(kwargs, param, new_value)
    236     if param in kwargs:
    237         if kwargs[param] != new_value:
--> 238             raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    239     else:
    240         kwargs[param] = new_value

ValueError: The parameter 'image_size' expected value 224 but got 384 instead.

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
        img_size=384,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )
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
        lambda x: x,
        lambda x: v2.functional.horizontal_flip(x),
        lambda x: v2.functional.vertical_flip(x),
        lambda x: v2.functional.rotate(
            x, angle=90, interpolation=InterpolationMode.BILINEAR
        ),
        lambda x: v2.functional.perspective(
            x,
            startpoints=[
                (0, 0),
                (x.size[0] - 1, 0),
                (x.size[0] - 1, x.size[1] - 1),
                (0, x.size[1] - 1),
            ],
            endpoints=[
                (10, 10),
                (x.size[0] - 11, 5),
                (x.size[0] - 5, x.size[1] - 11),
                (5, x.size[1] - 5),
            ],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, model_a_img_size, model_b_img_size, transform=test_transforms, ttas=ttas
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
linear_head.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)

        if tta:
            model_a_inputs = torch.cat(list(model_a_inputs), dim=0).to(
                device, non_blocking=True
            )
            model_b_inputs = torch.cat(list(model_b_inputs), dim=0).to(
                device, non_blocking=True
            )
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            num_tta = len(ttas)
            model_a_batch_logits = model_a_outputs.view(num_tta, bs, -1).mean(dim=0)
            model_b_batch_logits = model_b_outputs.view(num_tta, bs, -1).mean(dim=0)

            outputs = (model_a_batch_logits + model_b_batch_logits) / 2.0

            outputs = linear_head(outputs)

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)
            outputs = (model_a_outputs + model_b_outputs) / 2.0

            outputs = linear_head(outputs)

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3720974768.py in <cell line: 0>()
      2 all_preds = []
      3 
----> 4 model_a.eval()
      5 model_b.eval()
      6 linear_head.eval()

NameError: name 'model_a' is not defined

## === cell 4
sample_sub = pd.read_csv(sample_path)
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if merged["label"].isna().any():
    merged["label"] = merged["label"].fillna(0).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

merged.to_csv("submission.csv", index=False)
merged
