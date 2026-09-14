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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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

0.8513145965548504

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I remove the problematic TensorBoard import, make model loading robust when the checkpoint file is missing, filter the test image list to exclude directories, and adjust all model heads to output the five required classes. These fixes prevent import crashes, file‑not‑found errors, and directory‑reading issues, and they ensure predictions are in the correct label range, allowing the script to run end‑to‑end and generate a valid `submission.csv`.'
- What this solution (achieved 0.09865) has done: 'The fix adds all missing imports, defines the `root` path, and ensures the dataset, model, and utility functions have the required dependencies. With these corrections the script runs end‑to‑end, creates a proper `submission.csv`, and can now produce a valid Kaggle submission.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

root = os.getcwd()


def resolve_path(relative_path):
    """
    Return an existing absolute path for a given relative path.
    Checks the current working directory first, then the default Kaggle input directory.
    """
    candidates = [
        os.path.join(root, relative_path),
        os.path.join("/kaggle/input", relative_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Unable to locate path for {relative_path}. Tried: {candidates}"
    )


def get_transforms(aug: bool):
    """
    Return (train_transform, test_transform).
    If aug is True, include basic augmentations; otherwise only resize & normalize.
    All transforms output a torch tensor.
    """
    common = [
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2(),
    ]
    if aug:
        train_tf = A.Compose(
            [
                A.RandomResizedCrop((384, 384), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
                *common,
            ]
        )
    else:
        train_tf = A.Compose(common)

    test_tf = A.Compose(common)
    return train_tf, test_tf


def get_model(name: str, width_mult: float = 1.0, dropout: float = 0.0):
    """
    Build a MobileNetV3 Large model (pre‑trained) and adapt the classifier
    to output 5 classes. `width_mult` and `dropout` are passed to the torchvision
    constructor where applicable.
    """
    if width_mult != 1.0:
        net = models.mobilenet_v3_large(pretrained=False, width_mult=width_mult)
    else:
        try:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            net = models.mobilenet_v3_large(weights=weights, width_mult=width_mult)
        except AttributeError:
            net = models.mobilenet_v3_large(pretrained=True, width_mult=width_mult)

    in_features = net.classifier[3].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 5),
    )
    return net


def load_model(net, name: str, tag: str, save_dir: str, device):
    """
    Attempt to load a checkpoint; if not found or if loading fails (e.g., size mismatch),
    return the original net. Loading uses `strict=False` to ignore mismatched keys.
    """
    ckpt_path = os.path.join(save_dir, f"{name}_{tag}.pth")
    if os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(state["model_state_dict"], strict=False)
            print(f"Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(
                f"Failed to load checkpoint ({ckpt_path}) due to: {e}. Using fresh model."
            )
    else:
        print(f"No checkpoint found at {ckpt_path}; using fresh model.")
    return net


class CSVDataset(Dataset):
    """
    Generic CSV‑driven dataset. Returns a dict with an ``image`` tensor
    (and optionally a ``label``).
    """

    def __init__(
        self, annotations_df, img_dir, transform=None, target_transform=None, aug=True
    ):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed["image"]
        sample = {"image": image}
        return sample


class TrainDataset(CSVDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        label = self.img_labels.iloc[idx, 1]
        sample["label"] = int(label)
        return sample




## === cell 1
"""
Main execution: load data, model, and run inference
"""

args = {}
args["name"] = "mobilenet_384_randomcrop_width_mult_1.8"
args["batch_size"] = 32
args["width_mult"] = 1.8
args["dropout"] = 0.0
args["aug"] = True
args["model"] = "base"
args["gpu_id"] = 0

assert args["name"] is not None, "Must set experiment name before training"

data_dir = resolve_path("cassava-leaf-disease-classification")
save_dir = os.path.join(root, "pretrained1")
os.makedirs(save_dir, exist_ok=True)

train_csv_path = os.path.join(data_dir, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_img_dir = os.path.join(data_dir, "train_images")
train_transforms, _ = get_transforms(args["aug"])

train_dataset = TrainDataset(
    train_df, train_img_dir, transform=train_transforms, aug=args["aug"]
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

device = f'cuda:{args["gpu_id"]}' if torch.cuda.is_available() else "cpu"
print(f"Training samples: {len(train_dataset)} \t device: {device}")

net = get_model(
    args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
).to(device)

net = load_model(net, args["name"], "best", save_dir, device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

net.train()
for epoch in range(5):  # a few more epochs for better accuracy
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in train_loader:
        imgs = batch["image"].float().to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch [{epoch+1}/5]  Loss: {epoch_loss/total:.4f}  Acc: {correct/total:.4f}"
    )

net.eval()  # switch back to eval mode for inference

img_dir = os.path.join(data_dir, "test_images")
test_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
test_pd = pd.DataFrame({"image_id": test_filenames})
num_test = len(test_pd)

_, test_transforms = get_transforms(args["aug"])

test_dataset = CSVDataset(test_pd, img_dir, transform=test_transforms, aug=args["aug"])
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"test images: {num_test} \t device: {device}")

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000 and magnitude < 5:
        magnitude += 1
        num /= 1000.0
    return f'{num:.2f}{" KMGTP"[magnitude]}'


print(f"Number of total parameters: {human_format(num_params)}")

pred_list = []
with torch.no_grad():
    for data in test_dataloader:
        imgs = data["image"].float().to(device)
        outputs = net(imgs)
        pred_list.extend(outputs.argmax(dim=1).cpu().tolist())

test_pd["label"] = pred_list
print(test_pd.head())
test_pd.to_csv(os.path.join(root, "submission.csv"), index=False)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3900384050.py in <cell line: 0>()
     57 
     58         optimizer.zero_grad()
---> 59         outputs = net(imgs)
     60         loss = criterion(outputs, labels)
     61         loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/mobilenetv3.py in forward(self, x)
    218 
    219     def forward(self, x: Tensor) -> Tensor:
--> 220         return self._forward_impl(x)
    221 
    222 

/usr/local/lib/python3.11/dist-packages/torchvision/models/mobilenetv3.py in _forward_impl(self, x)
    213         x = torch.flatten(x, 1)
    214 
--> 215         x = self.classifier(x)
    216 
    217         return x

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1728 and 2304x5)
