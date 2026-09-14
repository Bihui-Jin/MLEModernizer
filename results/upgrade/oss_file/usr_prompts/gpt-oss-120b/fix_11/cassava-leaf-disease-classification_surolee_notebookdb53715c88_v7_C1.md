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
     38 print(f"Training samples: {len(train_dataset)} \t device: {device}")
     39 
---> 40 net = get_model(
     41     args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
     42 ).to(device)

/tmp/ipykernel_55/3715352040.py in get_model(name, width_mult, dropout)
     67         # torchvision 0.13+ uses `weights` instead of `pretrained`
     68         weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
---> 69         net = models.mobilenet_v3_large(weights=weights, width_mult=width_mult)
     70     except AttributeError:
     71         # Older torchvision versions

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/mobilenetv3.py in mobilenet_v3_large(weights, progress, **kwargs)
    390 
    391     inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
--> 392     return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)
    393 
    394 

/usr/local/lib/python3.11/dist-packages/torchvision/models/mobilenetv3.py in _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)
    283 
    284     if weights is not None:
--> 285         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    286 
    287     return model

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in load_state_dict(self, state_dict, strict, assign)
   2579 
   2580         if len(error_msgs) > 0:
-> 2581             raise RuntimeError(
   2582                 "Error(s) in loading state_dict for {}:\n\t{}".format(
   2583                     self.__class__.__name__, "\n\t".join(error_msgs)

RuntimeError: Error(s) in loading state_dict for MobileNetV3:
	size mismatch for features.0.0.weight: copying a param with shape torch.Size([16, 3, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 3, 3, 3]).
	size mismatch for features.0.1.weight: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.0.1.bias: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.0.1.running_mean: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.0.1.running_var: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.1.block.0.0.weight: copying a param with shape torch.Size([16, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([32, 1, 3, 3]).
	size mismatch for features.1.block.0.1.weight: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.1.block.0.1.bias: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.1.block.0.1.running_mean: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.1.block.0.1.running_var: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.1.block.1.0.weight: copying a param with shape torch.Size([16, 16, 1, 1]) from checkpoint, the shape in current model is torch.Size([32, 32, 1, 1]).
	size mismatch for features.1.block.1.1.weight: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.1.block.1.1.bias: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.1.block.1.1.running_mean: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.1.block.1.1.running_var: copying a param with shape torch.Size([16]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.2.block.0.0.weight: copying a param with shape torch.Size([64, 16, 1, 1]) from checkpoint, the shape in current model is torch.Size([112, 32, 1, 1]).
	size mismatch for features.2.block.0.1.weight: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([112]).
	size mismatch for features.2.block.0.1.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([112]).
	size mismatch for features.2.block.0.1.running_mean: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([112]).
	size mismatch for features.2.block.0.1.running_var: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([112]).
	size mismatch for features.2.block.1.0.weight: copying a param with shape torch.Size([64, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([112, 1, 3, 3]).
	size mismatch for features.2.block.1.1.weight: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([112]).
	size mismatch for features.2.block.1.1.bias: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([112]).
	size mismatch for features.2.block.1.1.running_mean: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([112]).
	size mismatch for features.2.block.1.1.running_var: copying a param with shape torch.Size([64]) from checkpoint, the shape in current model is torch.Size([112]).
	size mismatch for features.2.block.2.0.weight: copying a param with shape torch.Size([24, 64, 1, 1]) from checkpoint, the shape in current model is torch.Size([40, 112, 1, 1]).
	size mismatch for features.2.block.2.1.weight: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([40]).
	size mismatch for features.2.block.2.1.bias: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([40]).
	size mismatch for features.2.block.2.1.running_mean: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([40]).
	size mismatch for features.2.block.2.1.running_var: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([40]).
	size mismatch for features.3.block.0.0.weight: copying a param with shape torch.Size([72, 24, 1, 1]) from checkpoint, the shape in current model is torch.Size([128, 40, 1, 1]).
	size mismatch for features.3.block.0.1.weight: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.3.block.0.1.bias: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.3.block.0.1.running_mean: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.3.block.0.1.running_var: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.3.block.1.0.weight: copying a param with shape torch.Size([72, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([128, 1, 3, 3]).
	size mismatch for features.3.block.1.1.weight: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.3.block.1.1.bias: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.3.block.1.1.running_mean: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.3.block.1.1.running_var: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.3.block.2.0.weight: copying a param with shape torch.Size([24, 72, 1, 1]) from checkpoint, the shape in current model is torch.Size([40, 128, 1, 1]).
	size mismatch for features.3.block.2.1.weight: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([40]).
	size mismatch for features.3.block.2.1.bias: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([40]).
	size mismatch for features.3.block.2.1.running_mean: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([40]).
	size mismatch for features.3.block.2.1.running_var: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([40]).
	size mismatch for features.4.block.0.0.weight: copying a param with shape torch.Size([72, 24, 1, 1]) from checkpoint, the shape in current model is torch.Size([128, 40, 1, 1]).
	size mismatch for features.4.block.0.1.weight: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.0.1.bias: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.0.1.running_mean: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.0.1.running_var: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.1.0.weight: copying a param with shape torch.Size([72, 1, 5, 5]) from checkpoint, the shape in current model is torch.Size([128, 1, 5, 5]).
	size mismatch for features.4.block.1.1.weight: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.1.1.bias: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.1.1.running_mean: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.1.1.running_var: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.2.fc1.weight: copying a param with shape torch.Size([24, 72, 1, 1]) from checkpoint, the shape in current model is torch.Size([32, 128, 1, 1]).
	size mismatch for features.4.block.2.fc1.bias: copying a param with shape torch.Size([24]) from checkpoint, the shape in current model is torch.Size([32]).
	size mismatch for features.4.block.2.fc2.weight: copying a param with shape torch.Size([72, 24, 1, 1]) from checkpoint, the shape in current model is torch.Size([128, 32, 1, 1]).
	size mismatch for features.4.block.2.fc2.bias: copying a param with shape torch.Size([72]) from checkpoint, the shape in current model is torch.Size([128]).
	size mismatch for features.4.block.3.0.weight: copying a param with shape torch.Size([40, 72, 1, 1]) from checkpoint, the shape in current model is torch.Size([72, 128, 1, 1]).
	size mismatch for features.4.block.3.1.weight: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.4.block.3.1.bias: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.4.block.3.1.running_mean: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.4.block.3.1.running_var: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.5.block.0.0.weight: copying a param with shape torch.Size([120, 40, 1, 1]) from checkpoint, the shape in current model is torch.Size([216, 72, 1, 1]).
	size mismatch for features.5.block.0.1.weight: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.0.1.bias: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.0.1.running_mean: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.0.1.running_var: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.1.0.weight: copying a param with shape torch.Size([120, 1, 5, 5]) from checkpoint, the shape in current model is torch.Size([216, 1, 5, 5]).
	size mismatch for features.5.block.1.1.weight: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.1.1.bias: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.1.1.running_mean: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.1.1.running_var: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.2.fc1.weight: copying a param with shape torch.Size([32, 120, 1, 1]) from checkpoint, the shape in current model is torch.Size([56, 216, 1, 1]).
	size mismatch for features.5.block.2.fc1.bias: copying a param with shape torch.Size([32]) from checkpoint, the shape in current model is torch.Size([56]).
	size mismatch for features.5.block.2.fc2.weight: copying a param with shape torch.Size([120, 32, 1, 1]) from checkpoint, the shape in current model is torch.Size([216, 56, 1, 1]).
	size mismatch for features.5.block.2.fc2.bias: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.5.block.3.0.weight: copying a param with shape torch.Size([40, 120, 1, 1]) from checkpoint, the shape in current model is torch.Size([72, 216, 1, 1]).
	size mismatch for features.5.block.3.1.weight: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.5.block.3.1.bias: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.5.block.3.1.running_mean: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.5.block.3.1.running_var: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.6.block.0.0.weight: copying a param with shape torch.Size([120, 40, 1, 1]) from checkpoint, the shape in current model is torch.Size([216, 72, 1, 1]).
	size mismatch for features.6.block.0.1.weight: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.0.1.bias: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.0.1.running_mean: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.0.1.running_var: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.1.0.weight: copying a param with shape torch.Size([120, 1, 5, 5]) from checkpoint, the shape in current model is torch.Size([216, 1, 5, 5]).
	size mismatch for features.6.block.1.1.weight: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.1.1.bias: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.1.1.running_mean: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.1.1.running_var: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.2.fc1.weight: copying a param with shape torch.Size([32, 120, 1, 1]) from checkpoint, the shape in current model is torch.Size([56, 216, 1, 1]).
	size mismatch for features.6.block.2.fc1.bias: copying a param with shape torch.Size([32]) from checkpoint, the shape in current model is torch.Size([56]).
	size mismatch for features.6.block.2.fc2.weight: copying a param with shape torch.Size([120, 32, 1, 1]) from checkpoint, the shape in current model is torch.Size([216, 56, 1, 1]).
	size mismatch for features.6.block.2.fc2.bias: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.6.block.3.0.weight: copying a param with shape torch.Size([40, 120, 1, 1]) from checkpoint, the shape in current model is torch.Size([72, 216, 1, 1]).
	size mismatch for features.6.block.3.1.weight: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.6.block.3.1.bias: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.6.block.3.1.running_mean: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.6.block.3.1.running_var: copying a param with shape torch.Size([40]) from checkpoint, the shape in current model is torch.Size([72]).
	size mismatch for features.7.block.0.0.weight: copying a param with shape torch.Size([240, 40, 1, 1]) from checkpoint, the shape in current model is torch.Size([432, 72, 1, 1]).
	size mismatch for features.7.block.0.1.weight: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.7.block.0.1.bias: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.7.block.0.1.running_mean: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.7.block.0.1.running_var: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.7.block.1.0.weight: copying a param with shape torch.Size([240, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([432, 1, 3, 3]).
	size mismatch for features.7.block.1.1.weight: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.7.block.1.1.bias: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.7.block.1.1.running_mean: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.7.block.1.1.running_var: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.7.block.2.0.weight: copying a param with shape torch.Size([80, 240, 1, 1]) from checkpoint, the shape in current model is torch.Size([144, 432, 1, 1]).
	size mismatch for features.7.block.2.1.weight: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.7.block.2.1.bias: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.7.block.2.1.running_mean: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.7.block.2.1.running_var: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.8.block.0.0.weight: copying a param with shape torch.Size([200, 80, 1, 1]) from checkpoint, the shape in current model is torch.Size([360, 144, 1, 1]).
	size mismatch for features.8.block.0.1.weight: copying a param with shape torch.Size([200]) from checkpoint, the shape in current model is torch.Size([360]).
	size mismatch for features.8.block.0.1.bias: copying a param with shape torch.Size([200]) from checkpoint, the shape in current model is torch.Size([360]).
	size mismatch for features.8.block.0.1.running_mean: copying a param with shape torch.Size([200]) from checkpoint, the shape in current model is torch.Size([360]).
	size mismatch for features.8.block.0.1.running_var: copying a param with shape torch.Size([200]) from checkpoint, the shape in current model is torch.Size([360]).
	size mismatch for features.8.block.1.0.weight: copying a param with shape torch.Size([200, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([360, 1, 3, 3]).
	size mismatch for features.8.block.1.1.weight: copying a param with shape torch.Size([200]) from checkpoint, the shape in current model is torch.Size([360]).
	size mismatch for features.8.block.1.1.bias: copying a param with shape torch.Size([200]) from checkpoint, the shape in current model is torch.Size([360]).
	size mismatch for features.8.block.1.1.running_mean: copying a param with shape torch.Size([200]) from checkpoint, the shape in current model is torch.Size([360]).
	size mismatch for features.8.block.1.1.running_var: copying a param with shape torch.Size([200]) from checkpoint, the shape in current model is torch.Size([360]).
	size mismatch for features.8.block.2.0.weight: copying a param with shape torch.Size([80, 200, 1, 1]) from checkpoint, the shape in current model is torch.Size([144, 360, 1, 1]).
	size mismatch for features.8.block.2.1.weight: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.8.block.2.1.bias: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.8.block.2.1.running_mean: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.8.block.2.1.running_var: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.9.block.0.0.weight: copying a param with shape torch.Size([184, 80, 1, 1]) from checkpoint, the shape in current model is torch.Size([328, 144, 1, 1]).
	size mismatch for features.9.block.0.1.weight: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.9.block.0.1.bias: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.9.block.0.1.running_mean: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.9.block.0.1.running_var: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.9.block.1.0.weight: copying a param with shape torch.Size([184, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([328, 1, 3, 3]).
	size mismatch for features.9.block.1.1.weight: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.9.block.1.1.bias: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.9.block.1.1.running_mean: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.9.block.1.1.running_var: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.9.block.2.0.weight: copying a param with shape torch.Size([80, 184, 1, 1]) from checkpoint, the shape in current model is torch.Size([144, 328, 1, 1]).
	size mismatch for features.9.block.2.1.weight: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.9.block.2.1.bias: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.9.block.2.1.running_mean: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.9.block.2.1.running_var: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.10.block.0.0.weight: copying a param with shape torch.Size([184, 80, 1, 1]) from checkpoint, the shape in current model is torch.Size([328, 144, 1, 1]).
	size mismatch for features.10.block.0.1.weight: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.10.block.0.1.bias: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.10.block.0.1.running_mean: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.10.block.0.1.running_var: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.10.block.1.0.weight: copying a param with shape torch.Size([184, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([328, 1, 3, 3]).
	size mismatch for features.10.block.1.1.weight: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.10.block.1.1.bias: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.10.block.1.1.running_mean: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.10.block.1.1.running_var: copying a param with shape torch.Size([184]) from checkpoint, the shape in current model is torch.Size([328]).
	size mismatch for features.10.block.2.0.weight: copying a param with shape torch.Size([80, 184, 1, 1]) from checkpoint, the shape in current model is torch.Size([144, 328, 1, 1]).
	size mismatch for features.10.block.2.1.weight: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.10.block.2.1.bias: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.10.block.2.1.running_mean: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.10.block.2.1.running_var: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([144]).
	size mismatch for features.11.block.0.0.weight: copying a param with shape torch.Size([480, 80, 1, 1]) from checkpoint, the shape in current model is torch.Size([864, 144, 1, 1]).
	size mismatch for features.11.block.0.1.weight: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.0.1.bias: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.0.1.running_mean: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.0.1.running_var: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.1.0.weight: copying a param with shape torch.Size([480, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([864, 1, 3, 3]).
	size mismatch for features.11.block.1.1.weight: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.1.1.bias: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.1.1.running_mean: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.1.1.running_var: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.2.fc1.weight: copying a param with shape torch.Size([120, 480, 1, 1]) from checkpoint, the shape in current model is torch.Size([216, 864, 1, 1]).
	size mismatch for features.11.block.2.fc1.bias: copying a param with shape torch.Size([120]) from checkpoint, the shape in current model is torch.Size([216]).
	size mismatch for features.11.block.2.fc2.weight: copying a param with shape torch.Size([480, 120, 1, 1]) from checkpoint, the shape in current model is torch.Size([864, 216, 1, 1]).
	size mismatch for features.11.block.2.fc2.bias: copying a param with shape torch.Size([480]) from checkpoint, the shape in current model is torch.Size([864]).
	size mismatch for features.11.block.3.0.weight: copying a param with shape torch.Size([112, 480, 1, 1]) from checkpoint, the shape in current model is torch.Size([200, 864, 1, 1]).
	size mismatch for features.11.block.3.1.weight: copying a param with shape torch.Size([112]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for features.11.block.3.1.bias: copying a param with shape torch.Size([112]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for features.11.block.3.1.running_mean: copying a param with shape torch.Size([112]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for features.11.block.3.1.running_var: copying a param with shape torch.Size([112]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for features.12.block.0.0.weight: copying a param with shape torch.Size([672, 112, 1, 1]) from checkpoint, the shape in current model is torch.Size([1208, 200, 1, 1]).
	size mismatch for features.12.block.0.1.weight: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.0.1.bias: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.0.1.running_mean: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.0.1.running_var: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.1.0.weight: copying a param with shape torch.Size([672, 1, 3, 3]) from checkpoint, the shape in current model is torch.Size([1208, 1, 3, 3]).
	size mismatch for features.12.block.1.1.weight: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.1.1.bias: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.1.1.running_mean: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.1.1.running_var: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.2.fc1.weight: copying a param with shape torch.Size([168, 672, 1, 1]) from checkpoint, the shape in current model is torch.Size([304, 1208, 1, 1]).
	size mismatch for features.12.block.2.fc1.bias: copying a param with shape torch.Size([168]) from checkpoint, the shape in current model is torch.Size([304]).
	size mismatch for features.12.block.2.fc2.weight: copying a param with shape torch.Size([672, 168, 1, 1]) from checkpoint, the shape in current model is torch.Size([1208, 304, 1, 1]).
	size mismatch for features.12.block.2.fc2.bias: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.12.block.3.0.weight: copying a param with shape torch.Size([112, 672, 1, 1]) from checkpoint, the shape in current model is torch.Size([200, 1208, 1, 1]).
	size mismatch for features.12.block.3.1.weight: copying a param with shape torch.Size([112]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for features.12.block.3.1.bias: copying a param with shape torch.Size([112]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for features.12.block.3.1.running_mean: copying a param with shape torch.Size([112]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for features.12.block.3.1.running_var: copying a param with shape torch.Size([112]) from checkpoint, the shape in current model is torch.Size([200]).
	size mismatch for features.13.block.0.0.weight: copying a param with shape torch.Size([672, 112, 1, 1]) from checkpoint, the shape in current model is torch.Size([1208, 200, 1, 1]).
	size mismatch for features.13.block.0.1.weight: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.0.1.bias: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.0.1.running_mean: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.0.1.running_var: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.1.0.weight: copying a param with shape torch.Size([672, 1, 5, 5]) from checkpoint, the shape in current model is torch.Size([1208, 1, 5, 5]).
	size mismatch for features.13.block.1.1.weight: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.1.1.bias: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.1.1.running_mean: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.1.1.running_var: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.2.fc1.weight: copying a param with shape torch.Size([168, 672, 1, 1]) from checkpoint, the shape in current model is torch.Size([304, 1208, 1, 1]).
	size mismatch for features.13.block.2.fc1.bias: copying a param with shape torch.Size([168]) from checkpoint, the shape in current model is torch.Size([304]).
	size mismatch for features.13.block.2.fc2.weight: copying a param with shape torch.Size([672, 168, 1, 1]) from checkpoint, the shape in current model is torch.Size([1208, 304, 1, 1]).
	size mismatch for features.13.block.2.fc2.bias: copying a param with shape torch.Size([672]) from checkpoint, the shape in current model is torch.Size([1208]).
	size mismatch for features.13.block.3.0.weight: copying a param with shape torch.Size([160, 672, 1, 1]) from checkpoint, the shape in current model is torch.Size([288, 1208, 1, 1]).
	size mismatch for features.13.block.3.1.weight: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.13.block.3.1.bias: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.13.block.3.1.running_mean: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.13.block.3.1.running_var: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.14.block.0.0.weight: copying a param with shape torch.Size([960, 160, 1, 1]) from checkpoint, the shape in current model is torch.Size([1728, 288, 1, 1]).
	size mismatch for features.14.block.0.1.weight: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.0.1.bias: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.0.1.running_mean: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.0.1.running_var: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.1.0.weight: copying a param with shape torch.Size([960, 1, 5, 5]) from checkpoint, the shape in current model is torch.Size([1728, 1, 5, 5]).
	size mismatch for features.14.block.1.1.weight: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.1.1.bias: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.1.1.running_mean: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.1.1.running_var: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.2.fc1.weight: copying a param with shape torch.Size([240, 960, 1, 1]) from checkpoint, the shape in current model is torch.Size([432, 1728, 1, 1]).
	size mismatch for features.14.block.2.fc1.bias: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.14.block.2.fc2.weight: copying a param with shape torch.Size([960, 240, 1, 1]) from checkpoint, the shape in current model is torch.Size([1728, 432, 1, 1]).
	size mismatch for features.14.block.2.fc2.bias: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.14.block.3.0.weight: copying a param with shape torch.Size([160, 960, 1, 1]) from checkpoint, the shape in current model is torch.Size([288, 1728, 1, 1]).
	size mismatch for features.14.block.3.1.weight: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.14.block.3.1.bias: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.14.block.3.1.running_mean: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.14.block.3.1.running_var: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.15.block.0.0.weight: copying a param with shape torch.Size([960, 160, 1, 1]) from checkpoint, the shape in current model is torch.Size([1728, 288, 1, 1]).
	size mismatch for features.15.block.0.1.weight: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.0.1.bias: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.0.1.running_mean: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.0.1.running_var: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.1.0.weight: copying a param with shape torch.Size([960, 1, 5, 5]) from checkpoint, the shape in current model is torch.Size([1728, 1, 5, 5]).
	size mismatch for features.15.block.1.1.weight: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.1.1.bias: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.1.1.running_mean: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.1.1.running_var: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.2.fc1.weight: copying a param with shape torch.Size([240, 960, 1, 1]) from checkpoint, the shape in current model is torch.Size([432, 1728, 1, 1]).
	size mismatch for features.15.block.2.fc1.bias: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([432]).
	size mismatch for features.15.block.2.fc2.weight: copying a param with shape torch.Size([960, 240, 1, 1]) from checkpoint, the shape in current model is torch.Size([1728, 432, 1, 1]).
	size mismatch for features.15.block.2.fc2.bias: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.15.block.3.0.weight: copying a param with shape torch.Size([160, 960, 1, 1]) from checkpoint, the shape in current model is torch.Size([288, 1728, 1, 1]).
	size mismatch for features.15.block.3.1.weight: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.15.block.3.1.bias: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.15.block.3.1.running_mean: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.15.block.3.1.running_var: copying a param with shape torch.Size([160]) from checkpoint, the shape in current model is torch.Size([288]).
	size mismatch for features.16.0.weight: copying a param with shape torch.Size([960, 160, 1, 1]) from checkpoint, the shape in current model is torch.Size([1728, 288, 1, 1]).
	size mismatch for features.16.1.weight: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.16.1.bias: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.16.1.running_mean: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for features.16.1.running_var: copying a param with shape torch.Size([960]) from checkpoint, the shape in current model is torch.Size([1728]).
	size mismatch for classifier.0.weight: copying a param with shape torch.Size([1280, 960]) from checkpoint, the shape in current model is torch.Size([2304, 1728]).
	size mismatch for classifier.0.bias: copying a param with shape torch.Size([1280]) from checkpoint, the shape in current model is torch.Size([2304]).
	size mismatch for classifier.3.weight: copying a param with shape torch.Size([1000, 1280]) from checkpoint, the shape in current model is torch.Size([1000, 2304]).
