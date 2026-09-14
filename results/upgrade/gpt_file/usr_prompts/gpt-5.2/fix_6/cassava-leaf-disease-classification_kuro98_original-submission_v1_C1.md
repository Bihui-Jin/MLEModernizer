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

0.8850105772136597

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17975) has done: 'I fix the ViT input-size assertion by using the model’s native preprocessing (224×224) so tensor shapes match what `vit_b_16` expects. I also correct the TTA implementation: your current `ttas` list contains stochastic transforms that aren’t being applied as intended and it returns nested lists that break collation; I replace them with deterministic image-level flips/rotations that work reliably with the DataLoader and preserve the same “average over TTA” core logic. Finally, I point `test_dir` to the correct folder (your current path has a trailing `test_images/` that doesn’t exist under that base in this environment), ensuring predictions are generated for all sample submission rows and a valid `submission.csv` is written.'
- What this solution (achieved 0.72758) has done: 'Your current score indicates the model is effectively doing zero-shot ImageNet inference, because the ViT head is reinitialized to 5 classes but never trained on cassava labels. To move the accuracy toward the target, I add a minimal, standard fine-tuning step using `train.csv` and `train_images/` while keeping the same ViT-B/16 backbone, the same cross-entropy objective (implicit in `torch.nn.functional.cross_entropy`), and the same prediction/TTA averaging logic. I also switch the preprocessing to the model’s official `weights.transforms()` (still 224×224 + normalize) to match the pretrained distribution exactly, which is a small but reliable boost. The rest of the inference and submission-writing pipeline stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.76345) has done: 'To move accuracy up toward your 0.885 target without changing the core ViT-B/16 + CE fine-tuning + TTA averaging logic, I (1) add a simple stratified train/val split and report val accuracy each epoch to catch obvious under/overfitting and confirm training is actually improving, (2) apply lightweight label smoothing in cross-entropy (same loss family; often improves calibration/generalization on Cassava with minimal code change), and (3) freeze the ViT backbone for the first epoch and then unfreeze (same architecture/training loop, but stabilizes early training so the head adapts before full fine-tuning). These are minimal, standard tweaks that typically give a noticeable bump from ~0.72 without adding extra approximations or changing inference semantics. The submission writing remains identical and still guarantees alignment with `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

base_dir = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = f"{base_dir}/train.csv"
train_dir = f"{base_dir}/train_images"
test_dir = f"{base_dir}/test_images"
sample_sub_path = f"{base_dir}/sample_submission.csv"

img_size = 224
batch_size = 32
num_workers = 4
num_classes = 5

epochs = 4
lr = 3e-4

label_smoothing = 0.05

grad_clip_norm = 1.0



## === cell 2
from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.DEFAULT
vit_model = vit_b_16(weights=weights)
vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)
vit_model.to(device)




## === cell 3
class CassavaTrainDataset(VisionDataset):
    """Train dataset reading labels from train.csv and images from train_images/."""

    def __init__(self, data_dir, df, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)


class CassavaTestDataset(VisionDataset):
    """Test dataset with deterministic TTA list.

    Returns:
      imgs: Tensor [n_tta, C, H, W]
      filename: str
    """

    def __init__(self, data_dir, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )
        self.ttas = ttas if ttas is not None else [lambda x: x]

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        imgs = [self.transform(t(img)) for t in self.ttas]
        imgs = torch.stack(imgs, dim=0)
        return imgs, filename

    def __len__(self):
        return len(self.images)




## === cell 4
base_eval_tfms = weights.transforms()
train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.75, 1.0), ratio=(0.9, 1.1)
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
    ]
)
test_transforms = base_eval_tfms


def tta_identity(img):
    return img


def tta_hflip(img):
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def tta_vflip(img):
    return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def tta_rot90(img):
    return img.transpose(Image.Transpose.ROTATE_90)


def tta_rot180(img):
    return img.transpose(Image.Transpose.ROTATE_180)


def tta_rot270(img):
    return img.transpose(Image.Transpose.ROTATE_270)


ttas = [tta_identity, tta_hflip, tta_vflip, tta_rot90, tta_rot180, tta_rot270]

train_df = pd.read_csv(train_csv_path)


def stratified_split(df, label_col="label", val_frac=0.1, seed=3407):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_parts = []
    tr_parts = []
    for _, g in df.groupby(label_col):
        n_val = max(1, int(round(len(g) * val_frac)))
        val_parts.append(g.iloc[:n_val])
        tr_parts.append(g.iloc[n_val:])
    val_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    tr_df = (
        pd.concat(tr_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return tr_df, val_df


tr_df, val_df = stratified_split(train_df, val_frac=0.1, seed=3407)

train_dataset = CassavaTrainDataset(train_dir, tr_df, transform=train_transforms)
val_dataset = CassavaTrainDataset(train_dir, val_df, transform=test_transforms)

class_counts = tr_df["label"].value_counts().sort_index()
class_weights = (1.0 / class_counts).to_dict()
sample_weights = tr_df["label"].map(class_weights).astype("float32").values
sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  # sampler controls order
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

test_dataset = CassavaTestDataset(test_dir, transform=test_transforms, ttas=ttas)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/183136725.py in <cell line: 0>()
     12         v2.ToImage(),
     13         v2.ToDtype(torch.float32, scale=True),
---> 14         v2.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
     15     ]
     16 )

KeyError: 'mean'

## === cell 5
import torch.nn.functional as F


def set_backbone_trainable(model, trainable: bool):
    for name, p in model.named_parameters():
        if name.startswith("heads."):
            p.requires_grad = True
        else:
            p.requires_grad = trainable


set_backbone_trainable(vit_model, trainable=False)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, vit_model.parameters()), lr=lr, weight_decay=0.05
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs * max(1, len(train_loader))
)

for epoch in range(epochs):
    if epoch == 1:
        set_backbone_trainable(vit_model, trainable=True)
        optimizer = torch.optim.AdamW(vit_model.parameters(), lr=lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs - epoch) * max(1, len(train_loader))
        )

    vit_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = vit_model(xb)

        loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(vit_model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()
        scheduler.step()

        running_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum().item())
        total += int(xb.size(0))

    vit_model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = vit_model(xb)
            preds = logits.argmax(dim=1)
            val_correct += int((preds == yb).sum().item())
            val_total += int(xb.size(0))

    print(
        f"epoch {epoch+1}/{epochs} | "
        f"loss {running_loss/total:.4f} | "
        f"train_acc {correct/total:.4f} | "
        f"val_acc {val_correct/max(1,val_total):.4f}"
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1097200845.py in <cell line: 0>()
     17 
     18 scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
---> 19     optimizer, T_max=epochs * max(1, len(train_loader))
     20 )
     21 

NameError: name 'train_loader' is not defined

## === cell 6
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        bsz, n_tta, c, h, w = inputs.shape
        x = inputs.view(bsz * n_tta, c, h, w).to(device, non_blocking=True)

        probs = normalizer(vit_model(x))  # [B*n_tta, num_classes]
        probs = probs.view(bsz, n_tta, -1).mean(dim=1)  # [B, num_classes]

        pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3665773231.py in <cell line: 0>()
      5 
      6 with torch.no_grad():
----> 7     for batch_idx, (inputs, filenames) in enumerate(test_loader):
      8         bsz, n_tta, c, h, w = inputs.shape
      9         x = inputs.view(bsz * n_tta, c, h, w).to(device, non_blocking=True)

NameError: name 'test_loader' is not defined

## === cell 7
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(all_names, all_preds))

missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if len(missing) > 0:
    raise RuntimeError(
        f"Missing predictions for {len(missing)} test images. Example: {missing[:5]}"
    )

my_submission = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"],
        "label": sample_sub["image_id"].map(pred_map).astype(int),
    }
)
my_submission.to_csv("submission.csv", index=False)

print(my_submission.head())
print("Wrote submission.csv with rows:", len(my_submission))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/664250395.py in <cell line: 0>()
      6 ]
      7 if len(missing) > 0:
----> 8     raise RuntimeError(
      9         f"Missing predictions for {len(missing)} test images. Example: {missing[:5]}"
     10     )

RuntimeError: Missing predictions for 2676 test images. Example: ['1234294272.jpg', '1234332763.jpg', '1234375577.jpg', '1234555380.jpg', '1234571117.jpg']
