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

0.8939256572982774

# 6. Current score

0.61323

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11622) has done: 'I first fix the blocking runtime error by removing the dependency on missing external model files and replacing them with a simple, deterministic fallback that still produces valid 5-class logits. Then I fix the submission-length issue by (1) turning off `shuffle` for the test loader, (2) sorting/aligning predictions to the exact `sample_submission.csv` `image_id` order, and (3) ensuring every test image gets exactly one prediction. Finally, I make the dataset robust (RGB conversion + safe center-crop for smaller images) so inference doesn’t silently drop items or crash mid-epoch, guaranteeing a correctly formatted `submission.csv`.'
- What this solution (achieved 0.61958) has done: 'Your current score is extremely low because the “fallback” models are untrained, so predictions are essentially random. To move the accuracy up toward the target with minimal changes, I keep your inference pipeline structure (dataset, transforms, loaders, softmax/argmax, submission alignment) but add a small supervised training phase on `train.csv` using the same tiny CNN architecture and the same preprocessing as test. I also fix determinism settings slightly to avoid noisy swings and use the trained `model_a` (and optionally `model_b`) for inference on the test set exactly as you already do, producing the same `submission.csv` schema. This preserves the core model architecture and evaluation semantics while legitimately learning from labeled training data, which should move the score sharply upward toward your target band.'
- What this solution (achieved 0.61398) has done: 'Your score gap to the target is large (0.61958 → 0.89393), so we need a real (but minimal) improvement in generalization without changing your core model or training loop structure. The biggest issue is that your training transform doesn’t resize images (only test does), so the model is trained on 600×600 but evaluated on 384×384, which hurts a lot; I add the same resize in the train transform to match inference exactly. I also remove AMP for this tiny CNN (keeps semantics but avoids occasional instability/under-training) and add a lightweight LR scheduler that doesn’t change the loop structure but helps reach a better solution in 2 epochs. Finally, I fix your TTA branch averaging (it currently divides by 2 twice), though TTA is off by default.'
- What this solution (achieved 0.61248) has done: 'Your current gap to the target is large, so the smallest legitimate way to move accuracy upward is to improve generalization while keeping your exact tiny CNN + training loop intact. I make three minimal, score-relevant fixes: (1) correct the train preprocessing so it matches your test pipeline exactly (right now you effectively resize twice in train), (2) add the same center-crop/pad policy into the train transform path (so the model sees the same content framing as inference), and (3) add a tiny validation split to pick the better of model_a/model_b/ensemble for test inference (no early stopping, still 2 epochs). These changes don’t alter the model architecture or loss, but they remove train/test mismatch and reduce the risk of using the worse of the two independently trained tiny nets.'
- What this solution (achieved 0.61323) has done: 'Your gap to the target is large (0.61248 → 0.89393), so we should improve generalization without changing the tiny CNN, loss, or the overall training/inference structure. The smallest high-impact fix is to correct the biggest remaining train/test mismatch: training currently uses random flips but test/inference is deterministic center-crop only, so the model learns on augmented geometry that it never sees at inference; we keep your transforms pipeline but add the corresponding deterministic flip test-time augmentation (horizontal flip) and average logits, which typically boosts accuracy for image classification with minimal risk. I also add label-smoothing to the existing CrossEntropyLoss (same loss family, same semantics) to reduce overconfidence and usually improve validation accuracy for small CNNs. Finally, I keep your “pick best of a/b/ensemble on val” logic, but evaluate it under the same TTA setting used for test so the selection matches inference.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5

tta = True

train_epochs = 2
train_lr = 3e-4

val_frac = 0.10


class FallbackTinyNet(torch.nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model_a = FallbackTinyNet(num_classes=num_classes).to(device)
model_b = FallbackTinyNet(num_classes=num_classes).to(device)
linear_head = torch.nn.Identity().to(device)




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
        img_path = os.path.join(self.root, filename)

        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        if w < 600 or h < 600:
            pad_w = max(0, 600 - w)
            pad_h = max(0, 600 - h)
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = v2.Pad(padding, fill=0)(img)

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
    def __init__(self, csv_path, img_dir, img_size, transform=None):
        super().__init__(root=img_dir)
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self.cc = v2.CenterCrop((600, 600))
        self.resize = v2.Resize(
            (img_size, img_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img_path = os.path.join(self.img_dir, filename)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w < 600 or h < 600:
            pad_w = max(0, 600 - w)
            pad_h = max(0, 600 - h)
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = v2.Pad(padding, fill=0)(img)

        img = self.cc(img)
        img = self.resize(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
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

train_dataset_full = CassavaTrainDataset(
    csv_path=train_csv_path,
    img_dir=train_dir,
    img_size=model_a_img_size,
    transform=train_transforms,
)

n = len(train_dataset_full)
g = torch.Generator()
g.manual_seed(3407)
perm = torch.randperm(n, generator=g).tolist()
val_n = int(n * val_frac)
val_idx = perm[:val_n]
train_idx = perm[val_n:]

train_dataset = Subset(train_dataset_full, train_idx)
val_dataset = Subset(train_dataset_full, val_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
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

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=train_lr, weight_decay=1e-4)
optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=train_lr, weight_decay=1e-4)

scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_a, T_max=max(1, train_epochs)
)
scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_b, T_max=max(1, train_epochs)
)

scaler = torch.cuda.amp.GradScaler(enabled=False)

model_a.train()
model_b.train()

for epoch in range(train_epochs):
    running_loss_a = 0.0
    running_loss_b = 0.0
    correct_a = 0
    correct_b = 0
    seen = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer_a.zero_grad(set_to_none=True)
        optimizer_b.zero_grad(set_to_none=True)

        logits_a = model_a(imgs)
        logits_b = model_b(imgs)
        loss_a = criterion(logits_a, labels)
        loss_b = criterion(logits_b, labels)
        loss = loss_a + loss_b

        loss.backward()
        optimizer_a.step()
        optimizer_b.step()

        running_loss_a += float(loss_a.detach().cpu())
        running_loss_b += float(loss_b.detach().cpu())
        with torch.no_grad():
            pred_a = torch.argmax(logits_a, dim=1)
            pred_b = torch.argmax(logits_b, dim=1)
            correct_a += int((pred_a == labels).sum().detach().cpu())
            correct_b += int((pred_b == labels).sum().detach().cpu())
            seen += labels.numel()

    scheduler_a.step()
    scheduler_b.step()

    print(
        f"epoch={epoch+1}/{train_epochs} "
        f"loss_a={running_loss_a/max(1,len(train_loader)):.4f} "
        f"loss_b={running_loss_b/max(1,len(train_loader)):.4f} "
        f"acc_a={correct_a/max(1,seen):.4f} "
        f"acc_b={correct_b/max(1,seen):.4f} "
        f"lr_a={scheduler_a.get_last_lr()[0]:.2e}"
    )

model_a.eval()
model_b.eval()
with torch.no_grad():
    va_seen = 0
    va_ca = 0
    va_cb = 0
    va_cens = 0

    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if tta:
            imgs_flip = torch.flip(imgs, dims=[3])  # horizontal flip (N,C,H,W)
            la = 0.5 * model_a(imgs) + 0.5 * model_a(imgs_flip)
            lb = 0.5 * model_b(imgs) + 0.5 * model_b(imgs_flip)
        else:
            la = model_a(imgs)
            lb = model_b(imgs)

        pa = torch.argmax(la, dim=1)
        pb = torch.argmax(lb, dim=1)
        pens = torch.argmax(0.5 * la + 0.5 * lb, dim=1)

        va_ca += int((pa == labels).sum().detach().cpu())
        va_cb += int((pb == labels).sum().detach().cpu())
        va_cens += int((pens == labels).sum().detach().cpu())
        va_seen += labels.numel()

val_acc_a = va_ca / max(1, va_seen)
val_acc_b = va_cb / max(1, va_seen)
val_acc_ens = va_cens / max(1, va_seen)

if val_acc_ens >= val_acc_a and val_acc_ens >= val_acc_b:
    infer_mode = "ensemble"
elif val_acc_a >= val_acc_b:
    infer_mode = "a"
else:
    infer_mode = "b"

print(
    f"val_acc_a={val_acc_a:.4f} val_acc_b={val_acc_b:.4f} val_acc_ens={val_acc_ens:.4f} -> infer_mode={infer_mode}"
)



## === cell 5
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bsz = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bsz), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bsz), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            if infer_mode == "a":
                outputs = model_a_mean_logits
            elif infer_mode == "b":
                outputs = model_b_mean_logits
            else:
                outputs = 0.5 * model_a_mean_logits + 0.5 * model_b_mean_logits

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            if infer_mode == "a":
                outputs = model_a_outputs
            elif infer_mode == "b":
                outputs = model_b_outputs
            else:
                outputs = 0.5 * model_a_outputs + 0.5 * model_b_outputs

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 6
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
default_label = 0
aligned_labels = [
    int(pred_map.get(img_id, default_label))
    for img_id in sample_sub["image_id"].tolist()
]

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].tolist(), "label": aligned_labels}
)
my_submission.to_csv("submission.csv", index=False)

print(
    "Preds:",
    len(all_preds),
    "Unique files:",
    len(set(all_names)),
    "Submission rows:",
    len(my_submission),
)
my_submission.head()
