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

0.79522

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.2145) has done: 'I filter the dataset files to include only image files, preventing the DataLoader from trying to open the inner “test_images” directory as an image. This fixes the IsADirectoryError and ensures the submission length matches the number of test images, producing a valid .csv file.'
- What this solution (achieved 0.05531) has done: 'I replace the random‐output dummy model with a deterministic “majority‑class” dummy that always gives a very high logit for class 0 and low logits for the other classes. Since the Cassava dataset is heavily skewed toward class 0, this simple change raise the accuracy from the current ~0.21 toward the target (without altering the overall pipeline or model architecture).'
- What this solution (achieved 0.05531) has done: 'I keep the overall pipeline but add a small improvement: after loading the three models I use the supplied `linear_head` (if it is a real classifier) to combine the logits from `model_a` and `model_b` instead of only averaging them. When `linear_head` is still the dummy placeholder the code falls back to the original weighted average, so behaviour stays unchanged in that case. This tiny change can raise the accuracy noticeably without altering the core data‑handling or training logic.'
- What this solution (achieved 0.6917) has done: 'I added a lightweight training stage that uses a pretrained ResNet‑18 to learn a simple classifier on a subset of the real training data, then replace the dummy models with this trained model (both model_a and model_b). This keeps the original inference pipeline but gives far more informative logits, moving the accuracy much closer to the target while preserving the overall code structure.'
- What this solution (achieved 0.62481) has done: 'I train the ResNet‑18 on the full training set (instead of a small random subset), add light data‑augmentation, run a few more epochs, and optimize all model parameters rather than only the final linear layer. These modest changes keep the original pipeline intact while strengthening the learned classifier, which should raise the validation/test accuracy toward the target score.'
- What this solution (achieved 0.05531) has done: 'The change flips the training‑skip logic: if any model is a `DummyModel` we now skip the expensive 15‑epoch ResNet18 training, keeping the lightweight dummy models already loaded. This eliminates the dominant runtime cost while preserving the original inference pipeline and model handling semantics.'
- What this solution (achieved 0.79522) has done: 'I train a real ResNet‑18 on the full training set (instead of skipping because dummy models are present) and then use that trained model for both model_a and model_b. I also enable the simple test‑time‑augmentation flag so the inference averages a few transformed versions, which should lift the accuracy toward the target while keeping the original pipeline structure unchanged.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
from torch.backends import cudnn

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

model_a_img_size = 384
model_b_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True


class DummyModel(nn.Module):
    """
    Deterministic dummy model that strongly favors a chosen class.
    Returns large positive logits for the chosen class and large negative logits for others.
    """

    def __init__(self, out_features: int, fav_class: int = 0):
        super().__init__()
        self.out_features = out_features
        self.fav_class = fav_class

    def forward(self, x):
        batch = x.shape[0]
        high = torch.full((batch, 1), 5.0, device=x.device, dtype=torch.float32)
        low = torch.full(
            (batch, self.out_features - 1), -5.0, device=x.device, dtype=torch.float32
        )
        if self.fav_class == 0:
            return torch.cat([high, low], dim=1)
        else:
            parts = []
            for i in range(self.out_features):
                if i == self.fav_class:
                    parts.append(high)
                else:
                    parts.append(low[:, :1])
            return torch.cat(parts, dim=1)


def load_or_dummy(path: str, name: str):
    try:
        model = torch.load(path, map_location=device)
        print(f"{name} loaded from {path}")
    except FileNotFoundError:
        print(f"{path} not found – using {name} dummy model.")
        model = DummyModel(num_classes).to(device)
    except Exception as ex:
        print(f"Error loading {path} ({ex}) – using {name} dummy model.")
        model = DummyModel(num_classes).to(device)
    return model


model_a = load_or_dummy("/kaggle/input/vit-v1/vit_v1.pt", "model_a")
model_b = load_or_dummy("/kaggle/input/vit-boosted/vit_boosted.pt", "model_b")
linear_head = load_or_dummy("/kaggle/input/linear-head/linear_cls.pt", "linear_head")




## === cell 1
from torchvision.datasets import VisionDataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import random



class CassavaTrainDataset(VisionDataset):
    """Dataset for training – loads images and integer labels."""

    def __init__(self, img_dir, csv_path, img_size, transform=None):
        super().__init__(root=img_dir)
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.resize = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label


train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

full_train_dataset = CassavaTrainDataset(
    img_dir=train_img_dir,
    csv_path=train_csv_path,
    img_size=224,
    transform=train_transform,
)

train_loader = torch.utils.data.DataLoader(
    full_train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,  # keep workers alive across epochs
)

base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
base_model = base_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 8
base_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = base_model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(full_train_dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    scheduler.step()

model_a = base_model
model_b = base_model
linear_head = DummyModel(num_classes).to(device)  # keep dummy – not used




## === cell 2
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
from PIL import Image


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
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
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

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # keep deterministic ordering for submission
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 4
import pandas as pd

all_names = []
all_preds = []

model_a.eval()
model_b.eval()
linear_head.eval()  # dummy – not used

use_linear_head = False  # we keep the simple weighted average

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        batch_len = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            model_a_batch_logits = torch.stack(
                torch.split(model_a_outputs, batch_len), dim=0
            )
            model_b_batch_logits = torch.stack(
                torch.split(model_b_outputs, batch_len), dim=0
            )

            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            model_a_mean_logits = model_a(model_a_inputs)
            model_b_mean_logits = model_b(model_b_inputs)

        final_logits = 0.5 * model_a_mean_logits + 0.5 * model_b_mean_logits

        probs = normalizer(final_logits)
        pred_labels = torch.argmax(probs, dim=1).cpu().tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)




## === cell 5
import pandas as pd

submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
submission.head()
