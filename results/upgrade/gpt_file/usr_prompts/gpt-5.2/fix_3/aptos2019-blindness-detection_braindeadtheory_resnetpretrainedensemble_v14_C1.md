# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

base_inp = "/kaggle/input"
if os.path.exists(base_inp):
    print("Found /kaggle/input with datasets:", os.listdir(base_inp)[:20])



## === cell 1
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm



## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)

train_transform = (
    transform  # keep identical transforms to preserve evaluation semantics
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(
        self,
        csv_file,
        filetype,
        transform=None,
        root_dir="/kaggle/input/aptos2019-blindness-detection",
    ):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        row = self.eye_frame.iloc[idx]
        img_id = row["id_code"]

        if self.filetype == "train":
            img_name = os.path.join(self.root_dir, "train_images", img_id + ".png")
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(row["diagnosis"])
        else:
            img_name = os.path.join(self.root_dir, "test_images", img_id + ".png")
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, img_id




## === cell 3
DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")

test_dataset = APTOSDataset(
    csv_file=TEST_CSV, filetype="test", transform=transform, root_dir=DATA_ROOT
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

train_dataset = APTOSDataset(
    csv_file=TRAIN_CSV, filetype="train", transform=train_transform, root_dir=DATA_ROOT
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 4
def build_resnet(model_name: str, num_classes: int = 5):
    if model_name == "resnet152":
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet152(weights=weights)
    elif model_name == "resnet101":
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet101(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


model0 = build_resnet("resnet152", 5).to(device)
model1 = build_resnet("resnet101", 5).to(device)
model2 = build_resnet("resnet101", 5).to(device)
model3 = build_resnet("resnet101", 5).to(device)
model4 = build_resnet("resnet101", 5).to(device)




## === cell 5
def freeze_backbone_train_fc(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def compute_class_weights_from_csv(train_csv: str, num_classes: int = 5):
    df = pd.read_csv(train_csv)
    counts = (
        df["diagnosis"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values.astype(np.float64)
    )
    weights = 1.0 / np.clip(counts, 1.0, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_fc_head(
    model: nn.Module,
    train_loader,
    device,
    epochs: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
):
    freeze_backbone_train_fc(model)
    model.train()

    class_w = compute_class_weights_from_csv(TRAIN_CSV, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=lr, weight_decay=weight_decay
    )

    for ep in range(epochs):
        running_loss = 0.0
        n = 0
        for inputs, labels in tqdm(
            train_loader, desc=f"Train(fc) ep{ep+1}/{epochs}", leave=False
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n += bs

        print(f"Epoch {ep+1}/{epochs} - loss: {running_loss / max(n,1):.4f}")


train_fc_head(model0, train_loader, device, epochs=2, lr=1e-3)
train_fc_head(model1, train_loader, device, epochs=2, lr=1e-3)
train_fc_head(model2, train_loader, device, epochs=2, lr=1e-3)
train_fc_head(model3, train_loader, device, epochs=2, lr=1e-3)
train_fc_head(model4, train_loader, device, epochs=2, lr=1e-3)




## === cell 6
def compute_predictions(model, model_type, data_loader, device):
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Predict(train)", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.detach().cpu())
            num_examples += labels.size(0)
            correct_pred += (preds == labels).sum().item()
        return predictions, correct_pred / num_examples * 100.0
    else:
        predictions = []
        img_ids = []
        out = []
        for inputs, img_id in tqdm(data_loader, desc="Predict(test)", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().cpu().tolist())
            img_ids.extend(list(img_id))
            out.extend(outputs.detach().cpu())
        final_predictions_df = pd.DataFrame(
            {"id_code": img_ids, "diagnosis": predictions}
        )
        return final_predictions_df, out, img_ids


with torch.inference_mode():
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    print("Computing Test Predictions")
    test_predictions0, out0, ids0 = compute_predictions(
        model0, "test", test_loader, device
    )
    test_predictions1, out1, ids1 = compute_predictions(
        model1, "test", test_loader, device
    )
    test_predictions2, out2, ids2 = compute_predictions(
        model2, "test", test_loader, device
    )
    test_predictions3, out3, ids3 = compute_predictions(
        model3, "test", test_loader, device
    )
    test_predictions4, out4, ids4 = compute_predictions(
        model4, "test", test_loader, device
    )



## === cell 7
out = (
    torch.stack(out0)
    + torch.stack(out1)
    + torch.stack(out2)
    + torch.stack(out3)
    + torch.stack(out4)
) / 5.0

img_ids = np.array(ids0)
_, predictions = torch.max(out, 1)
predictions = predictions.numpy().astype(int)

final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
final_predictions.to_csv("submission.csv", index=False)

print(final_predictions.head())
print("Wrote submission.csv with shape:", final_predictions.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "diagnosis value counts:\n",
    final_predictions["diagnosis"].value_counts().sort_index(),
)
