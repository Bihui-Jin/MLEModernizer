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

3.7

# 3. Installed packages

geopandas==0.14.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

# 5. Target score

0.7783388837413378

# 6. Current score

0.85995

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I import NumPy (as np) at the top of the script so the submission code can create the prediction array without raising a NameError. This small addition fixes the runtime error and enables the notebook to generate a valid `submission.csv` file while preserving all existing logic.'
- What this solution (achieved 0.0) has done: 'The fixes add the missing imports, define a minimal EfficientNet‑like model when the real class isn’t available, and ensure all variables (`torch`, `os`, `np`, `pd`) are defined before they’re used. This resolves the runtime NameErrors, lets the pipeline run end‑to‑end, and produces a correctly‑formatted `submission.csv` file.'
- What this solution (achieved -0.42637) has done: 'The update switches the fallback EfficientNet‑B4 to use ImageNet‑pretrained weights (instead of random initialization). This gives the model much more meaningful feature representations, which should raise the validation kappa from the near‑zero baseline toward the target score while preserving all existing logic and data‑flow.'
- What this solution (achieved 0.85995) has done: 'I add a lightweight training stage using the same EfficientNet‑B4 backbone (now with a 5‑class output) and adjust the inference to return class logits that are averaged over eight simple test‑time augmentations. After training for a few epochs the model produce meaningful class predictions, replacing the previous random‑like regression that gave a negative kappa. The changes keep the overall architecture and data handling intact while moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torchvision.models import efficientnet_b4

try:
    from efficientnet_pytorch import EfficientNet  # type: ignore
except Exception:

    class EfficientNet(nn.Module):
        """Minimal EfficientNet‑B4 wrapper with configurable output classes."""

        def __init__(self, num_classes=5, width_coefficient=1.4, depth_coefficient=1.8):
            super().__init__()
            self.backbone = efficientnet_b4(pretrained=True)
            self.backbone.classifier[1] = nn.Linear(
                self.backbone.classifier[1].in_features, num_classes
            )

        def forward(self, x):
            return self.backbone(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./best_model.pth"  # optional checkpoint

best_model = EfficientNet(num_classes=5)

if os.path.exists(model_path):
    try:
        best_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pretrained checkpoint.")
    except Exception as e:
        print("Failed to load checkpoint:", e)
else:
    print("Checkpoint not found; using ImageNet pretrained weights.")

best_model = best_model.to(device)




## === cell 1
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
)
from PIL import Image
import torch.utils.data as data


class ImageDataset(data.Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        self.root = root
        self.path_list = path_list
        self.targets = torch.LongTensor(targets) if targets is not None else None
        self.transform = transform
        self.extension = extension

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img_path = os.path.join(self.root, path + self.extension)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[idx]
        else:
            return img, torch.tensor([])

    def __len__(self):
        return len(self.path_list)


train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_transform = Compose(
    [
        Resize((380, 380), Image.BICUBIC),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

train_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/train_images",
    path_list=train_df.id_code.values,
    targets=train_df.diagnosis.values,
    transform=train_transform,
)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=max(0, os.cpu_count() - 1),
    pin_memory=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-4)

best_model.train()
for epoch in range(5):  # 5 epochs are enough for a noticeable lift
    epoch_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(imgs)  # logits shape (B, 5)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss / len(train_dataset):.4f}")

best_model.eval()




## === cell 2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

test_transform = Compose(
    [
        Resize((380, 380), Image.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/test_images",
    path_list=df_test.id_code.values,
    transform=test_transform,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=max(0, os.cpu_count() - 1),
    pin_memory=True,
)




## === cell 3
def tta(x):
    """8‑fold test‑time augmentation returning averaged logits."""
    preds = []
    for flip1 in range(2):
        xf = x.flip(2) if flip1 else x
        for flip2 in range(2):
            xf2 = xf.flip(3) if flip2 else xf
            for trans in range(2):
                xt = xf2.transpose(-1, -2) if trans else xf2
                logits = best_model(xt)  # shape (B, 5)
                preds.append(logits.unsqueeze(0))
    return torch.cat(preds).mean(dim=0)  # averaged logits (B, 5)




## === cell 4
from tqdm import tqdm

preds = []
with torch.no_grad():
    for imgs, _ in tqdm(test_loader, total=len(test_loader)):
        imgs = imgs.to(device)
        logits = tta(imgs)  # (B, 5)
        classes = logits.argmax(dim=1)  # predicted class indices
        preds.extend(classes.cpu().tolist())




## === cell 5
opt_preds = np.array(preds, dtype=np.int32)




## === cell 6
sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
assert len(opt_preds) == len(sub), f"Prediction length {len(opt_preds)} != {len(sub)}"
sub["diagnosis"] = opt_preds
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(sub), "rows.")
