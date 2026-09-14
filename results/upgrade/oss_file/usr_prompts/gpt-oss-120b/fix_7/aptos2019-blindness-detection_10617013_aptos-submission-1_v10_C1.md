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

3.10

# 3. Installed packages

albumentations==2.0.8
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
seaborn==0.12.2
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
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.6400144551911864

# 6. Current score

0.06502

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.17166) has done: 'Implemented fixes:
- Removed unavailable `imgaug` library and replaced augmentations with Albumentations.
- Added missing imports and corrected seed initialization.
- Simplified and corrected the dataset class (proper transform handling, tensor conversion, and optional label support).
- Initialized a pretrained ResNet‑18 model, replaced its classifier for 5 classes, and omitted loading a non‑existent checkpoint.
- Fixed paths, loader creation, and inference loop.
- Ensured the submission CSV is written with the correct column name and `.csv` extension.'
- What this solution (achieved 0.14585) has done: 'I add a lightweight training stage so the ResNet‑18 model is fine‑tuned on the provided train split before inference. This includes loading the training CSV, creating a stratified train/validation split, defining simple augmentation for training, training for a few epochs with CrossEntropyLoss, keeping the best model by quadratic weighted kappa, and then using that model for test predictions. The rest of the pipeline (data loading, test transforms, submission writing) stays unchanged, preserving the original core logic while moving the score toward the target.'
- What this solution (achieved 0.14585) has done: 'Implemented fixes and modest enhancements to boost validation QWK and generate a proper submission:
- Corrected `RandomResizedCrop` usage (expects a tuple for size).
- Added class‑balanced loss weights computed from the training split.
- Increased training epochs to allow better convergence.
- Integrated a ReduceLROnPlateau scheduler to adapt learning rate based on validation QWK.
- Kept the overall architecture and workflow unchanged while ensuring the final CSV is written correctly.'
- What this solution (achieved 0.14585) has done: 'Implemented a fix for the Albumentations `RandomResizedCrop` parameters (now uses the correct single‑size signature) and extended training from 12 to 20 epochs to allow the model to converge better, which should raise the quadratic weighted kappa toward the target. No core‑logic changes were made; the architecture, loss, and training loop remain unchanged, and the script now reliably produces a valid `submission.csv`.'
- What this solution (achieved 0.14585) has done: 'Implemented fixes:
- Corrected the `RandomResizedCrop` initialization to use a tuple size `(224, 224)`, matching Albumentations’ required signature.
- Slightly extended training to 30 epochs and adjusted the learning‑rate scheduler patience for a more stable convergence, keeping the core model and loss unchanged.
- Added a short comment explaining the change.'
- What this solution (achieved 0.06502) has done: 'I fixed the Albumentations `RandomResizedCrop` call (now uses the required `size` argument) and changed both validation and test inference to use the soft‑max expected rating rounded to an integer, which better matches the quadratic weighted kappa metric. These adjustments resolve the runtime error and should raise the validation QWK, moving the score toward the target while keeping the core model and training logic unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")



## === cell 1
test_path = "../input/aptos2019-blindness-detection/test.csv"
test_img_dir = "../input/aptos2019-blindness-detection/test_images"

test_df = pd.read_csv(test_path)
print(test_df.head())



## === cell 2
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 diagnosis classes
model = model.to(device)



## === cell 3
test_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 4
class AptosDataset(Dataset):
    """
    Generic dataset used for both train and test.
    For test mode `label_available=False` and only images are returned.
    """

    def __init__(self, df, img_dir, transform=None, label_available=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_available = label_available

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["id_code"]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.label_available:
            label = self.df.iloc[idx]["diagnosis"]
            return image, label
        else:
            return image

    def show(self, idx):
        img = self.__getitem__(idx)
        if isinstance(img, (list, tuple)):
            img = img[0]
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        plt.imshow(img_np)
        plt.axis("off")
        plt.show()




## === cell 5
batch_size = 64
test_dataset = AptosDataset(
    test_df, test_img_dir, transform=test_transform, label_available=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

train_path = "../input/aptos2019-blindness-detection/train.csv"
train_img_dir = "../input/aptos2019-blindness-detection/train_images"

train_df = pd.read_csv(train_path)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["diagnosis"],
    random_state=SEED,
)

train_transform = A.Compose(
    [
        A.RandomResizedCrop(224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_dataset = AptosDataset(
    train_split, train_img_dir, transform=train_transform, label_available=True
)
val_dataset = AptosDataset(
    val_split, train_img_dir, transform=val_transform, label_available=True
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

class_counts = train_split["diagnosis"].value_counts().sort_index()
weights = 1.0 / class_counts.values
weights = weights * (len(class_counts) / weights.sum())
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, verbose=True
)

num_epochs = 30  # extended training for better convergence
best_kappa = -1.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train"):
        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val"):
            imgs = imgs.to(device)
            outputs = model(imgs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            expected = torch.sum(probs * torch.arange(5, device=device).float(), dim=1)
            preds = torch.round(expected).long()
            preds = torch.clamp(preds, 0, 4)  # ensure valid class range

            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    kappa = cohen_kappa_score(val_targets, val_preds, weights="quadratic")
    print(f"Epoch {epoch+1} | Train loss: {epoch_loss:.4f} | Val QWK: {kappa:.4f}")

    if kappa > best_kappa:
        best_kappa = kappa
        torch.save(model.state_dict(), "best_model.pth")

    scheduler.step(kappa)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3916381060.py in <cell line: 0>()
     22 train_transform = A.Compose(
     23     [
---> 24         A.RandomResizedCrop(224, scale=(0.8, 1.0)),
     25         A.HorizontalFlip(p=0.5),
     26         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 6
predictions = []
with torch.no_grad():
    for imgs in tqdm(test_loader, desc="Inference"):
        imgs = imgs.to(device)
        logits = model(imgs)

        probs = torch.nn.functional.softmax(logits, dim=1)
        expected = torch.sum(probs * torch.arange(5, device=device).float(), dim=1)
        preds = torch.round(expected).long()
        preds = torch.clamp(preds, 0, 4)

        predictions.extend(preds.cpu().numpy().tolist())

print(f"Generated {len(predictions)} predictions.")



## === cell 7
submission_path = "submission.csv"
sample_sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample_sub["diagnosis"] = predictions
sample_sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
