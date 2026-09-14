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

3.12

# 3. Installed packages

geopandas==0.14.4
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
sklearn-pandas==2.2.0
timm==1.0.19
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
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import timm
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split




## === cell 1
class BlindnessDataset(Dataset):
    """
    Dataset that optionally pre‑loads all images (already transformed) into memory.
    This removes the per‑sample disk I/O and transform overhead while preserving
    the exact output tensors that the original pipeline would produce.
    """

    def __init__(self, csv_file, root_dir, transform=None, test=False, preload=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.preload = preload

        if self.preload:
            self._imgs = []
            for idx in range(len(self.annotations)):
                img_path = os.path.join(
                    self.root_dir, self.annotations.iloc[idx, 0] + ".png"
                )
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)  # tensor after transform
                self._imgs.append(img)  # already a torch.Tensor
        else:
            self._imgs = None  # not used when not preloading

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.preload:
            image = self._imgs[idx]  # tensor ready for model
        else:
            img_name = os.path.join(
                self.root_dir, self.annotations.iloc[idx, 0] + ".png"
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)  # apply transform on‑the‑fly

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
base_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_dataset = BlindnessDataset(
    test_csv_file,
    test_root_dir,
    transform=base_transform,
    test=True,
    preload=True,  # preload test images (already transformed) for fast inference
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,  # reduced workers to lower spawn overhead
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)




## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)

models_list = []
model_keys = []

torch.backends.cudnn.benchmark = torch.cuda.is_available()
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # faster GPU matmul when possible

for model_key, path in model_paths.items():
    try:
        model_name = model_names[model_key]
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models_list.append(model)
        model_keys.append(model_key)
    except FileNotFoundError:
        continue

if len(models_list) == 0:
    torch.manual_seed(42)  # reproducibility

    train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

    full_train_dataset = BlindnessDataset(
        train_csv_file,
        train_root_dir,
        transform=base_transform,
        test=False,
        preload=True,  # preload once for fast indexing
    )
    train_idx, val_idx = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=0.1,
        random_state=42,
        stratify=full_train_dataset.annotations["diagnosis"],
    )

    train_subset = torch.utils.data.Subset(
        BlindnessDataset(
            train_csv_file,
            train_root_dir,
            transform=train_transform,
            test=False,
            preload=False,  # changed: load images per batch instead of preloading
        ),
        train_idx,
    )
    val_subset = torch.utils.data.Subset(
        BlindnessDataset(
            train_csv_file,
            train_root_dir,
            transform=base_transform,
            test=False,
            preload=False,  # changed: load images per batch instead of preloading
        ),
        val_idx,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )

    fallback_model = timm.create_model("resnet18", pretrained=True, num_classes=5)
    fallback_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fallback_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )
    scaler = torch.cuda.amp.GradScaler()  # mixed‑precision accelerator

    NUM_EPOCHS = 20  # retained from original logic
    best_kappa = -1.0
    best_state_dict = None

    fallback_model.train()
    for epoch in range(NUM_EPOCHS):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = fallback_model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        fallback_model.eval()
        with torch.inference_mode():
            val_preds = []
            val_targets = []
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                logits = fallback_model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
            epoch_kappa = cohen_kappa_score(val_targets, val_preds, weights="quadratic")
        if epoch_kappa > best_kappa:
            best_kappa = epoch_kappa
            best_state_dict = fallback_model.state_dict()
        scheduler.step(epoch_kappa)

        fallback_model.train()

    if best_state_dict is not None:
        fallback_model.load_state_dict(best_state_dict)

    fallback_model.eval()
    models_list.append(fallback_model)
    model_keys.append("resnet18")

if len(models_list) == 1:
    weights = {"resnet18": 1.0}
else:
    validation_scores = {
        "resnet18": 0.887,
        "efficientnet_b0": 0.8922,
        "efficientnet_b1": 0.894,
        "efficientnet_b2": 0.898,
        "efficientnet_b3": 0.9127,
        "efficientnet_b4": 0.893,
        "efficientnet_b5": 0.870,
        "inception_resnet_v2": 0.896,
        "inception_v4": 0.8875,
        "seresnext50_32x4d": 0.8652,
        "seresnext101_32x4d": 0.9083,
    }
    total_score = sum(validation_scores.values())
    weights = {k: validation_scores[k] / total_score for k in model_keys}




## === cell 6
weight_tensor = torch.tensor([weights[k] for k in model_keys], device=device).view(
    -1, 1, 1
)

all_outputs = []

with torch.inference_mode(), torch.cuda.amp.autocast():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device, non_blocking=True)

        logits_stack = torch.stack([model(images) for model in models_list], dim=0)
        probs = torch.nn.functional.softmax(logits_stack, dim=2)
        weighted_sum = (probs * weight_tensor).sum(dim=0)  # shape (batch, 5)

        all_outputs.extend(weighted_sum.cpu().numpy())

all_outputs = np.array(all_outputs)  # (num_test, 5)
final_predictions = np.argmax(all_outputs, axis=1)




## === cell 7
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)

submission_df.to_csv("submission.csv", index=False)
