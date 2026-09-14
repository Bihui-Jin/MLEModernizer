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

# 5. Target score

0.8355883073049606

# 6. Current score

0.19827

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.19827) has done: 'The fix adds robust model loading (falls back to a pretrained EfficientNet‑b0 when the ensemble files are missing), builds a small training loop to fine‑tune this fallback model on the provided training set, and adjusts the inference logic so it works even with a single model.  We also compute uniform ensemble weights, ensure the prediction array is always populated, and finally write a valid `submission.csv`.  All changes keep the original pipeline structure while enabling the notebook to run end‑to‑end and produce a submission that can achieve a score close to the target.'

# 9. Code solution

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




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)



## === cell 4
model_paths = {
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/inception_v4.pth",
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
model_keys_used = []

for key, path in model_paths.items():
    try:
        model_name = model_names[key]
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded pretrained ensemble model: {key}")
    except Exception as e:
        print(f"Could not load {key} ({e}), falling back to EfficientNet‑b0.")
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=5)
    model.to(device)
    model.eval()
    models_list.append(model)
    model_keys_used.append(key)



## === cell 5
if len(models_list) == 1 and model_keys_used[0] not in model_paths:
    train_csv = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_root = "/kaggle/input/aptos2019-blindness-detection/train_images"
    full_train_dataset = BlindnessDataset(
        train_csv, train_root, transform=transform, test=False
    )

    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = models_list[0]  # the only model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())
    try:
        from sklearn.metrics import cohen_kappa_score

        qwk = cohen_kappa_score(all_true, all_preds, weights="quadratic")
        print(f"Validation QWK (quick): {qwk:.4f}")
    except Exception:
        pass



## === cell 6
num_models = len(models_list)
weights = {k: 1.0 / num_models for k in model_keys_used}



## === cell 7
all_outputs = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        model_outputs = []
        for model_key, model in zip(model_keys_used, models_list):
            out = nn.functional.softmax(model(images), dim=1)  # (B,5)
            weighted = weights[model_key] * out
            model_outputs.append(weighted.unsqueeze(0))  # (1,B,5)
        stacked = torch.cat(model_outputs, dim=0)  # (M,B,5)
        summed = torch.sum(stacked, dim=0)  # (B,5)
        all_outputs.extend(summed.cpu().numpy())

all_outputs = np.array(all_outputs)  # (N,5)
final_predictions = np.argmax(all_outputs, axis=1)



## === cell 8
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
