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
import os, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

train_csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    majority_class = int(train_df["diagnosis"].value_counts().idxmax())
else:
    majority_class = 0  # safe default if training file missing

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform
        self.base_dir = "/kaggle/input/aptos2019-blindness-detection"

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.base_dir,
            f"{self.filetype}_images",
            self.eye_frame.loc[idx, "id_code"] + ".png",
        )
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.filetype == "train":
            label = int(self.eye_frame.loc[idx, "diagnosis"])
            return image, label
        else:
            img_id = self.eye_frame.loc[idx, "id_code"]
            return image, img_id


test_dataset = APTOSDataset(
    csv_file="/kaggle/input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=transform,
)
test_loader = DataLoader(
    test_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True
)

test_order_path = "/kaggle/input/aptos2019-blindness-detection/test.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 2
def load_model(model_cls, weight_path, device, num_classes=5):
    """
    Load a model. If a checkpoint exists at weight_path, use it;
    otherwise load ImageNet weights and mark the model as lacking
    real fine‑tuned weights.
    """
    if os.path.exists(weight_path):
        model = model_cls(pretrained=False)
        state = torch.load(weight_path, map_location=device)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state)
        model._has_real_weights = True
    else:
        model = model_cls(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model._has_real_weights = False  # no fine‑tuned checkpoint
    model = model.to(device)
    model.eval()
    return model


ckpt_dir = "/kaggle/input/resnet"
model0 = load_model(
    models.resnet152, os.path.join(ckpt_dir, "FinalResnet152_0.pt"), device
)

model1 = load_model(
    models.resnet101, os.path.join(ckpt_dir, "FinalResnet02.pt"), device
)

model2 = load_model(
    models.resnet101, os.path.join(ckpt_dir, "FinalResnet01.pt"), device
)

model3 = load_model(
    models.resnet101, os.path.join(ckpt_dir, "FinalResnet00.pt"), device
)

model4 = load_model(models.resnet101, os.path.join(ckpt_dir, "FinalResnet0.pt"), device)


if not any(
    getattr(m, "_has_real_weights", False)
    for m in [model0, model1, model2, model3, model4]
):
    print("No real checkpoints found – performing quick fine‑tuning.")
    train_dataset = APTOSDataset(
        csv_file="/kaggle/input/aptos2019-blindness-detection/train.csv",
        filetype="train",
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam  # will be instantiated per model

    def fine_tune(model, epochs=3):
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        model.fc.requires_grad = True
        model.train()
        opt = optimizer(model.fc.parameters(), lr=1e-3)
        for epoch in range(epochs):
            running_loss = 0.0
            for imgs, labs in tqdm(
                train_loader, desc=f"Fine‑tune epoch {epoch+1}/{epochs}", leave=False
            ):
                imgs, labs = imgs.to(device), labs.to(device)
                opt.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labs)
                loss.backward()
                opt.step()
                running_loss += loss.item()
            print(
                f"Epoch {epoch+1}/{epochs} loss: {running_loss/len(train_loader):.4f}"
            )
        model.eval()
        model._has_real_weights = True
        return model

    model0 = fine_tune(model0)
    model1 = fine_tune(model1)
    model2 = fine_tune(model2)
    model3 = fine_tune(model3)
    model4 = fine_tune(model4)




## === cell 3
def compute_predictions(model, mode, data_loader, device):
    """
    mode == 'train'  -> returns list of predictions and accuracy%
    mode == 'test'   -> returns dataframe (id_code, diagnosis), raw logits list, id list
    """
    if mode == "train":
        predictions = []
        correct, total = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Training inference"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total * 100 if total > 0 else 0.0
        return predictions, acc

    else:  # test mode
        preds = []
        img_ids = []
        logits = []
        for inputs, ids in tqdm(data_loader, desc="Test inference"):
            inputs = inputs.to(device)
            out = model(inputs)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            img_ids.extend(ids)
            logits.append(out.detach().cpu())
        df = pd.DataFrame({"id_code": img_ids, "diagnosis": preds})
        return df, logits, img_ids




## === cell 4
if not any(
    getattr(m, "_has_real_weights", False)
    for m in [model0, model1, model2, model3, model4]
):
    print("No real model weights found – building intensity fallback.")
    train_dataset = APTOSDataset(
        csv_file="/kaggle/input/aptos2019-blindness-detection/train.csv",
        filetype="train",
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    intensity_vals = []
    label_vals = []
    with torch.no_grad():
        for imgs, labs in tqdm(train_loader, desc="Building intensity centroids"):
            batch_means = imgs.view(imgs.size(0), -1).mean(dim=1).cpu().numpy()
            intensity_vals.extend(batch_means)
            label_vals.extend(labs.cpu().numpy())
    intensity_vals = np.array(intensity_vals)
    label_vals = np.array(label_vals)
    class_intensity_centroid = {}
    for cls in np.unique(label_vals):
        class_intensity_centroid[int(cls)] = intensity_vals[label_vals == cls].mean()
    fallback_intensity_map = class_intensity_centroid
else:
    fallback_intensity_map = None




## === cell 5
with torch.no_grad():
    print("Computing test predictions for each model...")
    pred0, out0, ids0 = compute_predictions(model0, "test", test_loader, device)
    pred1, out1, ids1 = compute_predictions(model1, "test", test_loader, device)
    pred2, out2, ids2 = compute_predictions(model2, "test", test_loader, device)
    pred3, out3, ids3 = compute_predictions(model3, "test", test_loader, device)
    pred4, out4, ids4 = compute_predictions(model4, "test", test_loader, device)




## === cell 6
if fallback_intensity_map is not None:
    print("Using intensity‑based fallback predictions.")
    intensity_preds = []
    fallback_ids = []
    for inputs, ids in tqdm(test_loader, desc="Fallback intensity inference"):
        batch_means = inputs.view(inputs.size(0), -1).mean(dim=1).cpu().numpy()
        for m in batch_means:
            diffs = {c: abs(m - cen) for c, cen in fallback_intensity_map.items()}
            pred_class = min(diffs, key=diffs.get)
            intensity_preds.append(pred_class)
        fallback_ids.extend(ids)
    final_predictions = pd.DataFrame(
        {"id_code": fallback_ids, "diagnosis": intensity_preds}
    )
else:
    logits0 = torch.cat(out0, dim=0)
    logits1 = torch.cat(out1, dim=0)
    logits2 = torch.cat(out2, dim=0)
    logits3 = torch.cat(out3, dim=0)
    logits4 = torch.cat(out4, dim=0)

    ensemble_logits = (logits0 + logits1 + logits2 + logits3 + logits4) / 5.0
    probs = torch.softmax(ensemble_logits, dim=1)

    class_range = torch.arange(5, device=device).float()
    expected = (probs * class_range).sum(dim=1)
    ensemble_pred = torch.round(expected).clamp(0, 4).long().cpu().numpy()
    ensemble_ids = np.array(ids0)

    final_predictions = pd.DataFrame(
        {"id_code": ensemble_ids, "diagnosis": ensemble_pred}
    )

test_order = pd.read_csv(test_order_path)["id_code"]
final_predictions = final_predictions.set_index("id_code").loc[test_order].reset_index()

submission_path = "submission.csv"
final_predictions.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
