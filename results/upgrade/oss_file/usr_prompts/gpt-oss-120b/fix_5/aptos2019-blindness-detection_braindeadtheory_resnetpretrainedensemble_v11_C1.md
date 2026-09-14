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

0.89353670176906

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.01711) has done: 'The fix updates the data paths to use the absolute “/kaggle/input” location, replaces missing custom ResNet weight files with standard ImageNet‑pretrained weights, and adjusts the prediction routine to use a regular tqdm loop. These changes remove the file‑not‑found and undefined‑model errors, ensure the models are instantiated correctly, and finally write a proper `submission.csv` with the required columns. No core modeling logic is altered beyond loading the pretrained weights, keeping the original architecture and ensemble approach intact.'

# 9. Code solution

## === cell 0
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Dataset for both train and test images."""

    def __init__(self, csv_file, filetype, transform=None):
        self.df = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform
        self.base_path = os.path.join(
            "/kaggle/input",
            "aptos2019-blindness-detection",
            "train_images" if filetype == "train" else "test_images",
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.base_path, self.df.loc[idx, "id_code"] + ".png")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.filetype == "train":
            label = int(self.df.loc[idx, "diagnosis"])
            return image, label
        else:
            return image, self.df.loc[idx, "id_code"]




## === cell 1
test_dataset = APTOSDataset(
    csv_file="/kaggle/input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=transform,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=2,  # reduced to lower overhead
    pin_memory=True,
    persistent_workers=True,
)

train_dataset = APTOSDataset(
    csv_file="/kaggle/input/aptos2019-blindness-detection/train.csv",
    filetype="train",
    transform=transform,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,  # reduced to lower overhead
    pin_memory=True,
    persistent_workers=True,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 2
def build_resnet(model_name, pretrained=True):
    if model_name == "resnet152":
        model = torchvision.models.resnet152(pretrained=pretrained)
    else:  # resnet101
        model = torchvision.models.resnet101(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # five severity classes
    return model.to(device)


model0 = build_resnet("resnet152", pretrained=True)
model1 = build_resnet("resnet101", pretrained=True)
model2 = build_resnet("resnet101", pretrained=True)
model3 = build_resnet("resnet101", pretrained=True)

models = [model0, model1, model2, model3]




## === cell 3
criterion = nn.CrossEntropyLoss()
epochs = 2  # a small number sufficient to move the score upward

optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in models]

print("Training all models jointly (single data pass per epoch)")

for epoch in range(epochs):
    running_losses = [0.0 for _ in models]
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_losses[idx] += loss.item()

    for idx, loss_sum in enumerate(running_losses):
        avg_loss = loss_sum / len(train_loader)
        model_name = "resnet152" if idx == 0 else "resnet101"
        print(f"Model {idx} ({model_name}) – Epoch {epoch+1} loss: {avg_loss:.4f}")




## === cell 4
def compute_predictions(model, mode, data_loader, device):
    """
    mode = 'train' or 'test'
    Returns:
        - For test: (DataFrame with id_code & diagnosis, logits tensor, list of ids)
        - For train: (list of predictions, accuracy)
    """
    model.eval()
    if mode == "train":
        preds, correct, total = [], 0, 0
        for inputs, labels in tqdm(data_loader, desc="Train Predict"):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        accuracy = correct / total * 100
        return preds, accuracy
    else:  # test
        logits_list, img_ids = [], []
        for inputs, ids in tqdm(data_loader, desc="Test Predict"):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            logits_list.append(outputs.cpu())
            img_ids.extend(ids)
        all_logits = torch.cat(logits_list, dim=0)
        _, preds = torch.max(all_logits, 1)
        return (
            pd.DataFrame({"id_code": img_ids, "diagnosis": preds.numpy()}),
            all_logits,
            img_ids,
        )




## === cell 5
def compute_test_logits_multi(models, data_loader, device):
    """
    Runs all models over the test DataLoader in a single pass.
    Returns a of logits tensors (one per model) and the image ids.
    """
    for m in models:
        m.eval()
    logits_per_model = [[] for _ in models]
    img_ids = []
    for inputs, ids in tqdm(data_loader, desc="Test Predict (multi)"):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = [m(inputs).cpu() for m in models]
        for lst, out in zip(logits_per_model, outputs):
            lst.append(out)
        img_ids.extend(ids)
    logits_tensors = [torch.cat(lst, dim=0) for lst in logits_per_model]
    return logits_tensors, img_ids


with torch.set_grad_enabled(False):
    print("Computing test predictions for all models in a single pass...")
    out_list, ids = compute_test_logits_multi(models, test_loader)

avg_logits = sum(out_list) / len(out_list)
_, final_preds = torch.max(avg_logits, 1)
final_preds = final_preds.numpy()

submission = pd.DataFrame({"id_code": np.array(ids), "diagnosis": final_preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/661475574.py in <cell line: 0>()
     21 with torch.set_grad_enabled(False):
     22     print("Computing test predictions for all models in a single pass...")
---> 23     out_list, ids = compute_test_logits_multi(models, test_loader)
     24 
     25 avg_logits = sum(out_list) / len(out_list)

TypeError: compute_test_logits_multi() missing 1 required positional argument: 'device'
