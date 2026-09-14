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

0.0780772011140781

# 6. Current score

-0.03562

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix adds a safe fallback when the expected pretrained model file is missing: `load_model` now checks for the file and, if absent, creates a lightweight `DummyModel` that returns zero logits (uniform scores) so the ensemble still runs. A warning is printed to inform the user. This prevents the `FileNotFoundError` and allows the pipeline to generate a valid `submission.csv` with deterministic predictions. No core modeling logic is changed, and the output format remains correct.'
- What this solution (achieved 0.0) has done: 'I switch the models to use ImageNet‑pretrained weights (setting `pretrained=True`) and, when a custom checkpoint is missing, keep the pretrained model instead of falling back to a dummy that always predicts class 0. This gives the ensemble a sensible feature extractor, so the predictions become varied and the quadratic weighted kappa should move from 0.0 toward the target 0.078 while preserving the original architecture and training logic.'
- What this solution (achieved 0.0) has done: 'Implemented two lightweight adjustments to raise the validation score toward the target while keeping the original workflow intact:

* Added ImageNet‑style normalization to the test transform so the pretrained backbone receives inputs in the range it was trained on.
* Switched the final class decision from a simple arg‑max to an expected‑value rounding (continuous prediction → nearest integer). This often aligns better with the quadratic weighted kappa metric.'
- What this solution (achieved -0.00499) has done: 'I change the way the ensemble’s final class is derived: instead of converting the averaged class probabilities to an expected value and rounding (which often collapses many predictions to the same label), I take the arg‑max of the averaged probabilities. This simple tweak keeps the core model and workflow unchanged but usually yields a more diverse set of predictions, moving the quadratic weighted kappa closer to the modest target score.'
- What this solution (achieved 0.0) has done: 'I adjust the test preprocessing to match the model’s expected 224 × 224 input size and replace the arg‑max decision with an expected‑value calculation (rounded to the nearest integer). This small change keeps the core architecture unchanged while giving the model’s probability outputs a finer‑grained influence on the final class, which should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.04411) has done: 'I replace the expected‑value rounding with a simple arg‑max class selection, which gives a more diverse set of predictions and is expected to move the quadratic weighted kappa closer to the modest target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I change the final prediction step to use the expected‑value (probability‑weighted) rating instead of arg‑max. Computing the average class probabilities across the ensemble, then taking the weighted sum of class indices and rounding to the nearest integer typically aligns better with the quadratic weighted kappa metric, moving the score upward toward the target while keeping the core pipeline unchanged.'
- What this solution (achieved 0.04471) has done: 'I replace the expected‑value rounding used to obtain the final class with a simple arg‑max over the averaged class probabilities. This keeps the core model and preprocessing untouched while giving more varied predictions, which should raise the quadratic weighted kappa from 0.0 toward the target 0.078.'
- What this solution (achieved 0.0) has done: 'The update changes the final prediction step to use a probability‑weighted expected value (rounded to the nearest integer) instead of a simple arg‑max, which aligns better with the quadratic weighted kappa metric and should raise the score toward the target while leaving the overall pipeline unchanged.'
- What this solution (achieved -0.07246) has done: 'I add a small class‑frequency bias to the averaged softmax outputs and switch the final decision from expected‑value rounding to an arg‑max, which gives more varied predictions. I also expand the ensemble to three ImageNet‑pretrained models (the extra models fall back to their pretrained weights) to increase diversity. These minimal tweaks keep the original architecture and training untouched while moving the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.02232) has done: 'I remove the class‑prior bias that was pulling the predictions away from the true distribution and switch the ensemble’s final decision from a plain arg‑max to a probability‑weighted expected‑value rounded to the nearest integer. This small change aligns the output more closely with the quadratic weighted kappa metric and should raise the score toward the target without altering the core model architecture.'
- What this solution (achieved -0.02353) has done: 'I switch the ensemble’s final decision from the probability‑weighted expected value (rounded) to a simple arg‑max over the averaged class probabilities. This minor change keeps the core model and preprocessing unchanged while typically producing more varied predictions that better match the quadratic weighted kappa, moving the score upward toward the target.'
- What this solution (achieved -0.03562) has done: 'I replace the simple arg‑max conversion with a probability‑weighted expected value rounded to the nearest integer (and clipped to the valid range 0‑4). This aligns the ensemble output more closely with the quadratic weighted kappa metric, which usually benefits from using the expected rating rather than the most likely class, and should raise the score from a negative value toward the target. The rest of the pipeline remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torchvision import transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import cohen_kappa_score

import copy
import timm

train_csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_df = pd.read_csv(train_csv_path)
class_counts = train_df["diagnosis"].value_counts().sort_index()
class_prior = (class_counts.values / class_counts.values.sum()).astype(np.float32)




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        augmentations=None,
        max_count=None,
        test=False,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_count = max_count
        self.test = test

        if not test:
            self.class_counts = (
                self.annotations["diagnosis"].value_counts().sort_index()
            )
        else:
            self.class_counts = None

        if max_count:
            self.oversample(max_count)

    def oversample(self, max_count):  # Over sampling classes to balance
        samples = []
        for diagnosis in self.class_counts.index:
            class_samples = self.annotations[self.annotations["diagnosis"] == diagnosis]
            oversampled_class = class_samples.sample(max_count, replace=True)
            samples.append(oversampled_class)
        self.annotations = pd.concat(samples).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image

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
def select_model(model_name, input_size):
    if model_name == "efficientnet_b5":
        return timm.create_model(
            "efficientnet_b5", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "inception_resnet_v2":
        return timm.create_model(
            "inception_resnet_v2", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "inception_v4":
        return timm.create_model(
            "inception_v4", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "seresnext50_32x4d":
        return timm.create_model(
            "seresnext50_32x4d", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "seresnext101_32x4d":
        return timm.create_model(
            "seresnext101_32x4d", pretrained=True, num_classes=5, in_chans=3
        )
    elif model_name == "resnet18(WD_1e-3)_aptos":
        return timm.create_model("resnet18", pretrained=True, num_classes=5, in_chans=3)
    else:
        raise ValueError(f"Unknown model name {model_name}")




## === cell 4
class DummyModel(nn.Module):
    """Fallback model used when a pretrained weight file is missing."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.size(0)
        return torch.zeros(batch_size, self.num_classes, device=x.device)


def load_model(model_name, input_size, model_path, device):
    model = select_model(model_name, input_size).to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"Warning: model file '{model_path}' not found. Using ImageNet‑pretrained model."
        )
    model.eval()
    return model




## === cell 5
def load_all_models(model_paths, device):
    loaded = []
    for model_name, input_size, model_path in model_paths:
        model = load_model(model_name, input_size, model_path, device)
        loaded.append((model_name, model))
    return loaded




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1"
os.makedirs(save_dir, exist_ok=True)




## === cell 7
model_paths = [
    (
        "resnet18(WD_1e-3)_aptos",
        512,
        os.path.join(save_dir, "resnet18(WD_1e-3)_aptos.pth"),
    ),
    (
        "efficientnet_b5",
        224,
        os.path.join(save_dir, "efficientnet_b5.pth"),
    ),
    (
        "seresnext50_32x4d",
        224,
        os.path.join(save_dir, "seresnext50_32x4d.pth"),
    ),
]




## === cell 8
loaded_models = load_all_models(model_paths, device)




## === cell 9
def predict_ensemble(models, test_csv_file, test_root_dir, submission_file, device):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = BlindnessDataset(
        test_csv_file, test_root_dir, transform=transform, test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    predictions = []
    for model_name, model in models:
        model_preds = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.softmax(outputs, dim=1)
                model_preds.append(preds.cpu().numpy())
        model_preds = np.concatenate(model_preds, axis=0)
        predictions.append(model_preds)

    avg_predictions = np.mean(predictions, axis=0)

    class_indices = np.arange(5)
    expected_ratings = np.sum(avg_predictions * class_indices, axis=1)
    final_predictions = np.rint(expected_ratings).astype(int)
    final_predictions = np.clip(final_predictions, 0, 4)

    test_df = pd.read_csv(test_csv_file)
    submission_df = pd.DataFrame(
        {"id_code": test_df["id_code"], "diagnosis": final_predictions}
    )
    submission_df.to_csv(submission_file, index=False)




## === cell 10
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
submission_file = "/kaggle/working/submission.csv"
predict_ensemble(loaded_models, test_csv_file, test_root_dir, submission_file, device)
