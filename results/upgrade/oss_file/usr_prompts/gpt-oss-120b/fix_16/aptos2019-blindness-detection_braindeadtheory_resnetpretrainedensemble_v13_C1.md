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

# 5. Target score

0.8948393061819762

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.01762) has done: 'I fix the file‑path errors, add safe loading of the checkpoint files (fallback to ImageNet‑pretrained weights when a file is missing), replace the broken `tqdm_notebook` import with the standard `tqdm`, and correct the submission CSV saving path. These changes let the notebook run end‑to‑end and generate a valid `submission.csv` while preserving the original ensemble logic.'
- What this solution (achieved 0.0) has done: 'I keep the original model loading and inference pipeline unchanged and only adjust the way the ensemble logits are converted to final predictions.  
Instead of taking the arg‑max class (which can be noisy for QWK), I compute the soft‑max probabilities, calculate the expected rating as a weighted average of class indices 0‑4, round it to the nearest integer and clip to the valid range. This small calibration step usually yields predictions that better reflect the ordering of classes and should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.10849) has done: 'I replace the expected‑value rounding with a simple argmax over the soft‑max probabilities, which typically aligns better with the discrete class labels used by the Quadratic Weighted Kappa metric. This small change keeps the overall pipeline intact while nudging the predictions toward the target score.'
- What this solution (achieved 0.0) has done: 'I revert the final prediction step to use an expected‑value (soft‑average) of the class probabilities instead of a plain arg‑max. Computing the weighted mean of the probabilities (0‑4), rounding it and clipping to the valid range usually aligns better with the ordinal nature of the quadratic weighted kappa, moving the score upward from the current negative value toward the target. No other parts of the pipeline are changed.'
- What this solution (achieved 0.0) has done: 'I keep the existing data loading, model loading, and ensemble code unchanged but add a lightweight fallback predictor that learns a simple linear relationship between the average image intensity and the diagnosis using the training set. If the pretrained checkpoints are unavailable (which makes the ensemble predictions essentially random), this linear model provides sensible predictions that move the Quadratic Weighted Kappa score toward the target. The new code computes mean‑intensity features for both training and test images, fits a `LinearRegression`, and uses the rounded, clipped predictions for the submission file.'
- What this solution (achieved 0.56193) has done: 'I compute ensemble logits for the training set, fit a simple LinearRegression that maps the averaged logits to the true diagnoses, and then use this calibrated model to predict the test set from the averaged test logits (replacing the mean‑intensity baseline). This keeps the original ResNet architecture and inference unchanged while adding a lightweight calibration step that should move the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.0) has done: 'I replace the linear‑regression calibration with a direct expected‑value computation from the soft‑max of the averaged logits.  This keeps the ensemble and feature steps unchanged but uses a prediction rule that better respects the ordinal nature of the classes, which should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.4325) has done: 'I add a lightweight calibration step that fits a LinearRegression model on the averaged logits of the training set and uses it to predict the test set. This keeps the original ensemble and inference untouched while providing a more informed mapping from logits to diagnosis, which should move the Quadratic Weighted Kappa score closer to the target.'
- What this solution (achieved 0.0) has done: 'I replace the linear‑regression calibration with a direct soft‑max‑based expected‑value computation. After averaging the ensemble logits for the test set, the code now converts them to probabilities, computes the weighted average of class indices (0‑4), rounds to the nearest integer and clips to the valid range. This small change keeps the model and ensemble intact while providing predictions that better respect the ordinal nature of the target metric, moving the quadratic weighted kappa score upward toward the target.'
- What this solution (achieved 0.30137) has done: 'I add a lightweight calibration step that fits a LinearRegression model on the averaged training logits and true diagnoses, then uses this model to predict the test set from its averaged logits. This small change keeps the original ensemble and preprocessing unchanged while providing a more informed mapping from logits to the ordinal diagnosis, which should move the quadratic weighted kappa score closer to the target.'
- What this solution (achieved 0.69824) has done: 'I replace the simple averaging‑logits calibration with a richer linear‑regression that uses each model’s raw logits as separate features (30 features total). This keeps the core inference unchanged but gives the calibrator more information, which should raise the quadratic weighted kappa toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.5306) has done: 'I replace the plain LinearRegression calibrator with a Ridge regression (to reduce over‑fitting) and also compute an auxiliary prediction using the soft‑max expected value of the averaged logits. By blending the ridge‑based prediction with the expected‑value prediction (60 % ridge + 40 % expected), we make a modest but targeted adjustment that should raise the quadratic weighted kappa toward the target without altering the core model or ensemble logic.'
- What this solution (achieved 0.0) has done: 'I replace the Ridge calibrator with an unregularized LinearRegression (which often fits the logits‑to‑label mapping better) and shift the blend to rely more on the soft‑max expected‑value predictions (80 % expected, 20 % linear). These minimal tweaks keep the original ensemble and inference untouched while providing a calibration that is more likely to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.0) has done: 'The timeout is caused by loading the image dataset six times – once per model – which repeats expensive I/O and transformation work. By running all six models together on each batch we read each image only once, keep the same logits order, and therefore preserve the exact ensemble logic while cutting the runtime dramatically. The only changes are a new batched‑model inference helper and its use for both test and train predictions; all other logic, model architectures and post‑processing remain unchanged.'

# 9. Code solution

## === cell 0
import os, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

torch.backends.cudnn.benchmark = True




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

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
            label = self.eye_frame.loc[idx, "diagnosis"]
        else:
            img_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
            label = self.eye_frame.loc[idx, "id_code"]

        img_path = os.path.join(img_dir, self.eye_frame.loc[idx, "id_code"] + ".png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label




## === cell 2
test_dataset = APTOSDataset(
    csv_file="/kaggle/input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=transform,
)
test_loader = DataLoader(
    test_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 3
def load_model(resnet_type, checkpoint_path, pretrained_fallback=True):
    """Create a ResNet model, load checkpoint if it exists, otherwise fall back."""
    if resnet_type == 152:
        model = torchvision.models.resnet152(pretrained=pretrained_fallback)
    elif resnet_type == 101:
        model = torchvision.models.resnet101(pretrained=pretrained_fallback)
    else:
        raise ValueError("Unsupported ResNet type")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(
            f"Checkpoint not found: {checkpoint_path}. Using pretrained ImageNet weights."
        )
    model.to(device)
    model.eval()
    return model


base_ckpt = "/kaggle/input/resnet"  # may not exist
base_ckpt0 = "/kaggle/input/resnet0"  # may not exist

model0 = load_model(152, os.path.join(base_ckpt, "FinalResnet152_0.pt"))
model1 = load_model(101, os.path.join(base_ckpt, "FinalResnet02.pt"))
model2 = load_model(101, os.path.join(base_ckpt, "FinalResnet01.pt"))
model3 = load_model(101, os.path.join(base_ckpt, "FinalResnet00.pt"))
model4 = load_model(152, os.path.join(base_ckpt0, "pretrainedResnet151_0.pt"))
model5 = load_model(101, os.path.join(base_ckpt0, "pretrainedResnet1.pt"))

models_list = [model0, model1, model2, model3, model4, model5]




## === cell 4
def compute_all_logits(models, loader, device):
    """
    Run inference for *all* models on the same loader.
    Returns a list of tensors (one per model) and the list of image ids.
    """
    n_models = len(models)
    logits_buffers = [[] for _ in range(n_models)]
    img_ids = []

    with torch.no_grad():
        for inputs, ids in tqdm(loader, desc="Predict all models"):
            inputs = inputs.to(device)
            for i, mdl in enumerate(models):
                out = mdl(inputs)  # (B,5)
                logits_buffers[i].append(out.cpu())
            img_ids.extend(ids)

    logits_list = [torch.cat(buf, dim=0) for buf in logits_buffers]  # (N,5) each
    return logits_list, img_ids


all_logits, all_ids = compute_all_logits(models_list, test_loader, device)




## === cell 5
def extract_mean_intensity(loader):
    """Return mean pixel intensity per image and its id."""
    feats = []
    ids = []
    with torch.no_grad():
        for inputs, lbl in tqdm(loader, desc="Extract features"):
            batch_mean = inputs.mean(dim=[1, 2, 3])  # (B,)
            feats.append(batch_mean.cpu())
            ids.extend(lbl)  # keep ids for test set
    feats = torch.cat(feats).numpy().reshape(-1, 1)  # column vector
    return feats, ids


train_dataset = APTOSDataset(
    csv_file="/kaggle/input/aptos2019-blindness-detection/train.csv",
    filetype="train",
    transform=transform,
)
train_loader = DataLoader(
    train_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True
)

train_labels = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")[
    "diagnosis"
].values

train_logits, _ = compute_all_logits(models_list, train_loader, device)

train_feat = np.concatenate([lg.numpy() for lg in train_logits], axis=1)  # (N,30)
test_feat = np.concatenate([lg.numpy() for lg in all_logits], axis=1)  # (N,30)

calibrator = LinearRegression()
calibrator.fit(train_feat, train_labels)

linear_preds = calibrator.predict(test_feat)  # continuous
linear_preds = np.rint(linear_preds).astype(np.int64)
linear_preds = np.clip(linear_preds, 0, 4)

N = test_feat.shape[0]
logits_per_model = test_feat.reshape(N, 6, 5)  # 6 models, 5 classes each
avg_logits = logits_per_model.mean(axis=1)  # (N,5)

exp_logits = np.exp(avg_logits - np.max(avg_logits, axis=1, keepdims=True))
probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

class_indices = np.arange(5)
expected_vals = (probabilities * class_indices).sum(axis=1)
expected_preds = np.rint(expected_vals).astype(np.int64)
expected_preds = np.clip(expected_preds, 0, 4)

train_intensity, _ = extract_mean_intensity(train_loader)
test_intensity, _ = extract_mean_intensity(test_loader)

intensity_reg = LinearRegression()
intensity_reg.fit(train_intensity, train_labels)

intensity_preds = intensity_reg.predict(test_intensity)
intensity_preds = np.rint(intensity_preds).astype(np.int64)
intensity_preds = np.clip(intensity_preds, 0, 4)

blended_continuous = 0.1 * linear_preds + 0.1 * expected_preds + 0.8 * intensity_preds
test_preds = np.rint(blended_continuous).astype(np.int64)
test_preds = np.clip(test_preds, 0, 4)

submission = pd.DataFrame({"id_code": all_ids, "diagnosis": test_preds})
submission = submission.sort_values("id_code").reset_index(drop=True)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
