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

3.9

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

0.9084729482465616

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.05355) has done: 'Implemented fixes to resolve dimension mismatches in the model and ensure proper submission handling. Updated the `ThreeStage_Model` to dynamically use the backbone’s feature size (1792) for all linear layers, preventing the runtime matrix multiplication error. Added safety checks for the submission array conversion and ensured the final CSV is written correctly. These changes allow the script to run end‑to‑end and generate a valid `submission.csv` file.'
- What this solution (achieved -0.07075) has done: 'I switch the inference to use the classifier head (`c_out`) instead of the regressor output, because the classifier directly predicts the five discrete severity classes and typically yields a higher quadratic weighted kappa. This small change keeps the core model untouched while aligning predictions with the evaluation metric, moving the score toward the target.'
- What this solution (achieved -0.03813) has done: 'I modify the inference step to use the model’s combined final regressor output (which aggregates classifier, regressor, and ordinal heads) and convert that continuous prediction to a discrete class using the existing thresholds. This small change keeps the core model unchanged while providing predictions that are better aligned with the quadratic weighted kappa metric, moving the score toward the target.'
- What this solution (achieved -0.19894) has done: 'I switch the inference to use the model’s classifier head directly (argmax of its logits) instead of the final combined regressor output, because the classifier provides discrete class predictions that align with the quadratic weighted kappa metric and should move the score upward. The rest of the pipeline stays unchanged, ensuring a valid submission file is still written.'
- What this solution (achieved 0.00808) has done: 'I keep the model architecture unchanged and only adjust the inference step to use the model’s final combined regressor output (which aggregates classifier, regressor, and ordinal heads) and then map this continuous prediction to the discrete severity classes with the existing thresholds. This change aligns predictions better with the quadratic weighted kappa metric and should raise the score toward the target.'
- What this solution (achieved 0.07487) has done: 'I switch the inference to use the model’s classifier head (`c_out`) and take the arg‑max class instead of the combined regressor output. This aligns predictions with the discrete severity labels required for the quadratic weighted kappa metric and should raise the score toward the target while keeping the core architecture unchanged.'
- What this solution (achieved -0.1059) has done: 'I keep the existing model and preprocessing unchanged but modify the inference step to use the model’s final combined regressor output (a continuous 0‑4 prediction) and convert it to a discrete class with the defined thresholds via `regress2class`. This small change aligns predictions with the evaluation metric and is expected to raise the quadratic weighted kappa score toward the target while preserving the core architecture.'
- What this solution (achieved -0.00588) has done: 'I add a lightweight validation step that fine‑tunes the regression‑to‑class thresholds using a small sampled subset of the training data and the quadratic weighted kappa metric. The script compute the best thresholds, replace the default list, and then keep the original inference flow unchanged, so predictions stay aligned with the metric while staying within the existing model architecture.'
- What this solution (achieved 0.12192) has done: 'I fixed the list‑to‑tensor error when updating the thresholds and switched inference to use the model’s classifier head (arg‑max of its logits), which aligns predictions with the discrete severity classes required for quadratic weighted kappa. These minimal changes keep the core architecture unchanged while producing a valid CSV submission and should raise the score toward the target.'
- What this solution (achieved 0.00381) has done: 'I adjust the inference step to use the model’s final regressor output (which was already calibrated on a validation subset) and convert those continuous predictions to discrete classes with the optimized thresholds. This aligns the prediction pipeline with the thresholds that gave the best QWK on the validation subset, a minimal change that should raise the score toward the target. The rest of the script remains unchanged.'
- What this solution (achieved -0.06074) has done: 'I modify the inference step to use the model’s classifier head directly (the logits) and take the arg‑max as the predicted class, instead of the final regression output and threshold conversion. This aligns predictions with the discrete severity labels required for quadratic weighted kappa and should raise the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.01959) has done: 'I switch the test‑time inference to use the model’s final regression head (the continuous 0‑4.5 output) and convert that prediction to a discrete class with the thresholds that were already optimised on the validation subset. This aligns predictions with the quadratic weighted kappa metric while keeping the core architecture unchanged, and is expected to raise the score toward the target.'

# 9. Code solution

## === cell 0
import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """
    Convert regression output (continuous 0‑4) to integer class
    using the defined thresholds.
    """
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).squeeze()
    return prediction.long()


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"


class ThreeStage_Model(nn.Module):
    def __init__(self, pretrained_backbone=True):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=pretrained_backbone, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", 1792)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )
        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), dim=1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 3
test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids_df["id_code"].values.squeeze()

input_size = 380
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model(pretrained_backbone=True)
weight_path = "../input/weights/B4_3stage_50epoch_CLAHE.pkl"
if os.path.isfile(weight_path):
    try:
        net.load_state_dict(torch.load(weight_path, map_location=device))
    except Exception as e:
        print(
            f"Warning: could not load weights ({e}); using randomly initialised model."
        )
else:
    print("Weight file not found; using ImageNet‑pretrained backbone only.")

net = net.to(device)
if hasattr(net, "backbone"):
    net.backbone = net.backbone.to(device)

net.eval()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3213748950.py in <cell line: 0>()
      5 transform = transforms.Compose(
      6     [
----> 7         trim(),
      8         cropTo4_3(),
      9         transforms.Resize((input_size * 3 // 4, input_size)),

NameError: name 'trim' is not defined

## === cell 4
if not os.path.isfile(weight_path):
    from torch.utils.data import Dataset, DataLoader

    class RetinaDataset(Dataset):
        def __init__(self, ids, labels, img_dir, transform):
            self.ids = ids
            self.labels = labels
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            img_id = self.ids[idx]
            label = self.labels[idx]
            img_path = f"{self.img_dir}/{img_id}.png"
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, label

    full_train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
    full_ids = full_train_df["id_code"].values
    full_labels = full_train_df["diagnosis"].values.astype(np.int64)

    np.random.seed(42)
    perm = np.random.permutation(len(full_ids))
    split = int(0.9 * len(full_ids))
    train_idx, val_idx = perm[:split], perm[split:]

    train_dataset = RetinaDataset(
        full_ids[train_idx],
        full_labels[train_idx],
        "../input/aptos2019-blindness-detection/train_images",
        transforms.Compose(
            [
                photometric_distort(),
                trim(),
                cropTo4_3(),
                transforms.Resize((input_size * 3 // 4, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]
                ),
            ]
        ),
    )
    val_dataset = RetinaDataset(
        full_ids[val_idx],
        full_labels[val_idx],
        "../input/aptos2019-blindness-detection/train_images",
        transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.classifier.parameters(), lr=1e-4)

    best_val_score = -np.inf
    net.train()
    for epoch in range(3):  # short fine‑tune
        epoch_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            logits, _, _ = net(imgs, final=False)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        net.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                logits, _, _ = net(imgs, final=False)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(lbls.numpy())
        val_score = cohen_kappa_score(all_true, all_preds, weights="quadratic")
        print(f"Epoch {epoch+1}: train loss {avg_loss:.4f}, val QWK {val_score:.5f}")
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = net.state_dict()
        net.train()
    net.load_state_dict(best_state)
    net.eval()
    print(f"Fine‑tuning completed. Best validation QWK: {best_val_score:.5f}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4118177336.py in <cell line: 0>()
----> 1 if not os.path.isfile(weight_path):
      2     from torch.utils.data import Dataset, DataLoader
      3 
      4     class RetinaDataset(Dataset):
      5         def __init__(self, ids, labels, img_dir, transform):

NameError: name 'weight_path' is not defined

## === cell 5
from sklearn.metrics import cohen_kappa_score

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_ids = train_df["id_code"].values
train_labels = train_df["diagnosis"].values

np.random.seed(42)
subset_size = min(1000, len(train_ids))
sample_idx = np.random.choice(len(train_ids), size=subset_size, replace=False)
sample_ids = train_ids[sample_idx]
sample_labels = train_labels[sample_idx]

preds = []
valid_labels = []

for idx, lbl in zip(sample_ids, sample_labels):
    img_path = f"../input/aptos2019-blindness-detection/train_images/{idx}.png"
    if not os.path.isfile(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = net(img_tensor, final=False)
        pred_class = torch.argmax(logits, dim=1).item()
        preds.append(pred_class)
        valid_labels.append(int(lbl))

preds = np.array(preds)
valid_labels = np.array(valid_labels)


def pred_to_class(p, th):
    """Map continuous predictions to classes using thresholds `th`."""
    th = np.asarray(th)
    cls = np.zeros_like(p, dtype=int)
    for i, t in enumerate(th):
        cls += (p >= t).astype(int)
    return cls


best_score = -np.inf
best_thr = threshold  # fallback

t1_vals = np.arange(0.0, 1.0, 0.1)
t2_vals = np.arange(1.0, 2.0, 0.1)
t3_vals = np.arange(2.0, 3.0, 0.1)
t4_vals = np.arange(3.0, 4.0, 0.1)

for t1 in t1_vals:
    for t2 in t2_vals:
        if t2 <= t1:
            continue
        for t3 in t3_vals:
            if t3 <= t2:
                continue
            for t4 in t4_vals:
                if t4 <= t3:
                    continue
                thr = [t1, t2, t3, t4]
                pred_cls = pred_to_class(preds, thr)
                score = cohen_kappa_score(valid_labels, pred_cls, weights="quadratic")
                if score > best_score:
                    best_score = score
                    best_thr = thr

print(f"Optimal thresholds on validation subset: {best_thr}")
print(f"Best QWK on validation subset: {best_score:.5f}")

threshold = best_thr



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1020644729.py in <cell line: 0>()
     19         continue
     20     img = Image.open(img_path).convert("RGB")
---> 21     img_tensor = transform(img).unsqueeze(0).to(device)
     22     with torch.no_grad():
     23         logits, _, _ = net(img_tensor, final=False)

NameError: name 'transform' is not defined

## === cell 6
submission = []
for i, idx in enumerate(test_ids):
    if i % 50 == 0:
        print(f"Processing {i}/{len(test_ids)}")
    image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}; predicting class 0")
        pred_class = 0
    else:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _, _ = net(img_tensor, final=False)
            pred_class = torch.argmax(logits, dim=1).item()
    submission.append([idx, int(pred_class)])

submission = np.array(submission)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1159173448.py in <cell line: 0>()
      9     else:
     10         img = Image.open(image_path).convert("RGB")
---> 11         img_tensor = transform(img).unsqueeze(0).to(device)
     12         with torch.no_grad():
     13             logits, _, _ = net(img_tensor, final=False)

NameError: name 'transform' is not defined

## === cell 7
if not isinstance(submission, np.ndarray) or submission.size == 0:
    dummy_id = test_ids[0] if len(test_ids) > 0 else "dummy_id"
    submission = np.array([[dummy_id, 0]])

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv with", len(df), "rows.")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
