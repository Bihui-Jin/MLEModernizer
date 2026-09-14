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

3.9

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

0.8971545795095285

# 6. Current score

0.74198

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03288) has done: 'I fix the runtime errors by (1) making the device selection CPU‑compatible, (2) correcting the `trim` transform to always return an image, (3) using a pretrained EfficientNet backbone instead of missing weight files, and (4) adjusting the inference loop to use the classifier logits for predictions. These changes ensure the script runs end‑to‑end and writes a non‑empty `submission.csv` while preserving the original model structure.'
- What this solution (achieved 0.59274) has done: 'We keep the original model architecture but add a quick training step that extracts backbone features for the labeled training images and fits a simple multinomial LogisticRegression classifier. During inference we replace the random arg‑max of the untrained classifier with predictions from this trained LogisticRegression, which should raise the Quadratic Weighted Kappa from the negative value toward the target. The rest of the pipeline (image transforms, device handling, CSV output) remains unchanged.'
- What this solution (achieved 0.57586) has done: 'I keep the overall model and feature‑extraction pipeline unchanged but improve the downstream classifier.  
1. Train the LogisticRegression with `class_weight='balanced'` to reduce bias from the imbalanced diagnosis classes.  
2. At inference, instead of a hard arg‑max, use the predicted class probabilities to compute an expected rating and round it – this often yields predictions that better align with the quadratic weighted kappa metric.  
These minimal adjustments should raise the validation score toward the target while still producing a correct `submission.csv`.'
- What this solution (achieved 0.76719) has done: 'The changes pre‑compute the frozen backbone features once for the whole training set (and reuse them for each epoch), and enable parallel image loading with multiple workers. This eliminates the costly backbone forward pass inside every training epoch while keeping the exact same classifier architecture, loss, and optimizer, so model predictions remain unchanged. Small tweaks (setting `torch.backends.cudnn.benchmark=True` and increasing `num_workers`) further speed up data loading without affecting results.'
- What this solution (achieved 0.74198) has done: 'I add a deterministic seed, extend the classifier fine‑tuning to more epochs and include a simple learning‑rate scheduler so the linear classifier can learn a stronger mapping from the frozen EfficientNet features. These changes keep the original backbone frozen and the inference logic unchanged, but they should raise the Quadratic Weighted Kappa toward the target without altering the core model design.'

# 9. Code solution

## === cell 0
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # enable cudnn auto‑tuning for faster conv ops




## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction




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
    def __init__(self):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
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
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 3
class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        else:
            return image




## === cell 4
train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
train_img_dir = "../input/aptos2019-blindness-detection/train_images"
test_img_dir = "../input/aptos2019-blindness-detection/test_images"

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

train_ids = train_df["id_code"].values
train_labels = train_df["diagnosis"].values
test_ids = test_df["id_code"].values

input_size = 300
transform = transforms.Compose(
    [
        trim(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()
net = net.to(device)




## === cell 5
for param in net.backbone.parameters():
    param.requires_grad = False


class AptosDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        img_path = f"{self.img_dir}/{id_code}.png"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, id_code


raw_dataset = AptosDataset(train_ids, train_img_dir, transform)
raw_loader = DataLoader(
    raw_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

print("Pre‑computing backbone features for training set...")
feat_list = []
with torch.no_grad():
    for imgs, _ in raw_loader:
        imgs = imgs.to(device)
        feats = net.backbone(imgs)  # (B, 1000)
        feat_list.append(feats.cpu())
feat_tensor = torch.cat(feat_list, dim=0)
label_tensor = torch.tensor(train_labels, dtype=torch.long)

train_feat_dataset = TensorDataset(feat_tensor, label_tensor)
train_feat_loader = DataLoader(
    train_feat_dataset, batch_size=32, shuffle=True, num_workers=0
)

class_counts = np.bincount(train_labels, minlength=5)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights * (class_weights.mean())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(net.classifier.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

net.train()
print("Starting classifier fine‑tuning on pre‑computed features...")
num_epochs = 15  # extended from 5 to better capture patterns
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for feats, labels in train_feat_loader:
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = net.classifier(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(train_feat_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

net.eval()
print("Classifier fine‑tuning completed.")




## === cell 6
print("Running inference on test images...")
submission = []
batch_size = 64


def predict_batch(ids):
    imgs = []
    for idx in ids:
        img_path = f"{test_img_dir}/{idx}.png"
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        imgs.append(img)
    imgs_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = net.backbone(imgs_tensor)  # (B, 1000)
        logits = net.classifier(feats)  # (B, 5)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    expected = np.dot(probs, np.arange(5))
    preds = np.rint(expected).astype(int)
    preds = np.clip(preds, 0, 4)
    return preds


batch = []
for i, idx in enumerate(test_ids):
    batch.append(idx)
    if len(batch) == batch_size or i == len(test_ids) - 1:
        preds = predict_batch(batch)
        for pid, pred in zip(batch, preds):
            submission.append([pid, int(pred)])
        batch = []

submission = np.array(submission)




## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written.")
