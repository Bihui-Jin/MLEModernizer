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

0.9127199468129416

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The update preloads all images into memory, enables cuDNN benchmarking and mixed‑precision (AMP) to accelerate GPU work, and raises the batch size while removing unnecessary extra workers. These changes keep the exact preprocessing, model architecture, and training loop logic intact, so predictions remain identical apart from negligible floating‑point variance, and the total runtime now fits within the 600 s limit.'

# 9. Code solution

## === cell 0
try:
    import subprocess, sys, pathlib

    wheel_path = pathlib.Path("../input/weights/timm-0.3.1-py3-none-any.whl")
    if wheel_path.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", str(wheel_path)])
except Exception as e:
    print("Optional wheel install skipped:", e)




## === cell 1
import random, math, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2, os
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import timm

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """Convert regression output tensor to integer class predictions."""
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).float()
    return prediction.long()




## === cell 3
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
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
            return out.squeeze(1)
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 4
class AptosDataset(Dataset):
    """
    Pre‑load all images into memory once during initialization.
    This removes per‑item disk I/O while preserving the exact same
    preprocessing pipeline, keeping predictions unchanged.
    """

    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.targets = torch.tensor(df["diagnosis"].values, dtype=torch.float32)
        self.transform = transform

        self.images = []
        for idx in self.df["id_code"]:
            img_path = os.path.join(img_dir, f"{idx}.png")
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            self.images.append(img_tensor)
        self.images = torch.stack(self.images)  # shape (N, C, H, W)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]
        random.shuffle(distortions)
        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    factor = random.uniform(0.7, 1.3)
                image = d(image, factor)
        return image


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
        return image


input_size = 384
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)




## === cell 5
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

train_path = "../input/aptos2019-blindness-detection/train_images"
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["diagnosis"]
)

train_dataset = AptosDataset(train_split, train_path, transform)
val_dataset = AptosDataset(val_split, train_path, transform)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)

net = ThreeStage_Model().to(device)

weights_path = "../input/weights/B4_3stage_25epoch_320.pkl"
if os.path.exists(weights_path):
    try:
        net.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded external weights.")
    except Exception as e:
        print("Failed to load external weights:", e)
else:
    print("Weight file not found – training from pretrained backbone.")

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()  # mixed‑precision scaler

best_kappa = -1.0
best_state = None
for epoch in range(5):  # small number of epochs to stay within time limits
    net.train()
    epoch_losses = []
    for imgs, targets in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = net(imgs, final=True)  # regression output
            loss = criterion(preds, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_losses.append(loss.item())
    avg_train_loss = np.mean(epoch_losses)

    net.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                c_out, r_out, o_out = net(imgs)  # intermediate regression output
            pred_cls = regress2class(r_out.squeeze(1))
            all_preds.extend(pred_cls.cpu().numpy())
            all_labels.extend(targets.cpu().numpy().astype(int))
    val_kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    print(
        f"Epoch {epoch+1}: Train loss {avg_train_loss:.4f} | Val Kappa {val_kappa:.4f}"
    )

    if val_kappa > best_kappa:
        best_kappa = val_kappa
        best_state = net.state_dict()

print("Best validation Kappa:", best_kappa)
if best_state is not None:
    net.load_state_dict(best_state)




## === cell 6
test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids["id_code"].values

test_path = "../input/aptos2019-blindness-detection/test_images"

test_imgs = []
for idx in test_ids:
    img_path = os.path.join(test_path, f"{idx}.png")
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    test_imgs.append(img_tensor)
test_imgs = torch.stack(test_imgs)  # (N, C, H, W)

submission = []

net.eval()
with torch.no_grad():
    batch_sz = 32
    for start in range(0, len(test_ids), batch_sz):
        end = start + batch_sz
        batch_tensor = test_imgs[start:end].to(device)
        with torch.cuda.amp.autocast():
            _, r_out, _ = net(batch_tensor)
        preds = regress2class(r_out.squeeze(1)).cpu().numpy()
        for idx, pred in zip(test_ids[start:end], preds):
            submission.append([idx, int(pred)])
        if (end) % 100 == 0 or end == len(test_ids):
            print(f"Processed {end}/{len(test_ids)} images")

submission = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(submission), "rows.")
