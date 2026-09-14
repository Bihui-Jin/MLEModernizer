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

# 5. Code solution

## === cell 0
import random, math, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.squeeze(1)
    pred = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
    for t in threshold:
        pred += (out >= t).long()
    return pred


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
        val = out[i].item()
        if val < 4.0:
            l1 = int(math.floor(val))
            l2 = int(math.ceil(val))
            pred_prob[i, l1] = 1 - (val - l1)
            pred_prob[i, l2] = 1 - (l2 - val)
        else:
            pred_prob[i, 4] = 1.0
    return pred_prob




## === cell 2
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
    def __init__(self, pretrained_backbone=True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=pretrained_backbone, features_only=False
        )
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
            out = torch.sigmoid(out) * 4.0
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.0
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out


net = ThreeStage_Model(pretrained_backbone=True)

weight_path = "../input/weights/B4_3stage_42epoch_CLAHE.pkl"
try:
    state = torch.load(weight_path, map_location=device)
    net.load_state_dict(state)
    print("Custom weights loaded.")
except FileNotFoundError:
    print(f"Weight file not found at {weight_path}. Using pretrained backbone only.")
except Exception as e:
    print(f"Error loading weights ({e}). Using pretrained backbone only.")

for param in net.parameters():
    param.requires_grad = False
for param in net.classifier.parameters():
    param.requires_grad = True

net = net.to(device)
net.eval()



## === cell 3
input_size = 380
transform = transforms.Compose(
    [
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids_df["id_code"].values.squeeze()
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
majority_class = train_df["diagnosis"].mode().iloc[0]
print(f"Majority class from training data: {majority_class}")


def compute_mean_intensity(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    return arr.mean()


train_image_dir = "../input/aptos2019-blindness-detection/train_images"
intensities = []
labels = []
print("Computing mean intensities for the training set (brightness fallback)...")
for idx, label in zip(train_df["id_code"], train_df["diagnosis"]):
    path = f"{train_image_dir}/{idx}.png"
    try:
        mean_int = compute_mean_intensity(path)
        intensities.append(mean_int)
        labels.append(label)
    except Exception:
        continue

intensities = np.array(intensities)
labels = np.array(labels)

medians = []
for cls in range(5):
    cls_vals = intensities[labels == cls]
    medians.append(np.median(cls_vals) if len(cls_vals) > 0 else np.nan)
medians = np.array(medians)

valid = ~np.isnan(medians)
sorted_cls = np.argsort(medians[valid])
sorted_medians = medians[valid][sorted_cls]

brightness_thresholds = []
for i in range(len(sorted_medians) - 1):
    brightness_thresholds.append((sorted_medians[i] + sorted_medians[i + 1]) / 2.0)
brightness_thresholds = np.array(brightness_thresholds)

ordered_labels = np.arange(5)[valid][sorted_cls]


def brightness_to_class(mean_intensity):
    if len(brightness_thresholds) == 0:
        return majority_class
    idx = np.searchsorted(brightness_thresholds, mean_intensity, side="right")
    return (
        int(ordered_labels[idx])
        if idx < len(ordered_labels)
        else int(ordered_labels[-1])
    )




## === cell 4
class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id_code"].values
        self.labels = df["diagnosis"].values
        self.dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = f"{self.dir}/{self.ids[idx]}.png"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


train_dataset = AptosDataset(train_df, train_image_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

net.train()
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0.0
    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        c_out, _, _ = net(imgs)  # use classifier output
        loss = criterion(c_out, lbls)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)

    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

net.eval()  # return to eval mode for inference



## === cell 5
submission = []
test_image_dir = "../input/aptos2019-blindness-detection/test_images"

for i, idx in enumerate(test_ids):
    print(f"Processing {i+1}/{len(test_ids)}: {idx}")
    image_path = f"{test_image_dir}/{idx}.png"
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            c_out, _, _ = net(img_tensor)
            probs = torch.softmax(c_out, dim=1)
            pred_class = probs.argmax(dim=1).item()
    except Exception as e:
        print(f"Model inference failed for {idx} ({e}), using brightness heuristic.")
        try:
            mean_int = compute_mean_intensity(image_path)
            pred_class = brightness_to_class(mean_int)
        except Exception:
            pred_class = majority_class

    pred_class = max(0, min(4, int(pred_class)))
    submission.append([idx, pred_class])

if not submission:
    raise RuntimeError("Submission list is empty after processing all test images.")

submission = np.array(submission)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")
