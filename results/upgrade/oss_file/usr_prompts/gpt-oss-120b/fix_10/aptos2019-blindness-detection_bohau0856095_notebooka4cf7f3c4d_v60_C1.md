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

0.9082151572788508

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.10585) has done: 'I fix the path issues, make the code run on CPU, safely handle missing weight files by falling back to pretrained EfficientNet weights, and ensure a non‑empty submission CSV is written.'
- What this solution (achieved 0.07431) has done: 'I replace the regression‑based prediction with the classifier logits, taking the arg‑max class as the final prediction. This small change keeps the model architecture unchanged while using a more appropriate output for the discrete diagnosis task, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.00141) has done: 'I adjust the weight‑loading logic to point to the correct directory (inside the dataset folder) and try a few fallback locations. This lets the model load the pretrained three‑stage weights instead of falling back to an untrained backbone, which should raise the Quadratic Weighted Kappa toward the target score while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.50799) has done: 'The changes enable parallel data loading, turn on CuDNN benchmarking, and batch the test‑time inference so the model processes many images per forward pass instead of one‑by‑one. This dramatically cuts I/O and GPU overhead while keeping the exact model architecture, training loop, and prediction logic unchanged.'

# 9. Code solution

## === cell 0
import os, random, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).squeeze()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
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
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


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
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)
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




## === cell 4
base_path = "/kaggle/input/aptos2019-blindness-detection"
test_csv_path = os.path.join(base_path, "test.csv")
test_ids = pd.read_csv(test_csv_path)["id_code"].values.squeeze()

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

net = ThreeStage_Model()
candidate_paths = [
    os.path.join(base_path, "weights", "B4_3stage_59epoch_CLAHE.pkl"),
    "/kaggle/working/aptos2019-blindness-detection/weights/B4_3stage_59epoch_CLAHE.pkl",
    "/kaggle/input/weights/B4_3stage_59epoch_CLAHE.pkl",
]
weights_path = None
for p in candidate_paths:
    if os.path.exists(p):
        weights_path = p
        break

if weights_path:
    try:
        net.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Custom weights loaded from {weights_path}.")
    except Exception as e:
        print(f"Failed to load custom weights from {weights_path}: {e}")
else:
    print("Custom weight file not found; will train a light classifier.")

net = net.to(device)




## === cell 5
if not weights_path:
    for param in net.backbone.parameters():
        param.requires_grad = False

    train_csv_path = os.path.join(base_path, "train.csv")
    train_df = pd.read_csv(train_csv_path)
    train_ids = train_df["id_code"].values
    train_labels = train_df["diagnosis"].values.astype(np.int64)

    class AptosDataset(Dataset):
        def __init__(self, ids, labels, img_dir, transform):
            self.ids = ids
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            img_id = self.ids[idx]
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, self.labels[idx]

    img_dir = os.path.join(base_path, "train_images")
    X_train, X_val, y_train, y_val = train_test_split(
        train_ids, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    train_dataset = AptosDataset(X_train, y_train, img_dir, transform)
    val_dataset = AptosDataset(X_val, y_val, img_dir, transform)

    num_workers = min(4, os.cpu_count() or 1)
    feature_loader_train = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    feature_loader_val = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    net.eval()  # ensure backbone is in eval mode
    with torch.no_grad():
        train_feats = []
        train_lbls = []
        for imgs, lbls in feature_loader_train:
            imgs = imgs.to(device, non_blocking=True)
            feats = net.backbone(imgs)  # shape (B, 1000)
            train_feats.append(feats.cpu())
            train_lbls.append(lbls)
        train_feats = torch.cat(train_feats)
        train_lbls = torch.cat(train_lbls)

        val_feats = []
        val_lbls = []
        for imgs, lbls in feature_loader_val:
            imgs = imgs.to(device, non_blocking=True)
            feats = net.backbone(imgs)
            val_feats.append(feats.cpu())
            val_lbls.append(lbls)
        val_feats = torch.cat(val_feats)
        val_lbls = torch.cat(val_lbls)

    train_feat_dataset = TensorDataset(train_feats, train_lbls)
    val_feat_dataset = TensorDataset(val_feats, val_lbls)

    train_loader = DataLoader(
        train_feat_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # small tensors, no need for extra workers
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_feat_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.classifier.parameters(), lr=1e-4)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    epochs = 8
    net.train()  # only classifier will be in train mode
    for epoch in range(epochs):
        running_loss = 0.0
        for feats, labels in train_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = net.classifier(feats)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = net.classifier(feats)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * feats.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        net.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device, non_blocking=True)
                logits = net.classifier(feats)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(labels.numpy())
        val_kappa = cohen_kappa_score(all_true, all_preds, weights="quadratic")
        print(
            f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f} - Val QWK: {val_kappa:.4f}"
        )
        net.train()
    net.eval()
else:
    net.eval()
