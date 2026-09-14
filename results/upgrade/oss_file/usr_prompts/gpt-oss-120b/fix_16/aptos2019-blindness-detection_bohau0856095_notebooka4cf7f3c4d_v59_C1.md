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

0.9092099815223468

# 6. Current score

0.56788

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.18314) has done: 'I fixed the shape mismatch caused by the EfficientNet backbone output size by dynamically using the backbone’s feature dimension for all linear layers. I also corrected the submission‑empty check to work with a NumPy array. The rest of the logic remains unchanged, ensuring the script runs end‑to‑end and writes a proper `submission.csv`.'
- What this solution (achieved 0.00245) has done: 'I switch the inference to use the model’s classifier head instead of the regressor thresholds, because the classifier directly predicts the five discrete severity classes and is far more aligned with the Quadratic Weighted Kappa metric. This small change keeps the overall architecture and training untouched while providing a much better calibrated prediction, moving the score toward the target. The rest of the pipeline (data loading, transforms, and CSV writing) remains unchanged.'
- What this solution (achieved -0.21819) has done: 'I keep the model architecture unchanged and only modify the inference step so that predictions combine the classifier logits with the ordinal‑head probabilities, which are better aligned with the quadratic weighted‑kappa metric. By averaging the softmax of the classifier and the ordinal‑derived class probabilities before taking the arg‑max, the predictions become more calibrated and should move the score closer to the target. The rest of the pipeline (loading, transforms, CSV writing) remains the same.'
- What this solution (achieved -0.08042) has done: 'I adjust the inference step to rely solely on the classifier head, which is directly trained for the five severity classes and aligns best with the Quadratic Weighted Kappa metric. By removing the averaging with the ordinal‑head probabilities, the predictions become more calibrated, moving the score upward toward the target while keeping the model architecture and training unchanged.'
- What this solution (achieved -0.02182) has done: 'I tweak the inference step to combine the classifier, regressor, and ordinal predictions instead of using only the classifier logits. By averaging their probability distributions before taking the arg‑max we obtain a more calibrated class prediction, which should raise the quadratic weighted kappa toward the target while keeping the model architecture and training untouched. The rest of the pipeline stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.08072) has done: 'I make the model loading robust – if the weight file is missing we simply skip loading and keep the randomly‑initialized network. I also ensure the network is moved to the correct device before inference. Finally, I replace the final‑regression inference with the classifier head (argmax of its logits), which aligns with the 5‑class target and should improve the quadratic weighted kappa while keeping the core architecture unchanged.'
- What this solution (achieved 0.38965) has done: 'I pre‑compute the frozen backbone features once (using deterministic transforms) and train the classifier on those tensors instead of repeatedly passing every image through the heavy EfficientNet backbone each epoch. This eliminates the expensive repeated image loading and backbone forward passes while keeping the same model architecture and loss; only the data pipeline changes, preserving correctness of the learned classifier. The validation loop is similarly updated to use the pre‑computed features, and the test inference still runs the backbone on the test images as before.'
- What this solution (achieved 0.50194) has done: 'I train the classifier head longer and use class‑balanced cross‑entropy to better handle the imbalanced severity labels. Adding a small learning‑rate scheduler (step decay) keeps the same architecture and training loop while giving the model more opportunity to improve, which should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.55441) has done: 'I increase the training length to give the classifier more opportunity to learn and add a simple horizontal‑flip test‑time augmentation during inference, averaging the classifier probabilities before taking the final class. These minimal changes keep the model architecture and core training loop intact while aiming to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.58406) has done: 'I keep the overall architecture and training loop unchanged but add two small enhancements that are expected to raise the quadratic weighted‑kappa score: (1) track and restore the model weights from the epoch with the highest validation accuracy, so inference uses the best‑performing checkpoint; (2) extend test‑time augmentation by also using a vertical flip and averaging the three softmax probability maps (original, horizontal flip, vertical flip) before taking the arg‑max. These changes are minimal, preserve the core logic, and should move the current 0.55441 score closer to the target 0.9092.'
- What this solution (achieved 0.56788) has done: 'I unfreeze the last EfficientNet block so the model can fine‑tune high‑level visual features while keeping the rest of the backbone frozen. This small change adds a few trainable parameters that often improve discriminative power, especially when the classifier head is already trained on frozen features. I also raise the epoch count modestly to let the extra parameters converge. The rest of the pipeline, including data handling, loss, and test‑time augmentation, stays unchanged.'

# 9. Code solution

## === cell 0
try:
    import subprocess, sys, os

    wheel_path = "../input/weights/timm-0.3.1-py3-none-any.whl"
    if os.path.isfile(wheel_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])
except Exception:
    pass



## === cell 1
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
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from PIL import Image, ImageChops
import timm



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True



## === cell 3
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze()
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




## === cell 4
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
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()
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
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 5
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




## === cell 6
class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id_code"].values
        self.labels = df["diagnosis"].values if "diagnosis" in df.columns else None
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{id_code}.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if self.labels is not None:
            label = int(self.labels[idx])
            return image, label
        else:
            return image, id_code


input_size = 380
train_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

val_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_img_dir = "../input/aptos2019-blindness-detection/train_images"

full_dataset = AptosDataset(train_df, train_img_dir, train_transform)
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

num_workers = min(8, os.cpu_count() or 2)



## === cell 7
net = ThreeStage_Model()
weight_path = "../input/weights/B4_3stage_58epoch_CLAHE.pkl"
if os.path.isfile(weight_path):
    try:
        state_dict = torch.load(weight_path, map_location=device)
        net.load_state_dict(state_dict, strict=True)
        print("Pretrained weights loaded.")
    except Exception as e:
        print(f"Warning: failed to load weights ({e}); using random init.")
else:
    print("Weight file not found; proceeding with random initialization.")

net = net.to(device)

for param in net.backbone.parameters():
    param.requires_grad = False

if hasattr(net.backbone, "blocks"):
    for param in net.backbone.blocks[-1].parameters():
        param.requires_grad = True
else:
    for name, param in net.backbone.named_parameters():
        if "conv_head" in name:
            param.requires_grad = True

net.eval()
with torch.no_grad():

    def extract_features(dataset):
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        feats = []
        labs = []
        for imgs, labels in loader:
            imgs = imgs.to(device)
            f = net.backbone(imgs)  # frozen (mostly) backbone output
            feats.append(f.cpu())
            labs.append(labels)
        return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

    train_feats, train_labels = extract_features(train_dataset)
    val_feats, val_labels = extract_features(val_dataset)

train_feat_dataset = TensorDataset(train_feats, train_labels)
val_feat_dataset = TensorDataset(val_feats, val_labels)

train_loader = DataLoader(
    train_feat_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
val_loader = DataLoader(
    val_feat_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

class_counts = torch.bincount(train_labels)
class_weights = 1.0 / (class_counts.float() + 1e-6)
class_weights = class_weights / class_weights.sum() * len(class_counts)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 45
best_val_acc = -1.0
best_state_dict = None

net.train()
for epoch in range(epochs):
    running_loss = 0.0
    for feats, labels in train_loader:
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        c_out = net.classifier(feats)  # only the classifier head
        loss = criterion(c_out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * feats.size(0)

    epoch_loss = running_loss / train_size

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            c_out = net.classifier(feats)
            preds = torch.argmax(c_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = net.state_dict()

    scheduler.step()
    net.train()

if best_state_dict is not None:
    net.load_state_dict(best_state_dict)
    print(f"Best model restored (Val Acc = {best_val_acc:.4f})")
net.eval()



## === cell 8
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_img_dir = "../input/aptos2019-blindness-detection/test_images"

test_dataset = AptosDataset(test_df, test_img_dir, val_transform)


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    ids = [item[1] for item in batch]
    return imgs, ids


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

submission = []
net.zero_grad()
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device)

        c_out, _, _ = net(imgs, final=False)
        probs = F.softmax(c_out, dim=1)

        imgs_h = torch.flip(imgs, dims=[3])
        c_out_h, _, _ = net(imgs_h, final=False)
        probs_h = F.softmax(c_out_h, dim=1)

        imgs_v = torch.flip(imgs, dims=[2])
        c_out_v, _, _ = net(imgs_v, final=False)
        probs_v = F.softmax(c_out_v, dim=1)

        avg_probs = (probs + probs_h + probs_v) / 3.0
        preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

        for idx, pred in zip(ids, preds):
            submission.append([idx, int(pred)])

submission = np.array(submission)

if len(submission) == 0:
    raise RuntimeError("Submission array is empty; check data loading.")
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
