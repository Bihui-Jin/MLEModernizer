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

0.8960656533846909

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the model load safely even when the weight file is missing, switch to CPU when CUDA isn’t available, fix the typo‑related variable name, guard the image‑trimming function, and ensure the script always writes a non‑empty `submission.csv`. These changes remove the runtime errors while keeping the original architecture unchanged, so the resulting predictions (though from an un‑trained model) produce a valid submission file.'
- What this solution (achieved -0.07883) has done: 'I load the EfficientNet backbone with ImageNet‑pretrained weights (changing only the `pretrained` flag) so the model starts from a sensible feature extractor instead of random weights. This tiny adjustment keeps the exact architecture and training logic unchanged while giving the regression head far more meaningful predictions, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.05096) has done: 'I keep the original model and data pipeline unchanged but replace the naive rounding of the regression output with the provided `regress2class` function, which applies calibrated thresholds. This small tweak can improve the class distribution of the predictions and therefore raise the quadratic weighted kappa toward the target without altering the architecture or training logic.'
- What this solution (achieved 0.17634) has done: 'I replace the regression‑based prediction with the classifier output that the model already produces. By taking the argmax of the classifier logits we obtain a direct class prediction, which is usually more reliable than converting a single regression output with fixed thresholds. This small change keeps the architecture untouched, only adjusts the inference step, and should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.02903) has done: 'I keep the model architecture unchanged but improve inference by using the model’s final regression head together with the classifier logits and add a simple horizontal‑flip test‑time augmentation. The combined prediction (averaging the two views and then rounding) should raise the quadratic weighted kappa toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.05689) has done: 'I replace the current `predict_one` implementation with a simpler inference that only uses the classifier logits (averaged over the original and horizontally‑flipped image) and drops the noisy regression‑based rounding. This retains the model architecture and training logic, yet aligns the prediction step more closely with the intended classification output, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.19614) has done: 'I make the weight‑loading step robust by handling possible “module.” prefixes and alternative checkpoint formats, so that the fine‑tuned EfficientNet‑B4 weights are correctly loaded when they exist. This change keeps the model architecture and inference logic unchanged, but ensures the pretrained classifier/regressor heads are used, which is expected to move the quadratic weighted kappa score much closer to the target.'
- What this solution (achieved 0.09274) has done: 'I keep the model architecture and data pipeline unchanged, but improve the inference step.  
The new `predict_one` function now averages the classifier, regression, and ordinal outputs from the original and horizontally‑flipped images, converts each to a class prediction, and then selects the majority vote (with a fallback to the classifier). This modest change aligns the prediction more closely with the ordinal nature of the task and should raise the quadratic weighted kappa toward the target score while still producing a valid `submission.csv`.'
- What this solution (achieved 0.04575) has done: 'I simplify the inference step to rely only on the classifier logits (with horizontal‑flip TTA) because the regression and ordinal heads are randomly initialized when the fine‑tuned weights are missing, and their votes currently hurt performance. Using the classifier alone is a safe, minimal change that should raise the quadratic weighted kappa toward the target while keeping the model architecture unchanged.'

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
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image, ImageChops
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def trim(im):
    """Crop black borders; if no crop is found return the original image."""
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -10)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im




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
        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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
train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
train_img_dir = "../input/aptos2019-blindness-detection/train_images"
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
test_img_dir = "../input/aptos2019-blindness-detection/test_images"

input_size = 300
train_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class APTOSDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = Image.open(img_path).convert("RGB")
        img = trim(img)
        img = self.transform(img)
        label = int(row["diagnosis"])
        return img, label


net = ThreeStage_Model()
weights_path = "../input/weights/B4_3stage_75epoch.pkl"


def load_finetuned_weights(model, path):
    """Load finetuned weights, handling possible 'module.' prefixes."""
    try:
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        cleaned_state = {}
        for k, v in state_dict.items():
            cleaned_state[k.replace("module.", "")] = v

        model.load_state_dict(cleaned_state, strict=False)
        print("✅ Loaded finetuned weights successfully.")
    except Exception as e:
        print(
            f"⚠️ Could not load finetuned weights from '{path}'. Proceeding with random heads. ({e})"
        )


load_finetuned_weights(net, weights_path)
net = net.to(device)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2139067550.py in <cell line: 0>()
     45 
     46 # Model & weight loading
---> 47 net = ThreeStage_Model()
     48 weights_path = "../input/weights/B4_3stage_75epoch.pkl"
     49 

/tmp/ipykernel_55/4177756437.py in __init__(self)
     23     def __init__(self):
     24         super(ThreeStage_Model, self).__init__()
---> 25         self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
     26         self.backbone.global_pool = GeM(flatten=True)
     27 

NameError: name 'timm' is not defined

## === cell 4
net.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

full_train_ds = APTOSDataset(train_csv_path, train_img_dir, train_transform)

val_size = int(0.1 * len(full_train_ds))
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(
    full_train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

epochs = 5
best_val_acc = 0.0

for epoch in range(1, epochs + 1):
    net.train()
    epoch_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        c_out, _, _ = net(imgs)
        loss = criterion(c_out, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)

    avg_loss = epoch_loss / len(train_loader.dataset)

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            c_out, _, _ = net(imgs)
            preds = torch.argmax(c_out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total if total > 0 else 0.0
    print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), "best_finetuned.pt")

net.eval()
print("Fine‑tuning completed.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2608131563.py in <cell line: 0>()
      1 # --------- Light fine‑tuning on the training set ----------
----> 2 net.train()
      3 criterion = nn.CrossEntropyLoss()
      4 optimizer = optim.Adam(net.parameters(), lr=1e-4)
      5 

NameError: name 'net' is not defined

## === cell 5
def predict_one(img_tensor):
    """
    Return a class prediction using only the classifier logits.
    The prediction is obtained by averaging logits from the original
    and horizontally‑flipped images and taking the argmax.
    """
    with torch.no_grad():
        c_out, _, _ = net(img_tensor)
        flipped = torch.flip(img_tensor, dims=[-1])
        c_out_f, _, _ = net(flipped)

        avg_c = (c_out + c_out_f) / 2.0
        pred = torch.argmax(avg_c, dim=1).item()
    return pred


test_ids_df = pd.read_csv(test_csv_path)
test_ids = np.squeeze(test_ids_df.values)

submission = []
for i, idx in enumerate(test_ids):
    print(f"Processing {i+1}/{len(test_ids)}: {idx}")
    image_path = os.path.join(test_img_dir, f"{idx}.png")
    if not os.path.exists(image_path):
        print(f"  Image not found: {image_path}, skipping.")
        continue
    img = Image.open(image_path).convert("RGB")
    img = trim(img)
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    pred = predict_one(img_tensor)
    submission.append([idx, pred])

if len(submission) == 0:
    dummy_id = test_ids[0] if len(test_ids) > 0 else "dummy_id"
    submission.append([dummy_id, 0])

submission_array = np.array(submission)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3077431512.py in <cell line: 0>()
     29     img_tensor = test_transform(img).unsqueeze(0).to(device)
     30 
---> 31     pred = predict_one(img_tensor)
     32     submission.append([idx, pred])
     33 

/tmp/ipykernel_55/3077431512.py in predict_one(img_tensor)
      6     """
      7     with torch.no_grad():
----> 8         c_out, _, _ = net(img_tensor)
      9         flipped = torch.flip(img_tensor, dims=[-1])
     10         c_out_f, _, _ = net(flipped)

NameError: name 'net' is not defined

## === cell 6
df = pd.DataFrame(submission_array, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}, rows: {len(df)}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1288666390.py in <cell line: 0>()
----> 1 df = pd.DataFrame(submission_array, columns=["id_code", "diagnosis"])
      2 output_path = "submission.csv"
      3 df.to_csv(output_path, index=False)
      4 print(f"Submission file written to {output_path}, rows: {len(df)}")

NameError: name 'submission_array' is not defined
