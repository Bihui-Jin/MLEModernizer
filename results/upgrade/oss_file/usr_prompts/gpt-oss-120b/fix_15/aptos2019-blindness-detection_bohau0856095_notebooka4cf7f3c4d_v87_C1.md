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

0.922919661413528

# 6. Current score

0.53019

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.05431) has done: 'I fixed the pipeline so it runs on CPU when a GPU is unavailable, added a safe fallback for the missing weights file (using a pretrained EfficientNet‑B4 backbone), and wrapped the optional weight‑install step in a try/except. The inference loop now correctly extracts a scalar prediction from the tensor, and the submission CSV is always written with the required columns. These minimal changes keep the original model architecture while ensuring a valid non‑empty `submission.csv` is produced.'
- What this solution (achieved -0.16205) has done: 'I keep the model architecture unchanged and only adjust the inference step. Instead of converting the regression output to a class with the custom threshold logic, I use the classifier head’s logits, apply a soft‑max and take the arg‑max as the predicted diagnosis. This small change aligns the prediction method with the model’s trained classification output and is expected to raise the quadratic weighted kappa score toward the target while preserving the rest of the pipeline.'
- What this solution (achieved 0.17654) has done: 'I adjust the inference step to use the model’s final regression output (the final branch) and convert that scalar prediction to a rating with the existing regress2class threshold logic. This aligns the post‑processing with how the model was trained for regression, which should raise the quadratic weighted kappa toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.15859) has done: 'I keep the model architecture exactly as it is and only adjust the post‑processing of the regression output during inference.  
Instead of converting the regression value to a class with the custom `regress2class` thresholds, I clamp the output to the valid rating range [0, 4] and then round it to the nearest integer. This small change matches the way the model was trained (as a regression → rating) and is expected to raise the quadratic weighted kappa score toward the target while still producing a correct `submission.csv`.'
- What this solution (achieved 0.05085) has done: 'I adjust the inference step to use the model’s classification head rather than the combined regression output. By extracting the logits (`c_out`), applying a soft‑max and taking the arg‑max, the predicted class aligns with how the network was originally trained for classification, which should raise the quadratic weighted kappa toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -0.0438) has done: 'I keep the model architecture unchanged and only modify the inference step to combine the three heads (classification logits, regression output, ordinal probabilities) and to use a simple test‑time horizontal flip augmentation. By averaging these probability estimates before taking arg‑max we obtain a more robust prediction, which should raise the quadratic weighted kappa score toward the target without altering the core training logic.'
- What this solution (achieved 0.03248) has done: 'I modify the inference step to use the model’s final combined regression output (the `final=True` path) instead of averaging the three separate heads. This output is a single scalar in the range 0‑4.5, so I convert it to a class by rounding and clamping to [0, 4]. I also keep the horizontal‑flip test‑time augmentation and average the two scalar predictions before rounding. This change aligns the prediction method with how the network was trained for regression, which should raise the quadratic weighted kappa score toward the target while preserving the rest of the pipeline.'
- What this solution (achieved 0.0855) has done: 'I modify the inference step to use the model’s three prediction heads (classification logits, regression‑based probabilities, and ordinal probabilities) and combine their soft‑max distributions. By averaging these probabilities (and also averaging over a horizontal‑flip augmentation) we obtain a more robust class prediction, which should raise the quadratic weighted kappa toward the target while keeping the original model architecture unchanged.'
- What this solution (achieved -0.0386) has done: 'I adjust the inference step to rely only on the model’s classification head, which is the part the network was primarily trained to predict. By removing the regression‑ and ordinal‑based probabilities (which currently dilute the prediction) and averaging only the original and horizontally‑flipped classification probabilities, the predicted classes become better calibrated for the quadratic weighted kappa metric, moving the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -0.1299) has done: 'I replace the inference logic with the model’s combined regression output (`final=True`), average predictions from the original and horizontally‑flipped images, then convert the averaged scalar to a class by rounding and clamping to [0, 4]. This keeps the architecture unchanged while using the head that was trained for the final rating, which should move the quadratic weighted kappa score closer to the target.'
- What this solution (achieved -0.13468) has done: 'I keep the model architecture unchanged and only modify the inference step to use the classification logits together with the regression‑based probabilities. By converting both heads into class probability vectors, averaging them, and then taking the arg‑max, the predictions become better aligned with how the network was trained, which should move the quadratic weighted kappa score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I load the training labels to compute the most frequent diagnosis and use it as a safe fallback when the fine‑tuned checkpoint cannot be found. When the checkpoint loads successfully I simplify the inference by averaging only the classification probabilities (and their horizontal‑flip counterpart) because the regression head is unreliable without proper weights. This minimal change keeps the original architecture, fixes the missing‑weights issue, and gives a reasonable baseline that should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.73465) has done: 'I added a quick fine‑tuning step that runs when the pretrained checkpoint is not available.  
The model’s EfficientNet backbone (pre‑trained on ImageNet) is kept, but the classifier head is trained for a few epochs on the training data, after which `weights_loaded` is set to True so the original inference logic (averaging classification probabilities with a horizontal flip) is used. This modest training improves predictions without changing the overall architecture, keeping the pipeline intact and producing a valid `submission.csv`.'
- What this solution (achieved 0.53019) has done: 'I keep the existing model and training logic but improve inference by also using the regression head, converting its scalar output to a class probability distribution and averaging it with the classification probabilities (including the horizontal‑flip TTA). This adds useful signal from the regression branch and is expected to raise the quadratic weighted kappa toward the target while preserving the core pipeline.'

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
import random, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops, ImageFile
import cv2
from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """Convert regression output to integer class."""
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
    """Convert a scalar regression tensor (0‑4.5) to a soft class distribution."""
    batch_sz = out.size(0)
    prob = torch.zeros((batch_sz, 5), device=out.device)
    for i in range(batch_sz):
        val = out[i].item()
        if val >= 4.0:
            prob[i, 4] = 1.0
        else:
            l1 = int(math.floor(val))
            l2 = int(math.ceil(val))
            prob[i, l1] = 1 - (val - l1)
            prob[i, l2] = 1 - (l2 - val)
    return prob




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
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)
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




## === cell 4
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




## === cell 5
test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids_df["id_code"].astype(str).tolist()

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
mode_class = int(train_df["diagnosis"].mode()[0])

input_size = 512
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

ckpt_path = "../input/weights/B4_3stage_17epoch_finetune2_512.pkl"
weights_loaded = False
try:
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    weights_loaded = True
except Exception:
    net.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)

net = net.to(device)
net.eval()



## === cell 6
if not weights_loaded:

    class AptosDataset(Dataset):
        def __init__(self, csv_path, img_root, transform=None):
            self.df = pd.read_csv(csv_path)
            self.img_root = img_root
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = f"{self.img_root}/{row['id_code']}.png"
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = int(row["diagnosis"])
            return img, label

    train_dataset = AptosDataset(
        csv_path="../input/aptos2019-blindness-detection/train.csv",
        img_root="../input/aptos2019-blindness-detection/train_images",
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )

    net.backbone.eval()  # freeze backbone
    for param in net.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(net.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    net.train()
    epochs = 2
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            c_out, _, _ = net(imgs, final=False)  # classification logits
            loss = criterion(c_out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        print(
            f"Finetune epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(train_dataset):.4f}"
        )

    net.eval()
    weights_loaded = True  # treat fine‑tuned model as having weights



## === cell 7
submission = []


def augment_horizontal(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


net.eval()
with torch.no_grad():
    for idx in test_ids:
        if not weights_loaded:
            pred_int = mode_class
        else:
            img_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
            img = Image.open(img_path).convert("RGB")

            img_tensor = transform(img).unsqueeze(0).to(device)
            c_out, r_out, _ = net(img_tensor, final=False)
            prob_c = F.softmax(c_out, dim=1)  # classification probs
            prob_r = regress2class_prob(r_out.squeeze(1))  # regression‑derived probs

            img_flipped = augment_horizontal(img)
            img_tensor_f = transform(img_flipped).unsqueeze(0).to(device)
            c_out_f, r_out_f, _ = net(img_tensor_f, final=False)
            prob_c_f = F.softmax(c_out_f, dim=1)
            prob_r_f = regress2class_prob(r_out_f.squeeze(1))

            prob_final = (prob_c + prob_c_f + prob_r + prob_r_f) / 4.0
            pred_int = torch.argmax(prob_final, dim=1).item()

        submission.append([idx, pred_int])

submission = np.array(submission)



## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
